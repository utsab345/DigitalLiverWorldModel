"""
Evaluation harness: accuracy, constraint-violation rate, generalization probes.
"""

import torch
import torch.nn.functional as Func
from torch.utils.data import DataLoader

F, D, S, P, A, C, M, FL = range(8)


def constraint_violations(x, ercp=None):
    """Count per-field monotonicity and bound violations.

    Args:
        x: Single trajectory (T, 8).
        ercp: Optional ERCP mask (T,) — S decreases at ERCP positions
              are not counted as violations.

    Returns:
        (violations, total_checks) tuple.
    """
    violations = 0
    total = 0
    for idx in [F, D, P, M, S]:
        if idx == S and ercp is not None:
            mask = ~ercp.bool()
            v = (x[1:, idx] < x[:-1, idx] - 1e-6)[mask[:-1] & mask[1:]].sum().item()
        else:
            v = (x[1:, idx] < x[:-1, idx] - 1e-6).sum().item()
        violations += v
        total += x.size(0) - 1
    eps = 1e-6
    oob = ((x[..., :7] < -eps) | (x[..., :7] > 1 + eps)).sum().item() + (x[..., M] > 2 + eps).sum().item()
    violations += oob
    total += x.numel()
    return violations, total


@torch.no_grad()
def _latent_stats(z_all: torch.Tensor) -> dict[str, float]:
    """Compute representation quality metrics from a (N, T, D) latent batch."""
    z_2d = z_all.reshape(-1, z_all.size(-1))
    std = z_2d.std(dim=0).mean().item()
    z_centered = z_2d - z_2d.mean(dim=0, keepdim=True)
    cov = (z_centered.T @ z_centered) / (z_centered.size(0) - 1)
    off_diag = cov - torch.diag(cov.diag())
    n_off = off_diag.numel()
    cov_off = off_diag.abs().sum().item() / n_off if n_off > 0 else 0.0
    return {"latent_std": std, "latent_cov_offdiag": cov_off}


@torch.no_grad()
def evaluate(model, val_ds, generator, cfg, device, target_encoder=None, projection=None):
    model.eval()

    loader = DataLoader(val_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                        pin_memory=cfg.pin_memory)

    mae_raw = 0.0
    mae_enf = 0.0
    v_raw, t_raw = 0, 0
    v_enf, t_enf = 0, 0
    n = 0
    z_all = []
    latent_sim = 0.0
    n_sim = 0

    for batch in loader:
        ctx = batch["context"].to(device)
        target = batch["target"].to(device)
        context_tensor = batch.get("context_tensor")
        if context_tensor is not None:
            context_tensor = context_tensor.to(device)
        treatment = batch.get("treatment")
        if treatment is not None:
            treatment = treatment.to(device)
        ercp = batch.get("ercp")
        if ercp is not None:
            ercp = ercp.to(device)

        out = model(ctx, context=context_tensor, treatment=treatment, ercp=ercp)
        z_all.append(out["z"].cpu())
        pred_raw = out["x_pred_raw"][:, cfg.context_length:]
        pred = out["x_pred"][:, cfg.context_length:]

        mae_raw += Func.l1_loss(pred_raw, target, reduction="sum").item()
        n += target.numel()

        if target_encoder is not None and projection is not None:
            z_target = target_encoder(target, context_tensor)
            z_pred = out["z_pred"][:, -cfg.forecast_horizon:]
            p_pred = projection(z_pred)
            p_target = projection(z_target)
            sim = Func.cosine_similarity(p_pred, p_target, dim=-1).mean().item()
            latent_sim += sim * target.size(0)
            n_sim += target.size(0)

        for b in range(pred.size(0)):
            pred_single = pred_raw[b]
            ercp_single = ercp[b, cfg.context_length:] if ercp is not None else None

            vi, ti = constraint_violations(pred_single, ercp=ercp_single)
            v_raw += vi
            t_raw += ti

            mae_enf += Func.l1_loss(pred[b], target[b], reduction="sum").item()
            vi2, ti2 = constraint_violations(pred[b], ercp=ercp_single)
            v_enf += vi2
            t_enf += ti2

    results = {
        "mae": mae_raw / n,
        "mae_enforced": mae_enf / n,
        "violation_rate_raw": v_raw / max(t_raw, 1),
        "violation_rate_enforced": v_enf / max(t_enf, 1),
        "latent_cosine_sim": latent_sim / max(n_sim, 1),
    }
    if z_all:
        z_cat = torch.cat(z_all, dim=0)
        results.update(_latent_stats(z_cat))

    # Generalization probes
    gen = _generalization_probes(model, generator, cfg, device)
    results.update(gen)

    return results


@torch.no_grad()
def _make_treatment_ercp(udca_start, ercp_months, total_len):
    """Build treatment and ERCP masks for the first total_len steps."""
    treatment = torch.zeros(total_len)
    treatment[udca_start:] = 1.0
    ercp = torch.zeros(total_len)
    for m in ercp_months:
        if m < total_len:
            ercp[m] = 1.0
    return treatment, ercp


@torch.no_grad()
def _generalization_probes(model, generator, cfg, device):
    results = {}
    sus_configs = [(0.9, "high_sus"), (0.1, "low_sus")]
    for sus_val, label in sus_configs:
        params = generator.sample_params(20)
        params["susceptibility"] = torch.full((20,), sus_val)
        num_cases = 20
        mae = 0.0
        n = 0
        for i in range(num_cases):
            p = {k: v[i] for k, v in params.items()}
            udca_start = int(torch.randint(6, 36, (1,)).item())
            n_ercp = int(torch.randint(0, 4, (1,)).item())
            ercp_months = sorted([
                int(torch.randint(12, cfg.n_steps - 12, (1,)).item())
                for _ in range(n_ercp)
            ])
            traj = generator.generate(
                n_steps=cfg.n_steps, params=p,
                udca_start=udca_start, ercp_months=ercp_months,
            ).to(device)
            ctx = traj[:cfg.context_length].unsqueeze(0)
            target = traj[cfg.context_length:cfg.context_length + cfg.forecast_horizon].unsqueeze(0)
            total_len = cfg.context_length + cfg.forecast_horizon
            context_tensor = torch.tensor(
                [p["age"].item(), p["sex"].item(), p["responder"].item(),
                 p["disease_class"].item(), float(udca_start)],
                dtype=torch.float32, device=device,
            ).unsqueeze(0)
            treatment, ercp = _make_treatment_ercp(udca_start, ercp_months, total_len)
            treatment = treatment.unsqueeze(0).to(device)
            ercp = ercp.unsqueeze(0).to(device)
            out = model(ctx, context=context_tensor, treatment=treatment, ercp=ercp)
            pred = out["x_pred"][:, cfg.context_length:].squeeze(0)
            mae += Func.l1_loss(pred, target.squeeze(0), reduction="sum").item()
            n += target.numel()
        results[f"gen_{label}_mae"] = mae / n

    # Longer rollouts (24-step horizon)
    num_cases = 20
    mae_24 = 0.0
    n_elem = 0
    params = generator.sample_params(num_cases)
    for i in range(num_cases):
        p = {k: v[i] for k, v in params.items()}
        udca_start = int(torch.randint(6, 36, (1,)).item())
        n_ercp = int(torch.randint(0, 4, (1,)).item())
        ercp_months = sorted([
            int(torch.randint(12, cfg.n_steps - 12, (1,)).item())
            for _ in range(n_ercp)
        ])
        traj = generator.generate(
            n_steps=cfg.n_steps, params=p,
            udca_start=udca_start, ercp_months=ercp_months,
        ).to(device)
        ctx = traj[:cfg.context_length].unsqueeze(0)
        target = traj[cfg.context_length:cfg.context_length + 24].unsqueeze(0)
        total_len = cfg.context_length + 24
        context_tensor = torch.tensor(
            [p["age"].item(), p["sex"].item(), p["responder"].item(),
             p["disease_class"].item(), float(udca_start)],
            dtype=torch.float32, device=device,
        ).unsqueeze(0)
        treatment, ercp = _make_treatment_ercp(udca_start, ercp_months, total_len)
        treatment = treatment.unsqueeze(0).to(device)
        ercp = ercp.unsqueeze(0).to(device)
        z = model.encode(ctx, context=context_tensor)
        z_pred, _ = model.dynamics(z, n_pred=24, treatment=treatment, ercp=ercp)
        x_pred = model.decode(z_pred)
        for c in model.constraints:
            if hasattr(c, "enforce"):
                x_pred = c.enforce(x_pred, ercp_mask=ercp)
        pred = x_pred[:, cfg.context_length:].squeeze(0)
        mae_24 += Func.l1_loss(pred, target.squeeze(0), reduction="sum").item()
        drift = Func.l1_loss(pred[-1:], target.squeeze(0)[-1:], reduction="sum").item()
        results.setdefault("rollout_drift", 0.0)
        results["rollout_drift"] += drift
        n_elem += target.numel()
    results["gen_rollout_24_mae"] = mae_24 / n_elem
    results["rollout_drift"] /= num_cases

    return results
