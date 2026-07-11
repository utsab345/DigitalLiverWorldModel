"""
Training loop for the Digital Liver World Model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from config import Config
from generator import SyntheticGenerator
from dataset import make_loaders
from models.encoder import Encoder
from models.decoder import Decoder
from models.predictor import Predictor, ProjectionHead
from models.ema import EMA
from models.constraints import MonotonicConstraint
from models.world_model import WorldModel
from losses import JEPALoss, ReconstructionLoss
from evaluate import evaluate
from explain import explain_sample
from utils import set_seed, plot_training_snapshot


def train(cfg: Config):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    gen = SyntheticGenerator(seed=cfg.seed)
    trajectories, contexts, context_tensors = gen.generate_dataset(n=cfg.n_patients, n_steps=cfg.n_steps)
    print(f"Generated {trajectories.shape[0]} trajectories of length {trajectories.shape[1]}")

    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = make_loaders(
        trajectories, contexts, context_tensors, cfg.batch_size, cfg.train_frac,
        context_length=cfg.context_length, forecast_horizon=cfg.forecast_horizon,
        stride=cfg.stride, pin_memory=cfg.pin_memory, num_workers=cfg.num_workers,
    )
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}, Test samples: {len(test_ds)}")

    encoder = Encoder(latent_dim=cfg.latent_dim, hidden_dim=cfg.hidden_dim,
                      num_heads=cfg.num_heads).to(device)
    target_ema = EMA(encoder, momentum=cfg.target_momentum)
    dynamics = Predictor(latent_dim=cfg.latent_dim, hidden_dim=cfg.hidden_dim).to(device)
    decoder = Decoder(latent_dim=cfg.latent_dim).to(device)
    projection = ProjectionHead(input_dim=cfg.latent_dim, hidden_dim=cfg.predictor_hidden).to(device)
    constraints = nn.ModuleList()
    if cfg.enforce_monotonic:
        constraints.append(MonotonicConstraint(weight=cfg.monotonic_weight))
    model = WorldModel(encoder, dynamics, decoder, constraints,
                       forecast_horizon=cfg.forecast_horizon).to(device)

    jepa_loss_fn = JEPALoss(std_target=cfg.std_target,
                            variance_weight=cfg.variance_weight,
                            cov_weight=cfg.cov_weight)
    recon_loss_fn = ReconstructionLoss()
    opt = optim.AdamW(
        list(model.parameters()) + list(projection.parameters()),
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_epoch = 0
    for epoch in range(cfg.epochs):
        model.train()
        total_loss, total_jepa, total_recon, total_c = 0.0, 0.0, 0.0, 0.0
        for batch in train_loader:
            ctx = batch["context"].to(device)
            target = batch["target"].to(device)
            treatment = batch["treatment"].to(device)
            ercp = batch["ercp"].to(device)
            context_tensor = batch.get("context_tensor")
            if context_tensor is not None:
                context_tensor = context_tensor.to(device)
            opt.zero_grad()

            z = model.encode(ctx, context_tensor)
            with torch.no_grad():
                z_target = target_ema.target(target, context_tensor)

            # Teacher-forced rollout during training
            z_pred = model.predict_latent(
                z, treatment=treatment, ercp=ercp, target_z=z_target,
            )
            x_pred = model.decode(z_pred)

            # Constraint penalty on raw predictions (before hard enforcement)
            c_losses = model.constraint_loss(x_pred)
            c_loss = sum(c_losses.values()) if c_losses else ctx.new_tensor(0.0)

            fh = target.size(1)

            # Reconstruction on raw predictions (provides decoder gradients)
            recon_loss = recon_loss_fn(x_pred[:, -fh:], target)

            # Hard constraint enforcement (gradient-free projection, matches inference)
            for c in model.constraints:
                if hasattr(c, "enforce"):
                    with torch.no_grad():
                        x_pred = c.enforce(x_pred, ercp)

            p_pred = projection(z_pred[:, -fh:])
            with torch.no_grad():
                p_target = projection(z_target)
            jepa_loss = jepa_loss_fn(p_pred, p_target)

            loss = cfg.jepa_weight * jepa_loss + cfg.recon_weight * recon_loss + c_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(projection.parameters()), cfg.grad_clip)
            opt.step()
            target_ema.update()

            total_loss += loss.item()
            total_jepa += jepa_loss.item()
            total_recon += recon_loss.item()
            total_c += c_loss.item() if isinstance(c_loss, torch.Tensor) else c_loss

        sched.step()

        if (epoch + 1) % 10 == 0:
            val_loss = 0.0
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    ctx = batch["context"].to(device)
                    target = batch["target"].to(device)
                    treatment = batch["treatment"].to(device)
                    ercp = batch["ercp"].to(device)
                    context_tensor = batch.get("context_tensor")
                    if context_tensor is not None:
                        context_tensor = context_tensor.to(device)
                    out = model(ctx, context=context_tensor,
                                treatment=treatment, ercp=ercp)
                    fh = target.size(1)

                    recon_loss = recon_loss_fn(out["x_pred"][:, -fh:], target)

                    z_target = target_ema.target(target, context_tensor)
                    p_pred = projection(out["z_pred"][:, -fh:])
                    p_target = projection(z_target)
                    jepa_loss = jepa_loss_fn(p_pred, p_target)

                    c_losses = model.constraint_loss(out["x_pred_raw"])
                    c_loss = sum(c_losses.values()) if c_losses else torch.tensor(0.0, device=device)

                    loss = cfg.jepa_weight * jepa_loss + cfg.recon_weight * recon_loss + c_loss
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            n = len(train_loader)
            print(f"Epoch {epoch+1}/{cfg.epochs}  train={total_loss/n:.4f}  "
                  f"j={total_jepa/n:.4f} r={total_recon/n:.4f} c={total_c/n:.4f}  "
                  f"val={val_loss:.4f}")

            plot_training_snapshot(model, val_ds, cfg.context_length,
                                   cfg.forecast_horizon, epoch + 1,
                                   Path(cfg.figure_dir) / "training",
                                   device, n_samples=5)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                ckpt = {
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "projection": projection.state_dict(),
                    "target_ema": target_ema.target.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(),
                    "val_loss": val_loss,
                }
                torch.save(ckpt, Path(cfg.checkpoint_dir) / "model.pt")

            if epoch - best_epoch >= cfg.patience:
                print(f"Early stopping at epoch {epoch+1} "
                      f"(no improvement for {cfg.patience} epochs)")
                break

    print("\nTraining complete. Loading best checkpoint for evaluation...")
    ckpt = torch.load(Path(cfg.checkpoint_dir) / "model.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    projection.load_state_dict(ckpt["projection"])
    target_ema.target.load_state_dict(ckpt["target_ema"])

    results = evaluate(model, test_ds, gen, cfg, device,
                       target_encoder=target_ema.target, projection=projection)
    for k, v in results.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\nExplainability sample:")
    sample = test_ds[0]
    ctx = sample["context"].unsqueeze(0).to(device)
    ctx_t = sample.get("context_tensor")
    if ctx_t is not None:
        ctx_t = ctx_t.unsqueeze(0).to(device)
    model.train()
    attrs = explain_sample(model, ctx, context=ctx_t)
    print(f"  Feature attributions at last timestep: "
          f"{attrs[-1, :].abs().argsort(descending=True).tolist()}")

    return model, results


if __name__ == "__main__":
    cfg = Config()
    train(cfg)
