import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FIELD_NAMES = ["F", "D", "S", "P", "A", "C", "M", "flare"]
F, D, S, P, A, C, M, FL = range(8)

Y_LIMITS: list[tuple[float, float]] = [
    (-0.05, 1.05),
    (-0.05, 1.05),
    (-0.05, 1.05),
    (-0.05, 1.05),
    (-0.05, 1.05),
    (-0.05, 1.05),
    (-0.05, 2.05),
    (-0.05, 1.05),
]


def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _plot_layout(figsize=(12, 10)):
    fig, axes = plt.subplots(4, 2, figsize=figsize, sharex=True)
    axes = axes.flatten()
    for i, name in enumerate(FIELD_NAMES):
        ax = axes[i]
        ax.set_ylabel(name)
        ax.set_ylim(Y_LIMITS[i])
        ax.grid(True, alpha=0.25)
        if i >= 6:
            ax.set_xlabel("Month")
    return fig, axes


def plot_trajectory(traj, title=None, path=None):
    fig, axes = _plot_layout()
    for i, name in enumerate(FIELD_NAMES):
        ax = axes[i]
        ax.plot(to_numpy(traj[:, i]))
        ax.axhline(0, color="gray", ls="--", alpha=0.3)
        if name != "M":
            ax.axhline(1, color="gray", ls="--", alpha=0.3)
    fig.legend(["trajectory"], loc="upper right", fontsize=9)
    if title:
        fig.suptitle(title)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        plt.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_comparison(pred, target, path=None):
    fig, axes = _plot_layout()
    for i, name in enumerate(FIELD_NAMES):
        ax = axes[i]
        ax.plot(to_numpy(target[:, i]), label="target", color="blue")
        ax.plot(to_numpy(pred[:, i]), label="pred", color="red", ls="--")
    fig.legend(["target", "pred"], loc="upper right", fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_training_snapshot(model, val_ds, context_length, forecast_horizon,
                           epoch, out_dir, device, n_samples=5):
    """Generate a comparison grid during training showing model progress.

    Plots context (blue), ground truth (green), raw prediction (red),
    and constraint-enforced prediction (black dotted) for n_samples patients.
    """
    import torch
    model.eval()
    n_samples = min(n_samples, len(val_ds))
    fig, axes = plt.subplots(8, n_samples, figsize=(4 * n_samples, 20))
    if n_samples == 1:
        axes = axes.reshape(-1, 1)

    t_ctx = torch.arange(context_length)
    t_fc = torch.arange(context_length, context_length + forecast_horizon)

    for s in range(n_samples):
        sample = val_ds[s]
        ctx = sample["context"].unsqueeze(0).to(device)
        treatment = sample["treatment"].unsqueeze(0).to(device)
        ercp = sample["ercp"].unsqueeze(0).to(device)
        context_tensor = sample.get("context_tensor")
        if context_tensor is not None:
            context_tensor = context_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(ctx, context=context_tensor, treatment=treatment, ercp=ercp)
            pred = out["x_pred_raw"].squeeze(0)[-forecast_horizon:]
            pred_enf = out["x_pred"].squeeze(0)[-forecast_horizon:]

        for f in range(8):
            ax = axes[f, s]
            ax.plot(t_ctx, sample["context"][:, f], "b-", linewidth=1.5)
            ax.plot(t_fc, sample["target"][:, f], "g--", linewidth=1.5)
            ax.plot(t_fc, pred[:, f].cpu(), "r-", alpha=0.8, linewidth=1.5)
            ax.plot(t_fc, pred_enf[:, f].cpu(), "k:", linewidth=2, alpha=0.9)
            ax.set_ylabel(FIELD_NAMES[f], fontsize=9)
            if f == 0:
                ax.set_title(f"Patient {s}", fontsize=10)
            if f == 7:
                ax.set_xlabel("Month")
            ax.grid(True, alpha=0.3)
            if s == n_samples - 1 and f == 0:
                ax.legend(["context", "target", "raw pred", "enforced"],
                          fontsize=7, loc="upper left")

    fig.suptitle(f"Training Progress — Epoch {epoch}", fontsize=14, y=1.01)
    plt.tight_layout()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path / f"train_epoch_{epoch:04d}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    model.train()
