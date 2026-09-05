"""
Generate comparison figures: predictions vs ground truth for each feature.
"""

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from config import Config
from generator import SyntheticGenerator
from dataset import make_loaders
from models.encoder import Encoder
from models.predictor import Predictor
from models.decoder import Decoder
from models.constraints import MonotonicConstraint
from models.world_model import WorldModel
from utils import FIELD_NAMES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = Config()
cfg.epochs = 1

gen = SyntheticGenerator(seed=cfg.seed)
trajectories, contexts, context_tensors = gen.generate_dataset(n=cfg.n_patients, n_steps=cfg.n_steps)
_, _, _, _, val_ds, _ = make_loaders(
    trajectories, contexts, context_tensors,
    batch_size=cfg.batch_size, train_frac=cfg.train_frac,
    context_length=cfg.context_length, forecast_horizon=cfg.forecast_horizon,
)

encoder = Encoder(latent_dim=cfg.latent_dim, hidden_dim=cfg.hidden_dim,
                  num_heads=cfg.num_heads).to(device)
dynamics = Predictor(latent_dim=cfg.latent_dim, hidden_dim=cfg.hidden_dim).to(device)
decoder = Decoder(latent_dim=cfg.latent_dim).to(device)
model = WorldModel(encoder, dynamics, decoder,
                   constraints=nn.ModuleList([MonotonicConstraint(weight=cfg.monotonic_weight)]),
                   forecast_horizon=cfg.forecast_horizon).to(device)

ckpt = torch.load(Path(cfg.checkpoint_dir) / "model.pt", map_location=device)
model.load_state_dict(ckpt["model"])
model.eval()

out_dir = Path(cfg.figure_dir)
out_dir.mkdir(parents=True, exist_ok=True)

n_samples_grid = min(5, len(val_ds))
n_samples_individual = min(10, len(val_ds))
n_total = max(n_samples_grid, n_samples_individual)

t_ctx = torch.arange(cfg.context_length)
t_fc = torch.arange(cfg.context_length, cfg.context_length + cfg.forecast_horizon)

predictions = []
for s in range(n_total):
    sample = val_ds[s]
    ctx = sample["context"].unsqueeze(0).to(device)
    treatment = sample["treatment"].unsqueeze(0).to(device)
    ercp = sample["ercp"].unsqueeze(0).to(device)
    context_tensor = sample.get("context_tensor")
    if context_tensor is not None:
        context_tensor = context_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(ctx, context=context_tensor, treatment=treatment, ercp=ercp)
        pred = out["x_pred_raw"].squeeze(0)[-cfg.forecast_horizon:]
        pred_enf = out["x_pred"].squeeze(0)[-cfg.forecast_horizon:]

    predictions.append({
        "context": sample["context"],
        "target": sample["target"],
        "pred": pred.cpu(),
        "pred_enf": pred_enf.cpu(),
    })

fig, axes = plt.subplots(8, n_samples_grid, figsize=(4 * n_samples_grid, 20))
if n_samples_grid == 1:
    axes = axes.reshape(-1, 1)

for s in range(n_samples_grid):
    p = predictions[s]
    for f in range(8):
        ax = axes[f, s]
        ax.plot(t_ctx, p["context"][:, f], "b-", label="context", linewidth=1.5)
        ax.plot(t_fc, p["target"][:, f], "g--", label="target", linewidth=1.5)
        ax.plot(t_fc, p["pred"][:, f], "r-", label="raw pred", alpha=0.8, linewidth=1.5)
        ax.plot(t_fc, p["pred_enf"][:, f], "k:", label="enforced", linewidth=2, alpha=0.9)
        ax.set_ylabel(FIELD_NAMES[f], fontsize=9)
        if f == 0:
            ax.set_title(f"Patient {s}", fontsize=10)
        if f == 7:
            ax.set_xlabel("Month")
        ax.grid(True, alpha=0.3)
        if s == n_samples_grid - 1 and f == 0:
            ax.legend(fontsize=7, loc="upper left")

fig.suptitle("Predictions vs Ground Truth  (context + 12-month forecast)", fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig(out_dir / "comparison_grid.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved comparison_grid.png")

for s in range(n_samples_individual):
    p = predictions[s]
    fig2, axes2 = plt.subplots(8, 1, figsize=(10, 16))

    for f in range(8):
        ax = axes2[f]
        ax.plot(t_ctx, p["context"][:, f], "b-", label="context", linewidth=1.5)
        ax.plot(t_fc, p["target"][:, f], "g--", label="target", linewidth=1.5)
        ax.plot(t_fc, p["pred"][:, f], "r-", label="raw pred", alpha=0.8, linewidth=1.5)
        ax.plot(t_fc, p["pred_enf"][:, f], "k:", label="enforced", linewidth=2, alpha=0.9)
        ax.set_ylabel(FIELD_NAMES[f], fontsize=10)
        ax.set_xlabel("Month")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
        ax.set_title(f"{FIELD_NAMES[f]}", fontsize=10)

    fig2.suptitle(f"Patient {s} — 12-month forecast", fontsize=14)
    plt.tight_layout()
    fig2.savefig(out_dir / f"patient_{s}.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved patient_{s}.png")

print(f"\nAll figures saved to {out_dir.resolve()}")
