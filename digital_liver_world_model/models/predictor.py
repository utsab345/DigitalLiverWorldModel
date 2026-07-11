"""
GRU-based world model predictor with treatment conditioning.
Replaces the old MLP predictor with a recurrent dynamics model.

The Predictor models temporal evolution of latents, conditioned on
treatment/intervention signals (UDCA, ERCP). It supports teacher-forced
training and autoregressive rollout during inference.

The ProjectionHead is a small MLP used to project both predicted and
target latents into a shared space for the JEPA loss comparison.
"""

import torch
import torch.nn as nn


class Predictor(nn.Module):
    """
    Recurrent world model predictor for JEPA.

    Args:
        latent_dim: Dimension of the latent space (input and output).
        hidden_dim: Hidden dimension of the GRU.
        num_layers: Number of GRU layers.
        dropout: Dropout applied between GRU layers (if num_layers > 1).
    """
    def __init__(self, latent_dim=64, hidden_dim=128, num_layers=2, dropout=0.1):
        super().__init__()
        tx_hidden = hidden_dim // 4
        self.tx_proj = nn.Sequential(
            nn.Linear(2, tx_hidden),
            nn.GELU(),
        )
        self.gru = nn.GRU(
            latent_dim + tx_hidden, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.out_proj = nn.Linear(hidden_dim, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, z, treatment=None, ercp=None, hidden=None, n_pred=0, target_z=None):
        """
        Args:
            z: Context latents (B, T, D).
            treatment: UDCA treatment mask (B, T+FH) or (B, T).
            ercp: ERCP procedure mask (B, T+FH) or (B, T).
            hidden: Initial hidden state for the GRU.
            n_pred: Number of autoregressive prediction steps beyond T.
            target_z: Ground-truth future latents (B, FH, D) for teacher forcing.

        Returns:
            z_out: Predicted latents (B, T + n_pred, D).
            hidden: Final hidden state of the GRU (post-rollout if applicable).
        """
        B, T, D = z.shape
        device = z.device

        # Build (B, T, 2) intervention signal from treatment + ERCP
        tx = torch.zeros(B, T, 2, device=device)
        if treatment is not None:
            tx[..., 0] = treatment[:, :T].float()
        if ercp is not None:
            tx[..., 1] = ercp[:, :T].float()
        tx_emb = self.tx_proj(tx)

        gru_in = torch.cat([z, tx_emb], dim=-1)
        z_out, hidden = self.gru(gru_in, hidden)
        z_out = self.norm(z + self.out_proj(z_out))

        if n_pred > 0:
            preds = []
            h = hidden
            step_inp = z_out[:, -1:]
            for step in range(n_pred):
                # Teacher forcing: use target_z when available during training
                if target_z is not None and step < target_z.size(1):
                    step_inp = target_z[:, step:step + 1]

                step_idx = T + step
                tx_step = torch.zeros(B, 1, 2, device=device)
                if treatment is not None and treatment.size(-1) > step_idx:
                    tx_step[..., 0] = treatment[:, step_idx:step_idx + 1].float()
                elif treatment is not None:
                    tx_step[..., 0] = treatment[:, -1:].float()
                if ercp is not None and ercp.size(-1) > step_idx:
                    tx_step[..., 1] = ercp[:, step_idx:step_idx + 1].float()
                elif ercp is not None:
                    tx_step[..., 1] = ercp[:, -1:].float()
                tx_step_emb = self.tx_proj(tx_step)

                out, h = self.gru(torch.cat([step_inp, tx_step_emb], dim=-1), h)
                step_inp = self.norm(step_inp + self.out_proj(out))
                preds.append(step_inp)
            z_out = torch.cat([z_out] + preds, dim=1)
            hidden = h  # return post-rollout hidden state

        return z_out, hidden


class ProjectionHead(nn.Module):
    """
    MLP projection head for JEPA loss comparison.

    Maps latents into a representation space where the JEPA loss
    (Smooth L1 + VICReg regularizers) is computed. Both predicted
    and target latents are passed through the same ProjectionHead
    to obtain comparable representations.
    """
    def __init__(self, input_dim=64, hidden_dim=128, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, z):
        return self.net(z)
