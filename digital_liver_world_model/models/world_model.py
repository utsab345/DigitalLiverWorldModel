"""
World model: encoder -> predictor (dynamics) -> decoder with optional constraints.

Exposes latents and decoded outputs so the trainer can compute
JEPA, VICReg, reconstruction, and constraint losses externally.
"""

import torch
import torch.nn as nn


class WorldModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        dynamics: nn.Module,
        decoder: nn.Module,
        constraints: nn.ModuleList | None = None,
        forecast_horizon: int = 0,
    ):
        super().__init__()
        assert forecast_horizon >= 0, f"forecast_horizon must be >= 0, got {forecast_horizon}"
        self.encoder = encoder
        self.dynamics = dynamics
        self.decoder = decoder
        self.constraints = constraints or nn.ModuleList()
        self.forecast_horizon = forecast_horizon

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        treatment: torch.Tensor | None = None,
        ercp: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            x: Observed clinical state (B, T, 8).
            context: Patient metadata (B, 5) — age, sex, responder, disease_class, udca_start.
            treatment: UDCA treatment mask (B, T+FH).
            ercp: ERCP procedure mask (B, T+FH).

        Returns:
            dict with keys:
                z:      Context latents (B, T, D).
                z_pred: Predicted latents (B, T+FH, D).
                x_pred: Decoded clinical states (B, T+FH, 8),
                        with hard constraints enforced.
        """
        z = self.encoder(x, context)
        z_pred, _ = self.dynamics(
            z, treatment=treatment, ercp=ercp, n_pred=self.forecast_horizon,
        )
        x_pred = self.decoder(z_pred)
        x_pred_raw = x_pred
        for c in self.constraints:
            if hasattr(c, "enforce"):
                x_pred = c.enforce(x_pred, ercp)
        return {
            "z": z,
            "z_pred": z_pred,
            "x_pred": x_pred,
            "x_pred_raw": x_pred_raw,
        }

    def encode(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.encoder(x, context)

    def predict_latent(
        self,
        z: torch.Tensor,
        treatment: torch.Tensor | None = None,
        ercp: torch.Tensor | None = None,
        target_z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z_pred, _ = self.dynamics(
            z, treatment=treatment, ercp=ercp,
            n_pred=self.forecast_horizon, target_z=target_z,
        )
        return z_pred

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def constraint_loss(self, x_pred: torch.Tensor) -> dict[str, torch.Tensor]:
        return {c.name: c(x_pred) for c in self.constraints}

    def total_constraint_loss(self, x_pred: torch.Tensor) -> torch.Tensor:
        losses = self.constraint_loss(x_pred)
        if not losses:
            return x_pred.new_tensor(0.0)
        return sum(losses.values())
