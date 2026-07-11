"""
Loss functions: JEPA (latent prediction), reconstruction, constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class JEPALoss(nn.Module):
    def __init__(self, std_target=1.0, eps=1e-4, variance_weight=1.0, cov_weight=1.0):
        super().__init__()
        self.std_target = std_target
        self.eps = eps
        self.variance_weight = variance_weight
        self.cov_weight = cov_weight

    def forward(self, z_pred, z_target, mask=None):
        z_target = z_target.detach()

        loss = F.smooth_l1_loss(z_pred, z_target, reduction="none")
        if mask is not None:
            loss = loss.mean(dim=-1)
            loss = (loss * mask).sum() / (mask.sum() + self.eps)
        else:
            loss = loss.mean()

        B, T, D = z_pred.shape
        z = z_pred.reshape(-1, D)

        std = torch.sqrt(z.var(dim=0) + self.eps)
        var_penalty = torch.relu(self.std_target - std).mean()

        z = z - z.mean(dim=0)
        denom = max(z.size(0) - 1, 1)
        cov = (z.T @ z) / denom
        off_diag = cov - torch.diag(cov.diag())
        cov_penalty = off_diag.pow(2).mean()

        return loss + self.variance_weight * var_penalty + self.cov_weight * cov_penalty


class ReconstructionLoss(nn.Module):
    def forward(self, x_pred, x_target):
        return F.mse_loss(x_pred, x_target)
