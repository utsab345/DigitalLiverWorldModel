"""
Clinical decoder: maps latent -> 8-D clinical state vector.
Outputs are bounded by construction: sigmoid -> [0,1], M scaled to [0,2].
"""

import torch
import torch.nn as nn

M_IDX = 6


class Decoder(nn.Module):
    def __init__(self, latent_dim=32, hidden_dims=(64, 64), output_dim=8, dropout=0.1):
        super().__init__()
        layers = []
        prev = latent_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
        scale = torch.ones(output_dim)
        scale[M_IDX] = 2.0
        self.register_buffer("scale", scale)

    def forward(self, z):
        raw = self.net(z)
        out = torch.sigmoid(raw)
        return out * self.scale
