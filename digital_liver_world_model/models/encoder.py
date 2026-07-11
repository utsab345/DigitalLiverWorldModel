"""
Graph-Attention Encoder: attends over the 8 clinical features using
a causal graph mask that constrains information flow along biological edges.

Edges: A->F, C->F, F->M, C->M, F->P, C->P, A->flare, C->flare + self-loops.

Integrates patient context (age, sex, responder, disease_class, udca_start)
via a separate context MLP branch fused with the state embedding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.1):
        super().__init__()
        assert out_dim % num_heads == 0, f"out_dim ({out_dim}) not divisible by num_heads ({num_heads})"
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(in_dim, out_dim * 3, bias=False)
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, N, D = x.shape
        H = self.num_heads
        C = self.head_dim
        qkv = self.qkv(x).reshape(B, N, 3, H, C).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return self.out_proj(out)


class Encoder(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64, latent_dim=32,
                 num_layers=2, num_heads=4, dropout=0.1, context_dim=5):
        super().__init__()
        self.node_proj = nn.Linear(1, hidden_dim)
        self.layers = nn.ModuleList()
        prev = hidden_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else latent_dim
            self.layers.append(GATLayer(prev, out_dim, num_heads, dropout))
            prev = out_dim
        self.norm = nn.LayerNorm(latent_dim)
        self.register_buffer("causal_mask", self._build_mask())

        self.context_mlp = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.fusion = nn.Linear(latent_dim * 2, latent_dim)
        self.pool_attn = nn.Linear(latent_dim, 1)

    def _build_mask(self):
        m = torch.zeros(8, 8)
        edges = [(4, 0), (5, 0), (0, 6), (5, 6), (0, 3), (5, 3),
                 (4, 7), (5, 7)]
        for i in range(8):
            edges.append((i, i))
        for src, tgt in edges:
            m[tgt, src] = 1
        return m.bool()

    def forward(self, x, context=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        B, T, _ = x.shape

        x = x.unsqueeze(-1)
        x = self.node_proj(x)
        x = x.reshape(B * T, 8, -1)

        for layer in self.layers:
            residual = x
            x = layer(x, self.causal_mask)
            x = F.gelu(x)
            if x.shape[-1] == residual.shape[-1]:
                x = x + residual

        x = self.norm(x)

        scores = self.pool_attn(x)
        weights = F.softmax(scores, dim=1)
        x = (x * weights).sum(dim=1)
        x = x.reshape(B, T, -1)

        if context is not None:
            z_ctx = self.context_mlp(context)
            z_ctx = z_ctx.unsqueeze(1).expand(-1, T, -1)
            x = self.fusion(torch.cat([x, z_ctx], dim=-1))

        return x
