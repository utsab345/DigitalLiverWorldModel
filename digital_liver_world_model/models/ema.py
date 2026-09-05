"""
Exponential Moving Average target encoder for JEPA.

Maintains a slow-moving copy of the online encoder. The target encoder
is updated via EMA and receives no gradients, providing stable targets
for the JEPA prediction loss.
"""

import copy
import torch
import torch.nn as nn


class EMA:
    def __init__(self, model: nn.Module, momentum: float = 0.996):
        assert 0.0 <= momentum < 1.0, f"momentum must be in [0, 1), got {momentum}"
        self.model = model
        self.target = copy.deepcopy(model)
        self.target.eval()
        self.momentum = momentum
        for p in self.target.parameters():
            p.requires_grad = False

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """Alias for self.target(...) so the EMA wrapper can be used like the encoder."""
        return self.target(*args, **kwargs)

    @torch.no_grad()
    def update(self):
        """
        Update target encoder parameters via EMA:
            θ_target ← momentum * θ_target + (1 - momentum) * θ_online
        """
        for p, p_t in zip(self.model.parameters(), self.target.parameters()):
            p_t.data = self.momentum * p_t.data + (1 - self.momentum) * p.data
        for b, b_t in zip(self.model.buffers(), self.target.buffers()):
            b_t.copy_(b)
