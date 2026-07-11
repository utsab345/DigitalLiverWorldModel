"""
Constraint enforcement and penalty computation.

MonotonicConstraint enforces F, D, P, M, S non-decreasing by construction
via post-hoc clamping on decoder output.  S normally worsens over time
but may decrease at ERCP events (when ercp_mask is provided).
"""

import torch
import torch.nn as nn

F, D, S, P, A, C, M, FL = range(8)
_MONO_IDXS = [F, D, P, M, S]


class MonotonicConstraint(nn.Module):
    def __init__(self, weight=10.0):
        super().__init__()
        self.weight = weight
        self.name = "monotonic"

    def enforce(self, x, ercp_mask=None):
        """
        Enforce monotonicity by post-hoc clamping.

        If ercp_mask is provided, S is allowed to decrease at ERCP timesteps
        (then resumes non-decreasing behaviour thereafter).
        """
        x = x.clone()
        s_orig = x[..., S].clone() if ercp_mask is not None else None
        for idx in _MONO_IDXS:
            x[..., idx] = torch.cummax(x[..., idx], dim=1)[0]
        if ercp_mask is not None:
            ercp = ercp_mask.bool()
            for b in range(x.size(0)):
                ercp_t = torch.where(ercp[b])[0]
                for t in ercp_t:
                    x[b, t, S] = s_orig[b, t]
                    x[b, t:, S] = torch.cummax(x[b, t:, S], dim=0)[0]
        return x

    def forward(self, x):
        loss = x.new_tensor(0.0)
        for idx in _MONO_IDXS:
            loss = loss + torch.relu(x[..., :-1, idx] - x[..., 1:, idx]).mean()
        return self.weight * loss
