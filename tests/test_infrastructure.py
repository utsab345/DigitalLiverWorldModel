import torch

from digital_liver_world_model.models.constraints import MonotonicConstraint
from digital_liver_world_model.telemetry import RequestMetrics


def test_constraint_projection_preserves_monotonic_dimensions() -> None:
    values = torch.tensor([[[0.2, 0.1, 0.4, 0.0, 0.0, 0.0, 0.2, 0.0],
                           [0.1, 0.3, 0.2, 0.4, 0.0, 0.0, 0.1, 0.0]]])
    projected = MonotonicConstraint().enforce(values, None)
    assert torch.all(projected[:, 1:, 0] >= projected[:, :-1, 0])
    assert torch.all(projected[:, 1:, 1] >= projected[:, :-1, 1])


def test_request_metrics_snapshot() -> None:
    metrics = RequestMetrics()
    metrics.start("r1")
    metrics.finish("r1", violations=2)
    snapshot = metrics.snapshot()
    assert snapshot["requests"] == 1
    assert snapshot["constraint_violations"] == 2
