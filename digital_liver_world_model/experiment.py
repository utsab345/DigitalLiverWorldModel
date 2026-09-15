"""Reproducible experiment metadata and optional MLflow integration."""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from config import Config


def set_deterministic_seed(seed: int) -> None:
    """Seed every local RNG and request deterministic PyTorch kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


class ExperimentLogger:
    """Write an always-available JSONL log and mirror it to MLflow when installed."""

    def __init__(self, cfg: Config, run_name: str = "digital-liver") -> None:
        self.path = Path(cfg.log_dir) / "metrics.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._mlflow = None
        if os.getenv("MLFLOW_TRACKING_URI"):
            try:
                import mlflow  # type: ignore
                self._mlflow = mlflow
                mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "digital-liver"))
                mlflow.start_run(run_name=run_name)
                mlflow.log_params(asdict(cfg))
            except ImportError:
                pass

    def log(self, step: int, metrics: dict[str, float]) -> None:
        record: dict[str, Any] = {"step": step, **metrics}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if self._mlflow is not None:
            self._mlflow.log_metrics(metrics, step=step)

    def close(self) -> None:
        if self._mlflow is not None:
            self._mlflow.end_run()
