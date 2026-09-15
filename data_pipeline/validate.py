"""Schema and privacy checks for approved clinical trajectory adapters."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def validate_trajectories(rows: list[list[float]], dimensions: int = 8) -> None:
    if not rows or any(len(row) != dimensions for row in rows):
        raise ValueError(f"expected non-empty rows with exactly {dimensions} features")
    if any(value != value or abs(value) == float("inf") for row in rows for value in row):
        raise ValueError("trajectory contains NaN or infinite values")


def write_manifest(path: str | Path, source: str, row_count: int, feature_schema: list[str]) -> None:
    payload = {"source": source, "row_count": row_count, "feature_schema": feature_schema}
    payload["content_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
