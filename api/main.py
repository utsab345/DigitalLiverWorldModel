"""Production-facing API boundary for trajectory prediction.

The model implementation is injected through ``PREDICTOR_MODULE`` so this
service can be tested and deployed without coupling transport to training.
"""
from __future__ import annotations

import importlib
import os
import time
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from digital_liver_world_model.telemetry import RequestMetrics

app = FastAPI(title="Digital Liver World Model API", version="0.2.0")
metrics = RequestMetrics()


class PredictRequest(BaseModel):
    patient_id: str | None = None
    observation_window: list[list[float]] = Field(..., min_length=1)
    horizon: int = Field(12, ge=1, le=120)


class CounterfactualRequest(PredictRequest):
    intervention_schedule: list[dict[str, Any]] = Field(default_factory=list)


def _predict(payload: dict[str, Any]) -> dict[str, Any]:
    module_name = os.getenv("PREDICTOR_MODULE")
    if not module_name:
        raise RuntimeError("PREDICTOR_MODULE is not configured")
    predictor = importlib.import_module(module_name)
    return predictor.predict(payload)


def _dump(model: BaseModel) -> dict[str, Any]:
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "model_version": os.getenv("MODEL_VERSION", "unknown")}


@app.get("/metrics")
def get_metrics() -> dict[str, float | int]:
    return metrics.snapshot()


@app.post("/predict")
def predict(request: PredictRequest) -> dict[str, Any]:
    request_id = str(uuid4())
    metrics.start(request_id)
    started = time.perf_counter()
    try:
        result = _predict(_dump(request))
        result.setdefault("latency_ms", (time.perf_counter() - started) * 1000)
        return result
    except Exception as exc:
        metrics.finish(request_id, error=True)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        if request_id in metrics._started:
            metrics.finish(request_id)


@app.post("/counterfactual")
def counterfactual(request: CounterfactualRequest) -> dict[str, Any]:
    payload = _dump(request)
    payload["counterfactual"] = True
    return _predict(payload)
