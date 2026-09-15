"""Celery boundary for long-running predictions; API remains responsive."""
from celery import Celery
import os

celery_app = Celery("digital_liver", broker=os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0"))


@celery_app.task(name="digital_liver.predict")
def predict_async(payload: dict) -> dict:
    from api.main import _predict
    return _predict(payload)
