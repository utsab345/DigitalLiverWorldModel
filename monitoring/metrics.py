"""Prometheus metrics definitions (optional dependency at runtime)."""
try:
    from prometheus_client import Counter, Histogram
    REQUESTS = Counter("liver_requests_total", "Inference requests")
    ERRORS = Counter("liver_errors_total", "Inference errors")
    LATENCY = Histogram("liver_request_latency_seconds", "Inference latency")
except ImportError:  # pragma: no cover
    REQUESTS = ERRORS = LATENCY = None
