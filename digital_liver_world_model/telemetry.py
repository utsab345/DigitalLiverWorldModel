"""Small in-process telemetry primitives suitable for an inference API."""

from dataclasses import dataclass, field
import time


@dataclass
class RequestMetrics:
    requests: int = 0
    errors: int = 0
    total_latency_ms: float = 0.0
    constraint_violations: int = 0
    _started: dict[str, float] = field(default_factory=dict, repr=False)

    def start(self, request_id: str) -> None:
        self._started[request_id] = time.perf_counter()
        self.requests += 1

    def finish(self, request_id: str, error: bool = False, violations: int = 0) -> None:
        started = self._started.pop(request_id, None)
        if started is not None:
            self.total_latency_ms += (time.perf_counter() - started) * 1000
        self.errors += int(error)
        self.constraint_violations += violations

    def snapshot(self) -> dict[str, float | int]:
        return {
            "requests": self.requests,
            "errors": self.errors,
            "avg_latency_ms": self.total_latency_ms / self.requests if self.requests else 0.0,
            "constraint_violations": self.constraint_violations,
        }
