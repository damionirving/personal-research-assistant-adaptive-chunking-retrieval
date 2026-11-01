from dataclasses import dataclass
from threading import Lock


@dataclass(frozen=True)
class MetricsSnapshot:
    request_count: int
    error_count: int
    total_latency_ms: float
    llm_call_count: int

    @property
    def average_latency_ms(self) -> float:
        if self.request_count == 0:
            return 0.0
        return self.total_latency_ms / self.request_count


class MetricsRegistry:
    """Small in-memory metrics registry for local and Lambda runtime metrics."""

    def __init__(self) -> None:
        self._request_count = 0
        self._error_count = 0
        self._total_latency_ms = 0.0
        self._llm_call_count = 0
        self._lock = Lock()

    def record_request(self, latency_ms: float, is_error: bool) -> None:
        with self._lock:
            self._request_count += 1
            self._total_latency_ms += latency_ms
            if is_error:
                self._error_count += 1

    def record_llm_call(self, count: int = 1) -> None:
        with self._lock:
            self._llm_call_count += count

    def snapshot(self) -> MetricsSnapshot:
        with self._lock:
            return MetricsSnapshot(
                request_count=self._request_count,
                error_count=self._error_count,
                total_latency_ms=self._total_latency_ms,
                llm_call_count=self._llm_call_count,
            )
