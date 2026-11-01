from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter, process_time

try:
    import resource
except ImportError:  # pragma: no cover - not available on all platforms
    resource = None


@dataclass(frozen=True)
class RuntimeTelemetrySnapshot:
    uptime_seconds: float
    throughput_requests_per_minute: float
    cpu_time_seconds: float
    memory_rss_mb: float | None


class RuntimeTelemetryProbe:
    """Deterministic process-level telemetry for local runs and Lambda containers."""

    def __init__(self) -> None:
        self._started_at = perf_counter()

    def snapshot(self, completed_request_count: int) -> RuntimeTelemetrySnapshot:
        uptime_seconds = max(perf_counter() - self._started_at, 1e-6)
        requests_including_current = max(completed_request_count, 0)
        throughput_requests_per_minute = (
            requests_including_current / (uptime_seconds / 60.0)
            if uptime_seconds > 0
            else 0.0
        )
        return RuntimeTelemetrySnapshot(
            # Uptime is consumed by workflow telemetry, where short-lived
            # requests must still report a positive duration.
            uptime_seconds=uptime_seconds,
            throughput_requests_per_minute=round(throughput_requests_per_minute, 2),
            cpu_time_seconds=round(process_time(), 4),
            memory_rss_mb=_current_memory_rss_mb(),
        )


def _current_memory_rss_mb() -> float | None:
    if resource is None:
        return None

    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss = float(usage.ru_maxrss)
    # Linux reports kilobytes, macOS reports bytes.
    if rss > 1_000_000:
        return round(rss / (1024 * 1024), 2)
    return round(rss / 1024, 2)
