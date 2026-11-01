from pra_core.runtime_metrics import RuntimeTelemetryProbe


def test_runtime_telemetry_keeps_short_uptime_positive(monkeypatch) -> None:
    timestamps = iter([100.0, 100.000123])
    monkeypatch.setattr(
        "pra_core.runtime_metrics.perf_counter",
        lambda: next(timestamps),
    )

    snapshot = RuntimeTelemetryProbe().snapshot(completed_request_count=1)

    assert 0 < snapshot.uptime_seconds < 0.001


def test_runtime_telemetry_uses_positive_floor_when_clock_does_not_advance(monkeypatch) -> None:
    monkeypatch.setattr("pra_core.runtime_metrics.perf_counter", lambda: 100.0)

    snapshot = RuntimeTelemetryProbe().snapshot(completed_request_count=0)

    assert snapshot.uptime_seconds == 1e-6
