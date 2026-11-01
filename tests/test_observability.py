from fastapi import FastAPI
from fastapi.testclient import TestClient

from pra_api.main import create_app


def test_trace_id_header_is_propagated() -> None:
    client = TestClient(create_app())

    response = client.get("/health", headers={"x-trace-id": "trace-from-test"})

    assert response.status_code == 200
    assert response.headers["x-trace-id"] == "trace-from-test"


def test_metrics_endpoint_reports_requests() -> None:
    client = TestClient(create_app())

    client.get("/health")
    response = client.get("/metrics")

    assert response.status_code == 200
    payload = response.json()
    assert payload["request_count"] >= 1
    assert payload["error_count"] == 0
    assert payload["average_latency_ms"] >= 0


def test_error_handler_returns_trace_id_for_validation_error() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/chat",
        headers={"x-trace-id": "bad-chat"},
        json={"message": "", "session_id": "local"},
    )

    assert response.status_code == 422
    assert response.json()["trace_id"] == "bad-chat"
    assert response.headers["x-trace-id"] == "bad-chat"


def test_error_handler_returns_trace_id_for_unhandled_error() -> None:
    app = create_app()

    @app.get("/boom")
    def boom() -> None:
        raise RuntimeError("explode")

    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/boom", headers={"x-trace-id": "boom-trace"})

    assert response.status_code == 500
    assert response.json() == {
        "error": "Internal server error.",
        "trace_id": "boom-trace",
    }


def test_metrics_records_server_errors() -> None:
    app: FastAPI = create_app()

    @app.get("/boom")
    def boom() -> None:
        raise RuntimeError("explode")

    client = TestClient(app, raise_server_exceptions=False)
    client.get("/boom")

    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.json()["error_count"] == 1
