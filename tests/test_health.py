from fastapi.testclient import TestClient

from pra_api.main import create_app
from pra_core.config import Settings


def test_health_endpoint_returns_service_metadata() -> None:
    app = create_app(
        Settings(
            environment="test",
            service_name="personal-research-assistant",
            service_version="0.1.0",
            log_format="console",
        )
    )

    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "personal-research-assistant",
        "version": "0.1.0",
        "environment": "test",
    }

