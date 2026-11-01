from fastapi.testclient import TestClient

from pra_api.main import create_app
from pra_core.auth.cognito import CognitoAuthProvider
from pra_core.config import Settings


class FakeCognitoVerifier:
    def verify(self, token: str) -> dict[str, object]:
        assert token == "valid-cognito-token"
        return {"sub": "cognito-user-1", "cognito:groups": ["researcher"]}


AUTH_HEADERS = {"Authorization": "Bearer valid-cognito-token"}


def limited_app() -> TestClient:
    app = create_app(
        Settings(
            environment="test",
            document_upload_enabled=True,
            cognito_issuer="https://issuer.example.test",
            cognito_client_id="client-id",
            rate_limit_chat_requests=1,
            rate_limit_chat_window_seconds=60,
            rate_limit_upload_requests=1,
            rate_limit_upload_window_seconds=3600,
            rate_limit_public_requests=1,
            rate_limit_public_window_seconds=60,
            rate_limit_ragas_requests=1,
            rate_limit_ragas_window_seconds=3600,
            ragas_enabled=True,
        )
    )
    app.state.auth_provider = CognitoAuthProvider(
        app.state.settings,
        token_verifier=FakeCognitoVerifier(),
    )
    return TestClient(app)


def test_chat_is_rate_limited_per_authenticated_user() -> None:
    client = limited_app()

    first = client.post(
        "/chat",
        headers=AUTH_HEADERS,
        json={"message": "hello", "session_id": "rate-chat"},
    )
    second = client.post(
        "/chat",
        headers=AUTH_HEADERS,
        json={"message": "hello again", "session_id": "rate-chat"},
    )

    assert first.status_code == 200
    assert second.status_code == 429
    assert "Rate limit exceeded for /chat" in second.json()["error"]
    assert second.headers["Retry-After"] == "60"


def test_upload_is_rate_limited_per_authenticated_user() -> None:
    client = limited_app()

    first = client.post(
        "/documents/upload",
        headers=AUTH_HEADERS,
        files={"file": ("notes.md", b"alpha beta gamma delta epsilon zeta", "text/markdown")},
    )
    second = client.post(
        "/documents/upload",
        headers=AUTH_HEADERS,
        files={"file": ("notes-2.md", b"one two three four five six", "text/markdown")},
    )

    assert first.status_code == 200
    assert second.status_code == 429
    assert "Rate limit exceeded for /documents/upload" in second.json()["error"]
    assert second.headers["Retry-After"] == "3600"


def test_auth_config_is_rate_limited_by_public_client_ip() -> None:
    client = limited_app()

    first = client.get("/auth/config", headers={"x-forwarded-for": "203.0.113.10"})
    second = client.get("/auth/config", headers={"x-forwarded-for": "203.0.113.10"})

    assert first.status_code == 200
    assert second.status_code == 429
    assert "Rate limit exceeded for /auth/config" in second.json()["error"]
    assert 1 <= int(second.headers["Retry-After"]) <= 60


def test_ragas_is_rate_limited_per_authenticated_user() -> None:
    client = limited_app()

    first = client.post(
        "/rag/evaluate/ragas",
        headers=AUTH_HEADERS,
        json={"dataset": "default"},
    )
    second = client.post(
        "/rag/evaluate/ragas",
        headers=AUTH_HEADERS,
        json={"dataset": "default"},
    )

    assert first.status_code == 202
    assert second.status_code == 429
    assert "Rate limit exceeded for /rag/evaluate/ragas" in second.json()["error"]
    assert second.headers["Retry-After"] == "3600"
