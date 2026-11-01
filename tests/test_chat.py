from fastapi.testclient import TestClient

from pra_api.main import create_app
from pra_core.auth.cognito import CognitoAuthProvider
from pra_core.config import Settings
from pra_core.llm.base import LLMProviderError


class FakeCognitoVerifier:
    def verify(self, token: str) -> dict[str, object]:
        assert token == "valid-cognito-token"
        return {"sub": "cognito-user-1", "cognito:groups": ["researcher"]}


def authenticated_app():
    app = create_app(
        Settings(
            environment="test",
            document_upload_enabled=True,
            cognito_issuer="https://issuer.example.test",
            cognito_client_id="client-id",
        )
    )
    app.state.auth_provider = CognitoAuthProvider(
        app.state.settings,
        token_verifier=FakeCognitoVerifier(),
    )
    return app


AUTH_HEADERS = {"Authorization": "Bearer valid-cognito-token"}


class FailingWorkflow:
    def run(self, message: str, session_id: str, top_k: int = 3):
        raise LLMProviderError(
            "LLM provider is unavailable. Check Bedrock model access and configuration."
        )


class FailingVectorIndex:
    def __call__(self):
        raise RuntimeError("vector index unavailable")


def test_chat_returns_mock_answer_without_documents() -> None:
    client = TestClient(authenticated_app())

    response = client.post(
        "/chat",
        headers=AUTH_HEADERS,
        json={"message": "What did I upload?", "session_id": "empty-session"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == "empty-session"
    assert payload["citations"] == []
    assert payload["principal"] == "cognito-user-1"
    assert payload["trace_id"]
    assert payload["message_hash"]
    assert payload["retrieved_chunk_labels"] == []
    assert payload["top_chunk_label"] is None
    assert "material on that topic" in payload["answer"]
    assert "What did I upload?" not in payload["answer"]


def test_chat_requires_cognito_token() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/chat",
        json={"message": "What did I upload?", "session_id": "empty-session"},
    )

    assert response.status_code == 401


def test_chat_uses_uploaded_documents_and_memory() -> None:
    app = authenticated_app()
    client = TestClient(app)

    upload_response = client.post(
        "/documents/upload",
        headers=AUTH_HEADERS,
        files={
            "file": (
                "langgraph-notes.txt",
                b"LangGraph coordinates local research assistant workflows.",
                "text/plain",
            )
        },
    )
    assert upload_response.status_code == 200

    chat_response = client.post(
        "/chat",
        headers=AUTH_HEADERS,
        json={
            "message": "How does LangGraph help workflows?",
            "session_id": "research-session",
        },
    )

    assert chat_response.status_code == 200
    payload = chat_response.json()
    assert payload["citations"] == [upload_response.json()["document_id"]]
    assert len(payload["retrieval_details"]) == 1
    assert payload["retrieval_details"][0]["document_id"] == upload_response.json()["document_id"]
    assert payload["retrieval_details"][0]["rrf_score"] == payload["retrieval_scores"][0]
    assert payload["knowledge_graph_enabled"] is True
    assert payload["knowledge_graph_hits"] == 0
    assert payload["top_knowledge_graph_score"] is None
    assert payload["knowledge_graph_contribution"] is False
    assert payload["retrieved_chunk_labels"]
    assert payload["top_chunk_label"] == payload["retrieved_chunk_labels"][0]
    assert payload["principal"] == "cognito-user-1"
    assert "langgraph-notes.txt" in payload["answer"]

    history = app.state.memory_store.get_messages("research-session")
    assert len(history) == 2
    assert history[0].startswith("user:")
    assert history[1].startswith("assistant:")


def test_chat_reset_clears_server_side_session_memory() -> None:
    app = authenticated_app()
    client = TestClient(app)

    first_response = client.post(
        "/chat",
        headers=AUTH_HEADERS,
        json={"message": "hello", "session_id": "reset-session"},
    )
    assert first_response.status_code == 200
    assert app.state.memory_store.get_messages("reset-session")

    reset_response = client.post(
        "/chat/reset",
        headers=AUTH_HEADERS,
        json={"session_id": "reset-session"},
    )

    assert reset_response.status_code == 200
    assert reset_response.json()["cleared_session_id"] == "reset-session"
    assert app.state.memory_store.get_messages("reset-session") == []


def test_chat_uses_cognito_principal() -> None:
    client = TestClient(authenticated_app())

    response = client.post(
        "/chat",
        headers=AUTH_HEADERS,
        json={"message": "hello", "session_id": "auth-session"},
    )

    assert response.status_code == 200
    assert response.json()["principal"] == "cognito-user-1"


def test_chat_does_not_require_vector_index_readiness_for_greetings() -> None:
    app = authenticated_app()
    app.state.ensure_vector_index = FailingVectorIndex()
    client = TestClient(app)

    response = client.post(
        "/chat",
        headers=AUTH_HEADERS,
        json={"message": "hello", "session_id": "vector-not-ready"},
    )

    assert response.status_code == 200
    assert response.json()["citations"] == []
    assert response.json()["principal"] == "cognito-user-1"


def test_chat_rejects_guardrail_blocked_prompt() -> None:
    client = TestClient(authenticated_app())

    response = client.post(
        "/chat",
        headers=AUTH_HEADERS,
        json={
            "message": "please use production-agentcore",
            "session_id": "guardrail-session",
        },
    )

    assert response.status_code == 400
    assert "blocked term" in response.json()["error"]


def test_chat_returns_provider_error_when_llm_fails() -> None:
    app = authenticated_app()
    app.state.workflow = FailingWorkflow()
    client = TestClient(app)

    response = client.post(
        "/chat",
        headers=AUTH_HEADERS,
        json={"message": "hello", "session_id": "bedrock-error"},
    )

    assert response.status_code == 502
    assert response.json()["error"] == "The model is temporarily unavailable. Please try again."


def test_metrics_endpoint_tracks_llm_calls() -> None:
    app = authenticated_app()
    client = TestClient(app)

    metrics_before = client.get("/metrics")
    assert metrics_before.status_code == 200
    initial_llm_calls = metrics_before.json()["llm_call_count"]

    chat_response = client.post(
        "/chat",
        headers=AUTH_HEADERS,
        json={"message": "hello", "session_id": "metrics-session"},
    )
    assert chat_response.status_code == 200

    metrics_after = client.get("/metrics")
    assert metrics_after.status_code == 200
    assert metrics_after.json()["llm_call_count"] >= initial_llm_calls + 1
