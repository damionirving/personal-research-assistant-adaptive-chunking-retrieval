import shutil
import subprocess
from pathlib import Path

from fastapi.testclient import TestClient

from pra_api.main import create_app

ROOT = Path(__file__).resolve().parents[1]


def test_home_page_serves_chat_ui() -> None:
    response = TestClient(create_app()).get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Personal Research Assistant" in response.text
    assert "Sign in" in response.text
    assert "VOIP Numbers are not recommended for SMS 2FAC." in response.text
    assert 'href="static/favicon.ico"' in response.text
    assert 'href="static/favicon.svg"' in response.text
    assert 'href="static/favicon-96x96.png"' in response.text
    assert 'href="static/apple-touch-icon.png"' in response.text
    assert 'href="static/app.css?v=timestamp-font-9"' in response.text
    assert 'src="static/app.js?v=rag-eval-diagnostics-live"' in response.text
    assert 'id="new-chat-button"' in response.text
    assert 'id="session-id"' in response.text
    assert 'id="rag-eval-button"' in response.text
    assert 'id="ragas-eval-button"' in response.text
    assert 'id="stop-ragas-button"' in response.text
    assert 'id="rag-eval-result"' in response.text
    assert 'id="ragas-eval-result"' in response.text
    assert 'id="rag-eval-progress"' in response.text
    assert 'id="rag-eval-progress-label"' in response.text
    assert 'id="rag-evaluation-history"' in response.text
    assert 'id="rag-eval-jobs"' in response.text
    assert 'id="refresh-rag-jobs-button"' in response.text
    assert "Recent evaluations" in response.text


def test_static_frontend_assets_are_served() -> None:
    client = TestClient(create_app())

    page_response = client.get("/")
    css_response = client.get("/static/app.css")
    js_response = client.get("/static/app.js")
    ico_response = client.get("/static/favicon.ico")
    svg_response = client.get("/static/favicon.svg")
    png_response = client.get("/static/favicon-96x96.png")
    apple_response = client.get("/static/apple-touch-icon.png")

    assert css_response.status_code == 200
    assert ico_response.status_code == 200
    assert svg_response.status_code == 200
    assert png_response.status_code == 200
    assert apple_response.status_code == 200
    assert "text/css" in css_response.headers["content-type"]
    assert ico_response.headers["content-type"] in {
        "image/x-icon",
        "image/vnd.microsoft.icon",
    }
    assert "image/svg+xml" in svg_response.headers["content-type"]
    assert "image/png" in png_response.headers["content-type"]
    assert "image/png" in apple_response.headers["content-type"]
    assert "javascript" in js_response.headers["content-type"]
    assert "documents/upload" in js_response.text
    assert "rag/evaluate" in js_response.text
    assert "rag/evaluate/ragas" in js_response.text
    assert "auth/config" in js_response.text
    assert "You must sign in to use these features" in js_response.text
    assert "Please Sign in Again." in js_response.text
    assert "browserTimeZone" in js_response.text
    assert "formatTimestamp" in js_response.text
    assert "message-time" in js_response.text
    assert "formatSignInTimestamp" in js_response.text
    assert "response_type\", \"code\"" in js_response.text
    assert "code_challenge_method\", \"S256\"" in js_response.text
    assert "grant_type: \"authorization_code\"" in js_response.text
    assert "grant_type: \"refresh_token\"" in js_response.text
    assert "addThinkingMessage" in js_response.text
    assert "thinking-dots" in js_response.text
    assert "The model is temporarily unavailable. Please try again." in js_response.text
    assert "retryableChatStatuses" in js_response.text
    assert "new Set([429, 500, 502, 503, 504])" in js_response.text
    assert "automaticRetryableAttempts = 1" in js_response.text
    assert "automaticRetryDelayMilliseconds = 500" in js_response.text
    assert "sleepMilliseconds" in js_response.text
    assert "buildChatScoreMeta" in js_response.text
    assert "hallucination risk" in js_response.text
    assert "semantic:" in js_response.text
    assert "bm25:" in js_response.text
    assert "rrf:" in js_response.text
    assert "message-scores" in css_response.text
    assert "Try again" in js_response.text
    assert "sendChatMessage(text, { addUserMessage: false })" in js_response.text
    assert "chat/reset" in js_response.text
    assert "Active session:" in js_response.text
    assert "topic reset:" in js_response.text
    assert "New chat ready." in js_response.text
    assert "currentSessionId" in js_response.text
    assert "activeChatRequestKey" in js_response.text
    assert "That message is already being sent." in js_response.text
    assert "chat_request_started" in js_response.text
    assert "chat_request_finished" in js_response.text
    assert "buildChatDebugMeta" in js_response.text
    assert "Run RAG Performance Test" in page_response.text
    assert "Run RAGAS Test" in page_response.text
    assert "buildRagEvaluationSummary" in js_response.text
    assert "buildRagEvaluationDiagnostics" in js_response.text
    assert "diagnostic evaluation" in js_response.text
    assert "loss summary" in js_response.text
    assert "expected relevant IDs present in index" in js_response.text
    assert "buildRagEvaluationJobSummary" in js_response.text
    assert "setRagEvaluationProgress" in js_response.text
    assert "getStoredRagEvaluationJobs" in js_response.text
    assert "saveStoredRagEvaluationJobs" in js_response.text
    assert "upsertStoredRagEvaluationJob" in js_response.text
    assert "updateStoredRagEvaluationJob" in js_response.text
    assert "renderRagEvaluationJobs" in js_response.text
    assert "checkStoredRagEvaluationJob" in js_response.text
    assert "refresh-rag-jobs-button" in js_response.text
    assert "clear-rag-jobs-button" in js_response.text
    assert "clearStoredRagEvaluationJobs" in js_response.text
    assert "Saved RAG evaluation data cleared." in js_response.text
    assert "fetchRagEvaluationJob" in js_response.text
    assert "fetchRagasEvaluationJob" in js_response.text
    assert "pra_ragas_eval_jobs" in js_response.text
    assert "cancelRagasEvaluationJob" in js_response.text
    assert "Waiting for RAGAS worker capacity" in js_response.text
    assert "the evaluation retries automatically" in js_response.text
    assert "orchestration_message" in js_response.text
    assert "restoreStoredRagasEvaluationJobs" in js_response.text
    assert "activeRagasPollers" in js_response.text
    assert 'mode: "full"' in js_response.text
    assert "top_k: 25" in js_response.text
    assert "max_questions: 12" in js_response.text
    assert "formatApiError" in js_response.text
    assert "trace_id:" in js_response.text
    assert "average retrieval latency" in js_response.text
    assert "trace:" in js_response.text
    assert "top chunk:" in js_response.text
    assert "retrievedChunkLabels" in js_response.text
    assert "message-debug" in css_response.text
    assert "progress-panel" in css_response.text
    assert "progress-track" in css_response.text
    assert "progress-bar" in css_response.text
    assert "progress-slide" in css_response.text
    assert "recent-panel" in css_response.text
    assert "recent-jobs" in css_response.text
    assert "recent-job" in css_response.text
    assert "/logout" in js_response.text
    assert "chat" in js_response.text


def test_root_favicon_is_served() -> None:
    response = TestClient(create_app()).get("/favicon.ico")

    assert response.status_code == 200
    assert "image/x-icon" in response.headers["content-type"]


def test_frontend_javascript_has_valid_syntax() -> None:
    node = shutil.which("node")
    if not node:
        return

    result = subprocess.run(
        [node, "--check", str(ROOT / "apps/api/src/pra_api/static/app.js")],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_auth_config_endpoint_serves_cognito_settings() -> None:
    from pra_core.config import Settings

    app = create_app(
        Settings(
            environment="test",
            cognito_client_id="client-id",
            cognito_hosted_ui_base_url="https://auth.example.test",
        )
    )
    response = TestClient(app).get("/auth/config")

    assert response.status_code == 200
    assert response.json()["provider"] == "cognito"
    assert response.json()["cognito_client_id"] == "client-id"
    assert response.json()["document_upload_enabled"] is True
    assert response.json()["retrieval_diagnostics_enabled"] is True
    assert response.json()["rag_evaluation_enabled"] is True
    assert response.json()["ragas_enabled"] is True


def test_auth_config_reports_disabled_evaluation_features() -> None:
    from pra_core.config import Settings

    app = create_app(
        Settings(
            environment="test",
            rag_evaluation_enabled=False,
            ragas_enabled=False,
        )
    )
    response = TestClient(app).get("/auth/config")

    assert response.status_code == 200
    assert response.json()["rag_evaluation_enabled"] is False
    assert response.json()["ragas_enabled"] is False


def test_auth_config_reports_disabled_document_upload() -> None:
    from pra_core.config import Settings

    app = create_app(Settings(environment="test", document_upload_enabled=False))
    response = TestClient(app).get("/auth/config")

    assert response.status_code == 200
    assert response.json()["document_upload_enabled"] is False


def test_auth_config_reports_enabled_retrieval_diagnostics() -> None:
    from pra_core.config import Settings

    app = create_app(Settings(environment="test", retrieval_diagnostics_enabled=True))
    response = TestClient(app).get("/auth/config")

    assert response.status_code == 200
    assert response.json()["retrieval_diagnostics_enabled"] is True


def test_auth_config_supports_allowed_cors_origin() -> None:
    from pra_core.config import Settings

    app = create_app(Settings(environment="test"))
    response = TestClient(app).get(
        "/auth/config",
        headers={"Origin": "http://localhost:5000"},
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5000"


def test_auth_config_preflight_rejects_unlisted_origin() -> None:
    from pra_core.config import Settings

    app = create_app(Settings(environment="test"))
    response = TestClient(app).options(
        "/auth/config",
        headers={
            "Origin": "https://example.com",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers


def test_static_assets_are_served_with_configured_root_path_prefix() -> None:
    from pra_core.config import Settings

    app = create_app(Settings(environment="test", api_root_path="/prod"))
    client = TestClient(app)

    response = client.get("/prod/static/app.css")

    assert response.status_code == 200
    assert "text/css" in response.headers["content-type"]


def test_static_assets_are_served_when_stage_prefix_is_stripped() -> None:
    from pra_core.config import Settings

    app = create_app(Settings(environment="test", api_root_path="/prod"))
    client = TestClient(app)

    response = client.get("/static/app.css")

    assert response.status_code == 200
    assert "text/css" in response.headers["content-type"]


def test_openapi_uses_configured_root_path() -> None:
    from pra_core.config import Settings

    app = create_app(Settings(environment="test", api_root_path="/prod"))

    assert app.root_path == "/prod"
