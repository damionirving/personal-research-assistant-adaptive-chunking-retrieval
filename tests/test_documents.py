from fastapi.testclient import TestClient
import pytest

from pra_api.main import create_app
from pra_core.auth.cognito import CognitoAuthProvider
from pra_core.config import Settings
from pra_core.documents.parser import DocumentParser, UnsupportedDocumentTypeError
from pra_core.vectorstores.faiss import FaissVectorStore


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


class FailingIndexer:
    def index(self, document) -> int:
        raise RuntimeError("vector backend unavailable")


def test_document_parser_accepts_text_document() -> None:
    parsed = DocumentParser().parse(
        content=b"LangGraph helps orchestrate agent workflows.",
        filename="notes.txt",
        content_type="text/plain",
    )

    assert parsed.filename == "notes.txt"
    assert "LangGraph" in parsed.text


def test_document_parser_rejects_unsupported_document() -> None:
    with pytest.raises(UnsupportedDocumentTypeError):
        DocumentParser().parse(
            content=b"%PDF",
            filename="notes.pdf",
            content_type="application/pdf",
        )


def test_upload_document_stores_local_document() -> None:
    client = TestClient(authenticated_app())
    document_text = b"Local retrieval uses uploaded research notes."

    response = client.post(
        "/documents/upload",
        headers=AUTH_HEADERS,
        files={
            "file": (
                "research.md",
                document_text,
                "text/markdown",
            )
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["filename"] == "research.md"
    assert payload["content_type"] == "text/markdown"
    assert payload["character_count"] == len(document_text)
    assert payload["chunk_count"] == 1
    assert payload["indexed"] is True
    assert payload["already_uploaded"] is False
    assert payload["principal"] == "cognito-user-1"
    assert payload["document_id"]


def test_upload_document_requires_cognito_token() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "research.md",
                b"Local embeddings are deterministic.",
                "text/markdown",
            )
        },
    )

    assert response.status_code == 401


def test_upload_document_is_rejected_when_disabled() -> None:
    app = create_app(
        Settings(
            environment="test",
            document_upload_enabled=False,
            cognito_issuer="https://issuer.example.test",
            cognito_client_id="client-id",
        )
    )
    app.state.auth_provider = CognitoAuthProvider(
        app.state.settings,
        token_verifier=FakeCognitoVerifier(),
    )
    response = TestClient(app).post(
        "/documents/upload",
        headers=AUTH_HEADERS,
        files={"file": ("research.md", b"disabled upload", "text/markdown")},
    )

    assert response.status_code == 403
    assert response.json()["error"] == "Document upload is disabled."


def test_upload_document_uses_cognito_principal() -> None:
    client = TestClient(authenticated_app())

    response = client.post(
        "/documents/upload",
        headers=AUTH_HEADERS,
        files={
            "file": (
                "research.md",
                b"Local embeddings are deterministic.",
                "text/markdown",
            )
        },
    )

    assert response.status_code == 200
    assert response.json()["principal"] == "cognito-user-1"


def test_upload_document_rejects_oversized_payload() -> None:
    app = create_app(
        Settings(
            environment="test",
            document_upload_enabled=True,
            cognito_issuer="https://issuer.example.test",
            cognito_client_id="client-id",
            max_upload_bytes=32,
        )
    )
    app.state.auth_provider = CognitoAuthProvider(
        app.state.settings,
        token_verifier=FakeCognitoVerifier(),
    )
    client = TestClient(app)

    response = client.post(
        "/documents/upload",
        headers=AUTH_HEADERS,
        files={
            "file": (
                "research.md",
                b"This document is definitely longer than thirty-two bytes.",
                "text/markdown",
            )
        },
    )

    assert response.status_code == 413
    payload = response.json()
    assert "too large" in str(payload).lower() or "request entity too large" in str(
        payload
    ).lower()


def test_upload_document_succeeds_when_indexing_is_unavailable() -> None:
    app = authenticated_app()
    app.state.document_indexer = FailingIndexer()
    client = TestClient(app)

    response = client.post(
        "/documents/upload",
        headers=AUTH_HEADERS,
        files={
            "file": (
                "research.md",
                b"Document storage should not depend on vector indexing.",
                "text/markdown",
            )
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["filename"] == "research.md"
    assert payload["indexed"] is False
    assert payload["already_uploaded"] is False
    assert payload["chunk_count"] == 0


def test_duplicate_upload_reuses_existing_document_and_skips_reindexing(tmp_path) -> None:
    app = create_app(
        Settings(
            environment="test",
            document_upload_enabled=True,
            persistence_backend="file",
            persistence_data_dir=str(tmp_path),
            vector_store_backend="faiss",
            cognito_issuer="https://issuer.example.test",
            cognito_client_id="client-id",
        )
    )
    app.state.auth_provider = CognitoAuthProvider(
        app.state.settings,
        token_verifier=FakeCognitoVerifier(),
    )
    client = TestClient(app)

    first = client.post(
        "/documents/upload",
        headers=AUTH_HEADERS,
        files={
            "file": (
                "research.md",
                b"duplicate prevention should keep the vector index stable",
                "text/markdown",
            )
        },
    )
    second = client.post(
        "/documents/upload",
        headers=AUTH_HEADERS,
        files={
            "file": (
                "research-copy.md",
                b"duplicate prevention should keep the vector index stable",
                "text/markdown",
            )
        },
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert isinstance(app.state.vector_store, FaissVectorStore)
    assert app.state.vector_store.count() == 1
    assert second.json()["document_id"] == first.json()["document_id"]
    assert second.json()["already_uploaded"] is True
    assert second.json()["indexed"] is False
    assert second.json()["chunk_count"] == 0
    assert "already uploaded" in second.json()["message"]


def test_duplicate_upload_retries_indexing_when_existing_document_is_not_indexed(
    tmp_path,
) -> None:
    app = create_app(
        Settings(
            environment="test",
            document_upload_enabled=True,
            persistence_backend="file",
            persistence_data_dir=str(tmp_path),
            vector_store_backend="faiss",
            cognito_issuer="https://issuer.example.test",
            cognito_client_id="client-id",
        )
    )
    app.state.auth_provider = CognitoAuthProvider(
        app.state.settings,
        token_verifier=FakeCognitoVerifier(),
    )
    app.state.document_indexer = FailingIndexer()
    app.state.ensure_vector_index = lambda: None
    client = TestClient(app)

    first = client.post(
        "/documents/upload",
        headers=AUTH_HEADERS,
        files={
            "file": (
                "research.md",
                b"existing uploads should retry indexing when missing vectors",
                "text/markdown",
            )
        },
    )
    assert first.status_code == 200
    assert first.json()["indexed"] is False

    recovered_app = create_app(
        Settings(
            environment="test",
            document_upload_enabled=True,
            persistence_backend="file",
            persistence_data_dir=str(tmp_path),
            vector_store_backend="faiss",
        )
    )
    app.state.document_indexer = recovered_app.state.document_indexer
    app.state.vector_store = recovered_app.state.vector_store
    app.state.ensure_vector_index = recovered_app.state.ensure_vector_index

    second = client.post(
        "/documents/upload",
        headers=AUTH_HEADERS,
        files={
            "file": (
                "research-copy.md",
                b"existing uploads should retry indexing when missing vectors",
                "text/markdown",
            )
        },
    )

    assert second.status_code == 200
    assert second.json()["already_uploaded"] is True
    assert second.json()["indexed"] is True
    assert second.json()["chunk_count"] == 1
    assert "indexing retried" in second.json()["message"]
    assert isinstance(app.state.vector_store, FaissVectorStore)
    assert app.state.vector_store.count() == 1
