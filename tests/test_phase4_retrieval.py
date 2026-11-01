import re

from pra_core.auth.base import AuthError
from pra_core.auth.cognito import CognitoAuthProvider, sign_in_again_message
from pra_core.config import Settings
from pra_core.documents.chunking import TextChunker
from pra_core.documents.models import Document, ParsedDocument
from pra_core.documents.repository import LocalDocumentRepository
from pra_core.embeddings.local import LocalHashEmbeddingProvider
from pra_core.indexing import DocumentIndexer
from pra_core.retrieval.vector import VectorRetriever
from pra_core.vectorstores.local import LocalVectorStore


class FailingVectorStore(LocalVectorStore):
    def search(self, embedding: list[float], k: int = 3):
        raise RuntimeError("vector backend unavailable")


def test_text_chunker_creates_overlapping_chunks() -> None:
    document = Document(
        id="doc-1",
        filename="notes.txt",
        content_type="text/plain",
        text="one two three four five six",
    )

    chunks = TextChunker(chunk_size=4, overlap=2).chunk(document)

    assert [chunk.text for chunk in chunks] == [
        "one two three four",
        "three four five six",
    ]
    assert chunks[0].id == "notes.txt#chunk-0"
    assert chunks[1].index == 1


def test_local_embedding_provider_is_deterministic_and_normalized() -> None:
    provider = LocalHashEmbeddingProvider(dimensions=16)

    first = provider.embed("LangGraph retrieval")
    second = provider.embed("LangGraph retrieval")

    assert first == second
    assert len(first) == 16
    assert any(value != 0 for value in first)


def test_upload_pipeline_indexes_chunks_for_vector_retrieval() -> None:
    repository = LocalDocumentRepository()
    embedding_provider = LocalHashEmbeddingProvider(dimensions=32)
    vector_store = LocalVectorStore()
    indexer = DocumentIndexer(
        chunker=TextChunker(chunk_size=6, overlap=1),
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )
    document = repository.add(
        ParsedDocument(
            filename="retrieval.txt",
            content_type="text/plain",
            text="vector embeddings retrieve research documents",
        )
    )

    chunk_count = indexer.index(document)
    results = VectorRetriever(embedding_provider, vector_store).search(
        "research embeddings",
        k=1,
    )

    assert chunk_count == 1
    assert vector_store.count() == 1
    assert results[0].document_id == document.id
    assert results[0].chunk_id == "retrieval.txt#chunk-0"
    assert results[0].score > 0


def test_vector_retriever_returns_no_documents_when_backend_is_unavailable() -> None:
    retriever = VectorRetriever(
        LocalHashEmbeddingProvider(dimensions=16),
        FailingVectorStore(),
    )

    assert retriever.search("hello", k=3) == []


class FakeCognitoVerifier:
    def verify(self, token: str) -> dict[str, object]:
        assert token == "valid-cognito-token"
        return {"sub": "cognito-user-1", "cognito:groups": ["researcher"]}


def test_cognito_auth_provider_uses_verified_token_claims() -> None:
    provider = CognitoAuthProvider(
        Settings(
            environment="test",
            cognito_issuer="https://issuer.example.test",
            cognito_client_id="client-id",
        ),
        token_verifier=FakeCognitoVerifier(),
    )

    principal = provider.authenticate("Bearer valid-cognito-token")

    assert principal.subject == "cognito-user-1"
    assert principal.roles == ("researcher",)


def test_cognito_auth_provider_rejects_missing_token() -> None:
    provider = CognitoAuthProvider(
        Settings(
            environment="test",
            cognito_issuer="https://issuer.example.test",
            cognito_client_id="client-id",
        ),
        token_verifier=FakeCognitoVerifier(),
    )

    try:
        provider.authenticate(None)
    except AuthError as exc:
        assert "Missing authorization token" in str(exc)
    else:
        raise AssertionError("Expected missing token to be rejected.")


def test_sign_in_again_message_includes_timestamp() -> None:
    assert re.match(
        r"Please Sign in Again\. \d{2}/\d{2}/20\d{2} \d{2}:\d{2} (AM|PM)",
        sign_in_again_message(),
    )
