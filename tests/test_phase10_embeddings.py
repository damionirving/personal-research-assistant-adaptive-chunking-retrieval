from pathlib import Path

from fastapi.testclient import TestClient

from pra_api.main import create_app
from pra_api.main import _ensure_vector_index_matches_embedding_provider
from pra_core.config import Settings
from pra_core.documents.models import DocumentChunk, ParsedDocument
from pra_core.documents.persistent import PersistentDocumentRepository
from pra_core.documents.chunking import TextChunker
from pra_core.embeddings.bedrock import BedrockEmbeddingProvider
from pra_core.embeddings.local import LocalHashEmbeddingProvider
from pra_core.indexing import DocumentIndexer
from pra_core.persistence.object_store import FileJsonObjectStore
from pra_core.vectorstores.opensearch import OpenSearchServerlessVectorStore
from pra_core.vectorstores.persistent import PersistentVectorStore


class FakeBody:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload


class FakeBedrockEmbeddingClient:
    def __init__(self) -> None:
        self.request: dict | None = None

    def invoke_model(self, **request):
        self.request = request
        return {"body": FakeBody(b'{"embedding": [0.25, -0.5, 0.75]}')}


class FakeOpenSearchIndices:
    def exists(self, index: str) -> bool:
        return True


class FakeOpenSearchClient:
    def __init__(self) -> None:
        self.indices = FakeOpenSearchIndices()
        self.index_request: dict | None = None

    def index(self, **request):
        self.index_request = request
        return {"result": "created"}


def test_bedrock_embedding_provider_invokes_runtime_model() -> None:
    client = FakeBedrockEmbeddingClient()
    provider = BedrockEmbeddingProvider(
        model_id="amazon.titan-embed-text-v2:0",
        region_name="us-east-1",
        dimensions=1024,
        client=client,
    )

    embedding = provider.embed("Northstar route credit policy")

    assert embedding == [0.25, -0.5, 0.75]
    assert provider.provider_id == "bedrock:amazon.titan-embed-text-v2:0:1024"
    assert client.request is not None
    assert client.request["modelId"] == "amazon.titan-embed-text-v2:0"
    assert client.request["contentType"] == "application/json"


def test_opensearch_serverless_index_uses_generated_document_ids() -> None:
    client = FakeOpenSearchClient()
    store = OpenSearchServerlessVectorStore(
        endpoint="https://example.us-east-1.aoss.amazonaws.com",
        index_name="pra-chunks",
        dimensions=3,
        region_name="us-east-1",
        client=client,
    )

    store.add(
        chunk=DocumentChunk(
            id="document-1:0",
            document_id="document-1",
            filename="policy.md",
            text="Northstar policy",
            index=0,
        ),
        embedding=[0.1, 0.2, 0.3],
    )

    assert client.index_request is not None
    assert "id" not in client.index_request
    assert "refresh" not in client.index_request
    assert client.index_request["body"]["chunk_id"]


def test_reindexes_persistent_documents_when_embedding_provider_changes(
    tmp_path: Path,
) -> None:
    object_store = FileJsonObjectStore(tmp_path)
    repository = PersistentDocumentRepository(object_store)
    document = repository.add(
        ParsedDocument(
            filename="policy.md",
            content_type="text/markdown",
            text="Copper Door approval is required for Lantern Pass incidents.",
        )
    )
    vector_store = PersistentVectorStore(object_store)
    first_indexer = DocumentIndexer(
        chunker=TextChunker(chunk_size=8, overlap=0),
        embedding_provider=LocalHashEmbeddingProvider(dimensions=8),
        vector_store=vector_store,
    )
    first_indexer.index(document)
    object_store.write_json(
        "vector_index_metadata.json",
        {"embedding_provider_id": "local-hash:8"},
    )

    second_indexer = DocumentIndexer(
        chunker=TextChunker(chunk_size=8, overlap=0),
        embedding_provider=LocalHashEmbeddingProvider(dimensions=16),
        vector_store=vector_store,
    )

    _ensure_vector_index_matches_embedding_provider(
        object_store=object_store,
        embedding_provider_id="local-hash:16",
        document_repository=repository,
        document_indexer=second_indexer,
        vector_store=vector_store,
    )

    metadata = object_store.read_json("vector_index_metadata.json", {})
    assert metadata["embedding_provider_id"] == "local-hash:16"
    assert metadata["indexed_document_count"] == 1
    assert metadata["indexed_chunk_count"] == vector_store.count()
    assert vector_store.count() > 0


def test_deployed_embedding_vector_config_does_not_block_ui_startup(
    tmp_path: Path,
) -> None:
    app = create_app(
        Settings(
            environment="test",
            persistence_backend="file",
            persistence_data_dir=str(tmp_path),
            vector_store_backend="opensearch",
            opensearch_endpoint="https://example.us-east-1.aoss.amazonaws.com",
            embedding_provider="bedrock",
        )
    )
    app.state.ensure_vector_index = lambda: (_ for _ in ()).throw(
        AssertionError("vector index should not be touched by the UI route")
    )

    response = TestClient(app).get("/")

    assert response.status_code == 200
    assert "Personal Research Assistant" in response.text
