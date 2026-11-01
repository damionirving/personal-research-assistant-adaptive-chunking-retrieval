from pra_api.main import create_app
from pra_core.config import Settings
from pra_core.documents.models import DocumentChunk
from pra_core.persistence.object_store import FileJsonObjectStore
from pra_core.vectorstores.faiss import FaissVectorStore


def test_faiss_vector_store_survives_recreation(tmp_path) -> None:
    object_store = FileJsonObjectStore(tmp_path)
    vector_store = FaissVectorStore(object_store)
    chunk = DocumentChunk(
        id="chunk-1",
        document_id="doc-1",
        filename="notes.md",
        text="faiss backed vector chunk",
        index=0,
        metadata={"strategy": "sentence_aware", "sentence_count": 1},
    )

    vector_store.add(chunk=chunk, embedding=[1.0, 0.0, 0.0])

    recreated = FaissVectorStore(FileJsonObjectStore(tmp_path))
    results = recreated.search([1.0, 0.0, 0.0], k=1)

    assert recreated.count() == 1
    assert results[0].chunk == chunk
    assert results[0].score == 1.0
    assert results[0].chunk.metadata["strategy"] == "sentence_aware"


def test_app_uses_faiss_vector_backend_with_persistent_object_store(tmp_path) -> None:
    app = create_app(
        Settings(
            environment="test",
            persistence_backend="file",
            persistence_data_dir=str(tmp_path),
            vector_store_backend="faiss",
        )
    )

    assert isinstance(app.state.vector_store, FaissVectorStore)
