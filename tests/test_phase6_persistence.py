from pra_api.main import create_app
from pra_core.config import Settings
from pra_core.documents.models import DocumentChunk, ParsedDocument
from pra_core.documents.persistent import PersistentDocumentRepository
from pra_core.memory.persistent import PersistentMemoryStore
from pra_core.persistence.object_store import FileJsonObjectStore
from pra_core.vectorstores.persistent import PersistentVectorStore


def test_persistent_document_repository_survives_recreation(tmp_path) -> None:
    object_store = FileJsonObjectStore(tmp_path)
    repository = PersistentDocumentRepository(object_store)

    document = repository.add(
        ParsedDocument(
            filename="notes.md",
            content_type="text/markdown",
            text="persistent documents survive restarts",
        )
    )

    recreated = PersistentDocumentRepository(FileJsonObjectStore(tmp_path))

    assert recreated.get(document.id) == document
    assert recreated.list()[0].filename == "notes.md"
    assert recreated.list()[0].content_hash == document.content_hash


def test_persistent_memory_store_survives_recreation(tmp_path) -> None:
    object_store = FileJsonObjectStore(tmp_path)
    memory = PersistentMemoryStore(object_store)

    memory.add_message("session-1", "user", "remember this")

    recreated = PersistentMemoryStore(FileJsonObjectStore(tmp_path))

    assert recreated.get_messages("session-1") == ["user: remember this"]


def test_persistent_memory_store_clears_session(tmp_path) -> None:
    object_store = FileJsonObjectStore(tmp_path)
    memory = PersistentMemoryStore(object_store)

    memory.add_message("session-1", "user", "remember this")
    memory.clear_session("session-1")

    recreated = PersistentMemoryStore(FileJsonObjectStore(tmp_path))

    assert recreated.get_messages("session-1") == []


def test_persistent_vector_store_survives_recreation(tmp_path) -> None:
    object_store = FileJsonObjectStore(tmp_path)
    vector_store = PersistentVectorStore(object_store)
    chunk = DocumentChunk(
        id="chunk-1",
        document_id="doc-1",
        filename="notes.md",
        text="persistent vector chunk",
        index=0,
        metadata={"strategy": "paragraph_aware", "paragraph_index": 0},
    )

    vector_store.add(chunk=chunk, embedding=[1.0, 0.0, 0.0])

    recreated = PersistentVectorStore(FileJsonObjectStore(tmp_path))
    results = recreated.search([1.0, 0.0, 0.0], k=1)

    assert recreated.count() == 1
    assert results[0].chunk == chunk
    assert results[0].score == 1.0
    assert results[0].chunk.metadata["strategy"] == "paragraph_aware"


def test_app_uses_file_persistence_backend(tmp_path) -> None:
    app = create_app(
        Settings(
            environment="test",
            persistence_backend="file",
            persistence_data_dir=str(tmp_path),
        )
    )

    assert isinstance(app.state.document_repository, PersistentDocumentRepository)
    assert isinstance(app.state.memory_store, PersistentMemoryStore)
    assert isinstance(app.state.vector_store, PersistentVectorStore)
