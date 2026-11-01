from threading import Lock

from pra_core.documents.models import DocumentChunk
from pra_core.persistence.object_store import JsonObjectStore
from pra_core.vectorstores.base import VectorSearchResult, VectorStore
from pra_core.vectorstores.local import _cosine_similarity


class PersistentVectorStore(VectorStore):
    """Vector index persisted through a JSON object store."""

    def __init__(self, object_store: JsonObjectStore, key: str = "vectors.json") -> None:
        self._object_store = object_store
        self._key = key
        self._lock = Lock()
        self._items = self._load()

    def add(self, chunk: DocumentChunk, embedding: list[float]) -> None:
        with self._lock:
            self._items.append((chunk, embedding))
            self._save()

    def search(self, embedding: list[float], k: int = 3) -> list[VectorSearchResult]:
        with self._lock:
            results = [
                VectorSearchResult(chunk=chunk, score=_cosine_similarity(embedding, stored))
                for chunk, stored in self._items
            ]
        return sorted(results, key=lambda result: result.score, reverse=True)[:k]

    def count(self) -> int:
        with self._lock:
            return len(self._items)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._save()

    def contains_document(self, document_id: str) -> bool:
        with self._lock:
            return any(chunk.document_id == document_id for chunk, _ in self._items)

    def _load(self) -> list[tuple[DocumentChunk, list[float]]]:
        rows = self._object_store.read_json(self._key, [])
        return [
            (
                DocumentChunk(
                    id=row["chunk"]["id"],
                    document_id=row["chunk"]["document_id"],
                    filename=row["chunk"]["filename"],
                    text=row["chunk"]["text"],
                    index=row["chunk"]["index"],
                    metadata=dict(row["chunk"].get("metadata", {})),
                ),
                row["embedding"],
            )
            for row in rows
        ]

    def _save(self) -> None:
        rows = [
            {
                "chunk": {
                    "id": chunk.id,
                    "document_id": chunk.document_id,
                    "filename": chunk.filename,
                    "text": chunk.text,
                    "index": chunk.index,
                    "metadata": chunk.metadata,
                },
                "embedding": embedding,
            }
            for chunk, embedding in self._items
        ]
        self._object_store.write_json(self._key, rows)
