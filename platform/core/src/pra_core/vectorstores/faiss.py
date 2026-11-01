from __future__ import annotations

from dataclasses import asdict
from threading import Lock

from pra_core.documents.models import DocumentChunk
from pra_core.persistence.object_store import JsonObjectStore
from pra_core.vectorstores.base import VectorSearchResult, VectorStore
from pra_core.vectorstores.local import _cosine_similarity


class FaissVectorStore(VectorStore):
    """Lambda-friendly FAISS-style vector store persisted in S3.

    The deployed AWS path uses this adapter to keep the active index in memory
    during a Lambda invocation while persisting the index state and metadata to
    S3 after every update. The implementation intentionally keeps the same
    VectorStore interface so the retrieval pipeline remains backend-agnostic.
    """

    def __init__(
        self,
        object_store: JsonObjectStore,
        index_key: str = "faiss_index.json",
        metadata_key: str = "faiss_index_metadata.json",
    ) -> None:
        self._object_store = object_store
        self._index_key = index_key
        self._metadata_key = metadata_key
        self._lock = Lock()
        self._items = self._load()
        self._metadata = self._load_metadata()

    def add(self, chunk: DocumentChunk, embedding: list[float]) -> None:
        with self._lock:
            dimension = self._metadata.get("dimension")
            if dimension is not None and len(embedding) != dimension:
                raise ValueError(
                    f"FAISS index dimension {dimension} does not match embedding length {len(embedding)}."
                )
            self._items = [
                item for item in self._items if item["chunk"]["id"] != chunk.id
            ]
            self._items.append(_encode_item(chunk, embedding))
            if self._metadata.get("dimension") is None:
                self._metadata["dimension"] = len(embedding)
            self._save()

    def search(self, embedding: list[float], k: int = 3) -> list[VectorSearchResult]:
        with self._lock:
            results = [
                VectorSearchResult(
                    chunk=_decode_chunk(item["chunk"]),
                    score=_cosine_similarity(embedding, item["embedding"]),
                )
                for item in self._items
            ]
        return sorted(results, key=lambda result: result.score, reverse=True)[:k]

    def clear(self) -> None:
        with self._lock:
            self._items = []
            self._metadata = _default_metadata()
            self._save()

    def count(self) -> int:
        with self._lock:
            return len(self._items)

    def contains_document(self, document_id: str) -> bool:
        with self._lock:
            return any(
                item["chunk"]["document_id"] == document_id for item in self._items
            )

    def _load(self) -> list[dict]:
        rows = self._object_store.read_json(self._index_key, [])
        return [dict(row) for row in rows]

    def _load_metadata(self) -> dict:
        metadata = self._object_store.read_json(self._metadata_key, {})
        if not isinstance(metadata, dict):
            return _default_metadata()
        return {
            "format": metadata.get("format", "faiss-flat-cosine"),
            "vector_count": int(metadata.get("vector_count", len(self._items))),
            "dimension": metadata.get("dimension")
            if metadata.get("dimension") is None
            else int(metadata.get("dimension")),
        }

    def _save(self) -> None:
        self._object_store.write_json(self._index_key, self._items)
        self._object_store.write_json(
            self._metadata_key,
            {
                "format": "faiss-flat-cosine",
                "vector_count": len(self._items),
                "dimension": self._metadata.get("dimension"),
            },
        )


def _encode_item(chunk: DocumentChunk, embedding: list[float]) -> dict:
    return {
        "chunk": asdict(chunk),
        "embedding": [float(value) for value in embedding],
    }


def _decode_chunk(payload: dict) -> DocumentChunk:
    return DocumentChunk(
        id=payload["id"],
        document_id=payload["document_id"],
        filename=payload["filename"],
        text=payload["text"],
        index=int(payload["index"]),
        metadata=dict(payload.get("metadata", {})),
    )


def _default_metadata() -> dict:
    return {"format": "faiss-flat-cosine", "vector_count": 0, "dimension": None}
