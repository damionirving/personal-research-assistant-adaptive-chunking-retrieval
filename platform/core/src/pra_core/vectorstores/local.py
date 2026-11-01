import math
from threading import Lock

from pra_core.documents.models import DocumentChunk
from pra_core.vectorstores.base import VectorSearchResult, VectorStore


class LocalVectorStore(VectorStore):
    """In-memory cosine-similarity vector store for Phase 4."""

    def __init__(self) -> None:
        self._items: list[tuple[DocumentChunk, list[float]]] = []
        self._lock = Lock()

    def add(self, chunk: DocumentChunk, embedding: list[float]) -> None:
        with self._lock:
            self._items.append((chunk, embedding))

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

    def contains_document(self, document_id: str) -> bool:
        with self._lock:
            return any(chunk.document_id == document_id for chunk, _ in self._items)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_mag = math.sqrt(sum(a * a for a in left))
    right_mag = math.sqrt(sum(b * b for b in right))
    if left_mag == 0 or right_mag == 0:
        return 0.0
    return numerator / (left_mag * right_mag)
