from abc import ABC, abstractmethod
from dataclasses import dataclass

from pra_core.documents.models import DocumentChunk


@dataclass(frozen=True)
class VectorSearchResult:
    chunk: DocumentChunk
    score: float


class VectorStore(ABC):
    @abstractmethod
    def add(self, chunk: DocumentChunk, embedding: list[float]) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, embedding: list[float], k: int = 3) -> list[VectorSearchResult]:
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def contains_document(self, document_id: str) -> bool:
        raise NotImplementedError
