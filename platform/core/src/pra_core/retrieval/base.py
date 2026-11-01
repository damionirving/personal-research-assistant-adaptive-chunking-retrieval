from abc import ABC, abstractmethod

from pra_core.documents.models import RetrievedDocument


class Retriever(ABC):
    @abstractmethod
    def search(self, query: str, k: int = 25) -> list[RetrievedDocument]:
        raise NotImplementedError
