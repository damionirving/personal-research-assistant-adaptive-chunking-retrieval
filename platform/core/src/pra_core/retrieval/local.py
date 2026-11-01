import re

from pra_core.documents.models import Document, RetrievedDocument
from pra_core.documents.repository import DocumentRepository
from pra_core.retrieval.base import Retriever


class LocalKeywordRetriever(Retriever):
    """Simple deterministic keyword retriever for local Phase 1 use."""

    def __init__(self, repository: DocumentRepository) -> None:
        self._repository = repository

    def search(self, query: str, k: int = 3) -> list[RetrievedDocument]:
        query_terms = _terms(query)
        ranked = [
            self._score_document(document=document, query_terms=query_terms)
            for document in self._repository.list()
        ]
        matches = [document for document in ranked if document.score > 0]
        return sorted(matches, key=lambda document: document.score, reverse=True)[:k]

    def _score_document(
        self,
        document: Document,
        query_terms: set[str],
    ) -> RetrievedDocument:
        document_terms = _terms(f"{document.filename} {document.text}")
        overlap = query_terms.intersection(document_terms)
        score = len(overlap) / max(len(query_terms), 1)
        return RetrievedDocument(
            document_id=document.id,
            filename=document.filename,
            text=document.text,
            score=score,
        )


def _terms(value: str) -> set[str]:
    return {term.lower() for term in re.findall(r"[a-zA-Z0-9]+", value)}
