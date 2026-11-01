from pra_core.documents.models import RetrievedDocument
from pra_core.embeddings.base import EmbeddingProvider
from pra_core.logging import get_logger
from pra_core.retrieval.base import Retriever
from pra_core.vectorstores.base import VectorStore


logger = get_logger(__name__)


class VectorRetriever(Retriever):
    """Retriever backed by the local embedding provider and vector store."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store

    def search(self, query: str, k: int = 3) -> list[RetrievedDocument]:
        try:
            query_embedding = self._embedding_provider.embed(query)
            results = self._vector_store.search(query_embedding, k=k)
        except Exception as exc:
            logger.exception(
                "retrieval_unavailable",
                extra={"error_type": exc.__class__.__name__},
            )
            return []
        return [
            RetrievedDocument(
                document_id=result.chunk.document_id,
                filename=result.chunk.filename,
                text=result.chunk.text,
                score=result.score,
                chunk_id=result.chunk.id,
                chunk_index=result.chunk.index,
            )
            for result in results
            if result.score > 0
        ]
