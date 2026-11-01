from pra_core.documents.chunking import TextChunker
from pra_core.documents.models import Document
from pra_core.embeddings.base import EmbeddingProvider
from pra_core.vectorstores.base import VectorStore


class DocumentIndexer:
    """Upload pipeline step: chunk, embed, and index a parsed document."""

    def __init__(
        self,
        chunker: TextChunker,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        self._chunker = chunker
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store

    def index(self, document: Document) -> int:
        chunks = self.chunks(document)
        for chunk in chunks:
            embedding = self._embedding_provider.embed(chunk.text)
            self._vector_store.add(chunk=chunk, embedding=embedding)
        return len(chunks)

    def chunks(self, document: Document):
        """Return the deterministic source chunks used by the indexer."""
        return self._chunker.chunk(document)
