from pra_core.documents.models import DocumentChunk
from pra_core.vectorstores.base import VectorSearchResult, VectorStore


class OpenSearchServerlessVectorStore(VectorStore):
    """OpenSearch Serverless vector store for deployed AWS retrieval."""

    def __init__(
        self,
        endpoint: str,
        index_name: str,
        dimensions: int,
        region_name: str,
        client=None,
    ) -> None:
        self._endpoint = endpoint.replace("https://", "").rstrip("/")
        self._index_name = index_name
        self._dimensions = dimensions
        self._region_name = region_name
        self._client = client
        self._index_ready = False

    def add(self, chunk: DocumentChunk, embedding: list[float]) -> None:
        self._ensure_index()
        self._client_instance().index(
            index=self._index_name,
            body={
                "document_id": chunk.document_id,
                "filename": chunk.filename,
                "text": chunk.text,
                "chunk_id": chunk.id,
                "chunk_index": chunk.index,
                "chunk_metadata": chunk.metadata,
                "embedding": embedding,
            },
        )

    def search(self, embedding: list[float], k: int = 3) -> list[VectorSearchResult]:
        self._ensure_index()
        response = self._client_instance().search(
            index=self._index_name,
            body={
                "size": k,
                "query": {
                    "knn": {
                        "embedding": {
                            "vector": embedding,
                            "k": k,
                        }
                    }
                },
            },
        )
        hits = response.get("hits", {}).get("hits", [])
        return [
            VectorSearchResult(
                chunk=DocumentChunk(
                    id=hit["_source"]["chunk_id"],
                    document_id=hit["_source"]["document_id"],
                    filename=hit["_source"]["filename"],
                    text=hit["_source"]["text"],
                    index=int(hit["_source"]["chunk_index"]),
                    metadata=dict(hit["_source"].get("chunk_metadata", {})),
                ),
                score=float(hit.get("_score", 0.0)),
            )
            for hit in hits
        ]

    def clear(self) -> None:
        client = self._client_instance()
        if client.indices.exists(index=self._index_name):
            client.indices.delete(index=self._index_name)
        self._index_ready = False

    def count(self) -> int:
        self._ensure_index()
        response = self._client_instance().count(index=self._index_name)
        return int(response.get("count", 0))

    def contains_document(self, document_id: str) -> bool:
        self._ensure_index()
        response = self._client_instance().count(
            index=self._index_name,
            body={"query": {"term": {"document_id": document_id}}},
        )
        return int(response.get("count", 0)) > 0

    def _ensure_index(self) -> None:
        if self._index_ready:
            return
        client = self._client_instance()
        if not client.indices.exists(index=self._index_name):
            client.indices.create(
                index=self._index_name,
                body={
                    "settings": {"index": {"knn": True}},
                    "mappings": {
                        "properties": {
                            "document_id": {"type": "keyword"},
                            "filename": {"type": "keyword"},
                            "text": {"type": "text"},
                            "chunk_id": {"type": "keyword"},
                            "chunk_index": {"type": "integer"},
                            "chunk_metadata": {"type": "object", "enabled": True},
                            "embedding": {
                                "type": "knn_vector",
                                "dimension": self._dimensions,
                                "method": {
                                    "name": "hnsw",
                                    "engine": "faiss",
                                    "space_type": "cosinesimil",
                                },
                            },
                        }
                    },
                },
            )
        self._index_ready = True

    def _client_instance(self):
        if self._client is None:
            import boto3
            from opensearchpy import AWSV4SignerAuth, OpenSearch, RequestsHttpConnection

            credentials = boto3.Session().get_credentials()
            auth = AWSV4SignerAuth(credentials, self._region_name, "aoss")
            self._client = OpenSearch(
                hosts=[{"host": self._endpoint, "port": 443}],
                http_auth=auth,
                use_ssl=True,
                verify_certs=True,
                connection_class=RequestsHttpConnection,
            )
        return self._client
