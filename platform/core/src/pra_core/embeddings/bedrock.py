import json

from pra_core.embeddings.base import EmbeddingProvider
from pra_core.logging import get_logger


logger = get_logger(__name__)


class BedrockEmbeddingProvider(EmbeddingProvider):
    """Bedrock Runtime embedding provider for deployed retrieval."""

    def __init__(
        self,
        model_id: str,
        region_name: str,
        dimensions: int = 1024,
        normalize: bool = True,
        client=None,
    ) -> None:
        self._model_id = model_id
        self._region_name = region_name
        self._dimensions = dimensions
        self._normalize = normalize
        self._client = client

    @property
    def provider_id(self) -> str:
        return f"bedrock:{self._model_id}:{self._dimensions}"

    def embed(self, text: str) -> list[float]:
        body = {
            "inputText": text,
            "dimensions": self._dimensions,
            "normalize": self._normalize,
        }
        response = self._bedrock().invoke_model(
            modelId=self._model_id,
            body=json.dumps(body).encode("utf-8"),
            contentType="application/json",
            accept="application/json",
        )
        payload = json.loads(response["body"].read().decode("utf-8"))
        embedding = payload.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            logger.error(
                "bedrock_embedding_empty",
                extra={
                    "bedrock_embedding_model_id": self._model_id,
                    "bedrock_region": self._region_name,
                },
            )
            raise ValueError("Bedrock embedding response did not include an embedding.")
        return [float(value) for value in embedding]

    def _bedrock(self):
        if self._client is None:
            import boto3
            from botocore.config import Config

            self._client = boto3.client(
                "bedrock-runtime",
                region_name=self._region_name,
                config=Config(read_timeout=60),
            )
        return self._client
