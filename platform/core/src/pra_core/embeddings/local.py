import hashlib
import math
import re

from pra_core.embeddings.base import EmbeddingProvider


class LocalHashEmbeddingProvider(EmbeddingProvider):
    """Deterministic hashing embedder for local tests and development."""

    def __init__(self, dimensions: int = 64) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive.")
        self._dimensions = dimensions

    @property
    def provider_id(self) -> str:
        return f"local-hash:{self._dimensions}"

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        for term in re.findall(r"[a-zA-Z0-9]+", text.lower()):
            digest = hashlib.sha256(term.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self._dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude == 0:
            return vector
        return [value / magnitude for value in vector]
