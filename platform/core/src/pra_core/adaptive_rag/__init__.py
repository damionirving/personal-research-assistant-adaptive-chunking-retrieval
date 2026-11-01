"""Feature-flagged adaptive chunking and retrieval for the research assistant."""

from pra_core.adaptive_rag.service import AdaptiveDocumentIndexer, AdaptiveRetriever, AdaptiveRagArtifacts

__all__ = ["AdaptiveDocumentIndexer", "AdaptiveRetriever", "AdaptiveRagArtifacts"]
