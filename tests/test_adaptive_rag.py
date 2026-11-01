from pathlib import Path

from pra_core.adaptive_rag.models import ChildArtifact, stable_id
from pra_core.adaptive_rag.service import (
    AdaptiveDocumentIndexer,
    AdaptiveRagArtifacts,
    AdaptiveRetriever,
    _analyze_and_chunk,
    _plan_query,
)
from pra_core.config import Settings
from pra_core.documents.chunking import TextChunker
from pra_core.documents.models import Document, ParsedDocument
from pra_core.documents.repository import LocalDocumentRepository
from pra_core.embeddings.local import LocalHashEmbeddingProvider
from pra_core.indexing import DocumentIndexer
from pra_core.retrieval.hybrid import HybridRetriever
from pra_core.vectorstores.local import LocalVectorStore


SAMPLE = """# Section 1 Definitions
CHF means congestive heart failure.

# Section 2 Clinical Table
| Term | Value |
| --- | --- |
| CHF | Heart failure |

# Section 3 References
See Section 1 for the definition of CHF.
"""


def _settings(**updates):
    return Settings(environment="test", chunk_target_tokens=20, chunk_min_tokens=5, chunk_max_tokens=80, chunk_overlap_tokens=5, **updates)


def test_adaptive_analysis_creates_hierarchy_tables_and_relationships():
    document = Document(id="doc-1", filename="guide.md", content_type="text/markdown", text=SAMPLE, content_hash="checksum")
    manifest, parents, children, tables, relationships = _analyze_and_chunk(document, _settings(table_processing_enabled=True))

    assert manifest.decision.strategy == "table_aware"
    assert manifest.domain.primary == "medical"
    assert len(parents) == 3
    assert children and all(child.parent_id for child in children)
    assert tables[0].rows[0] == ("Term", "Value")
    assert relationships


def test_query_planning_prefers_exact_and_relationship_retrieval():
    plan = _plan_query("How does Section 3 relate to Section 1 for CHF?", _settings())

    assert plan.intent == "cross_reference_analysis"
    assert plan.use_relationships is True
    assert "heart failure" in plan.expanded_terms


def test_query_planning_selects_adaptive_context_size_and_bm25_first():
    settings = _settings()

    assert _plan_query("What is CHF?", settings).final_context_top_k == 5
    assert _plan_query("What policy rule is required for CHF?", settings).final_context_top_k == 8
    complex_plan = _plan_query(
        "Who is the regional lead and why does the exception apply?", settings
    )
    assert complex_plan.final_context_top_k == 15
    assert complex_plan.bm25_first is True


def test_adaptive_index_and_retrieval_round_trip():
    settings = _settings(adaptive_chunking_enabled=True, adaptive_retrieval_enabled=True, table_processing_enabled=True)
    repository = LocalDocumentRepository()
    document = repository.add(ParsedDocument("guide.md", "text/markdown", SAMPLE))
    vector_store = LocalVectorStore()
    embedder = LocalHashEmbeddingProvider(dimensions=32)
    legacy_indexer = DocumentIndexer(TextChunker(20, 5), embedder, vector_store)
    artifacts = AdaptiveRagArtifacts(None)
    indexer = AdaptiveDocumentIndexer(legacy_indexer, embedder, vector_store, artifacts, settings)
    legacy_retriever = HybridRetriever(repository, embedder, vector_store)
    retriever = AdaptiveRetriever(legacy_retriever, repository, embedder, vector_store, artifacts, settings)

    assert indexer.index(document) > 0
    results = retriever.search("What is CHF?", k=5)

    assert results
    assert retriever.last_plan is not None
    assert results[0].chunk_id


def test_adaptive_retrieval_uses_hybrid_fallback_for_low_confidence_queries():
    settings = _settings(adaptive_chunking_enabled=True, adaptive_retrieval_enabled=True)
    repository = LocalDocumentRepository()
    document = repository.add(ParsedDocument("guide.md", "text/markdown", SAMPLE))
    vector_store = LocalVectorStore()
    embedder = LocalHashEmbeddingProvider(dimensions=32)
    legacy_indexer = DocumentIndexer(TextChunker(20, 5), embedder, vector_store)
    artifacts = AdaptiveRagArtifacts(None)
    indexer = AdaptiveDocumentIndexer(legacy_indexer, embedder, vector_store, artifacts, settings)
    retriever = AdaptiveRetriever(
        HybridRetriever(repository, embedder, vector_store),
        repository,
        embedder,
        vector_store,
        artifacts,
        settings,
    )

    indexer.index(document)
    retriever.search("Explain an unfamiliar semantic topic", k=15)

    assert retriever.last_context_manifest["low_confidence"] is True
    assert retriever.last_context_manifest["retrieval_mode"] == "hybrid_fallback"
    assert retriever.last_context_manifest["context_top_k"] == 15


def test_complex_incident_approval_query_includes_structural_evidence():
    settings = Settings(
        environment="test",
        adaptive_chunking_enabled=True,
        adaptive_retrieval_enabled=True,
        table_processing_enabled=True,
        chunk_target_tokens=300,
        chunk_overlap_tokens=60,
    )
    repository = LocalDocumentRepository()
    scenario = Path("docs/test-corpus/scenario-northstar-meridian.md").read_text()
    document = repository.add(
        ParsedDocument("scenario-northstar-meridian.md", "text/markdown", scenario)
    )
    vector_store = LocalVectorStore()
    embedder = LocalHashEmbeddingProvider(dimensions=32)
    artifacts = AdaptiveRagArtifacts(None)
    indexer = AdaptiveDocumentIndexer(
        DocumentIndexer(TextChunker(300, 60), embedder, vector_store),
        embedder,
        vector_store,
        artifacts,
        settings,
    )
    retriever = AdaptiveRetriever(
        HybridRetriever(repository, embedder, vector_store),
        repository,
        embedder,
        vector_store,
        artifacts,
        settings,
    )

    indexer.index(document)
    results = retriever.search(
        "Which incidents require Copper Door approval, and why?",
        k=15,
    )
    context = "\n".join(result.text for result in results)

    assert "## Incident Records" in context
    assert "## Copper Door Approval" in context
    assert "## Lantern Pass Users" in context
    assert "BF-C | Seal breach confirmed" in context
    assert "NM-1043" in context
    assert "NM-1046" in context
    assert "NM-1047" in context


def test_direct_record_approval_query_includes_record_and_rule_evidence():
    settings = Settings(
        environment="test",
        adaptive_chunking_enabled=True,
        adaptive_retrieval_enabled=True,
        table_processing_enabled=True,
        chunk_target_tokens=300,
        chunk_overlap_tokens=60,
    )
    repository = LocalDocumentRepository()
    scenario = Path("docs/test-corpus/scenario-northstar-meridian.md").read_text()
    document = repository.add(
        ParsedDocument("scenario-northstar-meridian.md", "text/markdown", scenario)
    )
    vector_store = LocalVectorStore()
    embedder = LocalHashEmbeddingProvider(dimensions=32)
    artifacts = AdaptiveRagArtifacts(None)
    indexer = AdaptiveDocumentIndexer(
        DocumentIndexer(TextChunker(300, 60), embedder, vector_store),
        embedder,
        vector_store,
        artifacts,
        settings,
    )
    retriever = AdaptiveRetriever(
        HybridRetriever(repository, embedder, vector_store),
        repository,
        embedder,
        vector_store,
        artifacts,
        settings,
    )

    indexer.index(document)
    results = retriever.search("Does NM-1045 require Copper Door approval?", k=15)
    context = "\n".join(result.text for result in results)

    assert "NM-1045" in context
    assert "## Incident Records" in context
    assert "Copper Door approval is required when" in context
    assert "A customer asks for a credit more than 14 calendar days after delivery" in context
    assert "BF-C | Seal breach confirmed" in context


def test_disabled_features_use_legacy_indexer_and_retriever():
    settings = _settings()
    repository = LocalDocumentRepository()
    document = repository.add(ParsedDocument("note.txt", "text/plain", "A short note about retrieval."))
    vector_store = LocalVectorStore()
    embedder = LocalHashEmbeddingProvider(dimensions=16)
    legacy_indexer = DocumentIndexer(TextChunker(20, 5), embedder, vector_store)
    indexer = AdaptiveDocumentIndexer(legacy_indexer, embedder, vector_store, AdaptiveRagArtifacts(None), settings)
    legacy_retriever = HybridRetriever(repository, embedder, vector_store)
    retriever = AdaptiveRetriever(legacy_retriever, repository, embedder, vector_store, AdaptiveRagArtifacts(None), settings)

    assert indexer.index(document) == 1
    assert retriever.search("retrieval")


def test_artifact_ids_are_deterministic():
    assert stable_id("doc", "parent", "0") == stable_id("doc", "parent", "0")
    assert stable_id("doc", "parent", "0") != stable_id("doc", "parent", "1")
