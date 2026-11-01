from dataclasses import replace

from pra_core.documents.models import Document, DocumentChunk, ParsedDocument
from pra_core.documents.repository import LocalDocumentRepository
from pra_core.retrieval.hybrid import HybridRetriever
from pra_core.retrieval.local import LocalKeywordRetriever


class StubEmbeddingProvider:
    def embed(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]


class StubVectorStore:
    def search(self, embedding: list[float], k: int = 25):
        return [
            type(
                "Result",
                (),
                {
                    "chunk": DocumentChunk(
                        id="chunk-vector",
                        document_id="doc-vector",
                        filename="vector.txt",
                        text="vector match",
                        index=0,
                    ),
                    "score": 0.9,
                },
            )(),
            type(
                "Result",
                (),
                {
                    "chunk": DocumentChunk(
                        id="chunk-hybrid",
                        document_id="doc-hybrid",
                        filename="hybrid.txt",
                        text="hybrid match",
                        index=0,
                    ),
                    "score": 0.5,
                },
            )(),
            type(
                "Result",
                (),
                {
                    "chunk": DocumentChunk(
                        id="chunk-vector-2",
                        document_id="doc-vector-2",
                        filename="vector-2.txt",
                        text="semantic alpha concept",
                        index=0,
                    ),
                    "score": 0.4,
                },
            )(),
        ][:k]


class DuplicateChunkVectorStore:
    def search(self, embedding: list[float], k: int = 25):
        return [
            type(
                "Result",
                (),
                {
                    "chunk": DocumentChunk(
                        id="chunk-hybrid-1",
                        document_id="doc-hybrid",
                        filename="hybrid.txt",
                        text="first hybrid chunk",
                        index=0,
                    ),
                    "score": 0.92,
                },
            )(),
            type(
                "Result",
                (),
                {
                    "chunk": DocumentChunk(
                        id="chunk-hybrid-2",
                        document_id="doc-hybrid",
                        filename="hybrid.txt",
                        text="second hybrid chunk",
                        index=1,
                    ),
                    "score": 0.73,
                },
            )(),
        ][:k]


class StubRepository:
    def __init__(self) -> None:
        self._documents = [
            Document(
                id="doc-bm25",
                filename="policy.txt",
                content_type="text/plain",
                text="alpha alpha beta",
            ),
            Document(
                id="doc-hybrid",
                filename="hybrid.txt",
                content_type="text/plain",
                text="hybrid match",
            ),
        ]

    def list(self) -> list[Document]:
        return self._documents


class EntityAnchorRepository:
    def __init__(self) -> None:
        self._documents = [
            Document(
                id="doc-orchid",
                filename="northstar.md",
                content_type="text/plain",
                text="Orchid Ledger is Northstar Meridian's main incident and route analytics system.",
            ),
            Document(
                id="doc-blue-finch",
                filename="blue-finch.md",
                content_type="text/plain",
                text="Blue Finch code BF-C identifies temperature escort exceptions.",
            ),
        ]

    def list(self) -> list[Document]:
        return self._documents


class CodeLookupRepository:
    def __init__(self) -> None:
        self._documents = [
            Document(
                id="doc-blue-finch",
                filename="scenario-northstar-meridian.md",
                content_type="text/markdown",
                text=(
                    "## Blue Finch Packaging Program\n\n"
                    "Blue Finch status codes:\n\n"
                    "| Code | Meaning |\n"
                    "| --- | --- |\n"
                    "| BF-A | Packaging arrived intact |\n"
                    "| BF-B | Outer carton damaged, seal intact |\n"
                    "| BF-C | Seal breach confirmed |\n"
                    "| BF-D | Temperature strip unreadable |\n\n"
                    "## Incident Records\n\n"
                    "| Incident ID | Issue Summary |\n"
                    "| --- | --- |\n"
                    "| NM-1045 | Blue Finch BF-C; customer reported 4 business days after delivery |"
                ),
            )
        ]

    def list(self) -> list[Document]:
        return self._documents


class CodeLookupVectorStore:
    def search(self, embedding: list[float], k: int = 25):
        return [
            type(
                "Result",
                (),
                {
                    "chunk": DocumentChunk(
                        id="scenario-northstar-meridian.md#chunk-9",
                        document_id="doc-blue-finch",
                        filename="scenario-northstar-meridian.md",
                        text=(
                            "## Incident Records | Incident ID | Issue Summary | "
                            "| --- | --- | "
                            "| NM-1045 | Blue Finch BF-C; customer reported 4 business days after delivery |"
                        ),
                        index=9,
                    ),
                    "score": 0.73,
                },
            )(),
        ][:k]


def test_local_keyword_retriever_returns_ranked_matches() -> None:
    repository = LocalDocumentRepository()
    first = repository.add(
        ParsedDocument(
            filename="agents.txt",
            content_type="text/plain",
            text="LangGraph agents coordinate workflows.",
        )
    )
    repository.add(
        ParsedDocument(
            filename="infra.txt",
            content_type="text/plain",
            text="Docker packages the local API.",
        )
    )

    results = LocalKeywordRetriever(repository).search("LangGraph workflows", k=1)

    assert len(results) == 1
    assert results[0].document_id == first.id
    assert results[0].score > 0


def test_hybrid_retriever_merges_bm25_and_vector_hits_and_reranks() -> None:
    retriever = HybridRetriever(
        repository=StubRepository(),
        embedding_provider=StubEmbeddingProvider(),
        vector_store=StubVectorStore(),
        bm25_top_k=25,
        vector_top_k=25,
        rrf_k=60,
        rerank_top_k=25,
    )

    results = retriever.search("alpha match", k=25)

    assert results[0].document_id == "doc-hybrid"
    assert results[1].document_id == "doc-vector"
    assert results[0].score >= results[1].score
    assert any(result.document_id == "doc-vector" for result in results)
    assert results[0].chunk_id == "chunk-hybrid"
    assert results[0].bm25_score is not None
    assert results[0].vector_score is not None
    assert results[0].rrf_score == results[0].score


def test_hybrid_retriever_uses_rrf_instead_of_raw_score_addition() -> None:
    retriever = HybridRetriever(
        repository=StubRepository(),
        embedding_provider=StubEmbeddingProvider(),
        vector_store=StubVectorStore(),
        bm25_top_k=25,
        vector_top_k=25,
        rrf_k=60,
        rerank_top_k=25,
    )

    results = retriever.search("alpha match", k=25)

    hybrid = next(item for item in results if item.document_id == "doc-hybrid")
    bm25_only = next(item for item in results if item.document_id == "doc-bm25")
    vector_only = next(item for item in results if item.document_id == "doc-vector")

    assert hybrid.score > bm25_only.score
    assert hybrid.score > vector_only.score
    assert hybrid.score < 1.0


def test_hybrid_retriever_applies_final_rerank_top_k_selection() -> None:
    retriever = HybridRetriever(
        repository=StubRepository(),
        embedding_provider=StubEmbeddingProvider(),
        vector_store=StubVectorStore(),
        rerank_top_k=2,
    )

    results = retriever.search("alpha match", k=25)

    assert len(results) == 2


def test_hybrid_retriever_records_stage_timings_and_candidate_counts() -> None:
    retriever = HybridRetriever(
        repository=StubRepository(),
        embedding_provider=StubEmbeddingProvider(),
        vector_store=StubVectorStore(),
        bm25_top_k=25,
        vector_top_k=25,
        rrf_k=60,
        rerank_top_k=25,
    )

    retriever.search("alpha match", k=25)
    trace = retriever.last_trace

    assert trace.bm25_latency_ms >= 0
    assert trace.vector_latency_ms >= 0
    assert trace.rrf_merge_latency_ms >= 0
    assert trace.reranker_latency_ms >= 0
    assert trace.retrieval_latency_ms >= 0
    assert trace.bm25_candidate_count >= 1
    assert trace.vector_candidate_count >= 1
    assert trace.merged_candidate_count >= 1
    assert trace.reranked_candidate_count >= 1


def test_hybrid_retriever_can_disable_reranking_with_feature_flag() -> None:
    retriever = HybridRetriever(
        repository=StubRepository(),
        embedding_provider=StubEmbeddingProvider(),
        vector_store=StubVectorStore(),
        bm25_top_k=25,
        vector_top_k=25,
        rrf_k=60,
        enable_reranking=False,
        rerank_top_k=25,
    )

    results = retriever.search("alpha match", k=25)
    trace = retriever.last_trace

    assert results[0].document_id == "doc-hybrid"
    assert trace.reranker_latency_ms == 0.0


def test_hybrid_retriever_can_disable_bm25_lexical_search() -> None:
    retriever = HybridRetriever(
        repository=StubRepository(),
        embedding_provider=StubEmbeddingProvider(),
        vector_store=StubVectorStore(),
        enable_bm25_lexical=False,
    )

    results = retriever.search("alpha match", k=25)
    trace = retriever.last_trace

    assert results
    assert all(result.bm25_score == 0.0 for result in results)
    assert trace.bm25_candidate_count == 0
    assert trace.vector_candidate_count >= 1


def test_hybrid_retriever_can_disable_vector_search() -> None:
    retriever = HybridRetriever(
        repository=StubRepository(),
        embedding_provider=StubEmbeddingProvider(),
        vector_store=StubVectorStore(),
        enable_vector_search=False,
    )

    results = retriever.search("alpha match", k=25)
    trace = retriever.last_trace

    assert results
    assert all(result.vector_score == 0.0 for result in results)
    assert trace.bm25_candidate_count >= 1
    assert trace.vector_candidate_count == 0


def test_hybrid_retriever_disables_rrf_when_a_retrieval_leg_is_disabled() -> None:
    retriever = HybridRetriever(
        repository=StubRepository(),
        embedding_provider=StubEmbeddingProvider(),
        vector_store=StubVectorStore(),
        enable_bm25_lexical=False,
        enable_vector_search=True,
        enable_reciprocal_rank_fusion=True,
    )

    assert retriever._enable_reciprocal_rank_fusion is False


def test_hybrid_retriever_keeps_best_cosine_score_per_document_for_metadata() -> None:
    retriever = HybridRetriever(
        repository=StubRepository(),
        embedding_provider=StubEmbeddingProvider(),
        vector_store=DuplicateChunkVectorStore(),
        bm25_top_k=25,
        vector_top_k=25,
        rrf_k=60,
        rerank_top_k=25,
    )

    results = retriever.search("hybrid match", k=25)

    hybrid = next(item for item in results if item.document_id == "doc-hybrid")
    assert hybrid.vector_score == 0.92
    assert hybrid.chunk_id == "chunk-hybrid-1"
    assert hybrid.chunk_index == 0
    assert hybrid.vector_score <= 1.0


def test_hybrid_retriever_prefers_exact_entity_anchor_for_lookup_queries() -> None:
    retriever = HybridRetriever(
        repository=EntityAnchorRepository(),
        embedding_provider=StubEmbeddingProvider(),
        vector_store=StubVectorStore(),
        bm25_top_k=125,
        vector_top_k=125,
        rrf_k=60,
        enable_reranking=True,
        rerank_top_k=25,
    )

    results = retriever.search("What is Orchid Ledger?", k=25)

    assert results[0].document_id == "doc-orchid"


def test_hybrid_retriever_repairs_exact_code_lookup_context() -> None:
    retriever = HybridRetriever(
        repository=CodeLookupRepository(),
        embedding_provider=StubEmbeddingProvider(),
        vector_store=CodeLookupVectorStore(),
        bm25_top_k=25,
        vector_top_k=25,
        rrf_k=60,
        preserve_bm25_candidates=True,
        enable_reranking=False,
        rerank_top_k=25,
    )

    results = retriever.search("What does Blue Finch code BF-C mean?", k=25)

    assert results[0].chunk_id == "scenario-northstar-meridian.md#chunk-9"
    assert "| BF-C | Seal breach confirmed |" in results[0].text


def test_hybrid_retriever_repairs_exact_table_field_lookup_context() -> None:
    repository = CodeLookupRepository()
    repository._documents[0] = replace(
        repository._documents[0],
        text=(
            "| Region Code | Region Name | Dispatch Hub | Regional Lead |\n"
            "| --- | --- | --- | --- |\n"
            "| NE-4 | Northeast Four | Albany Relay | Mara Voss |\n"
            "| SW-7 | Southwest Seven | Mesa Transfer | Rafi Ornelas |\n\n"
            "## Region-Specific Notes\n\nSW-7 uses Mountain time."
        ),
    )
    retriever = HybridRetriever(
        repository=repository,
        embedding_provider=StubEmbeddingProvider(),
        vector_store=CodeLookupVectorStore(),
        bm25_top_k=25,
        vector_top_k=25,
        rrf_k=60,
        preserve_bm25_candidates=True,
        enable_reranking=False,
        rerank_top_k=25,
    )

    results = retriever.search("Who is the regional lead for SW-7?", k=25)

    assert "| SW-7 | Southwest Seven | Mesa Transfer | Rafi Ornelas |" in results[0].text


def test_hybrid_retriever_repairs_exact_record_rule_lookup_context() -> None:
    retriever = HybridRetriever(
        repository=CodeLookupRepository(),
        embedding_provider=StubEmbeddingProvider(),
        vector_store=CodeLookupVectorStore(),
        bm25_top_k=25,
        vector_top_k=25,
        rrf_k=60,
        preserve_bm25_candidates=True,
        enable_reranking=False,
        rerank_top_k=25,
    )

    results = retriever.search("Does NM-1045 require Copper Door approval?", k=25)

    assert results[0].chunk_id == "scenario-northstar-meridian.md#chunk-9"
    assert "## Incident Records" in results[0].text
    assert "| NM-1045 | Blue Finch BF-C; customer reported 4 business days after delivery |" in results[0].text


def test_hybrid_retriever_supports_weighted_rrf_with_bm25_preservation() -> None:
    retriever = HybridRetriever(
        repository=StubRepository(),
        embedding_provider=StubEmbeddingProvider(),
        vector_store=StubVectorStore(),
        bm25_top_k=25,
        vector_top_k=25,
        rrf_k=30,
        fusion_strategy="weighted_rrf",
        bm25_fusion_weight=0.8,
        vector_fusion_weight=0.2,
        preserve_bm25_candidates=True,
        enable_reranking=False,
        rerank_top_k=25,
    )

    results = retriever.search("alpha match", k=25)

    assert results[0].document_id == "doc-bm25"
    assert results[1].document_id == "doc-hybrid"
    assert results[0].score >= results[1].score


def test_hybrid_retriever_supports_normalized_weighted_fusion() -> None:
    retriever = HybridRetriever(
        repository=StubRepository(),
        embedding_provider=StubEmbeddingProvider(),
        vector_store=StubVectorStore(),
        bm25_top_k=25,
        vector_top_k=25,
        rrf_k=30,
        fusion_strategy="normalized_weighted",
        bm25_fusion_weight=0.8,
        vector_fusion_weight=0.2,
        enable_reranking=False,
        rerank_top_k=25,
    )

    results = retriever.search("alpha match", k=25)

    assert results
    assert all(result.score >= 0 for result in results)
    assert results[0].rrf_score == results[0].score
