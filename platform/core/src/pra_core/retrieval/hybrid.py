from __future__ import annotations

import math
import re
from dataclasses import dataclass
from time import perf_counter

from pra_core.documents.models import Document, RetrievedDocument
from pra_core.documents.repository import DocumentRepository
from pra_core.embeddings.base import EmbeddingProvider
from pra_core.logging import get_logger
from pra_core.retrieval.base import Retriever
from pra_core.retrieval.fusion import FusionConfig, merge_retrieval_hits, no_rerank_key, rerank_key
from pra_core.vectorstores.base import VectorStore


logger = get_logger(__name__)


@dataclass(frozen=True)
class HybridRetrievalTrace:
    bm25_latency_ms: float
    vector_latency_ms: float
    rrf_merge_latency_ms: float
    reranker_latency_ms: float
    retrieval_latency_ms: float
    bm25_candidate_count: int
    vector_candidate_count: int
    merged_candidate_count: int
    reranked_candidate_count: int


class HybridRetriever(Retriever):
    """Hybrid BM25 + vector retriever with RRF merge and deterministic rerank."""

    def __init__(
        self,
        repository: DocumentRepository,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        bm25_top_k: int = 25,
        vector_top_k: int = 25,
        rrf_k: int = 60,
        fusion_strategy: str = "rrf",
        bm25_fusion_weight: float = 0.7,
        vector_fusion_weight: float = 0.3,
        preserve_bm25_candidates: bool = False,
        enable_bm25_lexical: bool = True,
        enable_vector_search: bool = True,
        enable_reciprocal_rank_fusion: bool = True,
        enable_reranking: bool = False,
        rerank_top_k: int = 25,
        graph_retriever=None,
    ) -> None:
        self._repository = repository
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._bm25_top_k = bm25_top_k
        self._vector_top_k = vector_top_k
        self._rrf_k = rrf_k
        self._fusion_strategy = fusion_strategy
        self._bm25_fusion_weight = bm25_fusion_weight
        self._vector_fusion_weight = vector_fusion_weight
        self._preserve_bm25_candidates = preserve_bm25_candidates
        self._enable_bm25_lexical = enable_bm25_lexical
        self._enable_vector_search = enable_vector_search
        self._enable_reciprocal_rank_fusion = (
            enable_reciprocal_rank_fusion and enable_bm25_lexical and enable_vector_search
        )
        self._enable_reranking = enable_reranking
        self._rerank_top_k = rerank_top_k
        self._graph_retriever = graph_retriever
        self._last_trace = HybridRetrievalTrace(
            bm25_latency_ms=0.0,
            vector_latency_ms=0.0,
            rrf_merge_latency_ms=0.0,
            reranker_latency_ms=0.0,
            retrieval_latency_ms=0.0,
            bm25_candidate_count=0,
            vector_candidate_count=0,
            merged_candidate_count=0,
            reranked_candidate_count=0,
        )

    @property
    def last_trace(self) -> HybridRetrievalTrace:
        return self._last_trace

    def search(self, query: str, k: int = 25) -> list[RetrievedDocument]:
        retrieval_started_at = perf_counter()
        documents = self._repository.list()
        query_terms = _terms(query)
        entity_phrases = _entity_phrases(query)
        bm25_started_at = perf_counter()
        bm25_hits = (
            self._bm25_search(
                query_terms,
                entity_phrases,
                documents,
                k=self._bm25_top_k,
            )
            if self._enable_bm25_lexical
            else []
        )
        bm25_latency_ms = _elapsed_ms(bm25_started_at)

        vector_started_at = perf_counter()
        vector_hits = (
            self._vector_search(query, k=self._vector_top_k)
            if self._enable_vector_search
            else []
        )
        vector_latency_ms = _elapsed_ms(vector_started_at)

        rrf_started_at = perf_counter()
        merged = (
            self._merge_results_rrf(bm25_hits, vector_hits)
            if self._enable_reciprocal_rank_fusion
            else self._merge_results_by_score(bm25_hits, vector_hits)
        )
        if self._graph_retriever is not None:
            merged = self._merge_graph_hits(
                merged,
                self._graph_retriever(query, k=self._rerank_top_k),
            )
        rrf_latency_ms = _elapsed_ms(rrf_started_at)

        if self._enable_reranking:
            rerank_started_at = perf_counter()
            ranked_candidates = sorted(merged.values(), key=rerank_key, reverse=True)
            rerank_latency_ms = _elapsed_ms(rerank_started_at)
        else:
            ranked_candidates = sorted(
                merged.values(),
                key=no_rerank_key,
                reverse=True,
            )
            rerank_latency_ms = 0.0

        final_k = min(k, self._rerank_top_k)
        documents_by_id = {document.id: document for document in documents}
        results = [
            _repair_exact_table_lookup_context(
                item,
                source_document=documents_by_id.get(item.document_id),
                query=query,
            )
            for item in ranked_candidates[:final_k]
            if item.score > 0
        ]
        self._last_trace = HybridRetrievalTrace(
            bm25_latency_ms=bm25_latency_ms,
            vector_latency_ms=vector_latency_ms,
            rrf_merge_latency_ms=rrf_latency_ms,
            reranker_latency_ms=rerank_latency_ms,
            retrieval_latency_ms=_elapsed_ms(retrieval_started_at),
            bm25_candidate_count=len(bm25_hits),
            vector_candidate_count=len(vector_hits),
            merged_candidate_count=len(merged),
            reranked_candidate_count=len(results),
        )
        return results

    def _merge_graph_hits(
        self,
        merged: dict[str, RetrievedDocument],
        graph_hits: list[RetrievedDocument],
    ) -> dict[str, RetrievedDocument]:
        for hit in graph_hits:
            key = hit.document_id
            graph_score = hit.score
            existing = merged.get(key)
            if existing is not None and existing.score >= graph_score:
                if hit.knowledge_graph_score is not None:
                    merged[key] = RetrievedDocument(
                        document_id=existing.document_id,
                        filename=existing.filename,
                        text=existing.text,
                        score=existing.score,
                        chunk_id=existing.chunk_id,
                        chunk_index=existing.chunk_index,
                        bm25_score=existing.bm25_score,
                        vector_score=existing.vector_score,
                        rrf_score=existing.rrf_score,
                        entity_anchor_score=existing.entity_anchor_score,
                        knowledge_graph_score=max(
                            existing.knowledge_graph_score or 0.0,
                            hit.knowledge_graph_score,
                        ),
                    )
                continue
            merged[key] = RetrievedDocument(
                document_id=hit.document_id,
                filename=hit.filename,
                text=hit.text,
                score=graph_score,
                chunk_id=hit.chunk_id,
                chunk_index=hit.chunk_index,
                bm25_score=hit.bm25_score or 0.0,
                vector_score=hit.vector_score or 0.0,
                rrf_score=graph_score,
                entity_anchor_score=hit.entity_anchor_score or 0.0,
                knowledge_graph_score=graph_score,
            )
        return merged

    def _bm25_search(
        self,
        query_terms: list[str],
        entity_phrases: list[str],
        documents: list[Document],
        k: int,
    ) -> list[RetrievedDocument]:
        if not query_terms or not documents:
            return []

        doc_tokens = [_terms(f"{document.filename} {document.text}") for document in documents]
        avg_doc_len = sum(len(tokens) for tokens in doc_tokens) / max(len(doc_tokens), 1)
        doc_freqs = _document_frequencies(doc_tokens)
        candidates: list[RetrievedDocument] = []

        for document, tokens in zip(documents, doc_tokens):
            score = _bm25_score(query_terms, tokens, doc_freqs, avg_doc_len, len(documents))
            entity_anchor_score = _entity_anchor_score(entity_phrases, document)
            score += entity_anchor_score
            if score <= 0:
                continue
            candidates.append(
                RetrievedDocument(
                    document_id=document.id,
                    filename=document.filename,
                    text=document.text,
                    score=score,
                    bm25_score=score,
                    vector_score=0.0,
                    rrf_score=0.0,
                    entity_anchor_score=entity_anchor_score,
                )
            )

        return sorted(candidates, key=lambda item: item.bm25_score or 0.0, reverse=True)[:k]

    def _vector_search(self, query: str, k: int) -> list[RetrievedDocument]:
        try:
            query_embedding = self._embedding_provider.embed(query)
            results = self._vector_store.search(query_embedding, k=k)
        except Exception as exc:
            logger.exception(
                "hybrid_vector_search_unavailable",
                extra={"error_type": exc.__class__.__name__},
            )
            return []

        return [
            RetrievedDocument(
                document_id=result.chunk.document_id,
                filename=result.chunk.filename,
                text=result.chunk.text,
                score=result.score,
                bm25_score=0.0,
                vector_score=result.score,
                rrf_score=0.0,
                chunk_id=result.chunk.id,
                chunk_index=result.chunk.index,
                entity_anchor_score=0.0,
                metadata=result.chunk.metadata,
            )
            for result in results
            if result.score > 0
        ]

    def _merge_results_rrf(
        self,
        bm25_hits: list[RetrievedDocument],
        vector_hits: list[RetrievedDocument],
    ) -> dict[str, RetrievedDocument]:
        return merge_retrieval_hits(
            bm25_hits=bm25_hits,
            vector_hits=vector_hits,
            config=FusionConfig(
                strategy=self._fusion_strategy,
                rrf_k=self._rrf_k,
                bm25_weight=self._bm25_fusion_weight,
                vector_weight=self._vector_fusion_weight,
                preserve_bm25_candidates=self._preserve_bm25_candidates,
            ),
            key_fn=lambda item: item.document_id,
            prefer_vector_chunk=True,
        )

    def _merge_results_by_score(
        self,
        bm25_hits: list[RetrievedDocument],
        vector_hits: list[RetrievedDocument],
    ) -> dict[str, RetrievedDocument]:
        merged: dict[str, RetrievedDocument] = {}
        for candidate in [*bm25_hits, *vector_hits]:
            existing = merged.get(candidate.document_id)
            candidate_score = max(
                candidate.score,
                candidate.rrf_score or 0.0,
                candidate.vector_score or 0.0,
                candidate.bm25_score or 0.0,
            )
            existing_score = (
                max(
                    existing.score,
                    existing.rrf_score or 0.0,
                    existing.vector_score or 0.0,
                    existing.bm25_score or 0.0,
                )
                if existing is not None
                else -1.0
            )
            if existing is not None and existing_score >= candidate_score:
                continue
            merged[candidate.document_id] = RetrievedDocument(
                document_id=candidate.document_id,
                filename=candidate.filename,
                text=candidate.text,
                score=candidate_score,
                bm25_score=candidate.bm25_score,
                vector_score=candidate.vector_score,
                rrf_score=candidate_score,
                chunk_id=candidate.chunk_id,
                chunk_index=candidate.chunk_index,
                entity_anchor_score=candidate.entity_anchor_score,
            )
        return merged


def _terms(value: str) -> list[str]:
    return [term.lower() for term in re.findall(r"[a-zA-Z0-9]+", value)]


def _document_frequencies(doc_tokens: list[list[str]]) -> dict[str, int]:
    frequencies: dict[str, int] = {}
    for tokens in doc_tokens:
        for term in set(tokens):
            frequencies[term] = frequencies.get(term, 0) + 1
    return frequencies


def _bm25_score(
    query_terms: list[str],
    document_terms: list[str],
    doc_freqs: dict[str, int],
    avg_doc_len: float,
    document_count: int,
) -> float:
    if not document_terms:
        return 0.0

    k1 = 1.5
    b = 0.75
    if document_count == 0:
        return 0.0

    score = 0.0
    doc_len = len(document_terms)
    for term in set(query_terms):
        frequency = document_terms.count(term)
        if frequency == 0:
            continue
        doc_frequency = doc_freqs.get(term, 0)
        if doc_frequency == 0:
            continue
        idf = math.log(1 + (document_count - doc_frequency + 0.5) / (doc_frequency + 0.5))
        numerator = frequency * (k1 + 1)
        denominator = frequency + k1 * (1 - b + b * (doc_len / max(avg_doc_len, 1.0)))
        score += idf * numerator / denominator
    return score


def _elapsed_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000, 2)


def _repair_exact_table_lookup_context(
    item: RetrievedDocument,
    *,
    source_document: Document | None,
    query: str,
) -> RetrievedDocument:
    identifier = _requested_table_identifier(query)
    if identifier is None or source_document is None:
        return item
    if _contains_table_row(item.text, identifier) or not _contains_table_row(
        source_document.text,
        identifier,
    ):
        return item
    return RetrievedDocument(
        document_id=item.document_id,
        filename=item.filename,
        text=source_document.text,
        score=item.score,
        chunk_id=item.chunk_id,
        chunk_index=item.chunk_index,
        bm25_score=item.bm25_score,
        vector_score=item.vector_score,
        rrf_score=item.rrf_score,
        entity_anchor_score=item.entity_anchor_score,
    )


def _requested_table_identifier(query: str) -> str | None:
    code_match = re.search(
        r"(?i)\bcode\s+([A-Z]{1,12}(?:-[A-Z0-9]{1,12})+)\b",
        query,
    )
    if code_match:
        return code_match.group(1).upper()
    fallback_match = re.search(r"\b[A-Z]{1,12}(?:-[A-Z0-9]{1,12})+\b", query)
    if fallback_match and re.search(
        r"(?i)\b(?:mean|means|meaning|define|what is|who is|does|do|require|required|requires|approval|exception|qualify|qualifies|eligible)\b",
        query,
    ):
        return fallback_match.group(0).upper()
    return None


def _contains_table_row(text: str, code: str) -> bool:
    return bool(
        re.search(
            rf"(?i)(?:^|\s)\|\s*{re.escape(code)}\s*\|\s*[^|]+?\s*\|",
            text,
        )
    )


def _entity_phrases(query: str) -> list[str]:
    phrases: list[str] = []
    title_case_matches = re.findall(r"\b[A-Z][a-z0-9-]*(?:\s+[A-Z][a-z0-9-]*)+\b", query)
    phrases.extend(title_case_matches)

    lookup_match = re.search(
        r"(?i)\b(?:what is|who is|define|explain|describe)\s+([A-Za-z0-9][A-Za-z0-9\s-]{1,80})",
        query,
    )
    if lookup_match:
        phrases.append(lookup_match.group(1).strip(" ?.!,:;\"'"))

    seen: set[str] = set()
    normalized: list[str] = []
    for phrase in phrases:
        lowered = " ".join(_terms(phrase))
        if len(lowered) < 3 or lowered in seen:
            continue
        seen.add(lowered)
        normalized.append(lowered)
    return normalized


def _entity_anchor_score(entity_phrases: list[str], document: Document) -> float:
    if not entity_phrases:
        return 0.0

    haystack = " ".join(_terms(f"{document.filename} {document.text}"))
    strongest = 0.0
    for phrase in entity_phrases:
        if phrase and phrase in haystack:
            strongest = max(strongest, 12.0)
    return strongest
