from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Literal
from time import perf_counter
from urllib.parse import urlparse
from uuid import uuid4

from pydantic import BaseModel, Field

from pra_core.documents.chunking import TextChunker
from pra_core.documents.filenames import normalize_filename
from pra_core.documents.models import Document, ParsedDocument, RetrievedDocument
from pra_core.documents.repository import LocalDocumentRepository
from pra_core.embeddings.base import EmbeddingProvider
from pra_core.indexing import DocumentIndexer
from pra_core.logging import get_logger
from pra_core.retrieval.base import Retriever
from pra_core.retrieval.fusion import FusionConfig, merge_retrieval_hits, no_rerank_key, rerank_key
from pra_core.vectorstores.local import LocalVectorStore

logger = get_logger(__name__)


class RagEvaluationItem(BaseModel):
    id: str
    question: str
    intent: str = ""
    source_files: list[str] = Field(default_factory=list)
    expected_sections: list[str] = Field(default_factory=list)
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    expected_answer_summary: str = ""
    expected_answer_terms: list[str] = Field(default_factory=list)
    expected_behavior: Literal["answer", "insufficient_evidence"] = "answer"


class RagEvaluationDataset(BaseModel):
    name: str
    default_top_k: int = 25
    eval_k: list[int] = Field(default_factory=lambda: [5, 10, 25])
    items: list[RagEvaluationItem] = Field(default_factory=list)


class RagEvaluationQuestionResult(BaseModel):
    id: str
    question: str
    labeled: bool
    retrieved_ids: list[str]
    relevant_chunk_ids: list[str]
    ground_truth_warnings: list[str] = Field(default_factory=list)
    metrics: dict[str, float | None]
    retrieval_latency_ms: float
    error: str | None = None


class RagEvaluationSummary(BaseModel):
    dataset: str
    total_questions: int
    labeled_questions: int
    unlabeled_questions: int
    failed_question_count: int
    eval_k: list[int]
    metrics: dict[str, float | None]
    latency: dict[str, float]
    results: list[RagEvaluationQuestionResult]


class RagEvaluationSourceDocument(BaseModel):
    requested_name: str
    resolved_name: str
    path: str


@dataclass(frozen=True)
class _EvaluationChunkEntry:
    document_id: str
    filename: str
    text: str
    chunk_id: str
    chunk_index: int
    tokens: list[str]
    metadata: dict


@dataclass(frozen=True)
class GroundTruthResolutionIssue:
    question_id: str
    issue: str
    source_file: str | None = None
    expected_section: str | None = None


@dataclass(frozen=True)
class ResolvedRelevantChunks:
    relevant_chunk_ids: list[str]
    warnings: list[str]
    issues: list[GroundTruthResolutionIssue]


def load_rag_evaluation_dataset(path: str | Path) -> RagEvaluationDataset:
    if _is_s3_uri(path):
        payload = json.loads(_read_s3_text(str(path)))
    else:
        dataset_path = Path(path)
        with dataset_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    return RagEvaluationDataset.model_validate(payload)


def resolve_dataset_path(configured_path: str) -> str | Path:
    if _is_s3_uri(configured_path):
        return configured_path
    candidate = Path(configured_path)
    if candidate.is_absolute():
        return candidate
    return Path.cwd() / candidate


def evaluate_retrieval_only(
    *,
    retriever: Retriever,
    dataset: RagEvaluationDataset,
    top_k: int,
    eval_k: list[int],
    repeat: int = 1,
    chunk_entries: list[_EvaluationChunkEntry] | None = None,
) -> RagEvaluationSummary:
    started_at = perf_counter()
    results: list[RagEvaluationQuestionResult] = []
    labeled_count = 0
    unlabeled_count = 0
    failed_count = 0
    latency_measurements: list[float] = []
    aggregate_metric_values: dict[str, list[float]] = {
        _metric_name(metric, k): []
        for k in eval_k
        for metric in ("recall", "precision", "mrr", "ndcg")
    }

    for _run_index in range(repeat):
        for item in dataset.items:
            question_started_at = perf_counter()
            resolved = resolve_relevant_chunks(
                item=item,
                retriever=retriever,
                chunk_entries=chunk_entries,
            )
            try:
                retrieved = retriever.search(item.question, k=top_k)
                retrieval_latency_ms = round((perf_counter() - question_started_at) * 1000, 2)
                latency_measurements.append(retrieval_latency_ms)
                retrieved_ids = [_retrieved_id(document) for document in retrieved]
                labeled = bool(resolved.relevant_chunk_ids)
                if labeled:
                    labeled_count += 1
                else:
                    unlabeled_count += 1
                result_metrics = _question_metrics(
                    retrieved_documents=retrieved,
                    relevant_chunk_ids=resolved.relevant_chunk_ids,
                    eval_k=eval_k,
                )
                if labeled:
                    for name, value in result_metrics.items():
                        if value is not None:
                            aggregate_metric_values[name].append(value)
                results.append(
                    RagEvaluationQuestionResult(
                        id=item.id,
                        question=item.question,
                        labeled=labeled,
                        retrieved_ids=retrieved_ids,
                        relevant_chunk_ids=resolved.relevant_chunk_ids,
                        ground_truth_warnings=resolved.warnings,
                        metrics=result_metrics,
                        retrieval_latency_ms=retrieval_latency_ms,
                    )
                )
            except Exception as exc:
                failed_count += 1
                labeled = bool(resolved.relevant_chunk_ids)
                if labeled:
                    labeled_count += 1
                else:
                    unlabeled_count += 1
                results.append(
                    RagEvaluationQuestionResult(
                        id=item.id,
                        question=item.question,
                        labeled=labeled,
                        retrieved_ids=[],
                        relevant_chunk_ids=resolved.relevant_chunk_ids,
                        ground_truth_warnings=resolved.warnings,
                        metrics=_null_metrics(eval_k),
                        retrieval_latency_ms=round((perf_counter() - question_started_at) * 1000, 2),
                        error=str(exc),
                    )
                )

    metrics = {
        name: round(mean(values), 4) if values else None
        for name, values in aggregate_metric_values.items()
    }
    duration_ms = round((perf_counter() - started_at) * 1000, 2)
    logger.info(
        "rag_evaluation_metrics_computed",
        extra={
            "dataset_name": dataset.name,
            "total_questions": len(results),
            "labeled_count": labeled_count,
            "unlabeled_count": unlabeled_count,
            "eval_k": eval_k,
            "duration_ms": duration_ms,
            "failed_question_count": failed_count,
        },
    )
    return RagEvaluationSummary(
        dataset=dataset.name,
        total_questions=len(results),
        labeled_questions=labeled_count,
        unlabeled_questions=unlabeled_count,
        failed_question_count=failed_count,
        eval_k=eval_k,
        metrics=metrics,
        latency={
            "avg_retrieval_latency_ms": round(mean(latency_measurements), 2)
            if latency_measurements
            else 0.0
        },
        results=results,
    )


def build_isolated_evaluation_retriever(
    *,
    dataset: RagEvaluationDataset,
    chunker: TextChunker,
    embedding_provider: EmbeddingProvider,
    docs_root: Path | None = None,
    bm25_top_k: int = 25,
    vector_top_k: int = 25,
    rrf_k: int = 60,
    enable_bm25_lexical: bool = True,
    enable_vector_search: bool = True,
    enable_reciprocal_rank_fusion: bool = True,
    enable_reranking: bool = False,
    fusion_strategy: str = "rrf",
    bm25_fusion_weight: float = 0.7,
    vector_fusion_weight: float = 0.3,
    preserve_bm25_candidates: bool = False,
    bm25_first: bool = False,
) -> tuple[Retriever, list[RagEvaluationSourceDocument]]:
    source_documents = load_evaluation_source_documents(dataset, docs_root=docs_root)
    repository = LocalDocumentRepository()
    vector_store = LocalVectorStore()
    indexer = DocumentIndexer(
        chunker=chunker,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    for source_document in source_documents:
        stored = repository.add(
            ParsedDocument(
                filename=source_document.resolved_name,
                content_type="text/markdown",
                text=Path(source_document.path).read_text(encoding="utf-8"),
            )
        )
        indexer.index(stored)

    return (
        EvaluationChunkHybridRetriever(
            repository=repository,
            chunker=chunker,
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            bm25_top_k=bm25_top_k,
            vector_top_k=vector_top_k,
            rrf_k=rrf_k,
            enable_bm25_lexical=enable_bm25_lexical,
            enable_vector_search=enable_vector_search,
            enable_reciprocal_rank_fusion=enable_reciprocal_rank_fusion,
            enable_reranking=enable_reranking,
            fusion_strategy=fusion_strategy,
            bm25_fusion_weight=bm25_fusion_weight,
            vector_fusion_weight=vector_fusion_weight,
            preserve_bm25_candidates=preserve_bm25_candidates,
            bm25_first=bm25_first,
        ),
        source_documents,
    )


def _retrieved_id(document) -> str:
    return document.chunk_id or document.document_id


def _retrieved_labels(document) -> set[str]:
    labels = {
        _retrieved_id(document),
        document.document_id,
        document.filename,
    }
    if document.chunk_index is not None:
        labels.add(f"{document.filename}#chunk-{document.chunk_index}")
    return {label for label in labels if label}


def _question_metrics(
    *,
    retrieved_documents,
    relevant_chunk_ids: list[str],
    eval_k: list[int],
) -> dict[str, float | None]:
    if not relevant_chunk_ids:
        return _null_metrics(eval_k)

    relevant = set(relevant_chunk_ids)
    retrieved_labels = [_retrieved_labels(document) for document in retrieved_documents]
    metrics: dict[str, float | None] = {}
    for k in eval_k:
        limited = retrieved_labels[:k]
        matched = set().union(*limited) & relevant if limited else set()
        true_positives = len(matched)
        metrics[_metric_name("precision", k)] = round(true_positives / k, 4)
        metrics[_metric_name("recall", k)] = round(true_positives / len(relevant), 4)
        metrics[_metric_name("mrr", k)] = round(_mrr_at_k(limited, relevant), 4)
        metrics[_metric_name("ndcg", k)] = round(_ndcg_at_k(limited, relevant), 4)
    return metrics


def resolve_relevant_chunks(
    *,
    item: RagEvaluationItem,
    retriever,
    chunk_entries: list[_EvaluationChunkEntry] | None = None,
) -> ResolvedRelevantChunks:
    return resolve_relevant_chunks_from_entries(
        item=item,
        chunk_entries=chunk_entries or _chunk_entries_from_retriever(retriever),
    )


def resolve_relevant_chunks_from_entries(
    *,
    item: RagEvaluationItem,
    chunk_entries: list[_EvaluationChunkEntry],
) -> ResolvedRelevantChunks:
    normalized_sources = [_normalize_source_name(name) for name in item.source_files]
    warnings: list[str] = []
    issues: list[GroundTruthResolutionIssue] = []
    relevant_ids: list[str] = []
    relevant_seen: set[str] = set()

    if not normalized_sources:
        warnings.append(f"{item.id}: no source_files configured for evaluation ground truth.")
        issues.append(GroundTruthResolutionIssue(question_id=item.id, issue="missing_source_files"))
        return ResolvedRelevantChunks(relevant_chunk_ids=[], warnings=warnings, issues=issues)

    if not item.expected_sections:
        warnings.append(f"{item.id}: no expected_sections configured for evaluation ground truth.")
        issues.append(GroundTruthResolutionIssue(question_id=item.id, issue="missing_expected_sections"))
        return ResolvedRelevantChunks(relevant_chunk_ids=[], warnings=warnings, issues=issues)

    entries_by_source: dict[str, list[_EvaluationChunkEntry]] = {}
    for entry in chunk_entries:
        entries_by_source.setdefault(_normalize_source_name(entry.filename), []).append(entry)

    for source in normalized_sources:
        source_entries = entries_by_source.get(source, [])
        if not source_entries:
            warnings.append(f"{item.id}: source file not found in live index: {source}.")
            issues.append(
                GroundTruthResolutionIssue(
                    question_id=item.id,
                    issue="source_file_not_found",
                    source_file=source,
                )
            )
            continue

        for section in item.expected_sections:
            selected = _select_section_chunks(item=item, section=section, chunks=source_entries)
            if not selected:
                warnings.append(
                    f"{item.id}: expected section not found in live index: {source} -> {section}."
                )
                issues.append(
                    GroundTruthResolutionIssue(
                        question_id=item.id,
                        issue="expected_section_not_found",
                        source_file=source,
                        expected_section=section,
                    )
                )
                continue
            for chunk in selected:
                if chunk.chunk_id in relevant_seen:
                    continue
                relevant_ids.append(chunk.chunk_id)
                relevant_seen.add(chunk.chunk_id)

    return ResolvedRelevantChunks(
        relevant_chunk_ids=relevant_ids,
        warnings=warnings,
        issues=issues,
    )


def _chunk_entries_from_retriever(retriever) -> list[_EvaluationChunkEntry]:
    direct_entries = getattr(retriever, "_chunk_entries", None)
    if direct_entries is not None:
        return list(direct_entries)

    repository = getattr(retriever, "_repository", None)
    chunker = getattr(retriever, "_chunker", None)
    if repository is None or chunker is None:
        return []

    entries: list[_EvaluationChunkEntry] = []
    for document in repository.list():
        for chunk in chunker.chunk(document):
            entries.append(
                _EvaluationChunkEntry(
                    document_id=chunk.document_id,
                    filename=chunk.filename,
                    text=chunk.text,
                    chunk_id=chunk.id,
                    chunk_index=chunk.index,
                    tokens=_terms(f"{chunk.filename} {chunk.text}"),
                    metadata=chunk.metadata,
                )
            )
    return entries


def _select_section_chunks(
    *,
    item: RagEvaluationItem,
    section: str,
    chunks: list[_EvaluationChunkEntry],
) -> list[_EvaluationChunkEntry]:
    scored = [
        (chunk, _section_ground_truth_score(section, item, chunk))
        for chunk in chunks
    ]
    matching = [
        (chunk, score)
        for chunk, score in scored
        if score > 0
    ]
    if not matching:
        return []

    matching.sort(key=lambda pair: (pair[1], -pair[0].chunk_index), reverse=True)
    exact_heading_matches = [
        chunk
        for chunk, _score in matching
        if _normalized_text(str(chunk.metadata.get("heading", ""))) == _normalized_text(section)
    ]
    if exact_heading_matches:
        return exact_heading_matches

    max_chunks = min(max(len(item.expected_sections), 1), 4)
    return [chunk for chunk, _score in matching[:max_chunks]]


def _section_ground_truth_score(
    section: str,
    item: RagEvaluationItem,
    chunk: _EvaluationChunkEntry,
) -> float:
    heading = str(chunk.metadata.get("heading", ""))
    normalized_section = _normalized_text(section)
    normalized_heading = _normalized_text(heading)
    normalized_text = _normalized_text(chunk.text)
    heading_tokens = set(_terms(heading))
    section_tokens = set(_terms(section))
    text_tokens = set(_terms(chunk.text))

    score = 0.0
    if normalized_section and normalized_heading == normalized_section:
        score += 100.0
    if normalized_section and normalized_section in normalized_text:
        score += 25.0
    score += len(heading_tokens & section_tokens) * 15.0
    score += len(text_tokens & section_tokens) * 3.0
    score += len(text_tokens & set(_terms(item.expected_answer_summary))) * 4.0
    score += len(text_tokens & set(_terms(item.question))) * 2.0
    return score


def _normalized_text(value: str) -> str:
    return " ".join(_terms(value))


def _null_metrics(eval_k: list[int]) -> dict[str, float | None]:
    return {
        _metric_name(metric, k): None
        for k in eval_k
        for metric in ("recall", "precision", "mrr", "ndcg")
    }


def _metric_name(metric: str, k: int) -> str:
    return f"{metric}_at_{k}"


def _mrr_at_k(retrieved_labels: list[set[str]], relevant: set[str]) -> float:
    for index, labels in enumerate(retrieved_labels, start=1):
        if labels & relevant:
            return 1.0 / index
    return 0.0


def _ndcg_at_k(retrieved_labels: list[set[str]], relevant: set[str]) -> float:
    dcg = 0.0
    for index, labels in enumerate(retrieved_labels, start=1):
        if labels & relevant:
            dcg += 1.0 / math.log2(index + 1)
    ideal_hits = min(len(relevant), len(retrieved_labels))
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def _is_s3_uri(value: str | Path) -> bool:
    return str(value).startswith("s3://")


def load_evaluation_source_documents(
    dataset: RagEvaluationDataset,
    docs_root: Path | None = None,
) -> list[RagEvaluationSourceDocument]:
    root = docs_root or _default_docs_root()
    available = _source_document_index(root)
    selected: list[RagEvaluationSourceDocument] = []
    seen: set[str] = set()

    for item in dataset.items:
        for requested_name in item.source_files:
            normalized = _normalize_source_name(requested_name)
            if normalized in seen:
                continue
            path = available.get(normalized)
            if path is None:
                continue
            seen.add(normalized)
            selected.append(
                RagEvaluationSourceDocument(
                    requested_name=requested_name,
                    resolved_name=normalized,
                    path=str(path),
                )
            )

    return selected


def _default_docs_root() -> Path:
    return Path(__file__).resolve().parents[5] / "docs"


def _source_document_index(root: Path) -> dict[str, Path]:
    if not root.exists():
        return {}

    index: dict[str, Path] = {}
    for path in root.rglob("*.md"):
        normalized = _normalize_source_name(path.name)
        index.setdefault(normalized, path)
    return index


def _normalize_source_name(name: str) -> str:
    return normalize_filename(name)


class EvaluationChunkHybridRetriever(Retriever):
    """Chunk-level hybrid retriever used only for isolated RAG evaluation."""

    def __init__(
        self,
        *,
        repository: LocalDocumentRepository,
        chunker: TextChunker,
        embedding_provider: EmbeddingProvider,
        vector_store: LocalVectorStore,
        bm25_top_k: int = 25,
        vector_top_k: int = 25,
        rrf_k: int = 60,
        enable_bm25_lexical: bool = True,
        enable_vector_search: bool = True,
        enable_reciprocal_rank_fusion: bool = True,
        enable_reranking: bool = False,
        fusion_strategy: str = "rrf",
        bm25_fusion_weight: float = 0.7,
        vector_fusion_weight: float = 0.3,
        preserve_bm25_candidates: bool = False,
        bm25_first: bool = False,
    ) -> None:
        self._repository = repository
        self._chunker = chunker
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._bm25_top_k = bm25_top_k
        self._vector_top_k = vector_top_k
        self._rrf_k = rrf_k
        self._enable_bm25_lexical = enable_bm25_lexical
        self._enable_vector_search = enable_vector_search
        self._enable_reciprocal_rank_fusion = (
            enable_reciprocal_rank_fusion and enable_bm25_lexical and enable_vector_search
        )
        self._enable_reranking = enable_reranking
        self._bm25_first = bm25_first
        self._fusion_strategy = fusion_strategy
        self._bm25_fusion_weight = bm25_fusion_weight
        self._vector_fusion_weight = vector_fusion_weight
        self._preserve_bm25_candidates = preserve_bm25_candidates
        self._chunk_entries = self._build_chunk_entries()
        self._avg_doc_len = (
            sum(len(entry.tokens) for entry in self._chunk_entries)
            / max(len(self._chunk_entries), 1)
        )
        self._doc_freqs = _document_frequencies(
            [entry.tokens for entry in self._chunk_entries]
        )
        self._query_cache: dict[str, list[RetrievedDocument]] = {}

    def search(self, query: str, k: int = 25) -> list[RetrievedDocument]:
        cached = self._query_cache.get(query)
        if cached is not None:
            return cached[:k]

        bm25_hits = self._bm25_search(query) if self._enable_bm25_lexical else []
        use_hybrid_fallback = self._should_use_hybrid_fallback(query, bm25_hits)
        vector_hits = (
            self._vector_search(query)
            if self._enable_vector_search and (not self._bm25_first or use_hybrid_fallback)
            else []
        )
        if self._bm25_first and not use_hybrid_fallback:
            ranked = sorted(bm25_hits, key=lambda item: item.bm25_score or 0.0, reverse=True)
            self._query_cache[query] = ranked
            return ranked[:k]
        merged = (
            self._merge_rrf(bm25_hits, vector_hits)
            if self._enable_reciprocal_rank_fusion
            else self._merge_by_score(bm25_hits, vector_hits)
        )
        if self._enable_reranking:
            ranked = sorted(merged.values(), key=rerank_key, reverse=True)
        else:
            ranked = sorted(merged.values(), key=no_rerank_key, reverse=True)
        self._query_cache[query] = ranked
        return ranked[:k]

    def _should_use_hybrid_fallback(self, query: str, bm25_hits: list[RetrievedDocument]) -> bool:
        if not self._bm25_first:
            return True
        scores = sorted((item.bm25_score or 0.0 for item in bm25_hits), reverse=True)
        if not scores:
            return True
        top_score = scores[0]
        score_gap = top_score - scores[1] if len(scores) > 1 else top_score
        strong_matches = sum(score >= max(1.0, top_score * 0.55) for score in scores)
        identifiers = [item.lower() for item in re.findall(r"\b[A-Z]{2,}-?\d+\b", query)]
        exact_anchor = bool(identifiers) and any(
            all(identifier in item.text.lower() for identifier in identifiers)
            for item in bm25_hits[:5]
        )
        return (
            (top_score < 1.0 and not exact_anchor)
            or (score_gap < 0.15 and strong_matches < 2)
        )

    def _build_chunk_entries(self) -> list[_EvaluationChunkEntry]:
        chunks: list[_EvaluationChunkEntry] = []
        for document in self._repository.list():
            for chunk in self._chunker.chunk(document):
                chunks.append(
                    _EvaluationChunkEntry(
                        document_id=chunk.document_id,
                        filename=chunk.filename,
                        text=chunk.text,
                        chunk_id=chunk.id,
                        chunk_index=chunk.index,
                        tokens=_terms(f"{chunk.filename} {chunk.text}"),
                        metadata=chunk.metadata,
                    )
                )
        return chunks

    def _bm25_search(self, query: str) -> list[RetrievedDocument]:
        query_terms = _terms(query)
        entity_phrases = _entity_phrases(query)
        if not query_terms:
            return []

        results: list[RetrievedDocument] = []

        for entry in self._chunk_entries:
            score = _bm25_score(
                query_terms,
                entry.tokens,
                self._doc_freqs,
                self._avg_doc_len,
                len(self._chunk_entries),
            )
            score += _entity_anchor_score(entity_phrases, entry.filename, entry.text)
            if score <= 0:
                continue
            results.append(
                RetrievedDocument(
                    document_id=entry.document_id,
                    filename=entry.filename,
                    text=entry.text,
                    score=score,
                    chunk_id=entry.chunk_id,
                    chunk_index=entry.chunk_index,
                    bm25_score=score,
                    vector_score=0.0,
                    rrf_score=0.0,
                )
            )

        return sorted(results, key=lambda item: item.bm25_score or 0.0, reverse=True)[: self._bm25_top_k]

    def _vector_search(self, query: str) -> list[RetrievedDocument]:
        query_embedding = self._embedding_provider.embed(query)
        vector_results = self._vector_store.search(query_embedding, k=self._vector_top_k)
        return [
            RetrievedDocument(
                document_id=result.chunk.document_id,
                filename=result.chunk.filename,
                text=result.chunk.text,
                score=result.score,
                chunk_id=result.chunk.id,
                chunk_index=result.chunk.index,
                bm25_score=0.0,
                vector_score=result.score,
                rrf_score=0.0,
            )
            for result in vector_results
            if result.score > 0
        ]

    def _merge_rrf(
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
            key_fn=lambda item: item.chunk_id or item.document_id,
            prefer_vector_chunk=True,
        )

    def _merge_by_score(
        self,
        bm25_hits: list[RetrievedDocument],
        vector_hits: list[RetrievedDocument],
    ) -> dict[str, RetrievedDocument]:
        merged: dict[str, RetrievedDocument] = {}
        for hit in [*bm25_hits, *vector_hits]:
            key = hit.chunk_id or hit.document_id or str(uuid4())
            hit_score = max(hit.rrf_score or 0.0, hit.vector_score or 0.0, hit.bm25_score or 0.0)
            existing = merged.get(key)
            existing_score = (
                max(
                    existing.rrf_score or 0.0,
                    existing.vector_score or 0.0,
                    existing.bm25_score or 0.0,
                )
                if existing
                else -1.0
            )
            if existing is not None and existing_score >= hit_score:
                continue
            merged[key] = RetrievedDocument(
                document_id=hit.document_id,
                filename=hit.filename,
                text=hit.text,
                score=hit_score,
                chunk_id=hit.chunk_id,
                chunk_index=hit.chunk_index,
                bm25_score=hit.bm25_score,
                vector_score=hit.vector_score,
                rrf_score=hit_score,
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
    if not document_terms or document_count == 0:
        return 0.0

    k1 = 1.5
    b = 0.75
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


def _entity_anchor_score(entity_phrases: list[str], filename: str, text: str) -> float:
    if not entity_phrases:
        return 0.0

    haystack = " ".join(_terms(f"{filename} {text}"))
    strongest = 0.0
    for phrase in entity_phrases:
        if phrase and phrase in haystack:
            strongest = max(strongest, 12.0)
    return strongest


def s3_object_exists(uri: str) -> bool:
    parsed = urlparse(uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    if not bucket or not key:
        return False

    import boto3
    from botocore.exceptions import ClientError

    try:
        boto3.client("s3").head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code", "")
        if error_code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise


def _read_s3_text(uri: str) -> str:
    parsed = urlparse(uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    if not bucket or not key:
        raise ValueError("S3 dataset path must be in the form s3://bucket/key")

    import boto3

    response = boto3.client("s3").get_object(Bucket=bucket, Key=key)
    return response["Body"].read().decode("utf-8")
