from __future__ import annotations

from pathlib import Path
from statistics import mean
from time import perf_counter

from pydantic import BaseModel, Field

from pra_core.documents.chunking import TextChunker
from pra_core.documents.models import ParsedDocument, RetrievedDocument
from pra_core.documents.repository import LocalDocumentRepository
from pra_core.embeddings.base import EmbeddingProvider
from pra_core.evaluation.rag_retrieval import (
    EvaluationChunkHybridRetriever,
    RagEvaluationDataset,
    RagEvaluationItem,
    build_isolated_evaluation_retriever,
    _metric_name,
    _normalize_source_name,
    _question_metrics,
    _retrieved_labels,
    load_evaluation_source_documents,
    resolve_relevant_chunks_from_entries,
)
from pra_core.indexing import DocumentIndexer
from pra_core.retrieval.hybrid import HybridRetriever
from pra_core.retrieval.fusion import no_rerank_key, rerank_key
from pra_core.vectorstores.local import LocalVectorStore


class RetrievalDiagnosticConfig(BaseModel):
    bm25_top_k: int = 50
    vector_top_k: int = 50
    rrf_k: int = 30
    fusion_strategy: str = "weighted_rrf"
    bm25_fusion_weight: float = 0.8
    vector_fusion_weight: float = 0.2
    preserve_bm25_candidates: bool = True
    enable_bm25_lexical: bool = True
    enable_vector_search: bool = True
    enable_reciprocal_rank_fusion: bool = True
    enable_reranking: bool = False
    rerank_top_k: int = 50
    llm_context_top_k: int = 10
    eval_k: list[int] = Field(default_factory=lambda: [5, 10, 25])


class ChunkDiagnosticConfig(BaseModel):
    chunk_size: int = 600
    chunk_overlap: int = 100


class LabelIntegrityIssue(BaseModel):
    question_id: str
    relevant_chunk_id: str
    issue: str
    expected_source_files: list[str]
    actual_source_file: str | None = None


class QuestionLabelIntegrity(BaseModel):
    question_id: str
    expected_source_files: list[str]
    present_ids: list[str]
    missing_ids: list[str]
    source_file_mismatches: dict[str, list[str]]


class StageDiagnosticMetrics(BaseModel):
    stage: str
    metrics: dict[str, float | None]
    retrieved_ids: list[str]


class RelevantChunkSurvival(BaseModel):
    relevant_chunk_id: str
    bm25_rank: int | None
    vector_rank: int | None
    rrf_rank: int | None
    reranked_rank: int | None
    included_in_final_llm_context: bool


class RetrievalLatencyBreakdown(BaseModel):
    bm25_latency_ms: float
    vector_latency_ms: float
    rrf_merge_latency_ms: float
    reranker_latency_ms: float
    total_retrieval_latency_ms: float


class QuestionRetrievalDiagnostic(BaseModel):
    id: str
    question: str
    label_integrity: QuestionLabelIntegrity
    stage_metrics: list[StageDiagnosticMetrics]
    relevant_chunk_survival: list[RelevantChunkSurvival]
    latency: RetrievalLatencyBreakdown


class RetrievalLossSummary(BaseModel):
    total_expected_relevant_ids: int
    present_in_index: int
    missing_from_index: int
    missing_in_bm25: int
    missing_in_vector: int
    missing_after_rrf: int
    missing_after_rerank: int
    missing_from_final_context: int


class RetrievalContributionSummary(BaseModel):
    bm25_only_relevant_hits: int
    vector_only_relevant_hits: int
    shared_relevant_hits: int
    unseen_relevant_hits: int


class RetrievalDiagnosticRun(BaseModel):
    name: str
    config: RetrievalDiagnosticConfig
    chunk_config: ChunkDiagnosticConfig
    label_integrity_issues: list[LabelIntegrityIssue]
    label_integrity_by_question: list[QuestionLabelIntegrity]
    stage_metric_means: dict[str, dict[str, float | None]]
    loss_summary: RetrievalLossSummary
    contribution_summary: RetrievalContributionSummary
    final_context_coverage_rate: float
    latency_means: dict[str, float]
    questions: list[QuestionRetrievalDiagnostic]


class RetrievalDiagnosticsReport(BaseModel):
    dataset: str
    total_questions: int
    base_run: RetrievalDiagnosticRun
    ablations: list[RetrievalDiagnosticRun]
    recommended_run_name: str


def run_retrieval_diagnostics(
    *,
    dataset: RagEvaluationDataset,
    embedding_provider: EmbeddingProvider,
    docs_root: Path | None = None,
    base_config: RetrievalDiagnosticConfig | None = None,
    base_chunk_config: ChunkDiagnosticConfig | None = None,
    rrf_k_values: list[int] | None = None,
    bm25_top_k_values: list[int] | None = None,
    vector_top_k_values: list[int] | None = None,
    chunk_configs: list[ChunkDiagnosticConfig] | None = None,
) -> RetrievalDiagnosticsReport:
    config = base_config or RetrievalDiagnosticConfig()
    chunk_config = base_chunk_config or ChunkDiagnosticConfig()
    base_run = _run_single_diagnostic(
        name="base",
        dataset=dataset,
        embedding_provider=embedding_provider,
        docs_root=docs_root,
        config=config,
        chunk_config=chunk_config,
    )

    ablations: list[RetrievalDiagnosticRun] = []
    for value in rrf_k_values or [20, 30, 60, 90]:
        ablations.append(
            _run_single_diagnostic(
                name=f"rrf_k={value}",
                dataset=dataset,
                embedding_provider=embedding_provider,
                docs_root=docs_root,
                config=config.model_copy(update={"rrf_k": value}),
                chunk_config=chunk_config,
            )
        )
    for value in bm25_top_k_values or [25, 50, 75, 100]:
        ablations.append(
            _run_single_diagnostic(
                name=f"bm25_top_k={value}",
                dataset=dataset,
                embedding_provider=embedding_provider,
                docs_root=docs_root,
                config=config.model_copy(update={"bm25_top_k": value}),
                chunk_config=chunk_config,
            )
        )
    for value in vector_top_k_values or [25, 50, 75, 100]:
        ablations.append(
            _run_single_diagnostic(
                name=f"vector_top_k={value}",
                dataset=dataset,
                embedding_provider=embedding_provider,
                docs_root=docs_root,
                config=config.model_copy(update={"vector_top_k": value}),
                chunk_config=chunk_config,
            )
        )
    for candidate in chunk_configs or [
        ChunkDiagnosticConfig(chunk_size=300, chunk_overlap=50),
        ChunkDiagnosticConfig(chunk_size=450, chunk_overlap=75),
        ChunkDiagnosticConfig(chunk_size=600, chunk_overlap=100),
    ]:
        ablations.append(
            _run_single_diagnostic(
                name=f"chunk={candidate.chunk_size}/{candidate.chunk_overlap}",
                dataset=dataset,
                embedding_provider=embedding_provider,
                docs_root=docs_root,
                config=config,
                chunk_config=candidate,
            )
        )
    for value in [False, True]:
        if value == config.enable_reranking:
            continue
        ablations.append(
            _run_single_diagnostic(
                name=f"enable_reranking={str(value).lower()}",
                dataset=dataset,
                embedding_provider=embedding_provider,
                docs_root=docs_root,
                config=config.model_copy(update={"enable_reranking": value}),
                chunk_config=chunk_config,
            )
        )
    for value in [25, 50, 75]:
        if value == config.rerank_top_k:
            continue
        ablations.append(
            _run_single_diagnostic(
                name=f"rerank_top_k={value}",
                dataset=dataset,
                embedding_provider=embedding_provider,
                docs_root=docs_root,
                config=config.model_copy(update={"rerank_top_k": value}),
                chunk_config=chunk_config,
            )
        )
    for value in [5, 8, 10, 15]:
        if value == config.llm_context_top_k:
            continue
        ablations.append(
            _run_single_diagnostic(
                name=f"llm_context_top_k={value}",
                dataset=dataset,
                embedding_provider=embedding_provider,
                docs_root=docs_root,
                config=config.model_copy(update={"llm_context_top_k": value}),
                chunk_config=chunk_config,
            )
        )
    ablations.extend(
        [
            _run_single_diagnostic(
                name="bm25_only",
                dataset=dataset,
                embedding_provider=embedding_provider,
                docs_root=docs_root,
                config=config.model_copy(
                    update={
                        "enable_bm25_lexical": True,
                        "enable_vector_search": False,
                        "enable_reciprocal_rank_fusion": False,
                    }
                ),
                chunk_config=chunk_config,
            ),
            _run_single_diagnostic(
                name="vector_only",
                dataset=dataset,
                embedding_provider=embedding_provider,
                docs_root=docs_root,
                config=config.model_copy(
                    update={
                        "enable_bm25_lexical": False,
                        "enable_vector_search": True,
                        "enable_reciprocal_rank_fusion": False,
                    }
                ),
                chunk_config=chunk_config,
            ),
            _run_single_diagnostic(
                name="weighted_rrf_0.7_0.3",
                dataset=dataset,
                embedding_provider=embedding_provider,
                docs_root=docs_root,
                config=config.model_copy(
                    update={
                        "fusion_strategy": "weighted_rrf",
                        "bm25_fusion_weight": 0.7,
                        "vector_fusion_weight": 0.3,
                    }
                ),
                chunk_config=chunk_config,
            ),
            _run_single_diagnostic(
                name="weighted_rrf_0.8_0.2_preserve_bm25",
                dataset=dataset,
                embedding_provider=embedding_provider,
                docs_root=docs_root,
                config=config.model_copy(
                    update={
                        "fusion_strategy": "weighted_rrf",
                        "bm25_fusion_weight": 0.8,
                        "vector_fusion_weight": 0.2,
                        "preserve_bm25_candidates": True,
                    }
                ),
                chunk_config=chunk_config,
            ),
            _run_single_diagnostic(
                name="normalized_weighted_0.8_0.2",
                dataset=dataset,
                embedding_provider=embedding_provider,
                docs_root=docs_root,
                config=config.model_copy(
                    update={
                        "fusion_strategy": "normalized_weighted",
                        "bm25_fusion_weight": 0.8,
                        "vector_fusion_weight": 0.2,
                    }
                ),
                chunk_config=chunk_config,
            ),
            _run_single_diagnostic(
                name="normalized_weighted_0.7_0.3",
                dataset=dataset,
                embedding_provider=embedding_provider,
                docs_root=docs_root,
                config=config.model_copy(
                    update={
                        "fusion_strategy": "normalized_weighted",
                        "bm25_fusion_weight": 0.7,
                        "vector_fusion_weight": 0.3,
                    }
                ),
                chunk_config=chunk_config,
            ),
            _run_single_diagnostic(
                name="normalized_weighted_0.6_0.4",
                dataset=dataset,
                embedding_provider=embedding_provider,
                docs_root=docs_root,
                config=config.model_copy(
                    update={
                        "fusion_strategy": "normalized_weighted",
                        "bm25_fusion_weight": 0.6,
                        "vector_fusion_weight": 0.4,
                    }
                ),
                chunk_config=chunk_config,
            ),
        ]
    )
    all_runs = [base_run, *ablations]
    recommended_run = _recommended_run(all_runs)
    return RetrievalDiagnosticsReport(
        dataset=dataset.name,
        total_questions=len(dataset.items),
        base_run=base_run,
        ablations=ablations,
        recommended_run_name=recommended_run.name,
    )


def format_retrieval_diagnostics_summary(report: RetrievalDiagnosticsReport) -> str:
    lines = [
        f"Dataset: {report.dataset}",
        f"Questions: {report.total_questions}",
        f"Recommended run: {report.recommended_run_name}",
        "",
        _format_run_summary(report.base_run),
    ]
    if report.ablations:
        lines.append("")
        lines.append("Ablations:")
        for run in report.ablations:
            lines.append(_format_run_summary(run))
    return "\n".join(lines)


def run_live_retrieval_diagnostics(
    *,
    dataset: RagEvaluationDataset,
    retriever,
    chunker: TextChunker,
    config: RetrievalDiagnosticConfig | None = None,
) -> RetrievalDiagnosticRun | None:
    if not isinstance(retriever, HybridRetriever):
        return None

    chunk_config = ChunkDiagnosticConfig(
        chunk_size=getattr(chunker, "_chunk_size"),
        chunk_overlap=getattr(chunker, "_overlap"),
    )
    diagnostic_retriever = EvaluationChunkHybridRetriever(
        repository=retriever._repository,
        chunker=chunker,
        embedding_provider=retriever._embedding_provider,
        vector_store=retriever._vector_store,
        bm25_top_k=retriever._bm25_top_k,
        vector_top_k=retriever._vector_top_k,
        rrf_k=retriever._rrf_k,
        fusion_strategy=getattr(retriever, "_fusion_strategy", "rrf"),
        bm25_fusion_weight=getattr(retriever, "_bm25_fusion_weight", 0.7),
        vector_fusion_weight=getattr(retriever, "_vector_fusion_weight", 0.3),
        preserve_bm25_candidates=getattr(retriever, "_preserve_bm25_candidates", False),
        enable_bm25_lexical=getattr(retriever, "_enable_bm25_lexical", True),
        enable_vector_search=getattr(retriever, "_enable_vector_search", True),
        enable_reciprocal_rank_fusion=getattr(
            retriever, "_enable_reciprocal_rank_fusion", True
        ),
        enable_reranking=getattr(retriever, "_enable_reranking", True),
    )
    active_config = config or RetrievalDiagnosticConfig(
        bm25_top_k=retriever._bm25_top_k,
        vector_top_k=retriever._vector_top_k,
        rrf_k=retriever._rrf_k,
        fusion_strategy=getattr(retriever, "_fusion_strategy", "rrf"),
        bm25_fusion_weight=getattr(retriever, "_bm25_fusion_weight", 0.7),
        vector_fusion_weight=getattr(retriever, "_vector_fusion_weight", 0.3),
        preserve_bm25_candidates=getattr(retriever, "_preserve_bm25_candidates", False),
        enable_bm25_lexical=getattr(retriever, "_enable_bm25_lexical", True),
        enable_vector_search=getattr(retriever, "_enable_vector_search", True),
        enable_reciprocal_rank_fusion=getattr(
            retriever, "_enable_reciprocal_rank_fusion", True
        ),
        enable_reranking=getattr(retriever, "_enable_reranking", True),
        rerank_top_k=retriever._rerank_top_k,
    )
    return _run_diagnostic_with_retriever(
        name="live_index",
        dataset=dataset,
        retriever=diagnostic_retriever,
        config=active_config,
        chunk_config=chunk_config,
    )


def run_isolated_retrieval_diagnostics(
    *,
    dataset: RagEvaluationDataset,
    chunker: TextChunker,
    embedding_provider: EmbeddingProvider,
    docs_root: Path | None = None,
    config: RetrievalDiagnosticConfig | None = None,
) -> RetrievalDiagnosticRun:
    active_config = config or RetrievalDiagnosticConfig()
    diagnostic_retriever, _source_documents = build_isolated_evaluation_retriever(
        dataset=dataset,
        chunker=chunker,
        embedding_provider=embedding_provider,
        docs_root=docs_root,
        bm25_top_k=active_config.bm25_top_k,
        vector_top_k=active_config.vector_top_k,
        rrf_k=active_config.rrf_k,
        fusion_strategy=active_config.fusion_strategy,
        bm25_fusion_weight=active_config.bm25_fusion_weight,
        vector_fusion_weight=active_config.vector_fusion_weight,
        preserve_bm25_candidates=active_config.preserve_bm25_candidates,
        enable_bm25_lexical=active_config.enable_bm25_lexical,
        enable_vector_search=active_config.enable_vector_search,
        enable_reciprocal_rank_fusion=active_config.enable_reciprocal_rank_fusion,
        enable_reranking=active_config.enable_reranking,
    )
    return _run_diagnostic_with_retriever(
        name="isolated_docs",
        dataset=dataset,
        retriever=diagnostic_retriever,
        config=active_config,
        chunk_config=ChunkDiagnosticConfig(
            chunk_size=getattr(chunker, "_chunk_size"),
            chunk_overlap=getattr(chunker, "_overlap"),
        ),
    )


def _run_single_diagnostic(
    *,
    name: str,
    dataset: RagEvaluationDataset,
    embedding_provider: EmbeddingProvider,
    docs_root: Path | None,
    config: RetrievalDiagnosticConfig,
    chunk_config: ChunkDiagnosticConfig,
) -> RetrievalDiagnosticRun:
    retriever = _build_diagnostic_retriever(
        dataset=dataset,
        embedding_provider=embedding_provider,
        docs_root=docs_root,
        config=config,
        chunk_config=chunk_config,
    )
    return _run_diagnostic_with_retriever(
        name=name,
        dataset=dataset,
        retriever=retriever,
        config=config,
        chunk_config=chunk_config,
    )


def _run_diagnostic_with_retriever(
    *,
    name: str,
    dataset: RagEvaluationDataset,
    retriever: EvaluationChunkHybridRetriever,
    config: RetrievalDiagnosticConfig,
    chunk_config: ChunkDiagnosticConfig,
) -> RetrievalDiagnosticRun:
    resolved_by_question = {
        item.id: resolve_relevant_chunks_from_entries(
            item=item,
            chunk_entries=retriever._chunk_entries,
        )
        for item in dataset.items
    }
    label_issues = _label_integrity_issues(dataset, resolved_by_question)
    label_integrity = _label_integrity_by_question(dataset, resolved_by_question)
    questions = [
        _diagnose_question(
            retriever,
            item,
            config,
            next(
                question_integrity
                for question_integrity in label_integrity
                if question_integrity.question_id == item.id
            ),
        )
        for item in dataset.items
    ]
    return RetrievalDiagnosticRun(
        name=name,
        config=config,
        chunk_config=chunk_config,
        label_integrity_issues=label_issues,
        label_integrity_by_question=label_integrity,
        stage_metric_means=_stage_metric_means(questions, config.eval_k),
        loss_summary=_loss_summary(questions),
        contribution_summary=_contribution_summary(questions),
        final_context_coverage_rate=_final_context_coverage_rate(questions),
        latency_means=_latency_means(questions),
        questions=questions,
    )


def _build_diagnostic_retriever(
    *,
    dataset: RagEvaluationDataset,
    embedding_provider: EmbeddingProvider,
    docs_root: Path | None,
    config: RetrievalDiagnosticConfig,
    chunk_config: ChunkDiagnosticConfig,
) -> EvaluationChunkHybridRetriever:
    chunker = TextChunker(
        chunk_size=chunk_config.chunk_size,
        overlap=chunk_config.chunk_overlap,
    )
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

    return EvaluationChunkHybridRetriever(
        repository=repository,
        chunker=chunker,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        bm25_top_k=config.bm25_top_k,
        vector_top_k=config.vector_top_k,
        rrf_k=config.rrf_k,
        fusion_strategy=config.fusion_strategy,
        bm25_fusion_weight=config.bm25_fusion_weight,
        vector_fusion_weight=config.vector_fusion_weight,
        preserve_bm25_candidates=config.preserve_bm25_candidates,
        enable_bm25_lexical=config.enable_bm25_lexical,
        enable_vector_search=config.enable_vector_search,
        enable_reciprocal_rank_fusion=config.enable_reciprocal_rank_fusion,
        enable_reranking=config.enable_reranking,
    )


def _diagnose_question(
    retriever: EvaluationChunkHybridRetriever,
    item: RagEvaluationItem,
    config: RetrievalDiagnosticConfig,
    label_integrity: QuestionLabelIntegrity,
) -> QuestionRetrievalDiagnostic:
    total_started_at = perf_counter()

    bm25_started_at = perf_counter()
    bm25_hits = retriever._bm25_search(item.question) if config.enable_bm25_lexical else []
    bm25_latency_ms = _elapsed_ms(bm25_started_at)

    vector_started_at = perf_counter()
    vector_hits = retriever._vector_search(item.question) if config.enable_vector_search else []
    vector_latency_ms = _elapsed_ms(vector_started_at)

    rrf_started_at = perf_counter()
    merged = (
        retriever._merge_rrf(bm25_hits, vector_hits)
        if config.enable_reciprocal_rank_fusion
        else retriever._merge_by_score(bm25_hits, vector_hits)
    )
    rrf_latency_ms = _elapsed_ms(rrf_started_at)

    pre_rerank = sorted(
        merged.values(),
        key=no_rerank_key,
        reverse=True,
    )
    if config.enable_reranking:
        reranker_started_at = perf_counter()
        reranked = sorted(merged.values(), key=rerank_key, reverse=True)[: config.rerank_top_k]
        reranker_latency_ms = _elapsed_ms(reranker_started_at)
    else:
        reranked = pre_rerank[: config.rerank_top_k]
        reranker_latency_ms = 0.0
    final_context = reranked[: config.llm_context_top_k]

    return QuestionRetrievalDiagnostic(
        id=item.id,
        question=item.question,
        label_integrity=label_integrity,
        stage_metrics=[
            _stage_metrics("bm25", bm25_hits, label_integrity.present_ids, config.eval_k),
            _stage_metrics("vector", vector_hits, label_integrity.present_ids, config.eval_k),
            _stage_metrics("hybrid_pre_rerank", pre_rerank, label_integrity.present_ids, config.eval_k),
            _stage_metrics("hybrid_post_rerank", reranked, label_integrity.present_ids, config.eval_k),
        ],
        relevant_chunk_survival=[
            RelevantChunkSurvival(
                relevant_chunk_id=label,
                bm25_rank=_rank_for_label(label, bm25_hits),
                vector_rank=_rank_for_label(label, vector_hits),
                rrf_rank=_rank_for_label(label, pre_rerank),
                reranked_rank=_rank_for_label(label, reranked),
                included_in_final_llm_context=_rank_for_label(label, final_context) is not None,
            )
            for label in label_integrity.present_ids
        ],
        latency=RetrievalLatencyBreakdown(
            bm25_latency_ms=bm25_latency_ms,
            vector_latency_ms=vector_latency_ms,
            rrf_merge_latency_ms=rrf_latency_ms,
            reranker_latency_ms=reranker_latency_ms,
            total_retrieval_latency_ms=_elapsed_ms(total_started_at),
        ),
    )


def _stage_metrics(
    stage: str,
    documents: list[RetrievedDocument],
    relevant_chunk_ids: list[str],
    eval_k: list[int],
) -> StageDiagnosticMetrics:
    return StageDiagnosticMetrics(
        stage=stage,
        metrics=_question_metrics(
            retrieved_documents=documents,
            relevant_chunk_ids=relevant_chunk_ids,
            eval_k=eval_k,
        ),
        retrieved_ids=[_primary_label(document) for document in documents],
    )


def _stage_metric_means(
    questions: list[QuestionRetrievalDiagnostic],
    eval_k: list[int],
) -> dict[str, dict[str, float | None]]:
    stage_values: dict[str, dict[str, list[float]]] = {}
    metric_names = [
        _metric_name(metric, k)
        for k in eval_k
        for metric in ("recall", "precision", "mrr", "ndcg")
    ]
    for question in questions:
        for stage in question.stage_metrics:
            values = stage_values.setdefault(stage.stage, {name: [] for name in metric_names})
            for metric_name, value in stage.metrics.items():
                if value is not None:
                    values[metric_name].append(value)

    return {
        stage: {
            metric_name: round(mean(values), 4) if values else None
            for metric_name, values in metrics.items()
        }
        for stage, metrics in stage_values.items()
    }


def _label_integrity_issues(
    dataset: RagEvaluationDataset,
    resolved_by_question,
) -> list[LabelIntegrityIssue]:
    issues: list[LabelIntegrityIssue] = []
    for item in dataset.items:
        expected_sources = [_normalize_source_name(name) for name in item.source_files]
        resolved = resolved_by_question[item.id]
        for issue in resolved.issues:
            issues.append(
                LabelIntegrityIssue(
                    question_id=item.id,
                    relevant_chunk_id=issue.expected_section or issue.source_file or "",
                    issue=issue.issue,
                    expected_source_files=expected_sources,
                    actual_source_file=issue.source_file,
                )
            )
    return issues


def _label_integrity_by_question(
    dataset: RagEvaluationDataset,
    resolved_by_question,
) -> list[QuestionLabelIntegrity]:
    output: list[QuestionLabelIntegrity] = []
    for item in dataset.items:
        expected_sources = [_normalize_source_name(name) for name in item.source_files]
        resolved = resolved_by_question[item.id]
        missing_ids = [
            issue.expected_section or issue.source_file or issue.issue
            for issue in resolved.issues
        ]
        output.append(
            QuestionLabelIntegrity(
                question_id=item.id,
                expected_source_files=expected_sources,
                present_ids=resolved.relevant_chunk_ids,
                missing_ids=missing_ids,
                source_file_mismatches={},
            )
        )
    return output


def _rank_for_label(label: str, documents: list[RetrievedDocument]) -> int | None:
    for index, document in enumerate(documents, start=1):
        if label in _retrieved_labels(document):
            return index
    return None


def _primary_label(document: RetrievedDocument) -> str:
    if document.chunk_index is not None:
        return f"{document.filename}#chunk-{document.chunk_index}"
    return document.chunk_id or document.document_id


def _elapsed_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000, 2)


def _loss_summary(
    questions: list[QuestionRetrievalDiagnostic],
) -> RetrievalLossSummary:
    total_expected = 0
    present_in_index = 0
    missing_from_index = 0
    missing_in_bm25 = 0
    missing_in_vector = 0
    missing_after_rrf = 0
    missing_after_rerank = 0
    missing_from_final_context = 0

    for question in questions:
        bm25_stage = next(stage for stage in question.stage_metrics if stage.stage == "bm25")
        vector_stage = next(stage for stage in question.stage_metrics if stage.stage == "vector")
        rrf_stage = next(
            stage for stage in question.stage_metrics if stage.stage == "hybrid_pre_rerank"
        )
        rerank_stage = next(
            stage for stage in question.stage_metrics if stage.stage == "hybrid_post_rerank"
        )
        bm25_labels = set(bm25_stage.retrieved_ids)
        vector_labels = set(vector_stage.retrieved_ids)
        rrf_labels = set(rrf_stage.retrieved_ids)
        rerank_labels = set(rerank_stage.retrieved_ids)

        total_expected += len(question.label_integrity.present_ids) + len(question.label_integrity.missing_ids)
        present_in_index += len(question.label_integrity.present_ids)
        missing_from_index += len(question.label_integrity.missing_ids)

        for label in question.label_integrity.present_ids:
            if label not in bm25_labels:
                missing_in_bm25 += 1
            if label not in vector_labels:
                missing_in_vector += 1
            if label not in rrf_labels:
                missing_after_rrf += 1
            if label not in rerank_labels:
                missing_after_rerank += 1
        for chunk in question.relevant_chunk_survival:
            if not chunk.included_in_final_llm_context:
                missing_from_final_context += 1

    return RetrievalLossSummary(
        total_expected_relevant_ids=total_expected,
        present_in_index=present_in_index,
        missing_from_index=missing_from_index,
        missing_in_bm25=missing_in_bm25,
        missing_in_vector=missing_in_vector,
        missing_after_rrf=missing_after_rrf,
        missing_after_rerank=missing_after_rerank,
        missing_from_final_context=missing_from_final_context,
    )


def _contribution_summary(
    questions: list[QuestionRetrievalDiagnostic],
) -> RetrievalContributionSummary:
    bm25_only = 0
    vector_only = 0
    shared = 0
    unseen = 0
    for question in questions:
        for chunk in question.relevant_chunk_survival:
            in_bm25 = chunk.bm25_rank is not None
            in_vector = chunk.vector_rank is not None
            if in_bm25 and in_vector:
                shared += 1
            elif in_bm25:
                bm25_only += 1
            elif in_vector:
                vector_only += 1
            else:
                unseen += 1
    return RetrievalContributionSummary(
        bm25_only_relevant_hits=bm25_only,
        vector_only_relevant_hits=vector_only,
        shared_relevant_hits=shared,
        unseen_relevant_hits=unseen,
    )


def _final_context_coverage_rate(
    questions: list[QuestionRetrievalDiagnostic],
) -> float:
    relevant_total = sum(len(question.relevant_chunk_survival) for question in questions)
    if relevant_total == 0:
        return 0.0
    included = sum(
        1
        for question in questions
        for chunk in question.relevant_chunk_survival
        if chunk.included_in_final_llm_context
    )
    return round(included / relevant_total, 4)


def _latency_means(
    questions: list[QuestionRetrievalDiagnostic],
) -> dict[str, float]:
    return {
        "bm25_latency_ms": round(mean(question.latency.bm25_latency_ms for question in questions), 2),
        "vector_latency_ms": round(mean(question.latency.vector_latency_ms for question in questions), 2),
        "rrf_merge_latency_ms": round(
            mean(question.latency.rrf_merge_latency_ms for question in questions), 2
        ),
        "reranker_latency_ms": round(
            mean(question.latency.reranker_latency_ms for question in questions), 2
        ),
        "total_retrieval_latency_ms": round(
            mean(question.latency.total_retrieval_latency_ms for question in questions), 2
        ),
    }


def _recommended_run(runs: list[RetrievalDiagnosticRun]) -> RetrievalDiagnosticRun:
    def run_key(run: RetrievalDiagnosticRun) -> tuple[float, float, float, float, float, float, float]:
        final_stage = run.stage_metric_means.get("hybrid_post_rerank") or run.stage_metric_means.get("bm25", {})
        return (
            final_stage.get("ndcg_at_10") or 0.0,
            final_stage.get("mrr_at_10") or 0.0,
            final_stage.get("recall_at_10") or 0.0,
            final_stage.get("precision_at_10") or 0.0,
            run.final_context_coverage_rate,
            -(run.latency_means.get("total_retrieval_latency_ms") or 0.0),
            -(
                run.config.bm25_top_k
                + run.config.vector_top_k
                + run.config.rerank_top_k
                + run.config.llm_context_top_k
            ),
        )

    return max(runs, key=run_key)


def _format_run_summary(run: RetrievalDiagnosticRun) -> str:
    lines = [
        (
            f"{run.name}: chunk={run.chunk_config.chunk_size}/{run.chunk_config.chunk_overlap}, "
            f"bm25={run.config.bm25_top_k}, vector={run.config.vector_top_k}, "
            f"rrf_k={run.config.rrf_k}, rerank={run.config.rerank_top_k}, "
            f"context={run.config.llm_context_top_k}, fusion={run.config.fusion_strategy}, "
            f"weights={run.config.bm25_fusion_weight:.1f}/{run.config.vector_fusion_weight:.1f}, "
            f"preserve_bm25={str(run.config.preserve_bm25_candidates).lower()}"
        ),
        f"  label issues: {len(run.label_integrity_issues)}",
        (
            "  loss summary: "
            f"missing_from_index={run.loss_summary.missing_from_index}, "
            f"missing_in_bm25={run.loss_summary.missing_in_bm25}, "
            f"missing_in_vector={run.loss_summary.missing_in_vector}, "
            f"missing_after_rrf={run.loss_summary.missing_after_rrf}, "
            f"missing_after_rerank={run.loss_summary.missing_after_rerank}, "
            f"missing_from_final_context={run.loss_summary.missing_from_final_context}"
        ),
        (
            "  contribution summary: "
            f"bm25_only={run.contribution_summary.bm25_only_relevant_hits}, "
            f"vector_only={run.contribution_summary.vector_only_relevant_hits}, "
            f"shared={run.contribution_summary.shared_relevant_hits}, "
            f"unseen={run.contribution_summary.unseen_relevant_hits}"
        ),
        (
            "  latency means: "
            f"bm25={run.latency_means['bm25_latency_ms']}ms, "
            f"vector={run.latency_means['vector_latency_ms']}ms, "
            f"merge={run.latency_means['rrf_merge_latency_ms']}ms, "
            f"rerank={run.latency_means['reranker_latency_ms']}ms, "
            f"total={run.latency_means['total_retrieval_latency_ms']}ms"
        ),
        f"  final context coverage: {run.final_context_coverage_rate}",
    ]
    for stage, metrics in run.stage_metric_means.items():
        rendered = ", ".join(
            f"{name}={value if value is not None else 'n/a'}"
            for name, value in metrics.items()
        )
        lines.append(f"  {stage}: {rendered}")
    return "\n".join(lines)
