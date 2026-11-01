from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from pra_core.evaluation.rag_retrieval import (
    RagEvaluationSummary,
    _chunk_entries_from_retriever,
    build_isolated_evaluation_retriever,
    evaluate_retrieval_only,
    load_rag_evaluation_dataset,
    s3_object_exists,
    _is_s3_uri,
)
from pra_core.evaluation.retrieval_diagnostics import (
    RetrievalDiagnosticConfig,
    RetrievalDiagnosticRun,
    run_isolated_retrieval_diagnostics,
)
from pra_core.logging import get_logger
from pra_core.persistence.object_store import JsonObjectStore

logger = get_logger(__name__)

RagEvaluationJobStatus = Literal["queued", "running", "completed", "failed"]


class RagEvaluationRequest(BaseModel):
    dataset: str = "default"
    top_k: int = Field(default=25, ge=1, le=25)
    eval_k: list[int] = Field(default_factory=lambda: [5, 10, 25])
    repeat: int = Field(default=1, ge=1, le=10)
    concurrency: int = Field(default=1, ge=1, le=5)


class RagEvaluationJobRecord(BaseModel):
    job_id: str
    status: RagEvaluationJobStatus
    request: RagEvaluationRequest
    principal: str
    created_at: datetime
    updated_at: datetime
    dataset_path: str
    trace_id: str | None = None
    summary: RagEvaluationSummary | None = None
    diagnostics: RetrievalDiagnosticRun | None = None
    error: str | None = None
    duration_ms: float | None = None
    total_questions: int | None = None
    labeled_questions: int | None = None
    unlabeled_questions: int | None = None
    failed_question_count: int | None = None


class RagEvaluationJobStore:
    def get(self, job_id: str) -> RagEvaluationJobRecord | None:
        raise NotImplementedError

    def put(self, record: RagEvaluationJobRecord) -> None:
        raise NotImplementedError


class MemoryRagEvaluationJobStore(RagEvaluationJobStore):
    def __init__(self) -> None:
        self._records: dict[str, RagEvaluationJobRecord] = {}
        self._lock = Lock()

    def get(self, job_id: str) -> RagEvaluationJobRecord | None:
        with self._lock:
            record = self._records.get(job_id)
            return record.model_copy(deep=True) if record else None

    def put(self, record: RagEvaluationJobRecord) -> None:
        with self._lock:
            self._records[record.job_id] = record.model_copy(deep=True)


class ObjectStoreRagEvaluationJobStore(RagEvaluationJobStore):
    def __init__(self, object_store: JsonObjectStore, prefix: str = "rag-evaluation/jobs") -> None:
        self._object_store = object_store
        self._prefix = prefix.strip("/")

    def get(self, job_id: str) -> RagEvaluationJobRecord | None:
        payload = self._object_store.read_json(self._key(job_id), None)
        if payload is None:
            return None
        return RagEvaluationJobRecord.model_validate(payload)

    def put(self, record: RagEvaluationJobRecord) -> None:
        self._object_store.write_json(
            self._key(record.job_id),
            record.model_dump(mode="json"),
        )

    def _key(self, job_id: str) -> str:
        return f"{self._prefix}/{job_id}.json" if self._prefix else f"{job_id}.json"


class RagEvaluationJobCoordinator:
    def __init__(
        self,
        *,
        job_store: RagEvaluationJobStore,
        retriever,
        chunker,
        embedding_provider,
        evaluation_embedding_provider=None,
        llm_context_top_k: int = 10,
        ensure_vector_index,
        dataset_loader=load_rag_evaluation_dataset,
        dataset_exists_checker=None,
        executor: ThreadPoolExecutor | None = None,
        queue_url: str | None = None,
        sqs_client=None,
        evaluation_docs_root: Path | None = None,
    ) -> None:
        self._job_store = job_store
        self._retriever = retriever
        self._chunker = chunker
        self._embedding_provider = embedding_provider
        self._evaluation_embedding_provider = evaluation_embedding_provider or embedding_provider
        self._llm_context_top_k = llm_context_top_k
        self._ensure_vector_index = ensure_vector_index
        self._dataset_loader = dataset_loader
        self._dataset_exists_checker = dataset_exists_checker or _default_dataset_exists_checker
        self._executor = executor
        self._queue_url = queue_url
        self._sqs_client = sqs_client
        self._evaluation_docs_root = evaluation_docs_root

    def submit(
        self,
        *,
        request: RagEvaluationRequest,
        principal: str,
        trace_id: str | None = None,
        dataset_path: str,
    ) -> RagEvaluationJobRecord:
        job_id = str(uuid4())
        now = datetime.now(timezone.utc)
        record = RagEvaluationJobRecord(
            job_id=job_id,
            status="queued",
            request=request,
            principal=principal,
            created_at=now,
            updated_at=now,
            dataset_path=dataset_path,
            trace_id=trace_id,
        )
        self._job_store.put(record)
        logger.info(
            "rag_evaluation_job_queued",
            extra={
                "job_id": job_id,
                "dataset_path": dataset_path,
                "principal": principal,
                "trace_id": trace_id,
            },
        )

        if self._queue_url:
            self._enqueue(job_id)
            return record

        if self._executor is not None:
            self._executor.submit(self.execute, job_id)
            return record

        return self.execute(job_id)

    def get(self, job_id: str) -> RagEvaluationJobRecord | None:
        return self._job_store.get(job_id)

    def execute(self, job_id: str) -> RagEvaluationJobRecord:
        record = self._job_store.get(job_id)
        if record is None:
            raise KeyError(f"Unknown rag evaluation job: {job_id}")

        running = record.model_copy(update={"status": "running", "updated_at": datetime.now(timezone.utc)})
        self._job_store.put(running)
        started_at = perf_counter()
        logger.info(
            "rag_evaluation_job_started",
            extra={
                "job_id": job_id,
                "dataset_path": record.dataset_path,
                "principal": record.principal,
                "trace_id": record.trace_id,
            },
        )

        try:
            summary, diagnostics = self._run_request(record.request, record.dataset_path)
            duration_ms = round((perf_counter() - started_at) * 1000, 2)
            completed = running.model_copy(
                update={
                    "status": "completed",
                    "updated_at": datetime.now(timezone.utc),
                    "summary": summary,
                    "diagnostics": diagnostics,
                    "duration_ms": duration_ms,
                    "total_questions": summary.total_questions,
                    "labeled_questions": summary.labeled_questions,
                    "unlabeled_questions": summary.unlabeled_questions,
                    "failed_question_count": summary.failed_question_count,
                    "error": None,
                }
            )
            self._job_store.put(completed)
            logger.info(
                "rag_evaluation_job_finished",
                extra={
                    "job_id": job_id,
                    "dataset_path": record.dataset_path,
                    "principal": record.principal,
                    "trace_id": record.trace_id,
                    "duration_ms": duration_ms,
                    "failed_question_count": summary.failed_question_count,
                },
            )
            return completed
        except Exception as exc:
            duration_ms = round((perf_counter() - started_at) * 1000, 2)
            failed = running.model_copy(
                update={
                    "status": "failed",
                    "updated_at": datetime.now(timezone.utc),
                    "duration_ms": duration_ms,
                    "error": str(exc),
                }
            )
            self._job_store.put(failed)
            logger.warning(
                "rag_evaluation_job_failed",
                extra={
                    "job_id": job_id,
                    "dataset_path": record.dataset_path,
                    "principal": record.principal,
                    "trace_id": record.trace_id,
                    "duration_ms": duration_ms,
                    "error": str(exc),
                },
            )
            return failed

    def _run_request(
        self,
        request: RagEvaluationRequest,
        dataset_path: str,
    ) -> tuple[RagEvaluationSummary, RetrievalDiagnosticRun | None]:
        if not self._dataset_exists_checker(dataset_path):
            raise FileNotFoundError(
                "RAG evaluation dataset was not found. "
                f"Place the dataset file at {dataset_path} or set PRA_RAG_EVAL_DATASET_PATH."
            )

        dataset = self._dataset_loader(dataset_path)
        normalized_eval_k = _normalized_eval_k(request.eval_k)
        self._ensure_vector_index()
        evaluation_retriever, _source_documents = build_isolated_evaluation_retriever(
            dataset=dataset,
            chunker=self._chunker,
            embedding_provider=self._evaluation_embedding_provider,
            docs_root=self._evaluation_docs_root,
            bm25_top_k=getattr(self._retriever, "_bm25_top_k", request.top_k),
            vector_top_k=getattr(self._retriever, "_vector_top_k", request.top_k),
            rrf_k=getattr(self._retriever, "_rrf_k", 30),
            fusion_strategy=getattr(self._retriever, "_fusion_strategy", "rrf"),
            bm25_fusion_weight=getattr(self._retriever, "_bm25_fusion_weight", 0.7),
            vector_fusion_weight=getattr(self._retriever, "_vector_fusion_weight", 0.3),
            preserve_bm25_candidates=getattr(self._retriever, "_preserve_bm25_candidates", False),
            enable_bm25_lexical=getattr(self._retriever, "_enable_bm25_lexical", True),
            enable_vector_search=getattr(self._retriever, "_enable_vector_search", True),
            enable_reciprocal_rank_fusion=getattr(
                self._retriever, "_enable_reciprocal_rank_fusion", True
            ),
            enable_reranking=getattr(self._retriever, "_enable_reranking", False),
            bm25_first=True,
        )
        isolated_chunk_entries = _chunk_entries_from_retriever(evaluation_retriever)
        summary = evaluate_retrieval_only(
            retriever=evaluation_retriever,
            dataset=dataset,
            top_k=request.top_k,
            eval_k=normalized_eval_k,
            repeat=request.repeat,
            chunk_entries=isolated_chunk_entries,
        )
        diagnostics = run_isolated_retrieval_diagnostics(
            dataset=dataset,
            chunker=self._chunker,
            embedding_provider=self._evaluation_embedding_provider,
            docs_root=self._evaluation_docs_root,
            config=RetrievalDiagnosticConfig(
                bm25_top_k=getattr(self._retriever, "_bm25_top_k", request.top_k),
                vector_top_k=getattr(self._retriever, "_vector_top_k", request.top_k),
                rrf_k=getattr(self._retriever, "_rrf_k", 30),
                fusion_strategy=getattr(self._retriever, "_fusion_strategy", "rrf"),
                bm25_fusion_weight=getattr(self._retriever, "_bm25_fusion_weight", 0.7),
                vector_fusion_weight=getattr(self._retriever, "_vector_fusion_weight", 0.3),
                preserve_bm25_candidates=getattr(self._retriever, "_preserve_bm25_candidates", False),
                enable_bm25_lexical=getattr(self._retriever, "_enable_bm25_lexical", True),
                enable_vector_search=getattr(self._retriever, "_enable_vector_search", True),
                enable_reciprocal_rank_fusion=getattr(
                    self._retriever, "_enable_reciprocal_rank_fusion", True
                ),
                enable_reranking=getattr(self._retriever, "_enable_reranking", False),
                rerank_top_k=getattr(self._retriever, "_rerank_top_k", request.top_k),
                llm_context_top_k=self._llm_context_top_k,
                eval_k=normalized_eval_k,
            ),
        )
        return summary, diagnostics

    def _enqueue(self, job_id: str) -> None:
        if not self._queue_url:
            return
        self._sqs().send_message(
            QueueUrl=self._queue_url,
            MessageBody=json.dumps({"job_id": job_id}),
        )

    def _sqs(self):
        if self._sqs_client is None:
            import boto3

            self._sqs_client = boto3.client("sqs")
        return self._sqs_client


def _default_dataset_exists_checker(path: str | Path) -> bool:
    if _is_s3_uri(path):
        return s3_object_exists(str(path))
    return Path(path).exists()


def _normalized_eval_k(values: list[int]) -> list[int]:
    normalized = sorted({value for value in values if 1 <= value <= 25})
    if not normalized:
        raise ValueError("eval_k must contain at least one value from 1 to 25.")
    return normalized
