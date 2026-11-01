from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Lock, Thread
from time import perf_counter
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from pra_core.evaluation.ragas import (
    FULL_RAGAS_METRICS,
    STANDARD_RAGAS_METRICS,
    RagasEvaluationSample,
    RagasEvaluationSummary,
    RagasQuestionResult,
    RagasRunConfig,
    evaluate_ragas,
)
from pra_core.evaluation.rag_retrieval import (
    RagEvaluationItem,
    _is_s3_uri,
    build_isolated_evaluation_retriever,
    load_rag_evaluation_dataset,
    s3_object_exists,
)
from pra_core.documents.filenames import normalize_filename
from pra_core.logging import get_logger
from pra_core.persistence.object_store import JsonObjectStore

logger = get_logger(__name__)

RagasJobStatus = Literal[
    "queued",
    "running",
    "cancel_requested",
    "cancelled",
    "completed",
    "failed",
]
# Leave enough time for a timed-out 14-minute Lambda invocation's SQS message
# to become visible again before a status read declares the job permanently stale.
_RAGAS_STALE_JOB_SECONDS = 35 * 60
_RAGAS_TIMEOUT_BUFFER_MS = 60_000
_RAGAS_HEARTBEAT_INTERVAL_SECONDS = 45


class RagasContinuationSchedulingError(RuntimeError):
    """Raised so SQS can retry after progress was safely checkpointed."""


def _merge_retrieval_details(details, retrieved) -> list[dict[str, object]]:
    """Attach retrieved text to workflow diagnostics for RAGAS metric contexts."""
    merged: list[dict[str, object]] = []
    for index, document in enumerate(retrieved):
        detail = dict(details[index]) if index < len(details) else {}
        detail.update(
            {
                "text": document.text,
                "document_id": detail.get("document_id", document.document_id),
                "filename": detail.get("filename", document.filename),
                "chunk_id": detail.get("chunk_id", document.chunk_id),
                "chunk_index": detail.get("chunk_index", document.chunk_index),
            }
        )
        merged.append(detail)
    return merged


class RagasEvaluationRequest(BaseModel):
    dataset: str = "default"
    mode: Literal["standard", "full"] = "full"
    top_k: int = Field(default=15, ge=1, le=25)
    max_questions: int = Field(default=12, ge=1, le=100)


class RagasQuestionState(BaseModel):
    """Durable, idempotent state for one RAGAS question."""

    question_id: str
    question_index: int
    sample: RagasEvaluationSample | None = None
    sample_generation_complete: bool = False
    completed_metrics: dict[str, float | None] = Field(default_factory=dict)
    metric_errors: dict[str, str] = Field(default_factory=dict)
    metric_retry_counts: dict[str, int] = Field(default_factory=dict)
    next_metric_index: int = 0
    question_complete: bool = False
    retrieval_count: int = 0
    answer_generation_count: int = 0


class RagasEvaluationJobRecord(BaseModel):
    job_id: str
    status: RagasJobStatus
    request: RagasEvaluationRequest
    principal: str
    created_at: datetime
    updated_at: datetime
    dataset_path: str
    trace_id: str | None = None
    summary: RagasEvaluationSummary | None = None
    error: str | None = None
    duration_ms: float | None = None
    completed_questions: int = 0
    total_questions: int | None = None
    question_states: list[RagasQuestionState] = Field(default_factory=list)
    orchestration_state: str | None = None
    orchestration_message: str | None = None
    capacity_wait_count: int = 0


class RagasEvaluationJobStore:
    def get(self, job_id: str) -> RagasEvaluationJobRecord | None:
        raise NotImplementedError

    def put(self, record: RagasEvaluationJobRecord) -> None:
        raise NotImplementedError


class MemoryRagasEvaluationJobStore(RagasEvaluationJobStore):
    def __init__(self) -> None:
        self._records: dict[str, RagasEvaluationJobRecord] = {}
        self._lock = Lock()

    def get(self, job_id: str) -> RagasEvaluationJobRecord | None:
        with self._lock:
            record = self._records.get(job_id)
            return record.model_copy(deep=True) if record else None

    def put(self, record: RagasEvaluationJobRecord) -> None:
        with self._lock:
            self._records[record.job_id] = record.model_copy(deep=True)


class ObjectStoreRagasEvaluationJobStore(RagasEvaluationJobStore):
    def __init__(self, object_store: JsonObjectStore, prefix: str = "ragas-evaluation/jobs") -> None:
        self._object_store = object_store
        self._prefix = prefix.strip("/")

    def get(self, job_id: str) -> RagasEvaluationJobRecord | None:
        payload = self._object_store.read_json(self._key(job_id), None)
        return RagasEvaluationJobRecord.model_validate(payload) if payload else None

    def put(self, record: RagasEvaluationJobRecord) -> None:
        self._object_store.write_json(self._key(record.job_id), record.model_dump(mode="json"))

    def _key(self, job_id: str) -> str:
        return f"{self._prefix}/{job_id}.json" if self._prefix else f"{job_id}.json"


class RagasEvaluationJobCoordinator:
    def __init__(
        self,
        *,
        job_store: RagasEvaluationJobStore,
        workflow,
        dataset_loader=load_rag_evaluation_dataset,
        dataset_exists_checker=None,
        executor: ThreadPoolExecutor | None = None,
        queue_url: str | None = None,
        sqs_client=None,
        chunker=None,
        evaluation_embedding_provider=None,
        evaluation_docs_root: Path | None = None,
        evaluation_workflow_factory=None,
        retrieval_config_source=None,
        evaluator_model_id: str | None = None,
        embedding_model_id: str | None = None,
        ragas_enabled: bool = True,
        allow_inline_continuation: bool = True,
        ensure_vector_index=None,
        batch_size: int = 4,
        evaluator_max_tokens: int = 768,
        metric_timeout_seconds: int = 120,
    ) -> None:
        self._job_store = job_store
        self._workflow = workflow
        self._dataset_loader = dataset_loader
        self._dataset_exists_checker = dataset_exists_checker or _default_dataset_exists_checker
        self._executor = executor
        self._queue_url = queue_url
        self._sqs_client = sqs_client
        self._chunker = chunker
        self._evaluation_embedding_provider = evaluation_embedding_provider
        self._evaluation_docs_root = evaluation_docs_root
        self._evaluation_workflow_factory = evaluation_workflow_factory
        self._retrieval_config_source = retrieval_config_source
        self._evaluator_model_id = evaluator_model_id
        self._embedding_model_id = embedding_model_id
        self._ragas_enabled = ragas_enabled
        self._allow_inline_continuation = allow_inline_continuation
        self._ensure_vector_index = ensure_vector_index or (lambda: None)
        self._batch_size = max(1, batch_size)
        self._evaluator_max_tokens = max(128, evaluator_max_tokens)
        self._metric_timeout_seconds = max(1, metric_timeout_seconds)
        # A warm Lambda can process several metric continuations. Reuse the
        # isolated corpus for those continuations instead of re-embedding it.
        self._full_evaluation_resources: dict[tuple[str, int], tuple[object, list[object]]] = {}

    def submit(
        self,
        *,
        request: RagasEvaluationRequest,
        principal: str,
        trace_id: str | None = None,
        dataset_path: str,
    ) -> RagasEvaluationJobRecord:
        if not self._ragas_enabled:
            raise RuntimeError("RAGAS evaluation is disabled. Set PRA_RAGAS_ENABLED=true to enable it.")
        job_id = str(uuid4())
        now = datetime.now(timezone.utc)
        record = RagasEvaluationJobRecord(
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
        if self._queue_url:
            self._enqueue(job_id)
        elif self._executor is not None:
            self._executor.submit(self.execute, job_id)
        else:
            return self.execute(job_id)
        return record

    def get(self, job_id: str) -> RagasEvaluationJobRecord | None:
        record = self._job_store.get(job_id)
        if record is None or record.status not in {"queued", "running"}:
            return record
        age_seconds = (datetime.now(timezone.utc) - record.updated_at).total_seconds()
        if age_seconds <= _RAGAS_STALE_JOB_SECONDS:
            return record
        age_minutes = max(1, round(age_seconds / 60))
        if record.status == "queued":
            stale_error = (
                "RAGAS evaluation was marked failed because no initial job checkpoint "
                f"was recorded within {age_minutes} minutes. The job may have been "
                "waiting for queue delivery, starter execution, or worker-capacity "
                "retries; review the Step Functions, starter, and worker logs."
            )
        else:
            progress = (
                f"{record.completed_questions}/{record.total_questions}"
                if record.total_questions is not None
                else str(record.completed_questions)
            )
            stale_error = (
                "RAGAS evaluation was marked failed because no persisted job checkpoint "
                f"was recorded for {age_minutes} minutes (progress: {progress}). "
                "A continuation may have been queued or run in another Lambda log "
                "stream, but no later result was saved to this job record."
            )
        failed = record.model_copy(
            update={
                "status": "failed",
                "updated_at": datetime.now(timezone.utc),
                "error": stale_error,
            }
        )
        self._job_store.put(failed)
        logger.warning(
            "ragas_worker_stale_job_detected",
            extra={"job_id": job_id, "previous_status": record.status, "age_seconds": age_seconds},
        )
        return failed

    def cancel(self, job_id: str) -> RagasEvaluationJobRecord | None:
        """Request cancellation without deleting the durable job record."""
        record = self._job_store.get(job_id)
        if record is None:
            return None
        if record.status in {"completed", "failed", "cancelled"}:
            return record
        if record.status == "cancel_requested":
            return record
        requested = record.model_copy(
            update={
                "status": "cancel_requested",
                "updated_at": datetime.now(timezone.utc),
            }
        )
        self._job_store.put(requested)
        logger.info("ragas_cancellation_requested", extra={"job_id": job_id})
        return requested

    def _cancel_if_requested(self, job_id: str) -> bool:
        record = self._job_store.get(job_id)
        if record is None:
            raise KeyError(f"Unknown RAGAS evaluation job: {job_id}")
        if record.status != "cancel_requested":
            return False
        cancelled = record.model_copy(
            update={
                "status": "cancelled",
                "updated_at": datetime.now(timezone.utc),
            }
        )
        self._job_store.put(cancelled)
        logger.info("ragas_job_cancelled", extra={"job_id": job_id})
        return True

    def execute(
        self,
        job_id: str,
        *,
        remaining_time_ms: int | None = None,
        worker_request_id: str | None = None,
        external_orchestrator: bool = False,
    ) -> RagasEvaluationJobRecord:
        record = self._job_store.get(job_id)
        if record is None:
            raise KeyError(f"Unknown RAGAS evaluation job: {job_id}")
        if record.status in {"completed", "failed", "cancelled"}:
            return record
        if self._cancel_if_requested(job_id):
            return self._job_store.get(job_id)
        running = record.model_copy(
            update={
                "status": "running",
                "updated_at": datetime.now(timezone.utc),
                "orchestration_state": "worker_running",
                "orchestration_message": "RAGAS worker capacity acquired; evaluation is running.",
            }
        )
        self._job_store.put(running)
        logger.info(
            "ragas_worker_job_claimed",
            extra={
                "job_id": job_id,
                "worker_request_id": worker_request_id,
                "previous_status": record.status,
                "new_status": running.status,
            },
        )
        started_at = perf_counter()
        deadline = (
            perf_counter() + (remaining_time_ms - _RAGAS_TIMEOUT_BUFFER_MS) / 1000
            if remaining_time_ms is not None
            else None
        )
        if record.request.mode == "full":
            try:
                return self._execute_full_metric_step(
                    running,
                    deadline=deadline,
                    worker_request_id=worker_request_id,
                    external_orchestrator=external_orchestrator,
                )
            except Exception as exc:
                latest = self._job_store.get(job_id)
                if latest is not None and latest.status == "cancel_requested":
                    self._cancel_if_requested(job_id)
                    return self._job_store.get(job_id) or latest
                base = latest or running
                failed = base.model_copy(
                    update={
                        "status": "failed",
                        "updated_at": datetime.now(timezone.utc),
                        "duration_ms": round((perf_counter() - started_at) * 1000, 2),
                        "error": str(exc),
                    }
                )
                self._job_store.put(failed)
                logger.warning(
                    "ragas_evaluation_job_failed",
                    extra={
                        "job_id": job_id,
                        "trace_id": record.trace_id,
                        "error": str(exc),
                    },
                )
                return failed
        try:
            summary, finished, completed_questions = self._run_request(
                record.request,
                record.dataset_path,
                record.job_id,
                deadline=deadline,
                existing_summary=record.summary,
                start_index=record.completed_questions,
                worker_request_id=worker_request_id,
            )
            if self._cancel_if_requested(job_id):
                return self._job_store.get(job_id)
            if not finished:
                latest = self._job_store.get(job_id)
                base = latest or running
                pending = base.model_copy(
                    update={
                        "status": "running",
                        "updated_at": datetime.now(timezone.utc),
                        "summary": summary,
                        "completed_questions": completed_questions,
                        "total_questions": summary.total_questions,
                    }
                )
                self._job_store.put(pending)
                if not external_orchestrator:
                    try:
                        scheduled = self._enqueue_or_continue(
                            job_id,
                            next_index=completed_questions,
                            remaining_question_count=max(
                                0,
                                (pending.total_questions or completed_questions)
                                - completed_questions,
                            ),
                        )
                    except Exception as exc:
                        logger.exception(
                            "ragas_worker_continuation_failed",
                            extra={
                                "job_id": job_id,
                                "next_index": completed_questions,
                                "completed_questions": completed_questions,
                            },
                        )
                        raise RagasContinuationSchedulingError(
                            "RAGAS continuation could not be scheduled; progress was checkpointed."
                        ) from exc
                    if not scheduled:
                        return self._job_store.get(job_id) or pending
                return pending
            latest = self._job_store.get(job_id)
            completed = (latest or running).model_copy(
                update={
                    "status": "completed",
                    "updated_at": datetime.now(timezone.utc),
                    "summary": summary,
                    "duration_ms": round((perf_counter() - started_at) * 1000, 2),
                    "error": None,
                    "completed_questions": completed_questions,
                    "total_questions": summary.total_questions,
                }
            )
            if self._cancel_if_requested(job_id):
                return self._job_store.get(job_id)
            self._job_store.put(completed)
            return completed
        except RagasContinuationSchedulingError:
            # Progress is already checkpointed; let SQS retry scheduling.
            raise
        except Exception as exc:
            latest = self._job_store.get(job_id)
            if latest is not None and latest.status == "cancel_requested":
                self._cancel_if_requested(job_id)
                return self._job_store.get(job_id) or latest
            base = latest or running
            failed = base.model_copy(
                update={
                    "status": "failed",
                    "updated_at": datetime.now(timezone.utc),
                    "duration_ms": round((perf_counter() - started_at) * 1000, 2),
                    "error": str(exc),
                }
            )
            self._job_store.put(failed)
            logger.warning(
                "ragas_evaluation_job_failed",
                extra={"job_id": job_id, "trace_id": record.trace_id, "error": str(exc)},
            )
            return failed

    def _execute_full_metric_step(
        self,
        record: RagasEvaluationJobRecord,
        *,
        deadline: float | None,
        worker_request_id: str | None,
        external_orchestrator: bool,
    ) -> RagasEvaluationJobRecord:
        """Run exactly one full-mode paid boundary, then checkpoint and continue."""
        if not self._dataset_exists_checker(record.dataset_path):
            raise FileNotFoundError(f"RAGAS evaluation dataset was not found: {record.dataset_path}")
        dataset = self._dataset_loader(record.dataset_path)
        items = self._select_items(dataset.items, record.request)
        states = list(record.question_states)
        if not states:
            states = [
                RagasQuestionState(question_id=item.id, question_index=index)
                for index, item in enumerate(items)
            ]
        state = next((item for item in states if not item.question_complete), None)
        if state is None:
            return self._checkpoint_full(record, states, items, completed=True)
        if self._cancel_if_requested(record.job_id):
            return self._job_store.get(record.job_id) or record
        metric_names = FULL_RAGAS_METRICS
        phase = "sample" if not state.sample_generation_complete else "metric"
        metric_name = None if phase == "sample" else metric_names[state.next_metric_index]
        if self._deadline_reached(deadline, record.job_id, phase=f"before_{phase}"):
            pending = self._checkpoint_full(record, states, items, completed=False)
            self._schedule_full_continuation(
                pending,
                state,
                phase=phase,
                metric_name=metric_name,
                external_orchestrator=external_orchestrator,
            )
            logger.info(
                "ragas_worker_deadline_deferred",
                extra={
                    "job_id": record.job_id,
                    "question_id": state.question_id,
                    "question_index": state.question_index,
                    "phase": phase,
                    "metric_name": metric_name,
                    "worker_request_id": worker_request_id,
                },
            )
            return pending
        if phase == "sample":
            logger.info(
                "ragas_sample_generation_started",
                extra={
                    "job_id": record.job_id,
                    "question_id": state.question_id,
                    "question_index": state.question_index,
                    "worker_request_id": worker_request_id,
                },
            )
            item = items[state.question_index]
            retriever, source_documents = self._get_full_evaluation_resources(
                dataset,
                record,
            )
            required_sources = {
                normalize_filename(name)
                for source_item in items
                for name in source_item.source_files
            }
            resolved_sources = {document.resolved_name for document in source_documents}
            if missing := sorted(required_sources - resolved_sources):
                raise FileNotFoundError(
                    "RAGAS evaluation source documents were not found: "
                    + ", ".join(missing)
                )
            workflow = self._build_evaluation_workflow(retriever)
            with self._periodic_heartbeat(
                job_id=record.job_id,
                completed=sum(item.question_complete for item in states),
                total=len(items),
                summary=record.summary,
                question_id=item.id,
                question_index=state.question_index,
                worker_request_id=worker_request_id,
            ):
                result = workflow.run(
                    message=item.question,
                    session_id=f"ragas-{record.job_id}-{item.id}",
                    top_k=record.request.top_k,
                )
            sample = RagasEvaluationSample(
                id=item.id,
                user_input=item.question,
                response=str(result.get("answer", "")),
                retrieved_contexts=[
                    str(value)
                    for value in result.get("ragas_contexts", [])
                    if str(value).strip()
                ],
                reference=item.expected_answer_summary or "",
                citations=[str(value) for value in result.get("citations", [])],
                retrieval_details=list(result.get("retrieval_details", [])),
                context_compression=dict(result.get("context_compression", {})),
                expected_behavior=item.expected_behavior,
            )
            updated = state.model_copy(
                update={
                    "sample": sample,
                    "sample_generation_complete": True,
                    "retrieval_count": 1,
                    "answer_generation_count": 1,
                }
            )
            states[state.question_index] = updated
            pending = self._checkpoint_full(record, states, items, completed=False)
            logger.info(
                "ragas_sample_generation_completed",
                extra={
                    "job_id": record.job_id,
                    "question_id": item.id,
                    "question_index": state.question_index,
                    "answer_generation_model": result.get("selected_model"),
                    "retrieval_count": 1,
                    "answer_generation_count": 1,
                },
            )
            self._schedule_full_continuation(
                pending,
                updated,
                phase="metric",
                metric_name=metric_names[0],
                external_orchestrator=external_orchestrator,
            )
            return pending

        if metric_name in state.completed_metrics:
            pending = self._advance_full_metric(
                record, states, items, state, external_orchestrator=external_orchestrator
            )
            return pending
        logger.info(
            "ragas_metric_started",
            extra={
                "job_id": record.job_id,
                "question_id": state.question_id,
                "question_index": state.question_index,
                "metric_name": metric_name,
                "phase": "metric",
                "completed_metric_count": len(state.completed_metrics),
                "total_metric_count": len(metric_names),
                "worker_request_id": worker_request_id,
            },
        )
        with self._periodic_heartbeat(
            job_id=record.job_id,
            completed=sum(item.question_complete for item in states),
            total=len(items),
            summary=record.summary,
            question_id=state.question_id,
            question_index=state.question_index,
            worker_request_id=worker_request_id,
        ):
            summary = self._evaluate_samples([state.sample], metric_names=(metric_name,))
        result = summary.results[0] if summary.results else None
        value = result.metrics.get(metric_name) if result else None
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            retries = state.metric_retry_counts.get(metric_name, 0)
            if retries < 1:
                updated = state.model_copy(
                    update={
                        "metric_retry_counts": {
                            **state.metric_retry_counts,
                            metric_name: retries + 1,
                        }
                    }
                )
                states[state.question_index] = updated
                pending = self._checkpoint_full(record, states, items, completed=False)
                logger.warning(
                    "ragas_metric_retry_started",
                    extra={
                        "job_id": record.job_id,
                        "question_id": state.question_id,
                        "question_index": state.question_index,
                        "metric_name": metric_name,
                        "phase": "metric",
                    },
                )
                self._schedule_full_continuation(
                    pending,
                    updated,
                    phase="metric",
                    metric_name=metric_name,
                    external_orchestrator=external_orchestrator,
                )
                return pending
            updated = state.model_copy(
                update={
                    "completed_metrics": {
                        **state.completed_metrics,
                        metric_name: None,
                    },
                    "metric_errors": {
                        **state.metric_errors,
                        metric_name: result.error if result else "metric returned no score",
                    },
                }
            )
        else:
            updated = state.model_copy(
                update={
                    "completed_metrics": {**state.completed_metrics, **result.metrics}
                }
            )
            logger.info(
                "ragas_metric_completed",
                extra={
                    "job_id": record.job_id,
                    "question_id": state.question_id,
                    "question_index": state.question_index,
                    "metric_name": metric_name,
                    "phase": "metric",
                    "completed_metric_count": len(updated.completed_metrics),
                    "total_metric_count": len(metric_names),
                },
            )
        states[state.question_index] = updated
        return self._advance_full_metric(
            record, states, items, updated, external_orchestrator=external_orchestrator
        )

    def _advance_full_metric(
        self,
        record,
        states,
        items,
        state,
        *,
        external_orchestrator: bool,
    ) -> RagasEvaluationJobRecord:
        next_index = state.next_metric_index + 1
        question_complete = next_index >= len(FULL_RAGAS_METRICS)
        updated = state.model_copy(
            update={
                "next_metric_index": next_index,
                "question_complete": question_complete,
            }
        )
        states[state.question_index] = updated
        complete = all(item.question_complete for item in states)
        pending = self._checkpoint_full(record, states, items, completed=complete)
        logger.info(
            "ragas_metric_checkpointed",
            extra={
                "job_id": record.job_id,
                "question_id": updated.question_id,
                "question_index": updated.question_index,
                "completed_metric_count": len(updated.completed_metrics),
                "total_metric_count": len(FULL_RAGAS_METRICS),
            },
        )
        if complete:
            logger.info(
                "ragas_question_completed",
                extra={
                    "job_id": record.job_id,
                    "question_id": updated.question_id,
                    "question_index": updated.question_index,
                },
            )
            return pending
        next_state = next(item for item in states if not item.question_complete)
        next_phase = "sample" if not next_state.sample_generation_complete else "metric"
        next_metric = None if next_phase == "sample" else FULL_RAGAS_METRICS[next_state.next_metric_index]
        self._schedule_full_continuation(
            pending,
            next_state,
            phase=next_phase,
            metric_name=next_metric,
            external_orchestrator=external_orchestrator,
        )
        return pending

    def _checkpoint_full(self, record, states, items, *, completed: bool) -> RagasEvaluationJobRecord:
        summary = self._full_summary(states, len(items))
        updated = record.model_copy(
            update={
                "status": "completed" if completed else "running",
                "updated_at": datetime.now(timezone.utc),
                "question_states": states,
                "summary": summary,
                "completed_questions": summary.completed_questions,
                "total_questions": len(items),
                "error": None,
            }
        )
        self._job_store.put(updated)
        return updated

    @staticmethod
    def _full_summary(states: list[RagasQuestionState], total: int) -> RagasEvaluationSummary:
        results = []
        for state in states:
            if state.sample is None:
                continue
            error = "; ".join(state.metric_errors.values()) or None
            results.append(
                RagasQuestionResult(
                    id=state.question_id,
                    question=state.sample.user_input,
                    answer=state.sample.response,
                    metrics=state.completed_metrics,
                    citations=state.sample.citations,
                    context_count=len(state.sample.retrieved_contexts),
                    context_compression=state.sample.context_compression,
                    error=error,
                )
            )
        completed = sum(item.question_complete and not item.metric_errors for item in states)
        aggregates = {}
        failed_metrics = {}
        for name in FULL_RAGAS_METRICS:
            values = [
                float(item.completed_metrics[name])
                for item in states
                if item.completed_metrics.get(name) is not None
            ]
            if values:
                aggregates[name] = round(sum(values) / len(values), 4)
            failures = sum(name in item.metric_errors for item in states)
            if failures:
                failed_metrics[name] = failures
        return RagasEvaluationSummary(
            total_questions=total,
            completed_questions=completed,
            failed_questions=sum(
                item.question_complete and bool(item.metric_errors) for item in states
            ),
            evaluation_success_rate=round(completed / total, 4) if total else 0.0,
            first_error=next(
                (error for item in states for error in item.metric_errors.values()),
                None,
            ),
            metrics=aggregates,
            failed_metric_counts=failed_metrics,
            results=results,
        )

    def _schedule_full_continuation(
        self,
        record,
        state,
        *,
        phase: str,
        metric_name: str | None,
        external_orchestrator: bool = False,
    ) -> None:
        if self._cancel_if_requested(record.job_id):
            return
        if external_orchestrator:
            logger.info(
                "ragas_metric_continuation_deferred_to_step_functions",
                extra={
                    "job_id": record.job_id,
                    "question_id": state.question_id,
                    "question_index": state.question_index,
                    "phase": phase,
                    "metric_name": metric_name,
                },
            )
            return
        if self._queue_url:
            self._enqueue(
                record.job_id,
                question_index=state.question_index,
                phase=phase,
                metric_name=metric_name,
            )
        elif self._executor is not None:
            self._executor.submit(self.execute, record.job_id)
        logger.info(
            "ragas_metric_continuation_scheduled",
            extra={
                "job_id": record.job_id,
                "question_id": state.question_id,
                "question_index": state.question_index,
                "phase": phase,
                "metric_name": metric_name,
            },
        )

    def _run_request(
        self,
        request: RagasEvaluationRequest,
        dataset_path: str,
        job_id: str,
        *,
        deadline: float | None = None,
        existing_summary: RagasEvaluationSummary | None = None,
        start_index: int = 0,
        worker_request_id: str | None = None,
    ) -> tuple[RagasEvaluationSummary, bool, int]:
        if not self._dataset_exists_checker(dataset_path):
            raise FileNotFoundError(
                "RAGAS evaluation dataset was not found. "
                f"Place the dataset file at {dataset_path} or set PRA_RAGAS_EVAL_DATASET_PATH."
            )
        dataset = self._dataset_loader(dataset_path)
        if self._cancel_if_requested(job_id):
            return existing_summary or RagasEvaluationSummary(
                total_questions=len(dataset.items), completed_questions=0, failed_questions=0,
                metrics={}, results=[]
            ), True, start_index
        items = self._select_items(dataset.items, request)
        batch_items = items[start_index : start_index + self._batch_size]
        logger.info(
            "ragas_worker_processing_started",
            extra={
                "job_id": job_id,
                "batch_size": len(batch_items),
                "mode": request.mode,
                "selected_question_count": len(items),
                "worker_request_id": worker_request_id,
                "metric_names": list(
                    FULL_RAGAS_METRICS if request.mode == "full" else STANDARD_RAGAS_METRICS
                ),
            },
        )
        evaluation_retriever, source_documents = self._build_evaluation_retriever(dataset, request)
        required_sources = {
            normalize_filename(source_name)
            for item in items
            for source_name in item.source_files
        }
        resolved_sources = {document.resolved_name for document in source_documents}
        missing_sources = sorted(required_sources - resolved_sources)
        if missing_sources:
            raise FileNotFoundError(
                "RAGAS evaluation source documents were not found: "
                + ", ".join(missing_sources)
            )
        logger.info(
            "ragas_evaluation_corpus_ready",
            extra={
                "job_id": job_id,
                "dataset_name": dataset.name,
                "source_document_count": len(source_documents),
                "selected_question_count": len(items),
                "mode": request.mode,
                "top_k": request.top_k,
            },
        )
        evaluation_workflow = self._build_evaluation_workflow(evaluation_retriever)
        samples: list[RagasEvaluationSample] = []
        for item in batch_items:
            if self._deadline_reached(deadline, job_id, phase="before_question"):
                return existing_summary or RagasEvaluationSummary(
                    total_questions=len(items), completed_questions=start_index,
                    failed_questions=0, metrics={}, results=[]
                ), False, start_index
            if self._cancel_if_requested(job_id):
                return existing_summary or RagasEvaluationSummary(
                    total_questions=len(items), completed_questions=0, failed_questions=0,
                    metrics={}, results=[]
                ), True, start_index
            self._heartbeat(
                job_id,
                completed=start_index,
                total=len(items),
                summary=existing_summary,
            )
            retrieved = evaluation_retriever.search(item.question, k=request.top_k)
            logger.info(
                "ragas_retrieval_preflight",
                extra={
                    "job_id": job_id,
                    "question_id": item.id,
                    "retrieved_context_count": len(retrieved),
                    "source_document_count": len(source_documents),
                    "top_k": request.top_k,
                },
            )
            if not retrieved:
                logger.warning(
                    "ragas_retrieval_preflight_failed",
                    extra={
                        "job_id": job_id,
                        "question_id": item.id,
                        "retrieved_context_count": 0,
                        "source_document_count": len(source_documents),
                        "top_k": request.top_k,
                    },
                )
                samples.append(
                    RagasEvaluationSample(
                        id=item.id,
                        user_input=item.question,
                        response="",
                        reference=item.expected_answer_summary or "",
                        citations=[],
                        retrieval_details=[],
                        expected_behavior=item.expected_behavior,
                    )
                )
                continue

            with self._periodic_heartbeat(
                job_id=job_id,
                completed=start_index,
                total=len(items),
                summary=existing_summary,
                question_id=item.id,
                question_index=start_index,
                worker_request_id=worker_request_id,
            ):


                result = evaluation_workflow.run(
                    message=item.question,
                    session_id=f"ragas-{job_id}-{item.id}",
                    top_k=request.top_k,
                )
            logger.info(
                "ragas_answer_generated",
                extra={
                    "job_id": job_id,
                    "question_id": item.id,
                    "answer_generation_model": result.get("selected_model"),
                    "routing_reason": result.get("routing_reason"),
                    "routing_confidence": result.get("routing_confidence"),
                    "ragas_evaluator_model": self._evaluator_model_id,
                },
            )
            details = _merge_retrieval_details(
                result.get("retrieval_details", []),
                retrieved,
            )
            samples.append(
                RagasEvaluationSample(
                    id=item.id,
                    user_input=item.question,
                    response=str(result.get("answer", "")),
                    retrieved_contexts=[
                        str(detail.get("text", ""))
                        for detail in details
                        if str(detail.get("text", "")).strip()
                    ],
                    reference=item.expected_answer_summary or "",
                    citations=[str(value) for value in result.get("citations", [])],
                    retrieval_details=details,
                    context_compression=result.get("context_compression", {}),
                    expected_behavior=item.expected_behavior,
                )
            )


        if self._deadline_reached(deadline, job_id, phase="before_judge"):
            return existing_summary or RagasEvaluationSummary(
                total_questions=len(items), completed_questions=start_index,
                failed_questions=0, metrics={}, results=[]
            ), False, start_index
        with self._periodic_heartbeat(
            job_id=job_id,
            completed=start_index,
            total=len(items),
            summary=existing_summary,
            question_id=None,
            question_index=start_index,
            worker_request_id=worker_request_id,
        ):
            batch_summary = self._evaluate_batch(samples, mode=request.mode)
        completed = min(start_index + len(batch_items), len(items))
        self._log_batch(
            job_id,
            batch_summary,
            request.mode,
            next_index=completed,
            remaining_question_count=max(0, len(items) - completed),
        )
        summary = self._merge_summaries(existing_summary, batch_summary, total=len(items))
        self._heartbeat(
            job_id,
            completed=completed,
            total=len(items),
            summary=summary,
        )
        if completed < len(items):
            self._deadline_reached(deadline, job_id, phase="after_checkpoint")
        return summary, completed >= len(items), completed

    def _build_evaluation_retriever(self, dataset, request):
        if self._chunker is None or self._evaluation_embedding_provider is None:
            raise RuntimeError(
                "RAGAS isolated retrieval dependencies are not configured."
            )
        source = self._retrieval_config_source
        return build_isolated_evaluation_retriever(
            dataset=dataset,
            chunker=self._chunker,
            embedding_provider=self._evaluation_embedding_provider,
            docs_root=self._evaluation_docs_root,
            bm25_top_k=getattr(source, "_bm25_top_k", request.top_k),
            vector_top_k=getattr(source, "_vector_top_k", request.top_k),
            rrf_k=getattr(source, "_rrf_k", 60),
            fusion_strategy=getattr(source, "_fusion_strategy", "rrf"),
            bm25_fusion_weight=getattr(source, "_bm25_fusion_weight", 0.7),
            vector_fusion_weight=getattr(source, "_vector_fusion_weight", 0.3),
            preserve_bm25_candidates=getattr(source, "_preserve_bm25_candidates", False),
            enable_bm25_lexical=getattr(source, "_enable_bm25_lexical", True),
            enable_vector_search=getattr(source, "_enable_vector_search", True),
            enable_reciprocal_rank_fusion=getattr(
                source, "_enable_reciprocal_rank_fusion", True
            ),
            enable_reranking=getattr(source, "_enable_reranking", False),
            bm25_first=True,
        )

    def _get_full_evaluation_resources(
        self,
        dataset,
        record: RagasEvaluationJobRecord,
    ) -> tuple[object, list[object]]:
        """Reuse a warm worker's isolated corpus across full-mode continuations."""
        cache_key = (record.dataset_path, record.request.top_k)
        resources = self._full_evaluation_resources.get(cache_key)
        if resources is None:
            resources = self._build_evaluation_retriever(dataset, record.request)
            self._full_evaluation_resources[cache_key] = resources
        return resources

    def _build_evaluation_workflow(self, retriever):
        if self._evaluation_workflow_factory is None:
            raise RuntimeError("RAGAS isolated workflow factory is not configured.")
        return self._evaluation_workflow_factory(retriever)
    def _evaluate_batch(
        self,
        samples: list[RagasEvaluationSample],
        *,
        mode: Literal["standard", "full"],
    ) -> RagasEvaluationSummary:
        grounded_samples: list[RagasEvaluationSample] = []
        refusal_samples: list[RagasEvaluationSample] = []
        preflight_failures: list[RagasQuestionResult] = []
        for sample in samples:
            if sample.retrieved_contexts:
                grounded_samples.append(sample)
            elif sample.expected_behavior == "insufficient_evidence":
                refusal_samples.append(sample)
            else:
                preflight_failures.append(
                    RagasQuestionResult(
                        id=sample.id,
                        question=sample.user_input,
                        answer=sample.response,
                        citations=sample.citations,
                        context_count=0,
                        error=(
                            "RAGAS judge calls skipped because retrieval returned no context "
                            "for a question that requires an answer."
                        ),
                        error_category="no_context",
                    )
                )

        summaries: list[RagasEvaluationSummary] = []
        if grounded_samples:
            summaries.append(
                self._evaluate_samples(
                    grounded_samples,
                    metric_names=(
                        FULL_RAGAS_METRICS if mode == "full" else STANDARD_RAGAS_METRICS
                    ),
                )
            )
        if refusal_samples:
            summaries.append(
                self._evaluate_samples(
                    refusal_samples,
                    metric_names=("answer_correctness",),
                )
            )
        if preflight_failures:
            summaries.append(
                RagasEvaluationSummary(
                    total_questions=len(preflight_failures),
                    completed_questions=0,
                    failed_questions=len(preflight_failures),
                    results=preflight_failures,
                )
            )

        merged: RagasEvaluationSummary | None = None
        for summary in summaries:
            merged = self._merge_summaries(merged, summary, total=len(samples))
        return merged or RagasEvaluationSummary(
            total_questions=0,
            completed_questions=0,
            failed_questions=0,
        )

    def _evaluate_samples(
        self,
        samples: list[RagasEvaluationSample],
        *,
        metric_names: tuple[str, ...],
    ) -> RagasEvaluationSummary:
        return evaluate_ragas(
            samples,
                config=RagasRunConfig(
                    evaluator_model_id=self._evaluator_model_id,
                    embedding_model_id=self._embedding_model_id,
                    metric_names=metric_names,
                    evaluator_max_tokens=self._evaluator_max_tokens,
                    metric_timeout_seconds=self._metric_timeout_seconds,
                    # Metric-level continuations own the one explicit recovery
                    # retry, so a single invocation never makes two judge calls.
                    max_retries=0,
                ),
        )

    @staticmethod
    def _select_items(
        items: list[RagEvaluationItem],
        request: RagasEvaluationRequest,
    ) -> list[RagEvaluationItem]:
        limit = min(request.max_questions, len(items))
        if request.mode == "full" or limit >= len(items):
            return items[:limit]

        complex_markers = (
            "multi_hop",
            "exception",
            "comparison",
            "global",
            "conditional",
            "combination",
            "authority",
            "precedence",
        )
        complex_items = [
            item for item in items if any(marker in item.intent.lower() for marker in complex_markers)
        ]
        simple_items = [item for item in items if item not in complex_items]
        complex_count = min(len(complex_items), (limit + 1) // 2)
        simple_count = min(len(simple_items), limit - complex_count)
        complex_count = min(len(complex_items), limit - simple_count)

        selected = {
            item.id: item
            for item in [
                *_evenly_spaced(simple_items, simple_count),
                *_evenly_spaced(complex_items, complex_count),
            ]
        }
        return [item for item in items if item.id in selected][:limit]

    def _heartbeat(
        self,
        job_id: str,
        *,
        completed: int,
        total: int,
        summary: RagasEvaluationSummary | None = None,
    ) -> None:
        record = self._job_store.get(job_id)
        if record is None:
            raise KeyError(f"Unknown RAGAS evaluation job: {job_id}")
        if record.status == "cancel_requested":
            return
        self._job_store.put(
            record.model_copy(
                update={
                    "updated_at": datetime.now(timezone.utc),
                    "completed_questions": completed,
                    "total_questions": total,
                    "summary": summary if summary is not None else record.summary,
                }
            )
        )

    @staticmethod
    def _check_deadline(deadline: float | None, job_id: str | None = None) -> None:
        if deadline is not None and perf_counter() >= deadline:
            logger.warning(
                "ragas_worker_deadline_approaching",
                extra={"job_id": job_id, "safety_window_ms": _RAGAS_TIMEOUT_BUFFER_MS},
            )
            raise TimeoutError("RAGAS worker reached its execution deadline before the next batch.")

    @staticmethod
    def _deadline_reached(
        deadline: float | None,
        job_id: str | None,
        *,
        phase: str,
    ) -> bool:
        if deadline is None or perf_counter() < deadline:
            return False
        logger.info(
            "ragas_worker_yielding_for_deadline",
            extra={
                "job_id": job_id,
                "phase": phase,
                "safety_window_ms": _RAGAS_TIMEOUT_BUFFER_MS,
            },
        )
        return True

    def _enqueue_or_continue(
        self,
        job_id: str,
        *,
        next_index: int,
        remaining_question_count: int,
    ) -> bool:
        current = self._job_store.get(job_id)
        if current is not None and current.status == "cancel_requested":
            self._cancel_if_requested(job_id)
            logger.info(
                "ragas_worker_continuation_suppressed_cancelled",
                extra={
                    "job_id": job_id,
                    "completed_questions": current.completed_questions,
                    "total_questions": current.total_questions,
                    "next_index": next_index,
                    "remaining_question_count": remaining_question_count,
                },
            )
            return False
        if current is not None and current.status in {"cancelled", "completed", "failed"}:
            return False
        if self._queue_url:
            self._enqueue(job_id)
            logger.info(
                "ragas_worker_continuation_scheduled",
                extra={
                    "job_id": job_id,
                    "completed_batch_size": self._batch_size,
                    "next_index": next_index,
                    "remaining_question_count": remaining_question_count,
                },
            )
        elif self._executor is not None:
            self._executor.submit(self.execute, job_id)
            logger.info(
                "ragas_worker_continuation_scheduled",
                extra={
                    "job_id": job_id,
                    "completed_batch_size": self._batch_size,
                    "next_index": next_index,
                    "remaining_question_count": remaining_question_count,
                },
            )
        elif self._allow_inline_continuation:
            self.execute(job_id)
        else:
            raise RuntimeError(
                "RAGAS continuation queue is not configured for the worker."
            )
        return True

    @staticmethod
    def _merge_summaries(
        previous: RagasEvaluationSummary | None,
        current: RagasEvaluationSummary,
        *,
        total: int,
    ) -> RagasEvaluationSummary:
        results = [*(previous.results if previous else []), *current.results]
        completed_questions = 0
        failed_questions = 0
        metric_values: dict[str, list[tuple[float, int]]] = {}
        failed_metric_counts: dict[str, int] = {}
        for summary in (previous, current):
            if summary is None:
                continue
            if summary.results:
                completed_questions += sum(1 for result in summary.results if not result.error)
                failed_questions += sum(1 for result in summary.results if result.error)
                for result in summary.results:
                    for name, value in result.metrics.items():
                        if value is not None:
                            metric_values.setdefault(name, []).append((float(value), 1))
                for name, count in summary.failed_metric_counts.items():
                    failed_metric_counts[name] = failed_metric_counts.get(name, 0) + count
            else:
                completed_questions += summary.completed_questions
                failed_questions += summary.failed_questions
                for name, value in summary.metrics.items():
                    if value is not None and summary.completed_questions:
                        metric_values.setdefault(name, []).append(
                            (float(value), summary.completed_questions)
                        )
                for name, count in summary.failed_metric_counts.items():
                    failed_metric_counts[name] = failed_metric_counts.get(name, 0) + count
        return RagasEvaluationSummary(
            total_questions=total,
            completed_questions=completed_questions,
            failed_questions=failed_questions,
            evaluation_success_rate=round(completed_questions / total, 4) if total else 0.0,
            first_error=next((result.error for result in results if result.error), None),
            metrics={
                name: sum(value * weight for value, weight in values) / sum(weight for _, weight in values)
                for name, values in metric_values.items()
            },
            failed_metric_counts=failed_metric_counts,
            results=results,
        )

    def _log_batch(
        self,
        job_id: str,
        summary: RagasEvaluationSummary,
        mode: Literal["standard", "full"],
        *,
        next_index: int,
        remaining_question_count: int,
    ) -> None:
        metric_names = FULL_RAGAS_METRICS if mode == "full" else STANDARD_RAGAS_METRICS
        logger.info(
            "ragas_batch_finished",
            extra={
                "job_id": job_id,
                "batch_size": summary.total_questions,
                "completed_questions": summary.completed_questions,
                "failed_questions": summary.failed_questions,
                "next_index": next_index,
                "remaining_question_count": remaining_question_count,
                "metric_names": list(metric_names),
                "evaluator_model_id": self._evaluator_model_id,
                "embedding_model_id": self._embedding_model_id,
            },
        )
        if summary.completed_questions == 0 and summary.failed_questions == summary.total_questions:
            logger.error(
                "ragas_all_questions_failed",
                extra={
                    "job_id": job_id,
                    "batch_size": summary.total_questions,
                    "first_error": summary.first_error,
                },
            )

    @contextmanager
    def _periodic_heartbeat(
        self,
        *,
        job_id: str,
        completed: int,
        total: int,
        summary: RagasEvaluationSummary | None,
        question_id: str | None,
        question_index: int,
        worker_request_id: str | None,
    ):
        stop = Event()

        def beat() -> None:
            while not stop.wait(_RAGAS_HEARTBEAT_INTERVAL_SECONDS):
                try:
                    self._heartbeat(
                        job_id,
                        completed=completed,
                        total=total,
                        summary=summary,
                    )
                    logger.info(
                        "ragas_worker_heartbeat",
                        extra={
                            "job_id": job_id,
                            "worker_request_id": worker_request_id,
                            "question_id": question_id,
                            "question_index": question_index,
                            "completed_questions": completed,
                            "total_questions": total,
                        },
                    )
                except Exception:
                    logger.exception(
                        "ragas_worker_heartbeat_failed",
                        extra={"job_id": job_id, "worker_request_id": worker_request_id},
                    )

        thread = Thread(target=beat, name="ragas-heartbeat", daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=1)

    def _enqueue(
        self,
        job_id: str,
        *,
        question_index: int | None = None,
        phase: str | None = None,
        metric_name: str | None = None,
    ) -> None:
        if self._sqs_client is None:
            import boto3

            self._sqs_client = boto3.client("sqs")
        payload: dict[str, object] = {"job_id": job_id}
        if question_index is not None:
            payload.update({"question_index": question_index, "phase": phase, "metric": metric_name})
        self._sqs_client.send_message(
            QueueUrl=self._queue_url,
            MessageBody=json.dumps(payload),
        )


def _default_dataset_exists_checker(path: str) -> bool:
    if _is_s3_uri(path):
        return s3_object_exists(path)
    from pathlib import Path

    return Path(path).exists()


def _evenly_spaced(
    items: list[RagEvaluationItem],
    count: int,
) -> list[RagEvaluationItem]:
    if count <= 0:
        return []
    if count >= len(items):
        return list(items)
    if count == 1:
        return [items[len(items) // 2]]
    last_index = len(items) - 1
    indexes = [round(position * last_index / (count - 1)) for position in range(count)]
    return [items[index] for index in indexes]
