from __future__ import annotations

import json
import os

from pra_core.logging import get_logger

logger = get_logger(__name__)

# Keep the low-cost isolated path as the default while allowing the deployed
# opt-in flag and production model settings to pass through unchanged.
RAGAS_GEMMA_MODEL_ID = "google.gemma-3-12b-it"
os.environ["PRA_RAGAS_EVALUATION_WORKER"] = "true"
os.environ.setdefault("PRA_RAGAS_MODEL_ONLY_ID", RAGAS_GEMMA_MODEL_ID)
os.environ.setdefault(
    "PRA_RAGAS_EVALUATOR_MODEL_ID", f"bedrock/{RAGAS_GEMMA_MODEL_ID}"
)

WORKER_APP = None


def _get_worker_app():
    """Defer expensive RAGAS application construction until an invocation begins."""
    global WORKER_APP
    if WORKER_APP is None:
        from pra_api.main import create_app

        WORKER_APP = create_app()
    return WORKER_APP


def _run_step(
    coordinator,
    job_id: str,
    *,
    aws_request_id: str | None,
    remaining_time_ms: int | None,
    external_orchestrator: bool,
) -> dict[str, object]:
    """Execute one durable RAGAS step and return only orchestration state."""
    loaded = coordinator.get(job_id)
    if loaded is None:
        logger.warning(
            "ragas_worker_skipped",
            extra={"aws_request_id": aws_request_id, "job_id": job_id, "reason": "job_not_found"},
        )
        return {"job_id": job_id, "status": "not_found"}

    if loaded.status in {"completed", "failed", "cancelled"}:
        return _step_result(loaded)
    if loaded.status not in {"queued", "running", "cancel_requested"}:
        raise RuntimeError(f"Invalid RAGAS job status: {loaded.status}")

    processed_record = coordinator.execute(
        job_id,
        remaining_time_ms=remaining_time_ms,
        worker_request_id=aws_request_id,
        external_orchestrator=external_orchestrator,
    )
    return _step_result(processed_record)


def _step_result(record) -> dict[str, object]:
    return {
        "job_id": record.job_id,
        "status": record.status,
        "completed_questions": record.completed_questions,
        "total_questions": record.total_questions,
    }


def handler(event, context):
    coordinator = _get_worker_app().state.ragas_coordinator
    aws_request_id = getattr(context, "aws_request_id", None)
    raw_records = event.get("Records", []) if isinstance(event, dict) else None
    records = raw_records if isinstance(raw_records, list) else []

    # Step Functions invokes this same image synchronously with only a job ID.
    # It owns the loop, so this path must never enqueue another SQS message.
    if isinstance(event, dict) and not records and isinstance(event.get("job_id"), str):
        job_id = event["job_id"]
        logger.info(
            "ragas_step_functions_step_received",
            extra={"aws_request_id": aws_request_id, "job_id": job_id},
        )
        remaining_time_ms = (
            context.get_remaining_time_in_millis()
            if context is not None and hasattr(context, "get_remaining_time_in_millis")
            else None
        )
        result = _run_step(
            coordinator,
            job_id,
            aws_request_id=aws_request_id,
            remaining_time_ms=remaining_time_ms,
            external_orchestrator=True,
        )
        logger.info(
            "ragas_step_functions_step_complete",
            extra={"aws_request_id": aws_request_id, **result},
        )
        return result

    logger.info(
        "ragas_worker_invocation_received",
        extra={
            "aws_request_id": aws_request_id,
            "record_count": len(records),
            "event_source_types": sorted(
                {str(record.get("eventSource")) for record in records if isinstance(record, dict)}
            ),
        },
    )
    if not records:
        reason = "no_records" if raw_records == [] else "unsupported_event"
        logger.warning(
            "ragas_worker_skipped",
            extra={"aws_request_id": aws_request_id, "reason": reason},
        )
        logger.info(
            "ragas_worker_invocation_complete",
            extra={"aws_request_id": aws_request_id, "processed_batch_count": 0},
        )
        return {"processed": []}

    if len(records) > 1:
        logger.error(
            "ragas_worker_skipped",
            extra={
                "aws_request_id": aws_request_id,
                "record_count": len(records),
                "reason": "multiple_records_not_allowed",
            },
        )
        raise RuntimeError("RAGAS worker accepts exactly one SQS record per invocation.")

    processed: list[dict[str, object]] = []
    for record in records:
        if not isinstance(record, dict):
            logger.error(
                "ragas_worker_skipped",
                extra={"aws_request_id": aws_request_id, "reason": "malformed_message"},
            )
            raise ValueError("Malformed RAGAS SQS record.")

        message_id = record.get("messageId")
        attributes = record.get("attributes") or {}
        logger.info(
            "ragas_worker_record_received",
            extra={
                "aws_request_id": aws_request_id,
                "message_id": message_id,
                "event_source": record.get("eventSource"),
                "approximate_receive_count": attributes.get("ApproximateReceiveCount"),
            },
        )
        try:
            body = json.loads(record.get("body", "{}"))
        except (TypeError, json.JSONDecodeError) as exc:
            logger.error(
                "ragas_worker_skipped",
                extra={
                    "aws_request_id": aws_request_id,
                    "message_id": message_id,
                    "reason": "malformed_message",
                    "error_type": type(exc).__name__,
                },
            )
            raise ValueError("Malformed RAGAS SQS message body.") from exc

        if not isinstance(body, dict):
            logger.error(
                "ragas_worker_skipped",
                extra={
                    "aws_request_id": aws_request_id,
                    "message_id": message_id,
                    "reason": "malformed_message",
                },
            )
            raise ValueError("RAGAS SQS message body must be an object.")

        job_id = body.get("job_id")
        logger.info(
            "ragas_worker_message_parsed",
            extra={
                "aws_request_id": aws_request_id,
                "message_id": message_id,
                "job_id": job_id,
                "mode": body.get("mode"),
                "metric_names": body.get("metric_names"),
                "max_questions": body.get("max_questions"),
            },
        )
        if not job_id:
            logger.warning(
                "ragas_worker_skipped",
                extra={
                    "aws_request_id": aws_request_id,
                    "message_id": message_id,
                    "reason": "missing_job_id",
                },
            )
            continue

        remaining_time_ms = (
            context.get_remaining_time_in_millis()
            if context is not None and hasattr(context, "get_remaining_time_in_millis")
            else None
        )
        try:
            processed_record = _run_step(
                coordinator,
                job_id,
                aws_request_id=aws_request_id,
                remaining_time_ms=remaining_time_ms,
                external_orchestrator=False,
            )
        except Exception:
            logger.exception(
                "ragas_worker_processing_failed",
                extra={
                    "aws_request_id": aws_request_id,
                    "message_id": message_id,
                    "job_id": job_id,
                },
            )
            raise
        processed.append({"job_id": processed_record["job_id"], "status": processed_record["status"]})
    logger.info(
        "ragas_worker_invocation_complete",
        extra={
            "aws_request_id": aws_request_id,
            "job_id": processed[0].get("job_id") if processed else None,
            "processed_batch_count": len(processed),
        },
    )
    return {"processed": processed}
