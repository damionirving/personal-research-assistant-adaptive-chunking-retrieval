from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

import boto3

logger = logging.getLogger(__name__)

_TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


def _s3_client():
    return boto3.client("s3")


def _job_key(job_id: str, prefix: str) -> str:
    base = prefix.strip("/")
    key = f"ragas-evaluation/jobs/{job_id}.json"
    return f"{base}/{key}" if base else key


def handler(event, context):
    """Persist low-cost Step Functions lifecycle checkpoints for one RAGAS job."""
    if not isinstance(event, dict):
        raise ValueError("RAGAS checkpoint event must be an object.")
    job_id = event.get("job_id")
    checkpoint = event.get("checkpoint")
    if not isinstance(job_id, str) or not job_id:
        raise ValueError("RAGAS checkpoint event requires a job_id.")
    if checkpoint not in {"orchestration_started", "waiting_for_capacity", "capacity_exhausted"}:
        raise ValueError("RAGAS checkpoint event has an unsupported checkpoint type.")

    bucket = os.environ["PRA_DOCUMENT_BUCKET_NAME"]
    prefix = os.environ.get("PRA_PERSISTENCE_S3_PREFIX", "pra-prod")
    key = _job_key(job_id, prefix)
    s3 = _s3_client()
    payload = json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8"))
    status = payload.get("status")
    if status in _TERMINAL_STATUSES or status == "cancel_requested":
        return {
            "job_id": job_id,
            "status": status,
            "capacity_wait_count": int(payload.get("capacity_wait_count", 0)),
            "checkpoint": "ignored_terminal_job",
        }

    capacity_wait_count = int(payload.get("capacity_wait_count", 0))
    if checkpoint == "orchestration_started":
        orchestration_state = "waiting_for_worker_capacity"
        message = "RAGAS evaluation is queued for the single cost-capped worker."
    elif checkpoint == "waiting_for_capacity":
        capacity_wait_count += 1
        orchestration_state = "waiting_for_worker_capacity"
        message = (
            "RAGAS evaluation is waiting for the single cost-capped worker. "
            f"Capacity retry {capacity_wait_count} is scheduled."
        )
    else:
        payload["status"] = "failed"
        orchestration_state = "capacity_timeout"
        message = "RAGAS evaluation could not acquire worker capacity before its retry limit."
        payload["error"] = message

    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    payload["orchestration_state"] = orchestration_state
    payload["orchestration_message"] = message
    payload["capacity_wait_count"] = capacity_wait_count
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    logger.info(
        "ragas_orchestration_checkpointed",
        extra={
            "job_id": job_id,
            "checkpoint": checkpoint,
            "status": payload["status"],
            "capacity_wait_count": capacity_wait_count,
        },
    )
    return {
        "job_id": job_id,
        "status": payload["status"],
        "capacity_wait_count": capacity_wait_count,
        "checkpoint": checkpoint,
    }
