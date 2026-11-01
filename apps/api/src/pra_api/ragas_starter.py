from __future__ import annotations

import json
import os

import boto3
from botocore.exceptions import ClientError

from pra_core.logging import get_logger

logger = get_logger(__name__)


def _step_functions_client():
    return boto3.client("stepfunctions")


def handler(event, context):
    """Start one idempotent Step Functions execution for each RAGAS job message."""
    state_machine_arn = os.environ["PRA_RAGAS_STATE_MACHINE_ARN"]
    records = event.get("Records", []) if isinstance(event, dict) else []
    if not isinstance(records, list) or len(records) != 1:
        raise ValueError("RAGAS starter accepts exactly one SQS record per invocation.")

    record = records[0]
    if not isinstance(record, dict):
        raise ValueError("Malformed RAGAS starter SQS record.")
    try:
        body = json.loads(record.get("body", "{}"))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Malformed RAGAS starter SQS message body.") from exc
    if not isinstance(body, dict) or not isinstance(body.get("job_id"), str):
        raise ValueError("RAGAS starter SQS message requires a job_id.")

    job_id = body["job_id"]
    execution_name = f"ragas-{job_id}"
    try:
        response = _step_functions_client().start_execution(
            stateMachineArn=state_machine_arn,
            name=execution_name,
            input=json.dumps({"job_id": job_id}),
        )
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") != "ExecutionAlreadyExists":
            raise
        logger.info(
            "ragas_step_functions_execution_already_running",
            extra={"job_id": job_id, "execution_name": execution_name},
        )
        return {"job_id": job_id, "status": "already_started"}

    logger.info(
        "ragas_step_functions_execution_started",
        extra={
            "job_id": job_id,
            "execution_name": execution_name,
            "execution_arn": response.get("executionArn"),
        },
    )
    return {"job_id": job_id, "status": "started"}
