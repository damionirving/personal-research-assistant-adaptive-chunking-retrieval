from __future__ import annotations

import json

from pra_api.main import create_app
from pra_core.logging import get_logger

logger = get_logger(__name__)
WORKER_APP = create_app()


def handler(event, context):
    coordinator = WORKER_APP.state.rag_evaluation_coordinator
    processed: list[dict[str, object]] = []
    for record in event.get("Records", []):
        body = json.loads(record.get("body", "{}"))
        job_id = body.get("job_id")
        if not job_id:
            logger.warning("rag_evaluation_job_missing_job_id", extra={"record": body})
            continue
        processed_record = coordinator.execute(job_id)
        processed.append(
            {
                "job_id": processed_record.job_id,
                "status": processed_record.status,
            }
        )
    return {"processed": processed}
