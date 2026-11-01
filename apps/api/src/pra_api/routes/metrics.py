from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter(tags=["metrics"])


class MetricsResponse(BaseModel):
    request_count: int
    error_count: int
    average_latency_ms: float
    llm_call_count: int


@router.get("/metrics", response_model=MetricsResponse)
def metrics(request: Request) -> MetricsResponse:
    snapshot = request.app.state.metrics.snapshot()
    return MetricsResponse(
        request_count=snapshot.request_count,
        error_count=snapshot.error_count,
        average_latency_ms=round(snapshot.average_latency_ms, 2),
        llm_call_count=snapshot.llm_call_count,
    )
