from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Header, HTTPException, Request

from pra_api.dependencies import get_auth_provider, get_rate_limiter
from pra_api.rate_limits import enforce_ragas_rate_limit
from pra_core.auth.base import AuthError, AuthProvider
from pra_core.evaluation.jobs import RagEvaluationJobRecord, RagEvaluationRequest
from pra_core.evaluation.ragas_jobs import RagasEvaluationJobRecord, RagasEvaluationRequest
from pra_core.evaluation.rag_retrieval import _is_s3_uri, resolve_dataset_path, s3_object_exists
from pra_core.logging import get_logger
from pra_core.rate_limit import FixedWindowRateLimiter

router = APIRouter(prefix="/rag", tags=["rag"])
logger = get_logger(__name__)


@router.post("/evaluate", response_model=RagEvaluationJobRecord, status_code=202)
def evaluate_rag(
    request: Request,
    payload: RagEvaluationRequest,
    authorization: str | None = Header(default=None),
    trace_id: str | None = Header(default=None, alias="x-trace-id"),
    auth_provider: AuthProvider = Depends(get_auth_provider),
) -> RagEvaluationJobRecord:
    if not request.app.state.settings.rag_evaluation_enabled:
        raise HTTPException(status_code=503, detail="RAG performance evaluation is disabled.")
    try:
        principal = auth_provider.authenticate(authorization)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    dataset_path = _dataset_path(request, payload.dataset)
    if not _dataset_exists(dataset_path):
        raise HTTPException(
            status_code=404,
            detail=(
                "RAG evaluation dataset was not found. "
                f"Place the dataset file at {dataset_path} or set PRA_RAG_EVAL_DATASET_PATH."
            ),
        )

    coordinator = request.app.state.rag_evaluation_coordinator
    record = coordinator.submit(
        request=payload,
        principal=principal.subject,
        trace_id=trace_id,
        dataset_path=str(dataset_path),
    )
    logger.info(
        "rag_evaluation_submitted",
        extra={
            "job_id": record.job_id,
            "dataset_path": str(dataset_path),
            "principal": principal.subject,
            "trace_id": trace_id,
        },
    )
    return record


@router.get("/evaluate/{job_id}", response_model=RagEvaluationJobRecord)
def get_rag_evaluation(
    job_id: str,
    request: Request,
    authorization: str | None = Header(default=None),
    auth_provider: AuthProvider = Depends(get_auth_provider),
) -> RagEvaluationJobRecord:
    try:
        principal = auth_provider.authenticate(authorization)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    record = request.app.state.rag_evaluation_coordinator.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"RAG evaluation job {job_id} was not found.")
    if record.principal != principal.subject:
        raise HTTPException(status_code=403, detail="You cannot access another user's RAG evaluation job.")
    return record


@router.post("/evaluate/ragas", response_model=RagasEvaluationJobRecord, status_code=202)
def evaluate_ragas(
    request: Request,
    payload: RagasEvaluationRequest,
    authorization: str | None = Header(default=None),
    trace_id: str | None = Header(default=None, alias="x-trace-id"),
    auth_provider: AuthProvider = Depends(get_auth_provider),
    rate_limiter: FixedWindowRateLimiter = Depends(get_rate_limiter),
) -> RagasEvaluationJobRecord:
    if not request.app.state.settings.ragas_enabled:
        raise HTTPException(status_code=503, detail="RAGAS evaluation is disabled.")
    try:
        principal = auth_provider.authenticate(authorization)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    enforce_ragas_rate_limit(request, principal, rate_limiter)

    dataset_path = _ragas_dataset_path(request, payload.dataset)
    if not _dataset_exists(dataset_path):
        raise HTTPException(
            status_code=404,
            detail=(
                "RAGAS evaluation dataset was not found. "
                f"Place the dataset file at {dataset_path} or set PRA_RAGAS_EVAL_DATASET_PATH."
            ),
        )
    try:
        question_limit = (
            request.app.state.settings.ragas_standard_max_questions
            if payload.mode == "standard"
            else request.app.state.settings.ragas_max_questions
        )
        top_k_limit = (
            request.app.state.settings.ragas_standard_top_k
            if payload.mode == "standard"
            else 25
        )
        payload = payload.model_copy(
            update={
                "max_questions": min(payload.max_questions, question_limit),
                "top_k": min(payload.top_k, top_k_limit),
            }
        )
        record = request.app.state.ragas_coordinator.submit(
            request=payload,
            principal=principal.subject,
            trace_id=trace_id,
            dataset_path=str(dataset_path),
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return record


@router.get("/evaluate/ragas/{job_id}", response_model=RagasEvaluationJobRecord)
def get_ragas_evaluation(
    job_id: str,
    request: Request,
    authorization: str | None = Header(default=None),
    auth_provider: AuthProvider = Depends(get_auth_provider),
) -> RagasEvaluationJobRecord:
    try:
        principal = auth_provider.authenticate(authorization)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    record = request.app.state.ragas_coordinator.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"RAGAS evaluation job {job_id} was not found.")
    if record.principal != principal.subject:
        raise HTTPException(status_code=403, detail="You cannot access another user's RAGAS evaluation job.")
    return record


@router.post("/evaluate/ragas/{job_id}/cancel", response_model=RagasEvaluationJobRecord)
def cancel_ragas_evaluation(
    job_id: str,
    request: Request,
    authorization: str | None = Header(default=None),
    auth_provider: AuthProvider = Depends(get_auth_provider),
) -> RagasEvaluationJobRecord:
    try:
        principal = auth_provider.authenticate(authorization)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    record = request.app.state.ragas_coordinator.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"RAGAS evaluation job {job_id} was not found.")
    if record.principal != principal.subject:
        raise HTTPException(status_code=403, detail="You cannot cancel another user's RAGAS evaluation job.")
    cancelled = request.app.state.ragas_coordinator.cancel(job_id)
    if cancelled is None:
        raise HTTPException(status_code=404, detail=f"RAGAS evaluation job {job_id} was not found.")
    return cancelled


def _dataset_path(request: Request, dataset_name: str) -> str | Path:
    if dataset_name != "default":
        raise HTTPException(
            status_code=400,
            detail="Only the default retrieval evaluation dataset is currently supported.",
        )
    return resolve_dataset_path(request.app.state.settings.rag_eval_dataset_path)


def _ragas_dataset_path(request: Request, dataset_name: str) -> str | Path:
    if dataset_name != "default":
        raise HTTPException(
            status_code=400,
            detail="Only the default RAGAS evaluation dataset is currently supported.",
        )
    return resolve_dataset_path(request.app.state.settings.ragas_eval_dataset_path)


def _dataset_exists(path: str | Path) -> bool:
    if _is_s3_uri(path):
        return s3_object_exists(str(path))
    return Path(path).exists()
