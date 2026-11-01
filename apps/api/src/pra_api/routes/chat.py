import hashlib

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field

from pra_api.dependencies import (
    get_auth_provider,
    get_guardrail,
    get_memory_store,
    get_rate_limiter,
    get_workflow,
)
from pra_api.rate_limits import enforce_chat_rate_limit
from pra_core.auth.base import AuthError, AuthProvider
from pra_core.agent.workflow import ResearchAssistantWorkflow
from pra_core.guardrails.base import Guardrail
from pra_core.llm.base import LLMProviderError
from pra_core.logging import get_logger
from pra_core.memory.store import MemoryStore
from pra_core.rate_limit import FixedWindowRateLimiter

router = APIRouter(prefix="/chat", tags=["chat"])
MODEL_UNAVAILABLE_MESSAGE = "The model is temporarily unavailable. Please try again."
logger = get_logger(__name__)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str = Field(default="default", min_length=1, max_length=128)
    top_k: int = Field(default=25, ge=1, le=25)


class ChatResponse(BaseModel):
    session_id: str
    trace_id: str = ""
    message_hash: str = ""
    answer: str
    citations: list[str]
    retrieved_chunk_labels: list[str] = Field(default_factory=list)
    top_chunk_label: str | None = None
    retrieval_scores: list[float] = Field(default_factory=list)
    retrieval_details: list[dict[str, object]] = Field(default_factory=list)
    knowledge_graph_enabled: bool = False
    knowledge_graph_hits: int = 0
    top_knowledge_graph_score: float | None = None
    knowledge_graph_contribution: bool = False
    min_retrieval_score: float | None = None
    max_retrieval_score: float | None = None
    avg_retrieval_score: float | None = None
    context_compression: dict[str, object] = Field(default_factory=dict)
    selected_model: str
    routing_reason: str
    routing_confidence: float
    hallucination_score: float
    hallucination_reason: str
    hallucination_action: str
    topic_reset_applied: bool = False
    principal: str


class ResetChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)


class ResetChatResponse(BaseModel):
    cleared_session_id: str
    principal: str


@router.post("", response_model=ChatResponse)
def chat(
    request: Request,
    payload: ChatRequest,
    authorization: str | None = Header(default=None),
    workflow: ResearchAssistantWorkflow = Depends(get_workflow),
    auth_provider: AuthProvider = Depends(get_auth_provider),
    guardrail: Guardrail = Depends(get_guardrail),
    rate_limiter: FixedWindowRateLimiter = Depends(get_rate_limiter),
) -> ChatResponse:
    try:
        principal = auth_provider.authenticate(authorization)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    enforce_chat_rate_limit(
        request=request,
        principal=principal,
        limiter=rate_limiter,
    )

    guardrail_result = guardrail.validate(payload.message)
    if not guardrail_result.allowed:
        raise HTTPException(status_code=400, detail=guardrail_result.reason)

    trace_id = getattr(request.state, "trace_id", "")
    message_hash = _message_hash(payload.message)
    logger.info(
        "chat_request_received",
        extra={
            "trace_id": trace_id,
            "session_id": payload.session_id,
            "message_hash": message_hash,
            "principal": principal.subject,
        },
    )

    try:
        result = workflow.run(
            message=payload.message,
            session_id=payload.session_id,
            top_k=payload.top_k,
        )
    except LLMProviderError as exc:
        raise HTTPException(status_code=502, detail=MODEL_UNAVAILABLE_MESSAGE) from exc
    retrieved_chunk_labels = [
        str(detail.get("chunk_label") or detail.get("filename") or "")
        for detail in result.get("retrieval_details", [])
        if detail.get("chunk_label") or detail.get("filename")
    ]
    result["trace_id"] = trace_id
    result["message_hash"] = message_hash
    result["retrieved_chunk_labels"] = retrieved_chunk_labels
    result["top_chunk_label"] = retrieved_chunk_labels[0] if retrieved_chunk_labels else None
    result["principal"] = principal.subject
    logger.info(
        "chat_retrieval_diagnostics",
        extra={
            "trace_id": trace_id,
            "session_id": payload.session_id,
            "message_hash": message_hash,
            "principal": principal.subject,
            "retrieved_chunk_labels": retrieved_chunk_labels,
            "top_chunk_label": result.get("top_chunk_label"),
            "retrieved_document_count": len(result.get("retrieval_details", [])),
        },
    )
    logger.info(
        "chat_response_sent",
        extra={
            "trace_id": trace_id,
            "session_id": payload.session_id,
            "message_hash": message_hash,
            "principal": principal.subject,
            "selected_model": result.get("selected_model"),
            "citation_count": len(result.get("citations", [])),
        },
    )
    return ChatResponse(**result)


@router.post("/reset", response_model=ResetChatResponse)
def reset_chat(
    request: Request,
    payload: ResetChatRequest,
    authorization: str | None = Header(default=None),
    auth_provider: AuthProvider = Depends(get_auth_provider),
    memory_store: MemoryStore = Depends(get_memory_store),
) -> ResetChatResponse:
    try:
        principal = auth_provider.authenticate(authorization)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    memory_store.clear_session(payload.session_id)
    logger.info(
        "chat_session_cleared",
        extra={
            "trace_id": getattr(request.state, "trace_id", ""),
            "session_id": payload.session_id,
            "principal": principal.subject,
        },
    )
    return ResetChatResponse(
        cleared_session_id=payload.session_id,
        principal=principal.subject,
    )


def _message_hash(message: str) -> str:
    return hashlib.sha256(message.encode("utf-8")).hexdigest()[:16]
