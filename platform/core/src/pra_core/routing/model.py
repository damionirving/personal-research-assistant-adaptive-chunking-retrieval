from dataclasses import dataclass

from pra_core.documents.models import RetrievedDocument

STRONG_RRF_SCORE_THRESHOLD = 0.03
BORDERLINE_RRF_SCORE_THRESHOLD = 0.02
STRONG_SEMANTIC_SCORE_THRESHOLD = 0.88
MIN_SUPPORTING_RRF_SCORE_THRESHOLD = 0.015
MIN_SUPPORTING_CHUNK_COUNT = 2
STRONG_SUPPORTING_RRF_SCORE_THRESHOLD = 0.02


@dataclass(frozen=True)
class ModelRoutingDecision:
    selected_model: str
    routing_reason: str
    routing_confidence: float


class DeterministicModelRouter:
    """Rule-based router for low-cost/default model selection."""

    def __init__(
        self,
        simple_model_id: str,
        reasoning_model_id: str,
    ) -> None:
        self._simple_model_id = simple_model_id
        self._reasoning_model_id = reasoning_model_id

    @property
    def simple_model_id(self) -> str:
        return self._simple_model_id

    @property
    def reasoning_model_id(self) -> str:
        return self._reasoning_model_id

    def select_model(
        self,
        message: str,
        documents: list[RetrievedDocument],
        history: list[str],
    ) -> ModelRoutingDecision:
        normalized = message.strip().lower()
        max_rrf_score = max((_document_rrf_score(document) for document in documents), default=None)
        max_semantic_score = max(
            (_document_semantic_score(document) for document in documents),
            default=None,
        )
        supporting_document_count = sum(
            1
            for document in documents
            if _document_rrf_score(document) >= MIN_SUPPORTING_RRF_SCORE_THRESHOLD
        )

        if _is_greeting(normalized):
            return ModelRoutingDecision(
                selected_model=self._simple_model_id,
                routing_reason="greeting_or_small_talk",
                routing_confidence=0.98,
            )

        if (
            _is_definition_question(normalized)
            and documents
            and max_rrf_score is not None
            and max_rrf_score >= BORDERLINE_RRF_SCORE_THRESHOLD
        ):
            return ModelRoutingDecision(
                selected_model=self._simple_model_id,
                routing_reason="definition_high_confidence_retrieval",
                routing_confidence=0.9,
            )

        if _is_high_value_reasoning(normalized):
            return ModelRoutingDecision(
                selected_model=self._reasoning_model_id,
                routing_reason="high_value_reasoning",
                routing_confidence=0.88,
            )

        if _is_conditional_reasoning_candidate(normalized):
            return ModelRoutingDecision(
                selected_model=self._reasoning_model_id,
                routing_reason="conditional_or_exception_reasoning",
                routing_confidence=0.86,
            )

        if _is_multi_record_rule_question(normalized):
            return ModelRoutingDecision(
                selected_model=self._reasoning_model_id,
                routing_reason="multi_record_rule_reasoning",
                routing_confidence=0.85,
            )

        if _requires_multi_chunk_reconciliation(normalized, documents):
            return ModelRoutingDecision(
                selected_model=self._reasoning_model_id,
                routing_reason="multi_chunk_reconciliation",
                routing_confidence=0.84,
            )

        if (
            documents
            and max_rrf_score is not None
            and max_rrf_score >= STRONG_RRF_SCORE_THRESHOLD
        ):
            return ModelRoutingDecision(
                selected_model=self._simple_model_id,
                routing_reason="simple_high_rrf_retrieval",
                routing_confidence=0.87,
            )

        if (
            documents
            and max_rrf_score is not None
            and max_rrf_score >= BORDERLINE_RRF_SCORE_THRESHOLD
            and max_semantic_score is not None
            and max_semantic_score >= STRONG_SEMANTIC_SCORE_THRESHOLD
        ):
            return ModelRoutingDecision(
                selected_model=self._simple_model_id,
                routing_reason="simple_borderline_rrf_strong_semantic_retrieval",
                routing_confidence=0.82,
            )

        if (
            documents
            and max_rrf_score is not None
            and max_rrf_score >= BORDERLINE_RRF_SCORE_THRESHOLD
            and supporting_document_count >= MIN_SUPPORTING_CHUNK_COUNT
        ):
            return ModelRoutingDecision(
                selected_model=self._simple_model_id,
                routing_reason="simple_borderline_rrf_multi_support_retrieval",
                routing_confidence=0.79,
            )

        if documents:
            return ModelRoutingDecision(
                selected_model=self._reasoning_model_id,
                routing_reason="weak_or_moderate_hybrid_retrieval",
                routing_confidence=0.74,
            )

        if _looks_like_research_question(normalized):
            return ModelRoutingDecision(
                selected_model=self._reasoning_model_id,
                routing_reason="reasoning_no_retrieval_question",
                routing_confidence=0.69,
            )

        if history:
            return ModelRoutingDecision(
                selected_model=self._simple_model_id,
                routing_reason="simple_conversation_with_history",
                routing_confidence=0.72,
            )

        return ModelRoutingDecision(
            selected_model=self._simple_model_id,
            routing_reason="simple_chat_no_retrieval",
            routing_confidence=0.8,
        )


def _is_greeting(message: str) -> bool:
    return message in {
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
    }


def _is_definition_question(message: str) -> bool:
    return (
        message.startswith(("what is ", "what are ", "who is ", "define "))
        or "what does " in message and message.endswith(" mean?")
        or " definition" in message
    )


def _is_high_value_reasoning(message: str) -> bool:
    high_value_terms = {
        "analyze",
        "analysis",
        "compare",
        "recommend",
        "recommendation",
        "evaluate",
        "strategy",
        "risk",
        "legal",
        "financial",
        "compliance",
        "contract",
        "policy tradeoff",
        "pros and cons",
        "decision",
    }
    return any(term in message for term in high_value_terms)


def _is_conditional_reasoning_candidate(message: str) -> bool:
    conditional_terms = {
        "always",
        "except",
        "unless",
        "only if",
        "does this still apply",
        "still apply",
        "whether",
        "under what conditions",
        "if ",
    }
    return any(term in message for term in conditional_terms)


def _is_multi_record_rule_question(message: str) -> bool:
    return (
        _looks_like_research_question(message)
        and any(
            term in message
            for term in (
                "incidents",
                "records",
                "claims",
                "shipments",
                "employees",
                "users",
                "items",
                "cases",
            )
        )
        and any(
            term in message
            for term in (
                "why",
                "require",
                "required",
                "requires",
                "approval",
                "qualify",
                "qualifies",
                "eligible",
                "exception",
                "violate",
                "violates",
            )
        )
    )


def _looks_like_research_question(message: str) -> bool:
    question_starters = (
        "what",
        "why",
        "how",
        "when",
        "where",
        "which",
        "who",
        "can",
        "could",
        "should",
        "would",
        "does",
        "do",
        "is",
        "are",
        "explain",
        "describe",
        "summarize",
        "tell me",
    )
    return "?" in message or message.startswith(question_starters)


def _requires_multi_chunk_reconciliation(
    message: str,
    documents: list[RetrievedDocument],
) -> bool:
    if len(documents) < 2:
        return False
    if not _is_conditional_reasoning_candidate(message) and not _looks_like_boolean_decision(message):
        return False

    strong_support_count = sum(
        1
        for document in documents
        if _document_rrf_score(document) >= STRONG_SUPPORTING_RRF_SCORE_THRESHOLD
    )
    return strong_support_count >= 2


def _looks_like_boolean_decision(message: str) -> bool:
    decision_starters = (
        "is ",
        "are ",
        "does ",
        "do ",
        "can ",
        "could ",
        "should ",
        "would ",
    )
    return message.startswith(decision_starters)


def _document_rrf_score(document: RetrievedDocument) -> float:
    if document.rrf_score is not None:
        return document.rrf_score
    return document.score


def _document_semantic_score(document: RetrievedDocument) -> float:
    if document.vector_score is not None:
        return document.vector_score
    return 0.0
