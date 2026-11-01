from dataclasses import dataclass

from pra_core.documents.models import RetrievedDocument

HIGH_RRF_SUPPORT_THRESHOLD = 0.03
MODERATE_RRF_SUPPORT_THRESHOLD = 0.015


@dataclass(frozen=True)
class GroundingScore:
    hallucination_score: float
    hallucination_reason: str


class GroundingScorer:
    """Heuristic grounding-risk scorer for MVP transparency."""

    def score(
        self,
        message: str,
        answer: str,
        documents: list[RetrievedDocument],
    ) -> GroundingScore:
        normalized_message = message.strip().lower()
        normalized_answer = answer.strip().lower()
        max_score = max((_document_rrf_score(document) for document in documents), default=None)

        if _is_small_talk(normalized_message):
            return GroundingScore(
                hallucination_score=0.1,
                hallucination_reason="Small talk does not require retrieved context.",
            )

        if not documents:
            if _is_uncertain_answer(normalized_answer):
                return GroundingScore(
                    hallucination_score=0.25,
                    hallucination_reason="No relevant context was retrieved and the answer expressed uncertainty.",
                )
            return GroundingScore(
                hallucination_score=0.82,
                hallucination_reason="No relevant context was retrieved for a factual answer.",
            )

        if max_score is None:
            return GroundingScore(
                hallucination_score=0.65,
                hallucination_reason="Retrieved context had no usable similarity score.",
            )

        if max_score >= HIGH_RRF_SUPPORT_THRESHOLD:
            return GroundingScore(
                hallucination_score=0.16,
                hallucination_reason="Answer is supported by strong fused retrieval context.",
            )

        if max_score >= MODERATE_RRF_SUPPORT_THRESHOLD:
            return GroundingScore(
                hallucination_score=0.38,
                hallucination_reason="Answer is supported by moderate fused retrieval context.",
            )

        return GroundingScore(
            hallucination_score=0.66,
            hallucination_reason="Retrieved context has weak fused retrieval support for the question.",
        )


def _is_small_talk(message: str) -> bool:
    return message in {"hi", "hello", "hey", "thanks", "thank you"}


def _is_uncertain_answer(answer: str) -> bool:
    uncertainty_phrases = {
        "i do not have",
        "i don't have",
        "not enough",
        "cannot answer",
        "can't answer",
        "no matching documents",
    }
    return any(phrase in answer for phrase in uncertainty_phrases)


def _document_rrf_score(document: RetrievedDocument) -> float:
    if document.rrf_score is not None:
        return document.rrf_score
    return document.score
