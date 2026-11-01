from __future__ import annotations

import re
from statistics import mean
from typing import Literal

from pydantic import BaseModel, Field

from pra_core.documents.filenames import normalize_filename
from pra_core.evaluation.rag_retrieval import RagEvaluationDataset, RagEvaluationItem


AnswerQualityMetricName = Literal[
    "answer_correctness",
    "citation_correctness",
    "groundedness",
    "multi_hop_reasoning",
    "refusal_insufficient_evidence",
]

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "by",
    "can",
    "does",
    "for",
    "from",
    "has",
    "if",
    "in",
    "into",
    "is",
    "it",
    "may",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "with",
}
_NEGATION_TERMS = {"cannot", "couldn't", "don't", "doesn't", "no", "not", "only", "unless", "without"}
_REFUSAL_PATTERNS = (
    "i do not know",
    "i don't know",
    "i do not have enough information",
    "i don't have enough information",
    "insufficient information",
    "insufficient evidence",
    "not enough information",
    "cannot answer",
    "can't answer",
    "unable to answer",
    "the document does not contain",
    "the document does not mention",
    "the context does not mention",
    "no relevant information",
    "no matching information",
)
_MULTI_HOP_INTENTS = {
    "authority_vs_eligibility",
    "comparison_reasoning",
    "exception_filtering",
    "exception_precedence",
    "global_scan",
    "multi_hop",
    "multi_hop_exception",
}


class AnswerQualityEvidence(BaseModel):
    source: str = ""
    text: str = ""


class AnswerQualityAnswer(BaseModel):
    """Captured answer data; this evaluator never calls an LLM."""

    id: str
    answer: str = ""
    citations: list[str] = Field(default_factory=list)
    citation_sources: list[str] = Field(default_factory=list)
    evidence: list[AnswerQualityEvidence] = Field(default_factory=list)
    retrieval_details: list[dict[str, object]] = Field(default_factory=list)


class AnswerQualityQuestionResult(BaseModel):
    id: str
    question: str
    expected_behavior: Literal["answer", "insufficient_evidence"]
    answer_present: bool
    metrics: dict[AnswerQualityMetricName, float | None]
    overall_score: float
    passed: bool
    matched_answer_terms: list[str] = Field(default_factory=list)
    missing_answer_terms: list[str] = Field(default_factory=list)
    aligned_sources: list[str] = Field(default_factory=list)
    unsupported_sources: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class AnswerQualitySummary(BaseModel):
    dataset: str
    total_questions: int
    answered_questions: int
    missing_answer_count: int
    metrics: dict[str, float | None]
    results: list[AnswerQualityQuestionResult]


def evaluate_answer_quality(
    *,
    dataset: RagEvaluationDataset,
    answers: list[AnswerQualityAnswer],
) -> AnswerQualitySummary:
    """Score captured answers with deterministic answer/evidence comparisons."""

    answers_by_id = {answer.id: answer for answer in answers}
    results = [
        _evaluate_item(item, answers_by_id.get(item.id))
        for item in dataset.items
    ]
    metric_names = [
        "answer_correctness",
        "citation_correctness",
        "groundedness",
        "multi_hop_reasoning",
        "refusal_insufficient_evidence",
    ]
    aggregate: dict[str, float | None] = {}
    for metric_name in metric_names:
        values = [
            result.metrics[metric_name]
            for result in results
            if result.metrics[metric_name] is not None
        ]
        aggregate[metric_name] = round(mean(values), 4) if values else None
    overall_values = [result.overall_score for result in results]
    aggregate["overall"] = round(mean(overall_values), 4) if overall_values else None
    return AnswerQualitySummary(
        dataset=dataset.name,
        total_questions=len(results),
        answered_questions=sum(result.answer_present for result in results),
        missing_answer_count=sum(not result.answer_present for result in results),
        metrics=aggregate,
        results=results,
    )


def _evaluate_item(
    item: RagEvaluationItem,
    captured: AnswerQualityAnswer | None,
) -> AnswerQualityQuestionResult:
    answer = captured or AnswerQualityAnswer(id=item.id)
    answer_terms = _content_terms(answer.answer)
    expected_terms = _expected_terms(item)
    matched_terms = sorted(expected_terms & answer_terms)
    missing_terms = sorted(expected_terms - answer_terms)
    refusal = _is_refusal(answer.answer)
    expected_refusal = item.expected_behavior == "insufficient_evidence"
    answer_present = bool(answer.answer.strip())
    notes: list[str] = []

    if expected_refusal:
        correctness = 1.0 if refusal else 0.0
        refusal_score = correctness
        if refusal:
            notes.append("The answer correctly used insufficient-evidence language.")
        else:
            notes.append("The answer should have refused or stated that evidence was insufficient.")
    else:
        correctness = _coverage(expected_terms, answer_terms)
        refusal_score = 0.0 if refusal else 1.0
        if refusal:
            notes.append("The answer refused despite an answerable reference case.")
        elif not expected_terms:
            notes.append("No expected answer terms were configured.")

    citation_score, aligned_sources, unsupported_sources = _citation_alignment(item, answer)
    groundedness = _groundedness(answer, answer_terms)
    multi_hop = _multi_hop_score(item, answer_terms)
    metrics: dict[AnswerQualityMetricName, float | None] = {
        "answer_correctness": correctness,
        "citation_correctness": citation_score,
        "groundedness": groundedness,
        "multi_hop_reasoning": multi_hop,
        "refusal_insufficient_evidence": refusal_score,
    }
    applicable_scores = [score for score in metrics.values() if score is not None]
    overall_score = round(mean(applicable_scores), 4) if applicable_scores else 0.0
    passed = all(score is None or score >= 0.7 for score in metrics.values())
    if not answer_present:
        passed = False
        notes.append("No captured answer was supplied for this question.")
    return AnswerQualityQuestionResult(
        id=item.id,
        question=item.question,
        expected_behavior=item.expected_behavior,
        answer_present=answer_present,
        metrics=metrics,
        overall_score=overall_score,
        passed=passed,
        matched_answer_terms=matched_terms,
        missing_answer_terms=missing_terms,
        aligned_sources=aligned_sources,
        unsupported_sources=unsupported_sources,
        notes=notes,
    )


def _expected_terms(item: RagEvaluationItem) -> set[str]:
    configured = item.expected_answer_terms or [item.expected_answer_summary]
    return {
        term
        for value in configured
        for term in _content_terms(value)
    }


def _content_terms(value: str) -> set[str]:
    terms = set(re.findall(r"[a-z0-9]+(?:[-/][a-z0-9]+)*", value.lower()))
    return {
        term
        for term in terms
        if term not in _STOPWORDS or term in _NEGATION_TERMS
    }


def _coverage(expected: set[str], actual: set[str]) -> float:
    if not expected:
        return 1.0
    return round(len(expected & actual) / len(expected), 4)


def _citation_alignment(
    item: RagEvaluationItem,
    answer: AnswerQualityAnswer,
) -> tuple[float | None, list[str], list[str]]:
    expected = {_source_name(source) for source in item.source_files}
    if not expected:
        return None, [], []
    provided = {
        _source_name(source)
        for source in _citation_source_values(answer)
        if _source_name(source)
    }
    if not provided:
        return 0.0, [], []
    aligned = sorted(expected & provided)
    unsupported = sorted(provided - expected)
    precision = len(aligned) / len(provided)
    recall = len(aligned) / len(expected)
    score = 0.0 if precision + recall == 0 else round(2 * precision * recall / (precision + recall), 4)
    return score, aligned, unsupported


def _citation_source_values(answer: AnswerQualityAnswer) -> list[str]:
    if answer.citation_sources:
        return [*answer.citations, *answer.citation_sources]
    details_by_document_id = {
        str(detail.get("document_id")): str(detail.get("filename", ""))
        for detail in answer.retrieval_details
        if detail.get("document_id") and detail.get("filename")
    }
    return [
        details_by_document_id.get(citation, citation)
        for citation in answer.citations
    ]


def _source_name(value: str) -> str:
    source = value.strip()
    if not source or ".md" not in source.lower():
        return ""
    source = source.split("#", 1)[0]
    return normalize_filename(source).lower()


def _groundedness(answer: AnswerQualityAnswer, answer_terms: set[str]) -> float | None:
    evidence_text = " ".join(item.text for item in answer.evidence if item.text.strip())
    if not evidence_text:
        evidence_text = " ".join(
            str(detail.get("text", ""))
            for detail in answer.retrieval_details
            if str(detail.get("text", "")).strip()
        )
    if not evidence_text:
        return None
    return _coverage(answer_terms, _content_terms(evidence_text))


def _multi_hop_score(
    item: RagEvaluationItem,
    answer_terms: set[str],
) -> float | None:
    is_multi_hop = item.intent in _MULTI_HOP_INTENTS or len(item.expected_sections) > 2
    if not is_multi_hop:
        return None
    if not item.expected_answer_summary:
        return _coverage(_expected_terms(item), answer_terms)
    clauses = [clause for clause in re.split(r"[.!?]+", item.expected_answer_summary) if clause.strip()]
    if not clauses:
        return _coverage(_expected_terms(item), answer_terms)
    clause_scores = [_coverage(_content_terms(clause), answer_terms) for clause in clauses]
    return round(mean(clause_scores), 4)


def _is_refusal(answer: str) -> bool:
    normalized = " ".join(answer.lower().split())
    return not normalized or any(pattern in normalized for pattern in _REFUSAL_PATTERNS)
