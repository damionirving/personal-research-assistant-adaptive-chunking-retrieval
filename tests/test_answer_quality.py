from pra_core.evaluation.answer_quality import (
    AnswerQualityAnswer,
    AnswerQualityEvidence,
    evaluate_answer_quality,
)
from pra_core.evaluation.rag_retrieval import RagEvaluationDataset


def test_answer_quality_schema_scores_correctness_citations_grounding_and_multihop() -> None:
    dataset = RagEvaluationDataset.model_validate(
        {
            "name": "answer-quality",
            "items": [
                {
                    "id": "q1",
                    "question": "Which incidents require approval, and why?",
                    "intent": "global_scan",
                    "source_files": ["scenario-northstar-meridian.md"],
                    "expected_sections": ["Approval", "Incident Records", "Leads"],
                    "expected_answer_summary": (
                        "NM-1043 requires approval because Tomas Greer is a Lantern Pass user. "
                        "NM-1046 requires approval if both credits are requested."
                    ),
                }
            ],
        }
    )
    report = evaluate_answer_quality(
        dataset=dataset,
        answers=[
            AnswerQualityAnswer(
                id="q1",
                answer=(
                    "NM-1043 requires approval because Tomas Greer is a Lantern Pass user. "
                    "NM-1046 requires approval if both credits are requested."
                ),
                citations=["scenario-northstar-meridian.md#chunk-4"],
                evidence=[
                    AnswerQualityEvidence(
                        source="scenario-northstar-meridian.md#chunk-4",
                        text=(
                            "NM-1043 requires approval because Tomas Greer is a Lantern Pass user. "
                            "NM-1046 requires approval if both credits are requested."
                        ),
                    )
                ],
            )
        ],
    )

    result = report.results[0]
    assert set(result.metrics) == {
        "answer_correctness",
        "citation_correctness",
        "groundedness",
        "multi_hop_reasoning",
        "refusal_insufficient_evidence",
    }
    assert result.metrics["answer_correctness"] == 1.0
    assert result.metrics["citation_correctness"] == 1.0
    assert result.metrics["groundedness"] == 1.0
    assert result.metrics["multi_hop_reasoning"] == 1.0
    assert result.passed is True
    assert report.metrics["overall"] == 1.0


def test_answer_quality_reports_citation_source_misalignment() -> None:
    dataset = RagEvaluationDataset.model_validate(
        {
            "name": "citation-alignment",
            "items": [
                {
                    "id": "q1",
                    "question": "What is BF-C?",
                    "source_files": ["scenario-northstar-meridian.md"],
                    "expected_answer_summary": "BF-C means seal breach confirmed.",
                }
            ],
        }
    )
    report = evaluate_answer_quality(
        dataset=dataset,
        answers=[
            AnswerQualityAnswer(
                id="q1",
                answer="BF-C means seal breach confirmed.",
                citations=["unrelated-policy.md#chunk-2"],
            )
        ],
    )

    result = report.results[0]
    assert result.metrics["answer_correctness"] == 1.0
    assert result.metrics["citation_correctness"] == 0.0
    assert result.aligned_sources == []
    assert result.unsupported_sources == ["unrelated-policy.md"]
    assert result.passed is False


def test_answer_quality_catches_missing_credit_amounts_for_nm_1046() -> None:
    dataset = RagEvaluationDataset.model_validate(
        {
            "name": "q10-completeness",
            "items": [
                {
                    "id": "uploaded-rag-015",
                    "question": "Incident NM-1046 had a 48-minute temperature excursion and a packaging seal breach confirmed. Can both credits be combined automatically?",
                    "intent": "combination_rule",
                    "source_files": ["scenario-northstar-meridian.md"],
                    "expected_answer_summary": (
                        "No. The 48-minute temperature excursion qualifies for the extended $275 credit, "
                        "and packaging seal breach may qualify for $90 if timely, but credits cannot be combined "
                        "for the same shipment unless Copper Door approval is recorded."
                    ),
                }
            ],
        }
    )
    incomplete = evaluate_answer_quality(
        dataset=dataset,
        answers=[
            AnswerQualityAnswer(
                id="uploaded-rag-015",
                answer=(
                    "No, credits cannot be combined automatically. Copper Door approval is required "
                    "because more than one credit category is requested."
                ),
                citations=["scenario-northstar-meridian.md#chunk-9"],
            )
        ],
    ).results[0]
    complete = evaluate_answer_quality(
        dataset=dataset,
        answers=[
            AnswerQualityAnswer(
                id="uploaded-rag-015",
                answer=(
                    "No. The 48-minute temperature excursion qualifies for the extended $275 credit, "
                    "and packaging seal breach may qualify for $90 if timely, but credits cannot be combined "
                    "for the same shipment unless Copper Door approval is recorded."
                ),
                citations=["scenario-northstar-meridian.md#chunk-9"],
            )
        ],
    ).results[0]

    assert "275" in incomplete.missing_answer_terms
    assert "90" in incomplete.missing_answer_terms
    assert incomplete.metrics["answer_correctness"] < 1.0
    assert complete.metrics["answer_correctness"] == 1.0


def test_answer_quality_maps_api_document_citations_through_retrieval_details() -> None:
    dataset = RagEvaluationDataset.model_validate(
        {
            "name": "api-citation-alignment",
            "items": [
                {
                    "id": "q1",
                    "question": "What is BF-C?",
                    "source_files": ["scenario-northstar-meridian.md"],
                    "expected_answer_summary": "BF-C means seal breach confirmed.",
                }
            ],
        }
    )
    result = evaluate_answer_quality(
        dataset=dataset,
        answers=[
            AnswerQualityAnswer(
                id="q1",
                answer="BF-C means seal breach confirmed.",
                citations=["document-uuid"],
                retrieval_details=[
                    {
                        "document_id": "document-uuid",
                        "filename": "scenario-northstar-meridian.md",
                    }
                ],
            )
        ],
    ).results[0]

    assert result.metrics["citation_correctness"] == 1.0
    assert result.aligned_sources == ["scenario-northstar-meridian.md"]


def test_answer_quality_scores_expected_insufficient_evidence_behavior() -> None:
    dataset = RagEvaluationDataset.model_validate(
        {
            "name": "refusal",
            "items": [
                {
                    "id": "q1",
                    "question": "Who is the lead for an unknown region?",
                    "expected_behavior": "insufficient_evidence",
                }
            ],
        }
    )

    refusal = evaluate_answer_quality(
        dataset=dataset,
        answers=[
            AnswerQualityAnswer(
                id="q1",
                answer="The document does not contain information about that region.",
            )
        ],
    ).results[0]
    unsupported_answer = evaluate_answer_quality(
        dataset=dataset,
        answers=[AnswerQualityAnswer(id="q1", answer="The lead is Rafi Ornelas.")],
    ).results[0]

    assert refusal.metrics["refusal_insufficient_evidence"] == 1.0
    assert refusal.metrics["answer_correctness"] == 1.0
    assert refusal.passed is True
    assert unsupported_answer.metrics["refusal_insufficient_evidence"] == 0.0
    assert unsupported_answer.passed is False
