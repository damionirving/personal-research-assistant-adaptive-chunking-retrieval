from unittest.mock import patch

from pra_core.agent.workflow import (
    _minimal_locked_fact_repair,
    apply_answer_shape_guardrail,
    answer_shape_diagnostics,
    build_answer_instruction,
    classify_answer_requirements,
    validate_and_repair_final_answer,
)
from pra_core.documents.models import RetrievedDocument


def _document(text: str) -> RetrievedDocument:
    return RetrievedDocument(
        document_id="doc-1",
        filename="policy.md",
        text=text,
        score=1.0,
        chunk_id="policy.md#chunk-0",
    )


def test_answer_shape_classifies_generic_multi_part_amount_and_approval_question() -> None:
    requirements = classify_answer_requirements(
        "What credit applies to case ZX-2043, and is manager approval required?"
    )

    assert requirements["question_type"] == "multi_part"
    assert requirements["multi_part"] is True
    assert "ZX-2043" in requirements["important_subject_terms"]
    assert "approval" in requirements["detected_rule_families"]
    assert "each_question_part" in requirements["required_slots"]
    assert "Answer each part explicitly" in build_answer_instruction(requirements)


def test_subject_extraction_removes_accidental_leading_question_words() -> None:
    requirements = classify_answer_requirements("Are Blue Finch approval rules required?")

    assert "Blue Finch" in requirements["extracted_entities"]
    assert "Are Blue Finch" not in requirements["extracted_entities"]
    assert requirements["entity_cleanup_applied"] is True


def test_answer_shape_diagnostics_report_missing_amount_and_subject_terms() -> None:
    requirements = classify_answer_requirements(
        "How much does the remote work stipend provide?"
    )

    diagnostics = answer_shape_diagnostics("It provides support.", requirements)

    assert "amount" in diagnostics["missing_slots"]
    assert diagnostics["subject_terms_missing_from_answer"]


def test_definition_shape_preserves_entity_and_function_guidance() -> None:
    requirements = classify_answer_requirements("What is Orchid Ledger?")

    assert requirements["question_type"] == "definition"
    assert "Orchid Ledger" in requirements["important_subject_terms"]
    instruction = build_answer_instruction(requirements)
    assert "type, role, and use" in instruction
    assert "Orchid Ledger" in instruction


def test_question_classification_prefers_yes_no_and_duration_over_amount_keywords() -> None:
    reimbursement = classify_answer_requirements(
        "Can an employee use the gym reimbursement to buy running shoes?"
    )
    duration = classify_answer_requirements(
        "How many days does a new hire have to use the remote work stipend?"
    )

    assert reimbursement["question_type"] == "yes_no_policy"
    assert "gym reimbursement" in reimbursement["important_subject_terms"]
    assert duration["question_type"] == "date_or_deadline"
    assert "remote work stipend" in duration["important_subject_terms"]


def test_person_lookup_and_wrong_binary_opening_are_reported() -> None:
    requirements = classify_answer_requirements("Who is responsible for the Orchid Ledger?")
    diagnostics = answer_shape_diagnostics("Yes. Morgan Lee owns it.", requirements)

    assert requirements["question_type"] == "person_lookup"
    assert diagnostics["answer_opening_type"] == "yes_no"
    assert diagnostics["unsupported_answer_shape_warning"] == (
        "binary opening does not match the requested answer shape"
    )


def test_guardrail_composes_bare_amount_without_returning_source_sentence() -> None:
    question = "How much money does a part-time employee get for the home office stipend?"
    result = apply_answer_shape_guardrail(
        question=question,
        answer="$600.",
        requirements=classify_answer_requirements(question),
        documents=[_document("Part-time employees receive $600 for the home office stipend.")],
    )

    assert result["answer_after_guardrail"] == (
        "A part-time employee receives $600 for the home office stipend."
    )
    assert result["repair_strategy"] == "supported_slot_answer"
    assert result["answer_shape_guardrail_applied"] is True


def test_guardrail_never_returns_markdown_heading_or_raw_policy_fragment() -> None:
    question = "How much money does a part-time employee get for the home office stipend?"
    source = (
        "## Home Office Stipend\n"
        "### Eligibility\n"
        "Part-time employees receive $600 for the home office stipend."
    )
    result = apply_answer_shape_guardrail(
        question=question,
        answer="$600.",
        requirements=classify_answer_requirements(question),
        documents=[_document(source)],
    )

    final_answer = str(result["answer_after_guardrail"])
    assert final_answer == "A part-time employee receives $600 for the home office stipend."
    assert final_answer not in source
    assert "#" not in final_answer
    assert "Eligibility" not in final_answer


def test_guardrail_replaces_bare_duration_and_preserves_condition() -> None:
    duration_question = "How many days does a new hire have to spend their remote work stipend?"
    duration = apply_answer_shape_guardrail(
        question=duration_question,
        answer="90 days.",
        requirements=classify_answer_requirements(duration_question),
        documents=[_document("A new hire has 90 days to spend the remote work stipend.")],
    )
    condition_question = "If an employee has worked for 5 years, how many weeks of paid sabbatical do they get?"
    condition = apply_answer_shape_guardrail(
        question=condition_question,
        answer="They receive 4 weeks of paid sabbatical.",
        requirements=classify_answer_requirements(condition_question),
        documents=[_document("After 5 years, the employee receives 4 weeks of paid sabbatical.")],
    )

    assert duration["answer_after_guardrail"] == (
        "A new hire has 90 days to spend the remote work stipend."
    )
    assert condition["answer_after_guardrail"] == (
        "The employee receives 4 weeks of paid sabbatical after 5 years."
    )


def test_guardrail_expands_eligibility_answer_from_supported_item_and_policy_evidence() -> None:
    question = "Can an employee use the gym reimbursement to buy running shoes?"
    result = apply_answer_shape_guardrail(
        question=question,
        answer="No, the reimbursement does not cover athletic apparel.",
        requirements=classify_answer_requirements(question),
        documents=[_document("Running shoes are athletic apparel. The gym reimbursement does not cover athletic apparel.")],
    )

    assert result["answer_after_guardrail"] == (
        "No. Running shoes are athletic apparel, and athletic apparel is not covered by the gym reimbursement."
    )


def test_guardrail_naturalizes_multi_part_answer_and_reports_diagnostics() -> None:
    question = "What credit applies, and is approval required?"
    result = apply_answer_shape_guardrail(
        question=question,
        answer="What credit applies? $210. Is approval required? Yes, Copper Door approval is required because the threshold applies.",
        requirements=classify_answer_requirements(question),
        documents=[_document("A $210 credit applies. Copper Door approval is required because the threshold applies.")],
    )

    assert result["answer_after_guardrail"].startswith(
        "A $210 credit applies, and Copper Door approval is required"
    )
    assert result["answer_shape_guardrail_applied"] is True
    assert result["answer_before_guardrail"].startswith("What credit applies?")
    assert result["high_quality_subject_terms"]
    assert "repair_strategy" in result


def test_subject_quality_filter_rejects_question_fragments() -> None:
    question = "If an employee has worked for 5 years, how many weeks of paid sabbatical do they get?"
    requirements = classify_answer_requirements(question)
    result = apply_answer_shape_guardrail(
        question=question,
        answer="4 weeks.",
        requirements=requirements,
        documents=[_document("After 5 years, the employee receives 4 weeks of paid sabbatical.")],
    )

    terms = result["high_quality_subject_terms"] + result["ignored_noisy_subject_terms"]
    assert "If" not in terms
    assert "do they" not in terms
    assert all("paid sabbatical do they" not in term for term in terms)


def test_guardrail_preserves_supported_condition_in_duration_answer() -> None:
    question = (
        "If an employee has worked at AcmeCorp for 5 years, "
        "how many weeks of paid sabbatical do they get?"
    )
    result = apply_answer_shape_guardrail(
        question=question,
        answer="They receive 4 weeks of paid sabbatical.",
        requirements=classify_answer_requirements(question),
        documents=[
            _document(
                "After 5 years at AcmeCorp, the employee receives 4 weeks of paid sabbatical."
            )
        ],
    )

    assert result["answer_after_guardrail"] == (
        "The employee receives 4 weeks of paid sabbatical after 5 years at AcmeCorp."
    )
    assert result["condition_guardrail_applied"] is True
    assert result["conditions_missing_from_answer"] == []
    assert "5 years" in " ".join(result["conditions_present_in_answer"])
    assert result["supported_question_conditions"] == result["supported_conditions"]
    assert result["answer_before_condition_guardrail"] == "They receive 4 weeks of paid sabbatical."
    assert result["condition_repair_failed_reason"] is None


def test_definition_meaning_answer_is_not_over_expanded() -> None:
    question = "If someone says an incident needs Copper Door, what does that mean?"
    result = apply_answer_shape_guardrail(
        question=question,
        answer=(
            "Copper Door is the manual manager approval path. "
            "It is required for multiple credit categories, late credit requests, "
            "high credit amounts, and Lantern Pass users."
        ),
        requirements=classify_answer_requirements(question),
        documents=[
            _document(
                "Copper Door is the manual manager approval path for incidents that require manager approval."
            )
        ],
    )

    assert result["answer_after_guardrail"] == (
        "Copper Door is the manual manager approval path."
    )
    assert result["repair_strategy"] == "definition_scope_trim"


def test_diagnostic_headings_are_removed_from_final_answer() -> None:
    question = "Incident ZX-9 involved condition A. What credit applies, and is approval required?"
    result = apply_answer_shape_guardrail(
        question=question,
        answer=(
            "Question Part 1: What credit applies? $210. "
            "Question Part 2: Is approval required? Yes, manager approval is required."
        ),
        requirements=classify_answer_requirements(question),
        documents=[_document("A $210 credit applies, and manager approval is required.")],
    )

    assert "Question Part" not in result["answer_after_guardrail"]
    assert "What credit applies?" not in result["answer_after_guardrail"]
    assert result["diagnostic_format_detected"] is True
    assert result["formatting_guardrail_applied"] is True
    assert result["diagnostic_format_guardrail_applied"] is True
    assert result["diagnostic_format_repair_failed_reason"] is None


def test_guardrail_naturalizes_credit_reason_and_secondary_answer_headers() -> None:
    question = "What credit should apply, and why?"
    result = apply_answer_shape_guardrail(
        question=question,
        answer=(
            "What credit should apply? $140. Why? A 22-minute temperature excursion "
            "qualifies for a standard credit of $140. Secondary answer? BF-B does not "
            "trigger a packaging seal failure because the seal was intact."
        ),
        requirements=classify_answer_requirements(question),
        documents=[
            _document(
                "A 22-minute temperature excursion qualifies for a standard credit of $140. "
                "BF-B does not trigger a packaging seal failure because the seal was intact."
            )
        ],
    )

    final_answer = str(result["answer_after_guardrail"])
    assert "$140" in final_answer
    assert "22-minute temperature excursion" in final_answer
    assert "BF-B" in final_answer
    assert "seal was intact" in final_answer
    assert "What credit should apply?" not in final_answer
    assert "Secondary answer?" not in final_answer
    assert result["mini_question_format_detected"] is True
def test_final_answer_validation_mandatorily_repairs_a_bare_duration() -> None:
    question = "How many days does a new hire have to spend their remote work stipend?"
    result = validate_and_repair_final_answer(
        question=question,
        answer="90 days.",
        requirements=classify_answer_requirements(question),
        documents=[_document("A new hire has 90 days to spend the remote work stipend.")],
    )

    assert result["final_answer_validation_applied"] is True
    assert result["final_answer_validation_failed"] is True
    assert "bare_value_answer" in result["final_answer_validation_failure_reasons"]
    assert result["answer_after_final_validation"] != "90 days."
    assert "90 days" in result["answer_after_final_validation"]
    assert result["answer_after_final_validation"] == (
        "A new hire has 90 days to spend the remote work stipend."
    )


def test_final_answer_validation_preserves_yes_no_subjects_and_conditions() -> None:
    policy_question = "Can an employee use the gym reimbursement to buy running shoes?"
    policy = validate_and_repair_final_answer(
        question=policy_question,
        answer=(
            "No, athletic apparel is not covered. The policy covers gym memberships, "
            "yoga studios, and martial arts classes."
        ),
        requirements=classify_answer_requirements(policy_question),
        documents=[
            _document(
                "Running shoes are athletic apparel. The gym reimbursement does not cover "
                "athletic apparel; it covers gym memberships, yoga studios, and martial arts classes."
            )
        ],
    )
    condition_question = (
        "If an employee has worked at AcmeCorp for 5 years, how many weeks of paid "
        "sabbatical do they get?"
    )
    condition = validate_and_repair_final_answer(
        question=condition_question,
        answer="They get 4 weeks of paid sabbatical.",
        requirements=classify_answer_requirements(condition_question),
        documents=[
            _document(
                "After 5 years at AcmeCorp, the employee gets 4 weeks of paid sabbatical."
            )
        ],
    )

    assert str(policy["answer_after_final_validation"]).startswith("No")
    assert str(policy["answer_after_final_validation"]) == (
        "No. Running shoes are athletic apparel, and athletic apparel is not covered by the gym reimbursement."
    )
    assert str(condition["answer_after_final_validation"]) == (
        "The employee receives 4 weeks of paid sabbatical after 5 years at AcmeCorp."
    )
    assert condition["condition_repair_applied"] is True


def test_final_answer_validation_removes_mini_question_headers() -> None:
    question = "What credit applies, and is approval required?"
    result = validate_and_repair_final_answer(
        question=question,
        answer=(
            "What credit applies? $210. Is approval required? Yes, Copper Door approval "
            "is required because Tomas Greer is a Lantern Pass user."
        ),
        requirements=classify_answer_requirements(question),
        documents=[
            _document(
                "A $210 credit applies, and Copper Door approval is required because Tomas "
                "Greer is a Lantern Pass user."
            )
        ],
    )

    final_answer = str(result["answer_after_final_validation"])
    assert "What credit applies?" not in final_answer
    assert "Is approval required?" not in final_answer
    assert "$210" in final_answer
    assert "Copper Door" in final_answer
    assert "Tomas Greer" in final_answer
    assert "Lantern Pass user" in final_answer
    assert result["mini_question_format_repair_applied"] is True


def test_final_answer_validation_trims_unasked_approval_from_credit_answer() -> None:
    question = (
        "Incident NM-1042 had a 22-minute temperature excursion and BF-B. "
        "What credit should apply, and why?"
    )
    result = validate_and_repair_final_answer(
        question=question,
        answer=(
            "A $140 standard temperature-excursion credit applies because the "
            "22-minute excursion falls in the 18-to-44-minute range. BF-B does not "
            "add a packaging-seal credit because the seal was intact. Copper Door "
            "approval is required because the customer is a Lantern Pass user."
        ),
        requirements=classify_answer_requirements(question),
        documents=[
            _document(
                "A 22-minute temperature excursion qualifies for the standard "
                "temperature-excursion credit of $140. The standard "
                "temperature-excursion credit applies between 18 and 44 minutes."
            ),
            _document(
                "BF-B does not add a packaging-seal credit because the seal was intact."
            ),
            _document(
                "Copper Door approval is required because the customer is a Lantern "
                "Pass user."
            ),
        ],
    )

    final_answer = str(result["answer_after_final_validation"])
    assert "Copper Door approval" not in final_answer
    assert "$140" in final_answer
    assert "BF-B" in final_answer
    assert result["extra_unasked_parts_detected"] is True
    assert result["scope_fit_passed"] is True


def test_answer_quality_gate_leaves_good_amount_and_deadline_answers_unchanged() -> None:
    cases = [
        (
            "How much does a part-time employee get for the home office stipend?",
            "Part-time employees receive $600 for the home office stipend.",
        ),
        (
            "How many days does a new hire have to spend the remote work stipend?",
            "A new hire has 90 days to spend the remote work stipend.",
        ),
    ]

    for question, answer in cases:
        result = validate_and_repair_final_answer(
            question=question,
            answer=answer,
            requirements=classify_answer_requirements(question),
            documents=[_document(answer)],
        )

        assert result["answer_after_final_validation"] == answer
        assert result["final_selected_answer_source"] == "original"


def test_final_validation_does_not_reject_concise_rule_answer_for_incidental_context() -> None:
    question = (
        "Incident ZX-2043 involved a 12-minute delay and was created by Morgan Lee. "
        "What credit applies, and is approval required?"
    )
    result = validate_and_repair_final_answer(
        question=question,
        answer=(
            "What credit applies? $210. Is approval required? Yes, Copper Door approval "
            "is required because Morgan Lee is a Lantern Pass user."
        ),
        requirements=classify_answer_requirements(question),
        documents=[
            _document(
                "Incident ZX-2043 has a 12-minute delay. A $210 credit applies. "
                "Copper Door approval is required because Morgan Lee is a Lantern Pass user."
            )
        ],
    )

    final_answer = str(result["answer_after_final_validation"])
    assert "$210" in final_answer
    assert "Copper Door approval" in final_answer
    assert "Morgan Lee" in final_answer
    assert result["final_selected_answer_source"] == "repaired"
    assert result["repair_rejected"] is False
    assert "ZX-2043" not in result["required_subject_terms"]
    assert "12-minute delay" not in result["required_question_conditions"]


def test_minimal_safe_repair_composes_a_sentence_without_copying_evidence() -> None:
    question = "How much money does a part-time employee get for the home office stipend?"
    source = (
        "Part-time employees receive $600 for the home office stipend after completing "
        "the enrollment form."
    )
    result = validate_and_repair_final_answer(
        question=question,
        answer="$600.",
        requirements=classify_answer_requirements(question),
        documents=[_document(source)],
    )

    final_answer = str(result["answer_after_final_validation"])
    assert final_answer == "A part-time employee receives $600 for the home office stipend."
    assert final_answer not in source
    assert result["locked_fact_preservation_passed"] is True


def test_minimal_safe_repair_uses_supported_slots_instead_of_a_bare_value() -> None:
    question = "How much money does a part-time employee get for the home office stipend?"
    source = "Part-time employees receive $600 for the home office stipend."

    repaired = _minimal_locked_fact_repair(
        question=question,
        answer="$600.",
        requirements=classify_answer_requirements(question),
        locked_facts=["$600"],
        documents=[_document(source)],
    )

    assert repaired == "A part-time employee receives $600 for the home office stipend."
    assert repaired != "$600."


def test_minimal_safe_repair_preserves_actor_and_action_for_a_duration() -> None:
    question = "How many days does a new hire have to spend their remote work stipend?"
    repaired = _minimal_locked_fact_repair(
        question=question,
        answer="90 days.",
        requirements=classify_answer_requirements(question),
        locked_facts=["90 days"],
        documents=[_document("New hires can spend the remote work stipend during a 90-day period.")],
    )

    assert repaired == "A new hire has 90 days to spend the remote work stipend."


def test_final_validation_selects_minimal_synthesis_after_main_repair_rejection() -> None:
    question = "How much money does a part-time employee get for the home office stipend?"
    requirements = classify_answer_requirements(question)
    rejected_guardrail = {
        "answer_after_guardrail": "The requested policy is $600.",
        "high_quality_subject_terms": ["home office stipend"],
        "detected_question_conditions": [],
        "supported_question_conditions": [],
        "required_question_conditions": [],
        "mini_question_format_detected": False,
        "bare_duration_detected": False,
        "repair_strategy": "invalid_main_candidate",
        "repair_failed_reason": None,
        "bare_duration_guardrail_applied": False,
        "answer_before_duration_guardrail": None,
        "answer_after_duration_guardrail": None,
        "raw_subject_terms": ["home office stipend"],
        "ignored_noisy_subject_terms": [],
        "condition_repair_failed_reason": None,
        "diagnostic_format_guardrail_applied": False,
        "answer_before_format_guardrail": None,
        "answer_after_format_guardrail": None,
        "diagnostic_format_repair_failed_reason": None,
    }

    with patch(
        "pra_core.agent.workflow.apply_answer_shape_guardrail",
        return_value=rejected_guardrail,
    ):
        result = validate_and_repair_final_answer(
            question=question,
            answer="$600.",
            requirements=requirements,
            documents=[_document("Part-time employees receive $600 for the home office stipend.")],
        )

    assert result["answer_after_final_validation"] == (
        "A part-time employee receives $600 for the home office stipend."
    )
    assert result["final_selected_answer_source"] == "minimal_safe_repair"
    assert result["minimal_safe_synthesis_applied"] is True
    assert result["minimal_safe_synthesis_passed"] is True


def test_final_validation_preserves_locked_duration_when_repair_evidence_has_other_values() -> None:
    question = "How many days does a new hire have to spend the remote work stipend?"
    result = validate_and_repair_final_answer(
        question=question,
        answer="90 days.",
        requirements=classify_answer_requirements(question),
        documents=[
            _document(
                "A new hire has 90 days to spend the remote work stipend. "
                "A separate equipment return must be reported within 3 business days."
            )
        ],
    )

    final_answer = str(result["answer_after_final_validation"])
    assert "90 days" in final_answer
    assert "3 business days" not in final_answer
    assert result["locked_facts_before_repair"] == ["90 days"]
    assert result["locked_facts_after_repair"] == ["90 days"]
    assert result["locked_fact_preservation_passed"] is True


def test_final_validation_restores_supported_subject_without_vague_placeholder() -> None:
    question = "Can an employee use the gym reimbursement to buy running shoes?"
    result = validate_and_repair_final_answer(
        question=question,
        answer="No, athletic apparel is not covered.",
        requirements=classify_answer_requirements(question),
        documents=[
            _document(
                "Running shoes are athletic apparel. The gym reimbursement does not cover athletic apparel."
            )
        ],
    )

    final_answer = str(result["answer_after_final_validation"])
    assert final_answer == (
        "No. Running shoes are athletic apparel, and athletic apparel is not covered by the gym reimbursement."
    )
    assert "requested policy" not in final_answer.lower()
    assert result["final_selected_answer_source"] == "repaired"
    assert result["answer_relevancy_gate_applied"] is True
    assert "required_subject_terms_missing" in result["answer_relevancy_gate_reason"]
    assert "running shoes" in result["required_subject_terms_inserted"]
    assert result["vague_placeholder_removed"] is False


def test_final_validation_normalizes_equivalent_service_condition() -> None:
    question = "If an employee has worked for five consecutive years, how many weeks of paid sabbatical do they get?"
    result = validate_and_repair_final_answer(
        question=question,
        answer="They get 4 weeks of paid sabbatical.",
        requirements=classify_answer_requirements(question),
        documents=[
            _document("After 5 years of service, an employee gets 4 weeks of paid sabbatical.")
        ],
    )

    assert "4 weeks" in str(result["answer_after_final_validation"])
    assert result["conditions_missing_before_repair"]
    assert result["condition_repair_applied"] is True


def test_final_validation_rejects_repair_that_drops_locked_entity() -> None:
    question = "What does Copper Door mean?"
    result = validate_and_repair_final_answer(
        question=question,
        answer="Copper Door is the manual manager approval path.",
        requirements=classify_answer_requirements(question),
        documents=[_document("Copper Door is the manual manager approval path.")],
    )

    assert "Copper Door" in str(result["answer_after_final_validation"])
    assert result["locked_fact_preservation_passed"] is True
    assert result["final_selected_answer_source"] == "original"


def test_final_validation_is_idempotent_for_repaired_answer() -> None:
    question = "How many days does a new hire have to spend the remote work stipend?"
    requirements = classify_answer_requirements(question)
    documents = [_document("A new hire has 90 days to spend the remote work stipend.")]
    first = validate_and_repair_final_answer(
        question=question,
        answer="90 days.",
        requirements=requirements,
        documents=documents,
    )
    second = validate_and_repair_final_answer(
        question=question,
        answer=str(first["answer_after_final_validation"]),
        requirements=requirements,
        documents=documents,
    )

    assert second["answer_after_final_validation"] == first["answer_after_final_validation"]
    assert second["final_selected_answer_source"] == "original"


def test_final_validation_does_not_change_an_already_good_answer() -> None:
    question = "Can an employee use the gym reimbursement to buy running shoes?"
    answer = "No. Running shoes are athletic apparel, and athletic apparel is not covered by the gym reimbursement."
    result = validate_and_repair_final_answer(
        question=question,
        answer=answer,
        requirements=classify_answer_requirements(question),
        documents=[
            _document(
                "Running shoes are athletic apparel, and athletic apparel is not covered by the gym reimbursement."
            )
        ],
    )

    assert result["answer_after_final_validation"] == answer
    assert result["final_selected_answer_source"] == "original"
    assert result["repair_rejected"] is False


def test_guardrail_trims_unrequested_policy_lists_after_a_complete_binary_answer() -> None:
    question = "Can an employee use the gym reimbursement to buy running shoes?"
    result = apply_answer_shape_guardrail(
        question=question,
        answer=(
            "No, running shoes are athletic apparel and are not covered by the gym reimbursement. "
            "The policy covers gym memberships, yoga studios, and martial arts classes."
        ),
        requirements=classify_answer_requirements(question),
        documents=[
            _document(
                "Running shoes are athletic apparel. The gym reimbursement does not cover athletic apparel; "
                "it covers gym memberships, yoga studios, and martial arts classes."
            )
        ],
    )

    assert result["answer_after_guardrail"] == (
        "No. Running shoes are athletic apparel and are not covered by the gym reimbursement."
    )
    assert result["repair_strategy"] == "policy_scope_trim"


def test_guardrail_naturalizes_all_mini_question_sections_without_dropping_facts() -> None:
    question = "What credit should apply, and why?"
    result = apply_answer_shape_guardrail(
        question=question,
        answer=(
            "What credit should apply? $140. Why? A 22-minute temperature excursion qualifies "
            "for a standard credit of $140. Secondary answer? BF-B does not impact the credit amount. "
            "Reason or condition? The temperature excursion is between 18 and 44 minutes."
        ),
        requirements=classify_answer_requirements(question),
        documents=[
            _document(
                "A 22-minute temperature excursion qualifies for a standard credit of $140. "
                "BF-B does not impact the credit amount."
            )
        ],
    )

    final_answer = str(result["answer_after_guardrail"])
    assert "$140" in final_answer
    assert "22-minute" in final_answer
    assert "BF-B" in final_answer
    assert not any(label in final_answer for label in ("What credit", "Why?", "Secondary answer", "Reason or condition"))


def test_guardrail_trims_plain_definition_to_its_first_supported_sentence() -> None:
    question = "What is Copper Door?"
    result = apply_answer_shape_guardrail(
        question=question,
        answer=(
            "Copper Door is the manual manager approval path. Copper Door approval is required "
            "when more than one credit category is requested, the credit exceeds $275, a request is late, "
            "or the incident was created by a Lantern Pass user."
        ),
        requirements=classify_answer_requirements(question),
        documents=[_document("Copper Door is the manual manager approval path.")],
    )

    assert result["answer_after_guardrail"] == "Copper Door is the manual manager approval path."
    assert result["repair_strategy"] == "definition_scope_trim"


def test_targeted_shape_repairs_are_idempotent() -> None:
    cases = [
        (
            "Can an employee use the gym reimbursement to buy running shoes?",
            "No, the gym reimbursement does not cover athletic apparel.",
            "Running shoes are athletic apparel. The gym reimbursement does not cover athletic apparel.",
        ),
        (
            "What credit should apply, and why?",
            (
                "What credit should apply? $140. Why? A 22-minute temperature excursion "
                "qualifies for a standard credit of $140. Secondary answer? BF-B does not impact the credit amount."
            ),
            (
                "A 22-minute temperature excursion qualifies for a standard credit of $140. "
                "BF-B does not impact the credit amount."
            ),
        ),
        (
            "What is Copper Door?",
            (
                "Copper Door is the manual manager approval path. Copper Door approval is required "
                "when more than one credit category is requested."
            ),
            "Copper Door is the manual manager approval path.",
        ),
    ]

    for question, answer, evidence in cases:
        requirements = classify_answer_requirements(question)
        documents = [_document(evidence)]
        first = validate_and_repair_final_answer(
            question=question,
            answer=answer,
            requirements=requirements,
            documents=documents,
        )
        second = validate_and_repair_final_answer(
            question=question,
            answer=str(first["answer_after_final_validation"]),
            requirements=requirements,
            documents=documents,
        )

        assert second["answer_after_final_validation"] == first["answer_after_final_validation"]
def test_q3_style_policy_repair_inserts_item_and_category_without_expansion() -> None:
    question = "Can an employee use the gym reimbursement to buy running shoes?"
    result = validate_and_repair_final_answer(
        question=question,
        answer="No, the gym reimbursement does not cover athletic apparel, according to the policy.",
        requirements=classify_answer_requirements(question),
        documents=[
            _document(
                "Running shoes are athletic apparel. The gym reimbursement does not cover athletic apparel."
            )
        ],
    )

    assert result["answer_after_final_validation"] == (
        "No. Running shoes are athletic apparel, and athletic apparel is not covered by the gym reimbursement."
    )
    assert result["answer_shape_repair_type"] == "subject_aware_policy_slot_answer"
    assert result["answer_shape_repair_passed"] is True


def test_q3_style_policy_repair_uses_conservative_item_fallback_without_category_mapping() -> None:
    question = "Can an employee use the travel allowance to buy an old laptop?"
    result = validate_and_repair_final_answer(
        question=question,
        answer="No, the travel allowance does not cover personal equipment.",
        requirements=classify_answer_requirements(question),
        documents=[_document("The travel allowance does not cover personal equipment.")],
    )

    assert result["answer_after_final_validation"] == "No. Old laptop is not covered by the travel allowance."
    assert "personal equipment" not in result["answer_after_final_validation"]


def test_q11_style_credit_repair_adds_category_threshold_and_secondary_effect() -> None:
    question = (
        "Incident NM-1042 had a 22-minute temperature excursion and BF-B. "
        "What credit should apply, and why?"
    )
    result = validate_and_repair_final_answer(
        question=question,
        answer=(
            "A $140 credit applies because A 22-minute temperature excursion qualifies for the credit. "
            "BF-B indicates outer carton damage, but the seal was intact, so it does not count as a packaging seal failure."
        ),
        requirements=classify_answer_requirements(question),
        documents=[
            _document(
                "A 22-minute temperature excursion qualifies for the standard temperature-excursion credit of $140. "
                "The standard temperature-excursion credit applies between 18 and 44 minutes."
            ),
            _document("BF-B indicates outer carton damage, but the seal was intact, so it does not count as a packaging seal failure."),
        ],
    )

    final_answer = str(result["answer_after_final_validation"])
    assert final_answer == (
        "A $140 standard temperature-excursion credit applies because the 22-minute excursion "
        "falls in the 18-to-44-minute range. BF-B does not add a packaging-seal credit because the seal was intact."
    )
    assert "because A" not in final_answer
    assert result["answer_shape_repair_type"] == "conditional_credit_slot_answer"


def test_q11_style_credit_repair_removes_redundant_single_label() -> None:
    question = (
        "Incident NM-1042 had a 22-minute temperature excursion and BF-B. "
        "What credit should apply, and why?"
    )
    result = validate_and_repair_final_answer(
        question=question,
        answer=(
            "A $140 single standard temperature excursion credit applies because the "
            "22-minute temperature excursion falls in the 18-to-44-minute range. "
            "BF-B does not add a packaging-seal credit because the seal was intact."
        ),
        requirements=classify_answer_requirements(question),
        documents=[
            _document(
                "A 22-minute temperature excursion qualifies for the standard "
                "temperature-excursion credit of $140. The standard "
                "temperature-excursion credit applies between 18 and 44 minutes."
            ),
            _document(
                "BF-B indicates outer carton damage, but the seal was intact, so it "
                "does not count as a packaging seal failure."
            ),
        ],
    )

    final_answer = str(result["answer_after_final_validation"])
    assert final_answer == (
        "A $140 standard temperature-excursion credit applies because the "
        "22-minute excursion falls in the 18-to-44-minute range. "
        "BF-B does not add a packaging-seal credit because the seal was intact."
    )
    assert "single standard" not in final_answer


def test_q11_style_mini_question_repair_preserves_secondary_code_effect() -> None:
    question = (
        "Incident NM-1042 had a 22-minute temperature excursion and BF-B. "
        "What credit should apply, and why?"
    )
    result = validate_and_repair_final_answer(
        question=question,
        answer=(
            "What credit should apply? $140. Why? A 22-minute temperature excursion "
            "qualifies for a standard credit of $140. Secondary answer? BF-B does not "
            "trigger a packaging seal failure. Reason or condition? Blue Finch BF-B "
            "indicates outer carton damage but intact seal, which does not count as a "
            "packaging seal failure."
        ),
        requirements=classify_answer_requirements(question),
        documents=[
            _document(
                "A 22-minute temperature excursion qualifies for the standard "
                "temperature-excursion credit of $140. The standard "
                "temperature-excursion credit applies between 18 and 44 minutes."
            ),
            _document(
                "BF-B indicates outer carton damage, but the seal was intact, so it "
                "does not count as a packaging seal failure."
            ),
        ],
    )

    final_answer = str(result["answer_after_final_validation"])
    assert final_answer == (
        "A $140 standard temperature-excursion credit applies because the "
        "22-minute excursion falls in the 18-to-44-minute range. "
        "BF-B does not add a packaging-seal credit because the seal was intact."
    )
    assert not any(
        label in final_answer
        for label in (
            "What credit should apply?",
            "Why?",
            "Secondary answer?",
            "Reason or condition?",
        )
    )


def test_q11_style_repair_uses_supported_secondary_reason_from_final_context() -> None:
    question = (
        "Incident NM-1042 had a 22-minute temperature excursion and BF-B. "
        "What credit should apply, and why?"
    )
    result = validate_and_repair_final_answer(
        question=question,
        answer=(
            "What credit should apply? $140. Why? A 22-minute temperature excursion "
            "qualifies for a standard credit of $140. Secondary answer? No additional "
            "credit applies because BF-B does not indicate a packaging seal failure."
        ),
        requirements=classify_answer_requirements(question),
        documents=[
            _document(
                "A 22-minute temperature excursion qualifies for the standard "
                "temperature-excursion credit of $140. The standard "
                "temperature-excursion credit applies between 18 and 44 minutes."
            ),
            _document(
                "BF-B indicates outer carton damage, but the seal was intact, so it "
                "does not count as a packaging seal failure."
            ),
        ],
    )

    assert result["answer_after_final_validation"] == (
        "A $140 standard temperature-excursion credit applies because the "
        "22-minute excursion falls in the 18-to-44-minute range. "
        "BF-B does not add a packaging-seal credit because the seal was intact."
    )


def test_conditional_credit_repair_preserves_a_generic_secondary_code() -> None:
    question = (
        "Incident OR-77 had a 12-minute delivery delay and PK-7. "
        "What credit should apply, and why?"
    )
    result = validate_and_repair_final_answer(
        question=question,
        answer=(
            "What credit should apply? $75. Why? A 12-minute delivery delay qualifies "
            "for a standard delivery credit of $75. Secondary answer? PK-7 does not "
            "trigger a damage failure."
        ),
        requirements=classify_answer_requirements(question),
        documents=[
            _document(
                "A 12-minute delivery delay qualifies for the standard delivery credit "
                "of $75. The standard delivery credit applies between 10 and 20 minutes."
            ),
            _document(
                "PK-7 indicates cosmetic scuffing, but the package was intact, so it "
                "does not count as a damage failure."
            ),
        ],
    )

    final_answer = str(result["answer_after_final_validation"])
    assert "$75" in final_answer
    assert "12-minute delivery delay" in final_answer
    assert "PK-7" in final_answer
    assert "does not add a damage credit" in final_answer
    assert "package was intact" in final_answer
    assert "Secondary answer?" not in final_answer


def test_q9_style_definition_trim_scopes_out_unasked_condition_facts() -> None:
    question = "If someone says an incident needs Copper Door, what does that mean?"
    answer = (
        "Copper Door refers to the manual manager approval path. It is required when more than one credit "
        "category is requested, the credit exceeds $275, a customer requests a credit more than 14 days after "
        "delivery, or the incident was created by a Lantern Pass user."
    )
    result = validate_and_repair_final_answer(
        question=question,
        answer=answer,
        requirements=classify_answer_requirements(question),
        documents=[_document(answer)],
    )

    assert result["answer_after_final_validation"] == "Copper Door refers to the manual manager approval path."
    assert "$275" not in result["scoped_locked_facts"]
    assert "$275" in result["trimmed_optional_facts"]
    assert result["repair_rejected"] is False


def test_definition_repair_adds_only_a_supported_operational_consequence() -> None:
    question = "If someone says an incident needs Copper Door, what does that mean?"
    result = validate_and_repair_final_answer(
        question=question,
        answer="Copper Door refers to the manual manager approval path.",
        requirements=classify_answer_requirements(question),
        documents=[
            _document("Copper Door is the manual manager approval path."),
            _document(
                "Any incident created by a Lantern Pass user requires Copper Door "
                "approval before a credit can be issued."
            ),
        ],
    )

    final_answer = str(result["answer_after_final_validation"])
    assert final_answer == (
        "Copper Door means the incident needs manual manager approval before the credit can be issued."
    )
    assert not any(term in final_answer for term in ("Lantern Pass", "$275", "14 days"))


def test_definition_condition_list_is_retained_when_the_question_asks_for_conditions() -> None:
    question = "When is Copper Door approval required?"
    answer = (
        "Copper Door approval is required when more than one credit category is requested or the credit exceeds $275."
    )
    result = validate_and_repair_final_answer(
        question=question,
        answer=answer,
        requirements=classify_answer_requirements(question),
        documents=[_document(answer)],
    )

    assert result["answer_after_final_validation"] == answer
    assert "$275" in result["scoped_locked_facts"]
    assert result["trimmed_optional_facts"] == []


def test_conditional_credit_repair_keeps_asked_approval_without_scaffolding() -> None:
    question = (
        "Incident NM-1043 involved medical nutrition packs, a 12-minute temperature "
        "excursion, and was created by Tomas Greer. What credit applies, and is approval required?"
    )
    result = validate_and_repair_final_answer(
        question=question,
        answer=(
            "Question Part 1: What credit applies? The $210 medical nutrition exception "
            "credit applies. Question Part 2: Is approval required? Yes, Copper Door "
            "approval is required because Tomas Greer is a Lantern Pass user."
        ),
        requirements=classify_answer_requirements(question),
        documents=[
            _document(
                "A medical nutrition exception amount of $210 applies when a verified "
                "temperature excursion is 10 minutes or longer."
            ),
            _document(
                "Copper Door approval is required because Tomas Greer is a Lantern Pass user."
            ),
        ],
    )

    final_answer = str(result["answer_after_final_validation"])
    assert "$210" in final_answer
    assert "medical nutrition exception credit" in final_answer
    assert "10 minutes or longer" in final_answer
    assert "Copper Door approval" in final_answer
    assert "Tomas Greer" in final_answer
    assert "Question Part" not in final_answer
    assert "What credit applies?" not in final_answer
    assert "Is approval required?" not in final_answer
    assert result["answer_relevancy_gate_applied"] is True
    assert result["forbidden_scaffold_removed"] is True
    assert result["primary_reason_added"] is True
    assert result["approval_scope_decision"] == "requested"

    repeated = validate_and_repair_final_answer(
        question=question,
        answer=final_answer,
        requirements=classify_answer_requirements(question),
        documents=[
            _document(
                "A medical nutrition exception amount of $210 applies when a verified "
                "temperature excursion is 10 minutes or longer."
            ),
            _document(
                "Copper Door approval is required because Tomas Greer is a Lantern Pass user."
            ),
        ],
    )
    assert repeated["answer_after_final_validation"] == final_answer


def test_low_cost_semantic_completion_repairs_exact_q12_labeled_answer() -> None:
    question = (
        "Incident NM-1043 involved medical nutrition packs, a 12-minute temperature "
        "excursion, and was created by Tomas Greer. What credit applies, and is approval required?"
    )
    answer = (
        "What credit applies? $210 medical nutrition exception. "
        "Is approval required? Yes, Copper Door approval is required because Tomas Greer "
        "is a Lantern Pass user."
    )
    documents = [
        _document(
            "A medical nutrition exception amount of $210 applies when a verified "
            "temperature excursion is 10 minutes or longer."
        ),
        _document(
            "Copper Door approval is required because Tomas Greer is a Lantern Pass user."
        ),
    ]

    result = validate_and_repair_final_answer(
        question=question,
        answer=answer,
        requirements=classify_answer_requirements(question),
        documents=documents,
    )
    final_answer = str(result["answer_after_final_validation"])

    assert final_answer == (
        "The $210 medical nutrition exception credit applies because verified medical "
        "nutrition temperature excursions of 10 minutes or longer qualify. Copper Door "
        "approval is required because Tomas Greer is a Lantern Pass user."
    )
    assert "$210" in final_answer
    assert "medical nutrition" in final_answer
    assert "credit" in final_answer
    assert "10 minutes" in final_answer or "10-minute" in final_answer
    assert "Copper Door approval" in final_answer
    assert "Tomas Greer" in final_answer or "Lantern Pass user" in final_answer
    assert "What credit applies?" not in final_answer
    assert "Is approval required?" not in final_answer
    assert "Question Part" not in final_answer
    assert result["semantic_completion_gate_applied"] is True
    assert result["semantic_completion_passed"] is True
    assert result["question_labels_detected"] == [
        "what_credit_applies",
        "approval_required",
    ]
    assert result["question_labels_removed"] is True
    assert result["semantic_reason_added"] is True
    assert result["semantic_reason_source"] is not None
    assert result["approval_part_detected"] is True
    assert result["approval_part_preserved"] is True
    assert result["approval_part_removed"] is False
    assert result["final_selected_answer_source"] == "semantic_completion"

    repeated = validate_and_repair_final_answer(
        question=question,
        answer=final_answer,
        requirements=classify_answer_requirements(question),
        documents=documents,
    )
    assert repeated["answer_after_final_validation"] == final_answer
    assert repeated["semantic_completion_gate_applied"] is False


def test_q3_semantic_completion_protection_removes_unasked_allowed_categories() -> None:
    question = "Can an employee use the gym reimbursement to buy running shoes?"
    result = validate_and_repair_final_answer(
        question=question,
        answer=(
            "No, athletic apparel is not covered. The policy covers gym memberships, "
            "yoga studios, and martial arts classes."
        ),
        requirements=classify_answer_requirements(question),
        documents=[
            _document(
                "Running shoes are athletic apparel. Athletic apparel is not covered "
                "by the gym reimbursement."
            )
        ],
    )

    assert result["answer_after_final_validation"] == (
        "No. Running shoes are athletic apparel, and athletic apparel is not covered "
        "by the gym reimbursement."
    )
    assert "gym memberships, yoga studios, and martial arts classes" not in str(
        result["answer_after_final_validation"]
    )
    assert result["semantic_completion_gate_applied"] is False


def test_q11_and_q9_already_good_answers_remain_unchanged() -> None:
    credit_question = (
        "Incident NM-1042 had a 22-minute temperature excursion and BF-B. "
        "What credit should apply, and why?"
    )
    credit_answer = (
        "A $140 standard temperature-excursion credit applies because the 22-minute temperature excursion "
        "falls in the 18-to-44-minute range. BF-B does not add a packaging-seal credit because the seal was intact."
    )
    definition_question = "If someone says an incident needs Copper Door, what does that mean?"
    definition_answer = "Copper Door means the incident needs manual manager approval before the credit can be issued."

    credit = validate_and_repair_final_answer(
        question=credit_question,
        answer=credit_answer,
        requirements=classify_answer_requirements(credit_question),
        documents=[
            _document(
                "A 22-minute temperature excursion qualifies for the standard temperature-excursion credit of $140. "
                "The standard temperature-excursion credit applies between 18 and 44 minutes."
            ),
            _document("BF-B does not add a packaging-seal credit because the seal was intact."),
        ],
    )
    definition = validate_and_repair_final_answer(
        question=definition_question,
        answer=definition_answer,
        requirements=classify_answer_requirements(definition_question),
        documents=[_document(definition_answer)],
    )

    assert credit["answer_after_final_validation"] == credit_answer
    assert definition["answer_after_final_validation"] == definition_answer
