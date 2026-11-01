import pytest

from pra_core.agent.workflow import (
    ResearchAssistantWorkflow,
    classify_answer_requirements,
    validate_answer_completeness,
)
from pra_core.documents.models import RetrievedDocument
from pra_core.llm.base import LLMProvider
from pra_core.memory.store import LocalMemoryStore
from pra_core.routing.model import DeterministicModelRouter
from pra_core.tracing.langsmith import NoOpLangSmithTracer


class StaticRetriever:
    def __init__(self, documents: list[RetrievedDocument]) -> None:
        self._documents = documents

    def search(self, query: str, k: int = 3) -> list[RetrievedDocument]:
        return self._documents[:k]


class NamedLLMProvider(LLMProvider):
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0

    def generate(
        self,
        prompt: str,
        context: list[RetrievedDocument],
        history: list[str],
    ) -> str:
        self.calls += 1
        return f"{self.name} answered."


class FixedAnswerProvider(NamedLLMProvider):
    def __init__(self, answer: str) -> None:
        super().__init__("fixed")
        self._answer = answer

    def generate(
        self,
        prompt: str,
        context: list[RetrievedDocument],
        history: list[str],
    ) -> str:
        self.calls += 1
        return self._answer


class FallbackAwareProvider(NamedLLMProvider):
    def __init__(self, name: str, resolved_name: str) -> None:
        super().__init__(name)
        self._resolved_name = resolved_name

    def resolved_model_id(self) -> str | None:
        return self._resolved_name


class HistoryCapturingProvider(NamedLLMProvider):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.last_history: list[str] = []

    def generate(
        self,
        prompt: str,
        context: list[RetrievedDocument],
        history: list[str],
    ) -> str:
        self.last_history = list(history)
        return super().generate(prompt=prompt, context=context, history=history)


class ContextCapturingProvider(NamedLLMProvider):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.last_context: list[RetrievedDocument] = []

    def generate(
        self,
        prompt: str,
        context: list[RetrievedDocument],
        history: list[str],
    ) -> str:
        self.last_context = list(context)
        return super().generate(prompt=prompt, context=context, history=history)


class FailingLLMProvider(NamedLLMProvider):
    def generate(
        self,
        prompt: str,
        context: list[RetrievedDocument],
        history: list[str],
    ) -> str:
        raise AssertionError("LLM should not be called for exact code table lookups.")


def test_router_sends_greetings_to_simple_model() -> None:
    router = DeterministicModelRouter(
        simple_model_id="google.gemma-3-12b-it",
        reasoning_model_id="opus",
    )

    decision = router.select_model(message="hello", documents=[], history=[])

    assert decision.selected_model == "google.gemma-3-12b-it"
    assert decision.routing_reason == "greeting_or_small_talk"


def test_router_sends_high_confidence_definition_to_simple_model() -> None:
    router = DeterministicModelRouter(
        simple_model_id="gemma",
        reasoning_model_id="mistral",
    )

    decision = router.select_model(
        message="If someone says an incident needs Copper Door, what does that mean?",
        documents=[
            RetrievedDocument(
                document_id="doc-1",
                filename="scenario.md",
                text="Copper Door means special approval.",
                score=8.11,
                rrf_score=8.11,
            )
        ],
        history=[],
    )

    assert decision.selected_model == "gemma"
    assert decision.routing_reason == "definition_high_confidence_retrieval"


def test_router_sends_high_value_reasoning_to_opus() -> None:
    router = DeterministicModelRouter(
        simple_model_id="google.gemma-3-12b-it",
        reasoning_model_id="opus",
    )

    decision = router.select_model(
        message="Compare the policies and recommend the lower risk option.",
        documents=[
            RetrievedDocument(
                document_id="doc-1",
                filename="policy.md",
                text="Policy text",
                score=0.91,
            )
        ],
        history=[],
    )

    assert decision.selected_model == "opus"
    assert decision.routing_reason == "high_value_reasoning"


def test_router_sends_conditional_reasoning_candidate_to_reasoning_model() -> None:
    router = DeterministicModelRouter(
        simple_model_id="google.gemma-3-12b-it",
        reasoning_model_id="opus",
    )

    decision = router.select_model(
        message="If a shipment is recovered before Monday at 10:00 AM, is the route delay credit always reduced to $25?",
        documents=[
            RetrievedDocument(
                document_id="doc-1",
                filename="policy.md",
                text="Conditional policy text.",
                score=0.04,
                vector_score=0.82,
                rrf_score=0.04,
            )
        ],
        history=[],
    )

    assert decision.selected_model == "opus"
    assert decision.routing_reason == "conditional_or_exception_reasoning"


def test_workflow_returns_model_routing_and_grounding_metadata() -> None:
    simple = NamedLLMProvider("simple")
    reasoning = NamedLLMProvider("reasoning")
    workflow = ResearchAssistantWorkflow(
        retriever=StaticRetriever(
            [
                RetrievedDocument(
                    document_id="doc-1",
                    filename="policy.md",
                    text="Remote stipend expires after 90 days.",
                    score=0.0328,
                    rrf_score=0.0328,
                )
            ]
        ),
        memory_store=LocalMemoryStore(),
        tracer=NoOpLangSmithTracer(project_name="test"),
        llm_provider=reasoning,
        llm_providers={
            "google.gemma-3-12b-it": simple,
            "opus": reasoning,
        },
        model_router=DeterministicModelRouter(
            simple_model_id="google.gemma-3-12b-it",
            reasoning_model_id="opus",
        ),
    )

    result = workflow.run(
        message="When does the stipend expire?",
        session_id="phase-11",
        top_k=3,
    )

    assert result["selected_model"] == "google.gemma-3-12b-it"
    assert result["routing_reason"] == "simple_high_rrf_retrieval"
    assert result["retrieval_scores"] == [0.0328]
    assert result["max_retrieval_score"] == 0.0328
    assert result["hallucination_score"] == 0.16
    assert result["hallucination_action"] == "none"
    assert simple.calls == 1
    assert reasoning.calls == 0


def test_workflow_cannot_return_a_bare_duration_without_final_validation() -> None:
    provider = FixedAnswerProvider("90 days.")
    workflow = ResearchAssistantWorkflow(
        retriever=StaticRetriever(
            [
                RetrievedDocument(
                    document_id="doc-1",
                    filename="remote-work.md",
                    text="A new hire has 90 days to spend the remote work stipend.",
                    score=0.8,
                    rrf_score=0.8,
                )
            ]
        ),
        memory_store=LocalMemoryStore(),
        tracer=NoOpLangSmithTracer(project_name="test"),
        llm_provider=provider,
        llm_providers={"google.gemma-3-12b-it": provider},
        model_router=DeterministicModelRouter(
            simple_model_id="google.gemma-3-12b-it",
            reasoning_model_id="opus",
        ),
    )

    result = workflow.run(
        message="How many days does a new hire have to spend their remote work stipend?",
        session_id="final-answer-validation",
        top_k=3,
    )

    assert result["answer"] == "A new hire has 90 days to spend the remote work stipend."
    validation = result["context_compression"]
    assert validation["final_answer_validation_applied"] is True
    assert validation["final_answer_validation_failed"] is True
    assert "bare_value_answer" in validation["final_answer_validation_failure_reasons"]


@pytest.mark.parametrize(
    ("question", "generated_answer", "context", "expected_fragment"),
    [
        (
                "How much money does a part-time employee get for the home office stipend?",
                "$600.",
                "Part-time employees receive $600 for the home office stipend.",
                "A part-time employee receives $600 for the home office stipend.",
        ),
        (
            "How many days does a new hire have to spend their remote work stipend?",
            "90 days.",
            "A new hire has 90 days to spend the remote work stipend.",
            "A new hire has 90 days to spend the remote work stipend.",
        ),
        (
            "What credit applies?",
            "What credit applies? $210.",
            "A $210 credit applies for the verified temperature excursion.",
            "A credit of $210 applies for the verified temperature excursion.",
        ),
        (
            "Is approval required?",
            "Is approval required? Yes.",
            "Copper Door approval is required because the request has multiple credit categories.",
            "Yes. Copper Door approval is required because the request has multiple credit categories.",
        ),
        (
            "If an employee has worked at AcmeCorp for 5 years, how many weeks of paid sabbatical do they get?",
            "They get 4 weeks of paid sabbatical.",
            "After 5 years at AcmeCorp, the employee gets 4 weeks of paid sabbatical.",
            "The employee receives 4 weeks of paid sabbatical after 5 years at AcmeCorp.",
        ),
    ],
    ids=["bare_amount", "bare_duration", "credit_mini_question", "approval_mini_question", "missing_condition"],
)
def test_workflow_final_return_boundary_never_returns_invalid_answer_shape(
    question: str,
    generated_answer: str,
    context: str,
    expected_fragment: str,
) -> None:
    provider = FixedAnswerProvider(generated_answer)
    workflow = ResearchAssistantWorkflow(
        retriever=StaticRetriever(
            [
                RetrievedDocument(
                    document_id="doc-1",
                    filename="policy.md",
                    text=context,
                    score=0.8,
                    rrf_score=0.8,
                )
            ]
        ),
        memory_store=LocalMemoryStore(),
        tracer=NoOpLangSmithTracer(project_name="test"),
        llm_provider=provider,
        llm_providers={"google.gemma-3-12b-it": provider},
        model_router=DeterministicModelRouter(
            simple_model_id="google.gemma-3-12b-it",
            reasoning_model_id="opus",
        ),
    )

    result = workflow.run(message=question, session_id="return-boundary", top_k=3)

    assert result["answer"] != generated_answer
    assert expected_fragment in result["answer"]
    validation = result["context_compression"]
    assert validation["final_answer_validation_applied"] is True
    assert validation["final_answer_validation_failed"] is True


def test_workflow_answers_exact_markdown_table_code_lookup_without_llm() -> None:
    simple = FailingLLMProvider("simple")
    workflow = ResearchAssistantWorkflow(
        retriever=StaticRetriever(
            [
                RetrievedDocument(
                    document_id="doc-1",
                    filename="scenario-northstar-meridian.md",
                    text=(
                        "## Blue Finch Packaging Program "
                        "Blue Finch status codes: "
                        "| Code | Meaning | | --- | --- | "
                        "| BF-A | Packaging arrived intact | "
                        "| BF-B | Outer carton damaged, seal intact | "
                        "| BF-C | Seal breach confirmed | "
                        "| BF-D | Temperature strip unreadable |"
                    ),
                    score=1.0,
                    rrf_score=1.0,
                )
            ]
        ),
        memory_store=LocalMemoryStore(),
        tracer=NoOpLangSmithTracer(project_name="test"),
        llm_provider=simple,
        llm_providers={"google.gemma-3-12b-it": simple, "opus": simple},
        model_router=DeterministicModelRouter(
            simple_model_id="google.gemma-3-12b-it",
            reasoning_model_id="opus",
        ),
    )

    result = workflow.run(
        message="What does Blue Finch code BF-C mean?",
        session_id="code-lookup",
        top_k=3,
    )

    assert result["answer"] == "Blue Finch code BF-C means Seal breach confirmed."
    assert result["llm_calls"] == 0
    assert result["hallucination_score"] == 0.16
    assert result["hallucination_action"] == "none"


def test_workflow_answers_exact_markdown_table_field_lookup_without_llm() -> None:
    simple = FailingLLMProvider("simple")
    workflow = ResearchAssistantWorkflow(
        retriever=StaticRetriever(
            [
                RetrievedDocument(
                    document_id="doc-1",
                    filename="scenario-northstar-meridian.md",
                    text=(
                        "| Region Code | Region Name | Dispatch Hub | Regional Lead |\n"
                        "| --- | --- | --- | --- |\n"
                        "| NE-4 | Northeast Four | Albany Relay | Mara Voss |\n"
                        "| SW-7 | Southwest Seven | Mesa Transfer | Rafi Ornelas |"
                    ),
                    score=1.0,
                    rrf_score=1.0,
                )
            ]
        ),
        memory_store=LocalMemoryStore(),
        tracer=NoOpLangSmithTracer(project_name="test"),
        llm_provider=simple,
        llm_providers={"google.gemma-3-12b-it": simple},
        model_router=DeterministicModelRouter(
            simple_model_id="google.gemma-3-12b-it",
            reasoning_model_id="opus",
        ),
    )

    result = workflow.run(
        message="Who is the regional lead for SW-7?",
        session_id="table-field-lookup",
        top_k=3,
    )

    assert result["answer"] == "The regional lead for SW-7 is Rafi Ornelas."
    assert result["llm_calls"] == 0


def test_workflow_answers_multi_record_rule_audit_without_llm() -> None:
    simple = FailingLLMProvider("simple")
    context = """
## Support Credit Rules

- Credits cannot be combined for the same shipment unless Copper Door approval is recorded.

Medical nutrition exception:

- If a shipment contains medical nutrition packs, any verified temperature excursion of 10 minutes or longer qualifies for **$210**.
- If the same shipment also has a packaging seal failure, the credit remains **$210** unless Copper Door approval is present.

## Copper Door Approval

Copper Door approval is required when:

- More than one credit category is requested for the same shipment.
- The requested credit exceeds **$275**.
- A customer asks for a credit more than 14 calendar days after delivery.
- The incident was created by a Lantern Pass user.

## Lantern Pass Users

| User | Active Through | Notes |
| --- | --- | --- |
| Lena Ortiz | 08/31/2026 | NE-4 field contractor |
| Tomas Greer | 09/15/2026 | GL-2 weekend contractor |
| Priya Shah | 07/20/2026 | SW-7 packaging audit contractor |

## Blue Finch Packaging Program

| Code | Meaning |
| --- | --- |
| BF-C | Seal breach confirmed |

## Incident Records

| Incident ID | Region | Shipment Type | Delivery Date | Created By | Issue Summary |
| --- | --- | --- | --- | --- | --- |
| NM-1042 | NE-4 | chilled produce | 07/03/2026 | Mara Voss | Temperature excursion 22 minutes; Blue Finch BF-B |
| NM-1043 | GL-2 | medical nutrition packs | 07/05/2026 | Tomas Greer | Temperature excursion 12 minutes; no packaging damage |
| NM-1044 | SW-7 | chilled dairy | 07/11/2026 | Priya Shah | Final delivery 2 hours 20 minutes late; recovered Sunday 8:40 PM |
| NM-1045 | SW-7 | chilled produce | 07/12/2026 | Rafi Ornelas | Blue Finch BF-C; customer reported 4 business days after delivery |
| NM-1046 | GL-2 | frozen bakery | 07/13/2026 | Imani Keller | Temperature excursion 48 minutes; packaging seal breach confirmed |
| NM-1047 | NE-4 | medical nutrition packs | 07/15/2026 | Lena Ortiz | Temperature excursion 16 minutes; Blue Finch BF-C |
"""
    workflow = ResearchAssistantWorkflow(
        retriever=StaticRetriever(
            [
                RetrievedDocument(
                    document_id="doc-1",
                    filename="scenario-northstar-meridian.md",
                    text=context,
                    score=1.0,
                    rrf_score=1.0,
                )
            ]
        ),
        memory_store=LocalMemoryStore(),
        tracer=NoOpLangSmithTracer(project_name="test"),
        llm_provider=simple,
        llm_providers={"google.gemma-3-12b-it": simple},
        model_router=DeterministicModelRouter(
            simple_model_id="google.gemma-3-12b-it",
            reasoning_model_id="opus",
        ),
    )

    result = workflow.run(
        message="Which incidents require Copper Door approval, and why?",
        session_id="copper-door-audit",
        top_k=3,
    )

    assert result["answer"] == "\n".join(
        [
            "NM-1043 requires Copper Door approval because Tomas Greer is a Lantern Pass user.",
            "NM-1046 requires Copper Door approval if both temperature and packaging credits are requested, because multiple credit categories are involved.",
            "NM-1047 requires Copper Door approval because Lena Ortiz is a Lantern Pass user and because the incident involves both medical nutrition temperature and packaging seal breach facts.",
            "NM-1045 may require Copper Door approval only if the customer asks for a credit more than 14 calendar days after delivery; the provided facts say 4 business days, so the normal packaging credit fails, but the 14-day Copper Door rule cannot be concluded from the given data.",
        ]
    )
    assert result["llm_calls"] == 0

    direct_result = workflow.run(
        message="Does NM-1045 require Copper Door approval?",
        session_id="copper-door-direct-record",
        top_k=3,
    )

    assert direct_result["answer"] == (
        "NM-1045 may require Copper Door approval only if the customer asks for a credit more than 14 calendar days after delivery; "
        "the provided facts say 4 business days, so the normal packaging credit fails, but the 14-day Copper Door rule cannot be concluded from the given data."
    )
    assert direct_result["llm_calls"] == 0


def test_workflow_compiles_table_rule_evidence_for_direct_record_questions() -> None:
    provider = ContextCapturingProvider("simple")
    context = """
## Reimbursement Rules

- A claim requires manager approval when the requested amount is above $1,000.
- A claim requires finance review when it has two exception flags.

## Claim Records

| Claim ID | Requested Amount | Exception Flags |
| --- | --- | --- |
| CL-17 | $1,250 | travel |
| CL-18 | $240 | none |
"""
    workflow = ResearchAssistantWorkflow(
        retriever=StaticRetriever(
            [
                RetrievedDocument(
                    document_id="doc-1",
                    filename="claims.md",
                    text=context,
                    score=1.0,
                    rrf_score=1.0,
                )
            ]
        ),
        memory_store=LocalMemoryStore(),
        tracer=NoOpLangSmithTracer(project_name="test"),
        llm_provider=provider,
        llm_providers={"google.gemma-3-12b-it": provider, "opus": provider},
        model_router=DeterministicModelRouter(
            simple_model_id="google.gemma-3-12b-it",
            reasoning_model_id="opus",
        ),
    )

    result = workflow.run(
        message="Does CL-17 require manager approval?",
        session_id="generic-direct-record-rule",
        top_k=3,
    )

    assert "manager approval" in result["answer"].lower()
    assert result["context_compression"]["final_answer_validation_applied"] is True
    assert provider.calls == 2
    assert provider.last_context[0].filename == "compiled-table-rule-evidence.md"
    assert "Primary record table: Claim Records" in provider.last_context[0].text
    assert "Claim ID=CL-17" in provider.last_context[0].text
    assert "A claim requires manager approval" in provider.last_context[0].text


def test_answer_shape_classifier_and_validator_cover_generic_question_types() -> None:
    cases = [
        ("Can the regional manager approve this request?", "No. The regional manager cannot approve it because the threshold is exceeded.", True),
        ("What credit amount applies and is approval required?", "Yes. $210 applies and approval is required.", True),
        ("Can the approver authorize this incident?", "Conditional: the manager may approve it.", True),
        ("Which incident receives the highest eligible credit?", "NM-2 receives $275 because it qualifies.", True),
    ]

    for question, answer, expected in cases:
        requirements = classify_answer_requirements(question)
        valid, _ = validate_answer_completeness(question, answer, requirements)
        assert valid is expected

    incomplete = classify_answer_requirements("Can the regional manager approve this request?")
    valid, reason = validate_answer_completeness("Can the regional manager approve this request?", "The threshold is exceeded.", incomplete)
    assert valid is False
    assert "Yes" in reason or "No" in reason


def test_router_sends_multi_record_rule_questions_to_reasoning_model() -> None:
    router = DeterministicModelRouter(
        simple_model_id="google.gemma-3-12b-it",
        reasoning_model_id="opus",
    )

    decision = router.select_model(
        message="Which incidents require Copper Door approval, and why?",
        documents=[
            RetrievedDocument(
                document_id="doc-1",
                filename="scenario.md",
                text="Copper Door approval is required when more than one credit category is requested.",
                score=1.0,
                rrf_score=1.0,
            )
        ],
        history=[],
    )

    assert decision.selected_model == "opus"
    assert decision.routing_reason == "multi_record_rule_reasoning"


def test_router_uses_rrf_score_threshold_for_simple_retrieval() -> None:
    router = DeterministicModelRouter(
        simple_model_id="google.gemma-3-12b-it",
        reasoning_model_id="opus",
    )

    decision = router.select_model(
        message="When does the stipend expire?",
        documents=[
            RetrievedDocument(
                document_id="doc-1",
                filename="policy.md",
                text="Remote stipend expires after 90 days.",
                score=0.012,
                rrf_score=0.031,
            )
        ],
        history=[],
    )

    assert decision.selected_model == "google.gemma-3-12b-it"
    assert decision.routing_reason == "simple_high_rrf_retrieval"


def test_router_sends_weak_rrf_retrieval_to_reasoning_model() -> None:
    router = DeterministicModelRouter(
        simple_model_id="google.gemma-3-12b-it",
        reasoning_model_id="opus",
    )

    decision = router.select_model(
        message="Summarize the issue.",
        documents=[
            RetrievedDocument(
                document_id="doc-1",
                filename="policy.md",
                text="Partial support only.",
                score=0.018,
                rrf_score=0.018,
            )
        ],
        history=[],
    )

    assert decision.selected_model == "opus"
    assert decision.routing_reason == "weak_or_moderate_hybrid_retrieval"


def test_router_sends_no_retrieval_research_question_to_reasoning_model() -> None:
    router = DeterministicModelRouter(
        simple_model_id="google.gemma-3-12b-it",
        reasoning_model_id="opus",
    )

    decision = router.select_model(
        message="What is a heat exchanger?",
        documents=[],
        history=[],
    )

    assert decision.selected_model == "opus"
    assert decision.routing_reason == "reasoning_no_retrieval_question"


def test_router_uses_semantic_check_for_borderline_rrf_retrieval() -> None:
    router = DeterministicModelRouter(
        simple_model_id="google.gemma-3-12b-it",
        reasoning_model_id="opus",
    )

    decision = router.select_model(
        message="Summarize the stipend rule.",
        documents=[
            RetrievedDocument(
                document_id="doc-1",
                filename="policy.md",
                text="Remote stipend expires after 90 days.",
                score=0.021,
                vector_score=0.91,
                rrf_score=0.021,
            )
        ],
        history=[],
    )

    assert decision.selected_model == "google.gemma-3-12b-it"
    assert decision.routing_reason == "simple_borderline_rrf_strong_semantic_retrieval"


def test_router_uses_support_count_tie_breaker_for_borderline_rrf_retrieval() -> None:
    router = DeterministicModelRouter(
        simple_model_id="google.gemma-3-12b-it",
        reasoning_model_id="opus",
    )

    decision = router.select_model(
        message="Summarize the stipend rule.",
        documents=[
            RetrievedDocument(
                document_id="doc-1",
                filename="policy.md",
                text="Remote stipend expires after 90 days.",
                score=0.022,
                vector_score=0.72,
                rrf_score=0.022,
            ),
            RetrievedDocument(
                document_id="doc-2",
                filename="faq.md",
                text="Employees must use the stipend within 90 days.",
                score=0.017,
                vector_score=0.69,
                rrf_score=0.017,
            ),
        ],
        history=[],
    )

    assert decision.selected_model == "google.gemma-3-12b-it"
    assert decision.routing_reason == "simple_borderline_rrf_multi_support_retrieval"


def test_router_escalates_multi_chunk_reconciliation_questions() -> None:
    router = DeterministicModelRouter(
        simple_model_id="google.gemma-3-12b-it",
        reasoning_model_id="opus",
    )

    decision = router.select_model(
        message="Does this still apply if there is an exception?",
        documents=[
            RetrievedDocument(
                document_id="doc-1",
                filename="policy.md",
                text="Base rule.",
                score=0.024,
                vector_score=0.78,
                rrf_score=0.024,
            ),
            RetrievedDocument(
                document_id="doc-2",
                filename="exceptions.md",
                text="Exception rule.",
                score=0.021,
                vector_score=0.74,
                rrf_score=0.021,
            ),
        ],
        history=[],
    )

    assert decision.selected_model == "opus"
    assert decision.routing_reason == "conditional_or_exception_reasoning"


def test_workflow_refines_simple_route_to_reasoning_for_multi_chunk_reconciliation() -> None:
    simple = NamedLLMProvider("simple")
    reasoning = NamedLLMProvider("reasoning")
    workflow = ResearchAssistantWorkflow(
        retriever=StaticRetriever(
            [
                RetrievedDocument(
                    document_id="doc-1",
                    filename="policy.md",
                    text="Base rule applies before Monday 10:00 AM.",
                    score=0.033,
                    vector_score=0.84,
                    rrf_score=0.033,
                    chunk_id="chunk-1",
                    chunk_index=0,
                ),
                RetrievedDocument(
                    document_id="doc-2",
                    filename="exceptions.md",
                    text="Exception in SW-7 during July.",
                    score=0.028,
                    vector_score=0.8,
                    rrf_score=0.028,
                    chunk_id="chunk-2",
                    chunk_index=1,
                ),
            ]
        ),
        memory_store=LocalMemoryStore(),
        tracer=NoOpLangSmithTracer(project_name="test"),
        llm_provider=reasoning,
        llm_providers={
            "google.gemma-3-12b-it": simple,
            "opus": reasoning,
        },
        model_router=DeterministicModelRouter(
            simple_model_id="google.gemma-3-12b-it",
            reasoning_model_id="opus",
        ),
    )

    result = workflow.run(
        message="If recovered before Monday at 10:00 AM, is the route delay credit always reduced to $25?",
        session_id="phase-11-reconcile",
        top_k=5,
    )

    assert result["selected_model"] == "opus"
    assert result["routing_reason"] in {
        "conditional_or_exception_reasoning",
        "post_retrieval_conditional_reconciliation",
    }
    assert simple.calls == 0
    assert reasoning.calls == 2


def test_workflow_applies_topic_reset_for_new_entity_lookup() -> None:
    provider = HistoryCapturingProvider("simple")
    memory_store = LocalMemoryStore()
    memory_store.add_message("topic-reset", "user", "What does Blue Finch code BF-C mean?")
    memory_store.add_message("topic-reset", "assistant", "Blue Finch is a logistics coding family.")
    workflow = ResearchAssistantWorkflow(
        retriever=StaticRetriever(
            [
                RetrievedDocument(
                    document_id="doc-1",
                    filename="northstar.md",
                    text="Orchid Ledger is the main analytics system.",
                    score=0.031,
                    rrf_score=0.031,
                )
            ]
        ),
        memory_store=memory_store,
        tracer=NoOpLangSmithTracer(project_name="test"),
        llm_provider=provider,
        llm_providers={"google.gemma-3-12b-it": provider},
        model_router=DeterministicModelRouter(
            simple_model_id="google.gemma-3-12b-it",
            reasoning_model_id="opus",
        ),
    )

    result = workflow.run(
        message="What is Orchid Ledger?",
        session_id="topic-reset",
        top_k=5,
    )

    assert result["topic_reset_applied"] is True
    assert provider.last_history == []


def test_workflow_uses_safe_fallback_for_very_high_grounding_risk() -> None:
    provider = NamedLLMProvider("simple")
    workflow = ResearchAssistantWorkflow(
        retriever=StaticRetriever([]),
        memory_store=LocalMemoryStore(),
        tracer=NoOpLangSmithTracer(project_name="test"),
        llm_provider=provider,
        llm_providers={"google.gemma-3-12b-it": provider},
        model_router=DeterministicModelRouter(
            simple_model_id="google.gemma-3-12b-it",
            reasoning_model_id="opus",
        ),
    )

    result = workflow.run(
        message="What is the reimbursement deadline?",
        session_id="phase-11-risk",
        top_k=3,
    )

    assert result["answer"] == (
        "I can't answer that confidently from what I have here yet. "
        "If you add material on that topic, I can take another pass."
    )
    assert result["hallucination_score"] == 0.82
    assert result["hallucination_action"] == "safe_fallback"


def test_workflow_reports_actual_fallback_reasoning_model_for_request() -> None:
    simple = NamedLLMProvider("simple")
    reasoning = FallbackAwareProvider(
        "reasoning",
        "arn:aws:bedrock:us-east-1::inference-profile/"
        "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    )
    workflow = ResearchAssistantWorkflow(
        retriever=StaticRetriever(
            [
                RetrievedDocument(
                    document_id="doc-1",
                    filename="policy.md",
                    text="Risk analysis requires a more careful answer.",
                    score=0.4,
                )
            ]
        ),
        memory_store=LocalMemoryStore(),
        tracer=NoOpLangSmithTracer(project_name="test"),
        llm_provider=reasoning,
        llm_providers={
            "google.gemma-3-12b-it": simple,
            "mistral.mistral-large-3-675b-instruct": reasoning,
            "arn:aws:bedrock:us-east-1::inference-profile/"
            "us.anthropic.claude-haiku-4-5-20251001-v1:0": reasoning,
        },
        model_router=DeterministicModelRouter(
            simple_model_id="google.gemma-3-12b-it",
            reasoning_model_id="mistral.mistral-large-3-675b-instruct",
        ),
    )

    result = workflow.run(
        message="Compare the risk options.",
        session_id="phase-14",
        top_k=3,
    )

    assert result["selected_model"] == (
        "arn:aws:bedrock:us-east-1::inference-profile/"
        "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    )
    assert result["routing_reason"] == "high_value_reasoning"
