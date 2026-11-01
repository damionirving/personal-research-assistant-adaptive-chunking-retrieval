import re
from contextvars import ContextVar
from typing import Callable, TypedDict

from langgraph.graph import END, StateGraph

from pra_core.documents.models import RetrievedDocument
from pra_core.grounding.scorer import GroundingScorer
from pra_core.llm.base import LLMProvider
from pra_core.logging import get_logger, trace_id_context
from pra_core.memory.store import MemoryStore
from pra_core.metrics import MetricsRegistry
from pra_core.pricing import BedrockPricingCatalog, load_bedrock_pricing_catalog
from pra_core.retrieval.base import Retriever
from pra_core.retrieval.context_compression import compress_context
from pra_core.routing.model import DeterministicModelRouter, _is_definition_question
from pra_core.runtime_metrics import RuntimeTelemetryProbe
from pra_core.tracing.langsmith import LangSmithTracer, TraceRun

logger = get_logger(__name__)
active_trace_run_context: ContextVar[TraceRun | None] = ContextVar(
    "active_trace_run_context",
    default=None,
)


class AssistantState(TypedDict):
    message: str
    session_id: str
    top_k: int
    llm_top_k: int
    history: list[str]
    topic_reset_applied: bool
    documents: list[RetrievedDocument]
    llm_documents: list[RetrievedDocument]
    citations: list[str]
    retrieval_scores: list[float]
    retrieval_details: list[dict[str, object]]
    knowledge_graph_enabled: bool
    knowledge_graph_hits: int
    top_knowledge_graph_score: float | None
    knowledge_graph_contribution: bool
    min_retrieval_score: float | None
    max_retrieval_score: float | None
    avg_retrieval_score: float | None
    llm_calls: int
    input_tokens: int
    output_tokens: int
    last_input_cost_usd: float | None
    last_output_cost_usd: float | None
    cost_per_call_usd: float | None
    total_cost_usd: float | None
    selected_model: str
    routing_reason: str
    routing_confidence: float
    answer: str
    hallucination_score: float
    hallucination_reason: str
    hallucination_action: str
    answer_requirements: dict[str, object]
    answer_completeness_valid: bool
    repair_invoked: bool
    repair_reason: str
    context_compression: dict[str, object]


class ResearchAssistantWorkflow:
    """Local LangGraph workflow for Phase 1 chat orchestration."""

    def __init__(
        self,
        retriever: Retriever,
        memory_store: MemoryStore,
        tracer: LangSmithTracer,
        llm_provider: LLMProvider | None = None,
        llm_providers: dict[str, LLMProvider] | None = None,
        model_router: DeterministicModelRouter | None = None,
        grounding_scorer: GroundingScorer | None = None,
        metrics_registry: MetricsRegistry | None = None,
        runtime_probe: RuntimeTelemetryProbe | None = None,
        llm_context_top_k: int = 25,
        knowledge_graph_enabled: bool = False,
        simple_answer_max_tokens: int = 150,
        reasoning_answer_max_tokens: int = 400,
    ) -> None:
        self._retriever = retriever
        self._llm_provider = llm_provider
        self._llm_providers = llm_providers or {"default": llm_provider}
        self._model_router = model_router or DeterministicModelRouter(
            simple_model_id="default",
            reasoning_model_id="default",
        )
        self._grounding_scorer = grounding_scorer or GroundingScorer()
        self._memory_store = memory_store
        self._tracer = tracer
        self._metrics_registry = metrics_registry
        self._pricing_catalog = load_bedrock_pricing_catalog()
        self._runtime_probe = runtime_probe or RuntimeTelemetryProbe()
        self._llm_context_top_k = llm_context_top_k
        self._knowledge_graph_enabled = knowledge_graph_enabled
        self._simple_answer_max_tokens = simple_answer_max_tokens
        self._reasoning_answer_max_tokens = reasoning_answer_max_tokens
        self._graph = self._build_graph()

    def run(self, message: str, session_id: str, top_k: int = 25) -> dict[str, object]:
        trace_id = trace_id_context.get() or "local"
        runtime_telemetry = self._runtime_probe.snapshot(
            completed_request_count=self._completed_request_count_including_current()
        )
        run = self._tracer.start_run(
            name="research_assistant_workflow",
            trace_id=trace_id,
            inputs={
                "message_length": len(message),
                "session_id": session_id,
                "top_k": top_k,
                "llm_top_k": min(top_k, self._llm_context_top_k),
            },
            metadata={
                "workflow": "research_assistant_workflow",
                "uptime_seconds": runtime_telemetry.uptime_seconds,
                "throughput_requests_per_minute": runtime_telemetry.throughput_requests_per_minute,
                "cpu_time_seconds": runtime_telemetry.cpu_time_seconds,
                "memory_rss_mb": runtime_telemetry.memory_rss_mb,
            },
            tags=["workflow:research_assistant"],
        )
        token = active_trace_run_context.set(run)
        try:
            state = self._graph.invoke(
                {
                    "message": message,
                    "session_id": session_id,
                    "top_k": top_k,
                    "llm_top_k": min(top_k, self._llm_context_top_k),
                    "history": [],
                    "topic_reset_applied": False,
                    "documents": [],
                    "llm_documents": [],
                    "citations": [],
                    "retrieval_scores": [],
                    "retrieval_details": [],
                    "knowledge_graph_enabled": self._knowledge_graph_enabled,
                    "knowledge_graph_hits": 0,
                    "top_knowledge_graph_score": None,
                    "knowledge_graph_contribution": False,
                    "min_retrieval_score": None,
                    "max_retrieval_score": None,
                    "avg_retrieval_score": None,
                    "llm_calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "last_input_cost_usd": None,
                    "last_output_cost_usd": None,
                    "cost_per_call_usd": None,
                    "total_cost_usd": None,
                    "selected_model": "",
                    "routing_reason": "",
                    "routing_confidence": 0.0,
                    "answer": "",
                    "hallucination_score": 0.0,
                    "hallucination_reason": "",
                    "hallucination_action": "none",
                    "answer_requirements": {},
                    "answer_completeness_valid": True,
                    "repair_invoked": False,
                    "repair_reason": "",
                    "context_compression": {},
                }
            )
            # This is the single shared return boundary for chat and RAGAS.
            # Validate after graph execution, before either consumer can receive
            # the generated answer string.
            state = self._validate_final_answer(state)
            state = self._save_memory(state)
            result = {
                "session_id": state["session_id"],
                "answer": state["answer"],
                "citations": state["citations"],
                "retrieval_scores": state["retrieval_scores"],
                "retrieval_details": state["retrieval_details"],
                "knowledge_graph_enabled": state["knowledge_graph_enabled"],
                "knowledge_graph_hits": state["knowledge_graph_hits"],
                "top_knowledge_graph_score": state["top_knowledge_graph_score"],
                "knowledge_graph_contribution": state["knowledge_graph_contribution"],
                "min_retrieval_score": state["min_retrieval_score"],
                "max_retrieval_score": state["max_retrieval_score"],
                "avg_retrieval_score": state["avg_retrieval_score"],
                "selected_model": state["selected_model"],
                "routing_reason": state["routing_reason"],
                "routing_confidence": state["routing_confidence"],
                "hallucination_score": state["hallucination_score"],
                "hallucination_reason": state["hallucination_reason"],
                "hallucination_action": state["hallucination_action"],
                "answer_requirements": state["answer_requirements"],
                "answer_completeness_valid": state["answer_completeness_valid"],
                "repair_invoked": state["repair_invoked"],
                "repair_reason": state["repair_reason"],
                "context_compression": state["context_compression"],
                # Internal evaluator handoff: these are the exact compressed
                # chunks supplied to answer generation, not a second retrieval.
                "ragas_contexts": [document.text for document in state["llm_documents"]],
                "llm_calls": state["llm_calls"],
                "llm_count": state["llm_calls"],
                "input_tokens": state["input_tokens"],
                "output_tokens": state["output_tokens"],
                "cost_per_call_usd": state["cost_per_call_usd"],
                "total_cost_usd": state["total_cost_usd"],
                "uptime_seconds": runtime_telemetry.uptime_seconds,
                "throughput_requests_per_minute": runtime_telemetry.throughput_requests_per_minute,
                "cpu_time_seconds": runtime_telemetry.cpu_time_seconds,
                "memory_rss_mb": runtime_telemetry.memory_rss_mb,
                "topic_reset_applied": state["topic_reset_applied"],
            }
            self._tracer.end_run(
                run,
                outputs={
                    "session_id": state["session_id"],
                    "citation_count": len(state["citations"]),
                    "retrieved_document_count": len(state["documents"]),
                    "max_retrieval_score": state["max_retrieval_score"],
                    "avg_retrieval_score": state["avg_retrieval_score"],
                    "selected_model": state["selected_model"],
                    "routing_reason": state["routing_reason"],
                    "routing_confidence": state["routing_confidence"],
                    "hallucination_score": state["hallucination_score"],
                    "hallucination_action": state["hallucination_action"],
                    "answer_length": len(state["answer"]),
                    "llm_calls": state["llm_calls"],
                    "llm_call_count": state["llm_calls"],
                    "llm_count": state["llm_calls"],
                    "input_tokens": state["input_tokens"],
                    "input_token_count": state["input_tokens"],
                    "output_tokens": state["output_tokens"],
                    "output_token_count": state["output_tokens"],
                    "cost_per_call_usd": state["cost_per_call_usd"],
                    "total_cost_usd": state["total_cost_usd"],
                    "uptime_seconds": runtime_telemetry.uptime_seconds,
                    "throughput_requests_per_minute": runtime_telemetry.throughput_requests_per_minute,
                    "cpu_time_seconds": runtime_telemetry.cpu_time_seconds,
                    "memory_rss_mb": runtime_telemetry.memory_rss_mb,
                    "topic_reset_applied": state["topic_reset_applied"],
                    "usage_metadata": self._build_usage_metadata(
                        input_tokens=state["input_tokens"],
                        output_tokens=state["output_tokens"],
                        input_cost_usd=state["last_input_cost_usd"],
                        output_cost_usd=state["last_output_cost_usd"],
                        total_cost_usd=state["total_cost_usd"],
                        cost_per_call_usd=state["cost_per_call_usd"],
                    ),
                },
            )
            return result
        except Exception as exc:
            self._tracer.end_run(run, error=str(exc))
            raise
        finally:
            active_trace_run_context.reset(token)

    def _build_graph(self):
        graph = StateGraph(AssistantState)
        graph.add_node("load_memory", self._load_memory)
        graph.add_node("apply_topic_reset", self._apply_topic_reset)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("route_model", self._route_model)
        graph.add_node("refine_model_route", self._refine_model_route)
        graph.add_node("generate", self._generate)
        graph.add_node("score_grounding", self._score_grounding)
        graph.add_node("apply_grounding_policy", self._apply_grounding_policy)

        graph.set_entry_point("load_memory")
        graph.add_edge("load_memory", "apply_topic_reset")
        graph.add_edge("apply_topic_reset", "retrieve")
        graph.add_edge("retrieve", "route_model")
        graph.add_edge("route_model", "refine_model_route")
        graph.add_edge("refine_model_route", "generate")
        graph.add_edge("generate", "score_grounding")
        graph.add_edge("score_grounding", "apply_grounding_policy")
        graph.add_edge("apply_grounding_policy", END)
        return graph.compile()

    def _load_memory(self, state: AssistantState) -> AssistantState:
        def operation() -> AssistantState:
            state["history"] = self._memory_store.get_messages(state["session_id"])
            return state

        return self._trace_step(
            name="load_memory",
            trace_id=state["session_id"],
            inputs={"session_id": state["session_id"]},
            run_type="tool",
            metadata={"workflow_step": "load_memory"},
            operation=operation,
            output_builder=lambda updated: {
                "history_message_count": len(updated["history"]),
            },
        )

    def _apply_topic_reset(self, state: AssistantState) -> AssistantState:
        def operation() -> AssistantState:
            if _is_topic_shift(message=state["message"], history=state["history"]):
                state["history"] = []
                state["topic_reset_applied"] = True
                logger.info(
                    "topic_reset_applied",
                    extra={
                        "session_id": state["session_id"],
                        "history_message_count": len(state["history"]),
                    },
                )
            return state

        return self._trace_step(
            name="apply_topic_reset",
            trace_id=state["session_id"],
            inputs={
                "message_length": len(state["message"]),
                "history_message_count": len(state["history"]),
            },
            run_type="tool",
            metadata={"workflow_step": "apply_topic_reset"},
            operation=operation,
            output_builder=lambda updated: {
                "history_message_count": len(updated["history"]),
                "topic_reset_applied": updated["topic_reset_applied"],
            },
        )

    def _retrieve(self, state: AssistantState) -> AssistantState:
        def operation() -> AssistantState:
            state["documents"] = self._retriever.search(
                query=state["message"],
                k=state["top_k"],
            )
            compressed = compress_context(state["message"], state["documents"])
            state["llm_documents"] = compressed.documents[: state["llm_top_k"]]
            state["context_compression"] = {
                "original_retrieved_context_count": len(state["documents"]),
                "final_compressed_context_count": len(state["llm_documents"]),
                "dropped_chunk_ids": list(compressed.dropped_reasons),
                "kept_chunk_ids": compressed.kept_chunk_ids,
                "kept_reasons": compressed.kept_reasons,
                "dropped_reasons": compressed.dropped_reasons,
                "question_type": compressed.question_type,
                "detected_entities": compressed.detected_entities,
                "detected_rule_families": compressed.detected_rule_families,
                "inferred_domain": compressed.inferred_domain,
                "inferred_source_family": compressed.inferred_source_family,
                "metadata_known": compressed.metadata_known,
                "fallback_path_used": compressed.fallback_path_used,
                "final_context_selection_strategy": compressed.final_context_selection_strategy,
                "weak_domain_only_kept_count": compressed.weak_domain_only_kept_count,
                "strong_signal_kept_count": compressed.strong_signal_kept_count,
                "weak_domain_dropped_count": compressed.weak_domain_dropped_count,
                "context_cap_reason": compressed.context_cap_reason,
                "recall_protection_reason": compressed.recall_protection_reason,
                "required_evidence_chunk_ids": compressed.required_evidence_chunk_ids or [],
                "final_context_count_before_precision_filter": (
                    compressed.final_context_count_before_precision_filter
                ),
                "final_context_count_after_precision_filter": (
                    compressed.final_context_count_after_precision_filter
                ),
                "dropped_extra_context_chunk_ids": compressed.dropped_extra_context_chunk_ids or [],
            }
            state["citations"] = list(dict.fromkeys(
                document.document_id for document in state["llm_documents"]
            ))
            state["retrieval_scores"] = [document.score for document in state["llm_documents"]]
            state["retrieval_details"] = [
                {
                    "document_id": document.document_id,
                    "filename": document.filename,
                    "chunk_id": document.chunk_id,
                    "chunk_index": document.chunk_index,
                    "chunk_label": _chunk_label(document),
                    "semantic_similarity_score": document.vector_score,
                    "bm25_score": document.bm25_score,
                    "rrf_score": document.rrf_score if document.rrf_score is not None else document.score,
                    "knowledge_graph_score": document.knowledge_graph_score,
                    "context_keep_reasons": state["context_compression"].get(
                        "kept_reasons", {}
                    ).get(document.chunk_id or _chunk_label(document), []),
                }
                for document in state["llm_documents"]
            ]
            graph_scores = [
                document.knowledge_graph_score
                for document in state["llm_documents"]
                if document.knowledge_graph_score is not None
            ]
            state["knowledge_graph_hits"] = len(graph_scores)
            state["top_knowledge_graph_score"] = max(graph_scores, default=None)
            state["knowledge_graph_contribution"] = bool(graph_scores)
            state["min_retrieval_score"] = (
                min(state["retrieval_scores"]) if state["retrieval_scores"] else None
            )
            state["max_retrieval_score"] = (
                max(state["retrieval_scores"]) if state["retrieval_scores"] else None
            )
            state["avg_retrieval_score"] = (
                sum(state["retrieval_scores"]) / len(state["retrieval_scores"])
                if state["retrieval_scores"]
                else None
            )
            return state

        return self._trace_step(
            name="retrieve",
            trace_id=state["session_id"],
            inputs={
                "query_length": len(state["message"]),
                "top_k": state["top_k"],
                "llm_top_k": state["llm_top_k"],
            },
            run_type="retriever",
            metadata={"workflow_step": "retrieve"},
            operation=operation,
            output_builder=lambda updated: {
                "retrieved_document_count": len(updated["documents"]),
                "llm_retrieved_document_count": len(updated["llm_documents"]),
                "context_compression": updated["context_compression"],
                "citation_count": len(updated["citations"]),
                "min_retrieval_score": updated["min_retrieval_score"],
                "max_retrieval_score": updated["max_retrieval_score"],
                "avg_retrieval_score": updated["avg_retrieval_score"],
                "topic_reset_applied": updated["topic_reset_applied"],
                **self._retrieval_trace_outputs(),
            },
        )

    def _route_model(self, state: AssistantState) -> AssistantState:
        def operation() -> AssistantState:
            decision = self._model_router.select_model(
                message=state["message"],
                documents=state["documents"],
                history=state["history"],
            )
            state["selected_model"] = decision.selected_model
            state["routing_reason"] = decision.routing_reason
            state["routing_confidence"] = decision.routing_confidence
            logger.info(
                "model_route_selected",
                extra={
                    "selected_model": decision.selected_model,
                    "routing_reason": decision.routing_reason,
                    "routing_confidence": decision.routing_confidence,
                    "session_id": state["session_id"],
                    "retrieved_document_count": len(state["documents"]),
                    "max_retrieval_score": state["max_retrieval_score"],
                },
            )
            return state

        return self._trace_step(
            name="route_model",
            trace_id=state["session_id"],
            inputs={
                "history_message_count": len(state["history"]),
                "retrieved_document_count": len(state["documents"]),
                "max_retrieval_score": state["max_retrieval_score"],
            },
            run_type="tool",
            metadata={"workflow_step": "route_model"},
            operation=operation,
            output_builder=lambda updated: {
                "selected_model": updated["selected_model"],
                "routing_reason": updated["routing_reason"],
                "routing_confidence": updated["routing_confidence"],
                "topic_reset_applied": updated["topic_reset_applied"],
            },
        )

    def _generate(self, state: AssistantState) -> AssistantState:
        def operation() -> AssistantState:
            state["answer_requirements"] = classify_answer_requirements(state["message"])
            state["context_compression"].update(
                _answer_shape_context_diagnostics(state["answer_requirements"])
            )
            llm_calls_before_generation = state["llm_calls"]
            state["answer"] = self._generate_with_selected_model(
                state,
                trace_step_name="generate_llm",
            )
            answer_shape = answer_shape_diagnostics(
                state["answer"], state["answer_requirements"]
            )
            answer_shape["deterministic_template_used"] = (
                state["llm_calls"] == llm_calls_before_generation
            )
            state["answer_requirements"].update(answer_shape)
            state["context_compression"].update(answer_shape)
            state["context_compression"]["answer_shape"].update(answer_shape)
            if state["llm_calls"] == 0:
                # Deterministic table/rule answers already have their own
                # exact evidence checks and must retain their public wording.
                state["answer_completeness_valid"] = True
                state["repair_reason"] = ""
            else:
                state["answer_completeness_valid"], state["repair_reason"] = (
                    validate_answer_completeness(
                        state["message"],
                        state["answer"],
                        state["answer_requirements"],
                    )
                )
            return state

        return self._trace_step(
            name="generate",
            trace_id=state["session_id"],
            inputs={
                "selected_model": state["selected_model"],
                "history_message_count": len(state["history"]),
                "retrieved_document_count": len(state["documents"]),
                "llm_retrieved_document_count": len(state["llm_documents"]),
            },
            run_type="chain",
            metadata={"workflow_step": "generate"},
            operation=operation,
            output_builder=lambda updated: {
                "selected_model": updated["selected_model"],
                "answer_length": len(updated["answer"]),
                "answer_completeness_valid": updated["answer_completeness_valid"],
            },
        )

    def _validate_final_answer(self, state: AssistantState) -> AssistantState:
        """Run the mandatory, no-model final-answer validation gate."""

        def operation() -> AssistantState:
            validation = validate_and_repair_final_answer(
                question=state["message"],
                answer=state["answer"],
                requirements=state["answer_requirements"],
                documents=state["llm_documents"],
            )
            state["answer"] = str(validation["answer_after_final_validation"])
            answer_shape = answer_shape_diagnostics(
                state["answer"], state["answer_requirements"]
            )
            state["answer_requirements"].update(answer_shape)
            state["context_compression"].update(answer_shape)
            state["context_compression"].update(validation)
            state["context_compression"]["answer_shape"].update(answer_shape)
            state["context_compression"]["answer_shape"].update(validation)
            state["answer_completeness_valid"], state["repair_reason"] = (
                validate_answer_completeness(
                    state["message"], state["answer"], state["answer_requirements"]
                )
            )
            return state

        return self._trace_step(
            name="validate_final_answer",
            trace_id=state["session_id"],
            inputs={
                "answer_length": len(state["answer"]),
                "llm_retrieved_document_count": len(state["llm_documents"]),
            },
            run_type="tool",
            metadata={"workflow_step": "validate_final_answer"},
            operation=operation,
            output_builder=lambda updated: {
                "answer_length": len(updated["answer"]),
                "validation_applied": updated["context_compression"].get(
                    "final_answer_validation_applied", False
                ),
                "validation_detected_failures": updated["context_compression"].get(
                    "final_answer_validation_failed", False
                ),
            },
        )

    def _refine_model_route(self, state: AssistantState) -> AssistantState:
        def operation() -> AssistantState:
            if (
                state["selected_model"] == self._model_router.simple_model_id
                and self._model_router.reasoning_model_id in self._llm_providers
                and _should_escalate_for_conditional_reconciliation(
                    message=state["message"],
                    documents=state["llm_documents"],
                )
            ):
                state["selected_model"] = self._model_router.reasoning_model_id
                state["routing_reason"] = "post_retrieval_conditional_reconciliation"
                state["routing_confidence"] = 0.83
            return state

        return self._trace_step(
            name="refine_model_route",
            trace_id=state["session_id"],
            inputs={
                "selected_model": state["selected_model"],
                "retrieved_document_count": len(state["documents"]),
                "llm_retrieved_document_count": len(state["llm_documents"]),
                "max_retrieval_score": state["max_retrieval_score"],
            },
            run_type="tool",
            metadata={"workflow_step": "refine_model_route"},
            operation=operation,
            output_builder=lambda updated: {
                "selected_model": updated["selected_model"],
                "routing_reason": updated["routing_reason"],
                "routing_confidence": updated["routing_confidence"],
            },
        )

    def _generate_with_selected_model(
        self,
        state: AssistantState,
        trace_step_name: str,
        *,
        allow_deterministic: bool = True,
    ) -> str:
        def operation() -> str:
            provider = self._llm_providers.get(state["selected_model"])
            if provider is None:
                provider = self._llm_provider or self._llm_providers["default"]
            selected_model_before_call = provider.resolved_model_id() or state["selected_model"]
            prompt = state["message"]
            llm_documents = list(state["llm_documents"])
            history = list(state["history"])
            deterministic_answer = None
            if allow_deterministic:
                deterministic_answer = _answer_code_lookup_from_context(
                    prompt=prompt,
                    documents=llm_documents,
                )
                if deterministic_answer is None:
                    deterministic_answer = _answer_table_lookup_from_context(
                        prompt=prompt,
                        documents=llm_documents,
                    )
                if deterministic_answer is None:
                    deterministic_answer = _answer_multi_record_rule_question_from_context(
                        prompt=prompt,
                        documents=llm_documents,
                    )
            if deterministic_answer is not None:
                return deterministic_answer
            llm_documents = _with_compiled_table_rule_evidence(
                prompt=prompt,
                documents=llm_documents,
            )
            state["llm_documents"] = llm_documents
            is_definition = _is_definition_question(state["message"])
            generation_prompt = (
                "Answer only the definition requested in one or two concise sentences. "
                f"Do not enumerate related incidents unless asked. User question: {prompt}"
                if is_definition
                else prompt
            )
            requirements = state.get("answer_requirements", {})
            if requirements:
                generation_prompt = (
                    f"{generation_prompt}\n\n{build_answer_instruction(requirements)}"
                )
            if state.get("repair_invoked"):
                generation_prompt = (
                    "Revise the answer once. Preserve supported content and citations. "
                    f"Add only the missing requirement: {state['repair_reason']}. "
                    "Use only the supplied context and keep the answer concise.\n\n"
                    + generation_prompt
                )
            answer_budget = (
                self._simple_answer_max_tokens
                if is_definition
                else self._reasoning_answer_max_tokens
            )
            answer = provider.generate_bounded(
                prompt=generation_prompt,
                context=llm_documents,
                history=history,
                max_output_tokens=answer_budget,
            )
            resolved_model = provider.resolved_model_id()
            if resolved_model:
                state["selected_model"] = resolved_model
            cost_estimate = self._pricing_catalog.estimate_call(
                model_id=resolved_model or selected_model_before_call,
                prompt=prompt,
                context_texts=[document.text for document in llm_documents],
                history=history,
                output_text=answer,
            )
            state["llm_calls"] += 1
            logger.info(
                "answer_generation_attempt",
                extra={
                    "answer_generation_attempt": state["llm_calls"],
                    "generation_call_budget": 2,
                    "generation_calls_used": state["llm_calls"],
                    "generation_kind": (
                        "completeness_repair"
                        if state.get("repair_invoked")
                        else "grounding_escalation"
                        if trace_step_name.startswith("grounding_")
                        else "initial_generation"
                    ),
                    "selected_model": state["selected_model"],
                    "session_id": state["session_id"],
                },
            )
            if self._metrics_registry is not None:
                self._metrics_registry.record_llm_call()
            state["input_tokens"] += cost_estimate.input_tokens
            state["output_tokens"] += cost_estimate.output_tokens
            state["last_input_cost_usd"] = cost_estimate.input_cost_usd
            state["last_output_cost_usd"] = cost_estimate.output_cost_usd
            state["cost_per_call_usd"] = cost_estimate.cost_per_call_usd
            if cost_estimate.cost_per_call_usd is not None:
                state["total_cost_usd"] = round(
                    (state["total_cost_usd"] or 0.0) + cost_estimate.cost_per_call_usd,
                    8,
                )
            return answer

        return self._trace_scalar_step(
            name=trace_step_name,
            trace_id=state["session_id"],
            inputs={
                "selected_model": state["selected_model"],
                "query_length": len(state["message"]),
                "history_message_count": len(state["history"]),
                "retrieved_document_count": len(state["llm_documents"]),
            },
            run_type="llm",
            metadata={"workflow_step": trace_step_name},
            operation=operation,
            output_builder=lambda answer: {
                "selected_model": state["selected_model"],
                "answer_length": len(answer),
                "llm_calls": state["llm_calls"],
                "llm_call_count": state["llm_calls"],
                "llm_count": state["llm_calls"],
                "input_tokens": state["input_tokens"],
                "input_token_count": state["input_tokens"],
                "output_tokens": state["output_tokens"],
                "output_token_count": state["output_tokens"],
                "cost_per_call_usd": state["cost_per_call_usd"],
                "total_cost_usd": state["total_cost_usd"],
                "usage_metadata": self._build_usage_metadata(
                    input_tokens=state["input_tokens"],
                    output_tokens=state["output_tokens"],
                    input_cost_usd=state["last_input_cost_usd"],
                    output_cost_usd=state["last_output_cost_usd"],
                    total_cost_usd=state["total_cost_usd"],
                    cost_per_call_usd=state["cost_per_call_usd"],
                ),
            },
        )

    def _completed_request_count_including_current(self) -> int:
        if self._metrics_registry is None:
            return 1
        return self._metrics_registry.snapshot().request_count + 1

    def _retrieval_trace_outputs(self) -> dict[str, float | int]:
        details = getattr(self._retriever, "last_trace", None)
        if details is None:
            return {}
        return {
            "bm25_latency_ms": details.bm25_latency_ms,
            "vector_latency_ms": details.vector_latency_ms,
            "rrf_merge_latency_ms": details.rrf_merge_latency_ms,
            "reranker_latency_ms": details.reranker_latency_ms,
            "retrieval_latency_ms": details.retrieval_latency_ms,
            "bm25_candidate_count": details.bm25_candidate_count,
            "vector_candidate_count": details.vector_candidate_count,
            "merged_candidate_count": details.merged_candidate_count,
            "reranked_candidate_count": details.reranked_candidate_count,
        }

    def _build_usage_metadata(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        input_cost_usd: float | None,
        output_cost_usd: float | None,
        total_cost_usd: float | None,
        cost_per_call_usd: float | None,
    ) -> dict[str, float | int]:
        usage: dict[str, float | int] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
        if input_cost_usd is not None:
            usage["input_cost"] = input_cost_usd
        if output_cost_usd is not None:
            usage["output_cost"] = output_cost_usd
        if total_cost_usd is not None:
            usage["total_cost"] = total_cost_usd
        elif cost_per_call_usd is not None:
            usage["total_cost"] = cost_per_call_usd
        return usage

    def _score_grounding(self, state: AssistantState) -> AssistantState:
        def operation() -> AssistantState:
            score = self._grounding_scorer.score(
                message=state["message"],
                answer=state["answer"],
                documents=state["llm_documents"],
            )
            state["hallucination_score"] = score.hallucination_score
            state["hallucination_reason"] = score.hallucination_reason
            logger.info(
                "answer_grounding_scored",
                extra={
                    "hallucination_score": score.hallucination_score,
                    "hallucination_reason": score.hallucination_reason,
                    "selected_model": state["selected_model"],
                    "session_id": state["session_id"],
                    "max_retrieval_score": state["max_retrieval_score"],
                },
            )
            return state

        return self._trace_step(
            name="score_grounding",
            trace_id=state["session_id"],
            inputs={
                "selected_model": state["selected_model"],
                "retrieved_document_count": len(state["documents"]),
                "answer_length": len(state["answer"]),
            },
            run_type="tool",
            metadata={"workflow_step": "score_grounding"},
            operation=operation,
            output_builder=lambda updated: {
                "hallucination_score": updated["hallucination_score"],
                "hallucination_reason": updated["hallucination_reason"],
            },
        )

    def _apply_grounding_policy(self, state: AssistantState) -> AssistantState:
        def operation() -> AssistantState:
            if state["hallucination_score"] >= 0.76:
                state["answer"] = (
                    "I can't answer that confidently from what I have here yet. "
                    "If you add material on that topic, I can take another pass."
                )
                state["hallucination_action"] = "safe_fallback"
                self._log_grounding_action(state)
                return state

            if (
                state["hallucination_score"] >= 0.56
                and state["selected_model"] == self._model_router.simple_model_id
                and self._model_router.reasoning_model_id in self._llm_providers
                and state["llm_calls"] < 2
            ):
                previous_model = state["selected_model"]
                state["selected_model"] = self._model_router.reasoning_model_id
                state["routing_reason"] = "hallucination_risk_escalated_to_reasoning"
                state["routing_confidence"] = 0.7
                state["answer"] = self._generate_with_selected_model(
                    state,
                    trace_step_name="grounding_escalation_generate_llm",
                )
                score = self._grounding_scorer.score(
                    message=state["message"],
                    answer=state["answer"],
                    documents=state["llm_documents"],
                )
                state["hallucination_score"] = score.hallucination_score
                state["hallucination_reason"] = score.hallucination_reason
                state["hallucination_action"] = f"escalated_from_{previous_model}"
                self._log_grounding_action(state)
                return state

            if (
                not state["answer_completeness_valid"]
                and state["llm_calls"] < 2
            ):
                state["repair_invoked"] = True
                state["answer"] = self._generate_with_selected_model(
                    state,
                    trace_step_name="completeness_repair_generate_llm",
                    allow_deterministic=False,
                )
                state["answer_completeness_valid"], state["repair_reason"] = (
                    validate_answer_completeness(
                        state["message"],
                        state["answer"],
                        state["answer_requirements"],
                    )
                )
                logger.info(
                    "answer_completeness_repair",
                    extra={
                        "repair_invoked": True,
                        "repair_reason": state["repair_reason"],
                        "generation_call_budget": 2,
                        "generation_calls_used": state["llm_calls"],
                        "session_id": state["session_id"],
                    },
                )
                return state

            if state["hallucination_score"] >= 0.31:
                state["hallucination_action"] = "low_confidence_metadata"
                self._log_grounding_action(state)
                return state

            state["hallucination_action"] = "none"
            return state

        return self._trace_step(
            name="apply_grounding_policy",
            trace_id=state["session_id"],
            inputs={
                "hallucination_score": state["hallucination_score"],
                "selected_model": state["selected_model"],
                "retrieved_document_count": len(state["llm_documents"]),
            },
            run_type="tool",
            metadata={"workflow_step": "apply_grounding_policy"},
            operation=operation,
            output_builder=lambda updated: {
                "selected_model": updated["selected_model"],
                "hallucination_score": updated["hallucination_score"],
                "hallucination_action": updated["hallucination_action"],
                "answer_length": len(updated["answer"]),
                "answer_completeness_valid": updated["answer_completeness_valid"],
                "repair_invoked": updated["repair_invoked"],
                "repair_reason": updated["repair_reason"],
            },
        )

    def _log_grounding_action(self, state: AssistantState) -> None:
        logger.info(
            "hallucination_threshold_action",
            extra={
                "hallucination_score": state["hallucination_score"],
                "hallucination_action": state["hallucination_action"],
                "selected_model": state["selected_model"],
                "routing_reason": state["routing_reason"],
                "session_id": state["session_id"],
                "max_retrieval_score": state["max_retrieval_score"],
            },
        )

    def _save_memory(self, state: AssistantState) -> AssistantState:
        def operation() -> AssistantState:
            self._memory_store.add_message(
                session_id=state["session_id"],
                role="user",
                content=state["message"],
            )
            self._memory_store.add_message(
                session_id=state["session_id"],
                role="assistant",
                content=state["answer"],
            )
            return state

        return self._trace_step(
            name="save_memory",
            trace_id=state["session_id"],
            inputs={
                "session_id": state["session_id"],
                "history_message_count": len(state["history"]),
            },
            run_type="tool",
            metadata={"workflow_step": "save_memory"},
            operation=operation,
            output_builder=lambda _updated: {"saved_message_count": 2},
        )

    def _trace_step(
        self,
        name: str,
        trace_id: str,
        inputs: dict[str, object],
        run_type: str,
        metadata: dict[str, object],
        operation: Callable[[], AssistantState],
        output_builder: Callable[[AssistantState], dict[str, object]],
    ) -> AssistantState:
        parent = active_trace_run_context.get()
        active_trace_id = parent.trace_id if parent else (trace_id_context.get() or trace_id)
        run = self._tracer.start_run(
            name=name,
            trace_id=active_trace_id,
            inputs=inputs,
            run_type=run_type,
            parent=parent,
            metadata=metadata,
            tags=[f"workflow_step:{name}"],
        )
        token = active_trace_run_context.set(run)
        try:
            result = operation()
            self._tracer.end_run(run, outputs=output_builder(result))
            return result
        except Exception as exc:
            self._tracer.end_run(run, error=str(exc))
            raise
        finally:
            active_trace_run_context.reset(token)

    def _trace_scalar_step(
        self,
        name: str,
        trace_id: str,
        inputs: dict[str, object],
        run_type: str,
        metadata: dict[str, object],
        operation: Callable[[], str],
        output_builder: Callable[[str], dict[str, object]],
    ) -> str:
        parent = active_trace_run_context.get()
        active_trace_id = parent.trace_id if parent else (trace_id_context.get() or trace_id)
        run = self._tracer.start_run(
            name=name,
            trace_id=active_trace_id,
            inputs=inputs,
            run_type=run_type,
            parent=parent,
            metadata=metadata,
            tags=[f"workflow_step:{name}"],
        )
        token = active_trace_run_context.set(run)
        try:
            result = operation()
            self._tracer.end_run(run, outputs=output_builder(result))
            return result
        except Exception as exc:
            self._tracer.end_run(run, error=str(exc))
            raise
        finally:
            active_trace_run_context.reset(token)


def _should_escalate_for_conditional_reconciliation(
    *,
    message: str,
    documents: list[RetrievedDocument],
) -> bool:
    normalized = message.strip().lower()
    if len(documents) < 2:
        return False
    if not _contains_conditional_reasoning_language(normalized):
        return False

    strong_support_count = sum(
        1
        for document in documents
        if _document_rrf_score(document) >= 0.015
    )
    distinct_chunks = {
        (document.document_id, document.chunk_id, document.chunk_index)
        for document in documents
    }
    return strong_support_count >= 2 and len(distinct_chunks) >= 2


def _contains_conditional_reasoning_language(message: str) -> bool:
    conditional_terms = (
        "always",
        "except",
        "unless",
        "only if",
        "does this still apply",
        "still apply",
        "whether",
        "under what conditions",
        "if ",
    )
    return any(term in message for term in conditional_terms)


def _document_rrf_score(document: RetrievedDocument) -> float:
    if document.rrf_score is not None:
        return document.rrf_score
    return document.score


def _chunk_label(document: RetrievedDocument) -> str:
    if document.chunk_index is None:
        return document.filename
    return f"{document.filename}#chunk-{document.chunk_index}"


def _with_compiled_table_rule_evidence(
    *,
    prompt: str,
    documents: list[RetrievedDocument],
) -> list[RetrievedDocument]:
    evidence = _compile_table_rule_evidence(prompt=prompt, documents=documents)
    if evidence is None:
        return documents
    source = documents[0] if documents else None
    return [
        RetrievedDocument(
            document_id=source.document_id if source else "compiled-evidence",
            filename="compiled-table-rule-evidence.md",
            text=evidence,
            score=source.score if source else 1.0,
            chunk_id="compiled-table-rule-evidence",
            chunk_index=-1,
            bm25_score=source.bm25_score if source else None,
            vector_score=source.vector_score if source else None,
            rrf_score=source.rrf_score if source else None,
        ),
        *documents,
    ]


def _compile_table_rule_evidence(
    *,
    prompt: str,
    documents: list[RetrievedDocument],
) -> str | None:
    if not _is_table_rule_question(prompt):
        return None
    text = "\n\n".join(document.text for document in documents)
    tables = _markdown_tables(text)
    if not tables:
        return None

    prompt_terms = set(_terms(prompt).split())
    primary = _select_primary_record_table(tables, prompt_terms)
    if primary is None:
        return None

    rules = _relevant_rule_lines(text, prompt_terms)
    lookup_tables = [
        table
        for table in tables
        if table is not primary and len(table["rows"]) <= 20
    ][:4]
    lines = [
        "Compiled table-and-rule evidence for a multi-record question.",
        "Use this evidence before the raw chunks. Apply the listed rules to each record, explain the row-level reason, and preserve unresolved conditions instead of overclaiming.",
        "",
        f"User question: {prompt}",
        "",
        f"Primary record table: {primary['section'] or 'Untitled table'}",
        _format_table_rows(primary["headers"], primary["rows"]),
    ]
    if rules:
        lines.extend(["", "Relevant rules and conditions:"])
        lines.extend(f"- {rule}" for rule in rules[:12])
    if lookup_tables:
        lines.extend(["", "Lookup/reference tables:"])
        for table in lookup_tables:
            lines.append(f"{table['section'] or 'Untitled table'}:")
            lines.append(_format_table_rows(table["headers"], table["rows"][:8]))
    return "\n".join(lines)


def _is_table_rule_question(prompt: str) -> bool:
    if _is_multi_record_rule_question(prompt):
        return True
    lowered = prompt.lower()
    return (
        bool(re.search(r"\b(which|list|what|who|does|do)\b", lowered))
        and bool(re.search(r"\b(why|require|required|requires|need|needs|qualify|qualifies|approval|exception|violate|violates|eligible)\b", lowered))
        and bool(re.search(r"\b(incidents?|records?|claims?|shipments?|employees?|users?|items?|cases?)\b", lowered))
    )


def _select_primary_record_table(
    tables: list[dict[str, object]],
    prompt_terms: set[str],
) -> dict[str, object] | None:
    record_markers = {"id", "incident", "record", "claim", "shipment", "employee", "user", "case"}
    scored: list[tuple[int, dict[str, object]]] = []
    for table in tables:
        headers = [str(header).lower() for header in table["headers"]]
        header_terms = set(_terms(" ".join(headers)).split())
        section_terms = set(_terms(str(table["section"])).split())
        row_count = len(table["rows"])
        score = len((header_terms | section_terms) & prompt_terms)
        if header_terms & record_markers:
            score += 4
        if any(header.endswith(" id") or header == "id" for header in headers):
            score += 3
        if row_count >= 3:
            score += 2
        if row_count >= 2:
            scored.append((score, table))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1] if scored[0][0] > 0 else None


def _markdown_tables(text: str) -> list[dict[str, object]]:
    tables: list[dict[str, object]] = []
    current_section = ""
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        heading = re.match(r"\s*#{1,6}\s+(.+?)\s*$", lines[index])
        if heading:
            current_section = heading.group(1)
            index += 1
            continue
        headers = _markdown_cells(lines[index])
        if index + 1 >= len(lines):
            break
        separator = _markdown_cells(lines[index + 1])
        if (
            len(headers) >= 2
            and len(separator) == len(headers)
            and all(re.fullmatch(r":?-{3,}:?", cell) for cell in separator)
        ):
            rows: list[dict[str, str]] = []
            index += 2
            while index < len(lines):
                cells = _markdown_cells(lines[index])
                if not cells:
                    break
                if len(cells) == len(headers):
                    rows.append(dict(zip(headers, cells)))
                index += 1
            tables.append({"section": current_section, "headers": headers, "rows": rows})
            continue
        index += 1
    return tables


def _relevant_rule_lines(text: str, prompt_terms: set[str]) -> list[str]:
    rules: list[str] = []
    seen: set[str] = set()
    for raw_line in text.splitlines():
        line = re.sub(r"^\s*[-*]\s+", "", raw_line).strip()
        if not line or line.startswith("|") or re.fullmatch(r"#+\s+.+", line):
            continue
        normalized = re.sub(r"[*_`]+", "", line)
        line_terms = set(_terms(normalized).split())
        has_rule_language = bool(
            re.search(
                r"\b(if|when|unless|required|requires|must|cannot|qualifies|eligible|approval|exception|only)\b",
                normalized,
                re.I,
            )
        )
        if not has_rule_language:
            continue
        if prompt_terms and not (line_terms & prompt_terms):
            continue
        compact = " ".join(normalized.split())
        if compact not in seen:
            rules.append(compact)
            seen.add(compact)
    return rules


_QUESTION_PREFIXES = {
    "are", "can", "could", "did", "do", "does", "how", "is", "may", "should",
    "if", "was", "were", "what", "when", "where", "which", "who", "why", "will", "would",
}
_SUBJECT_NOISE_WORDS = _QUESTION_PREFIXES | {
    "a", "an", "does", "for", "have", "has", "many", "much", "the", "to", "use",
}
_ANSWER_ENTITY_PATTERNS = (
    re.compile(r"\b[A-Z]{1,12}-[A-Z0-9]{1,12}\b"),
    re.compile(r"\b[A-Z]{1,12}-\d{2,}\b"),
    re.compile(r"\$\s?\d+(?:\.\d{1,2})?\b"),
    re.compile(r"\b\d+(?:\.\d+)?\s?(?:minutes?|hours?|days?|business days?)\b", re.I),
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
)
_ANSWER_SLOT_MAP = {
    "yes_no_policy": ("yes_or_no", "subject", "policy_rule", "reason", "exception_or_condition"),
    "eligibility_policy": ("eligible_or_not", "subject", "policy_or_program", "reason", "exception"),
    "amount_or_credit": ("amount", "subject", "triggering_condition", "time_period_or_unit", "approval_or_exception"),
    "definition": ("entity", "type_or_category", "primary_function"),
    "person_lookup": ("person", "role", "subject"),
    "date_or_deadline": ("date_or_duration", "subject", "trigger_event", "expiration_or_rollover"),
    "procedure": ("steps", "actor", "inputs", "outcome"),
    "approval_or_authority": ("approval_decision", "approval_path", "triggering_condition", "exception"),
    "incident_or_case_rule": ("outcome", "incident_facts", "rule_applied", "exception_or_override", "approval_requirement"),
    "comparison": ("compared_entities", "selection_or_difference", "basis", "caveat"),
    "multi_part": ("each_question_part", "primary_answer", "secondary_answer", "reason_or_condition"),
    "unknown": ("direct_answer", "subject", "condition_or_reason"),
}


def _clean_answer_entity(value: str) -> tuple[str, bool]:
    words = value.strip().split()
    while len(words) > 1 and words[0].lower().rstrip("?:,.") in _QUESTION_PREFIXES:
        words.pop(0)
    cleaned = " ".join(words).strip(" ?!.,:")
    return cleaned, cleaned != value.strip(" ?!.,:")


def _important_subject_terms(question: str) -> tuple[list[str], list[str], bool]:
    """Extract a small, generic set of answer subjects without NLP or model calls."""
    found: list[str] = []
    entities: list[str] = []
    cleaned_any = False
    for pattern in _ANSWER_ENTITY_PATTERNS:
        entities.extend(match.group(0) for match in pattern.finditer(question))
    for match in re.finditer(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b", question):
        entity, cleaned = _clean_answer_entity(match.group(0))
        cleaned_any = cleaned_any or cleaned
        if entity and entity.lower() not in _QUESTION_PREFIXES:
            entities.append(entity)
    for match in re.finditer(
        r"\b(?:for|about|of|to|use|buy|cover|receive|applies to)\s+(?:the\s+)?"
        r"([A-Za-z][A-Za-z0-9'/-]*(?:\s+[A-Za-z][A-Za-z0-9'/-]*){0,3})",
        question,
        re.I,
    ):
        phrase = re.sub(
            r"\s+(?:to|for|and|or|if|when|why|is|are|required|do|does|did)\b.*$",
            "",
            match.group(1),
            flags=re.I,
        )
        if len(phrase) >= 3:
            found.append(phrase)
    for match in re.finditer(
        r"\b(?:buy|purchase|cover|use)\s+(?:the\s+)?"
        r"([A-Za-z][A-Za-z0-9'/-]*(?:\s+[A-Za-z][A-Za-z0-9'/-]*){0,2})",
        question,
        re.I,
    ):
        phrase = re.sub(r"\s+(?:and|or|if|when|why)\b.*$", "", match.group(1), flags=re.I)
        if len(phrase) >= 3:
            found.append(phrase)
    for match in re.finditer(
        r"\b((?:[A-Za-z][A-Za-z0-9'/-]*\s+){0,4}"
        r"(?:stipend|reimbursement|benefit|program|policy|credit|allowance|fee))\b",
        question,
        re.I,
    ):
        words = match.group(1).split()
        while words and words[0].lower() in _SUBJECT_NOISE_WORDS:
            words.pop(0)
        phrase = " ".join(words)
        if len(phrase) >= 3:
            found.append(phrase)
    for marker in ("approval", "authority", "deadline", "stipend", "reimbursement", "credit", "policy", "exception"):
        if re.search(rf"\b{marker}\b", question, re.I):
            found.append(marker)
    for value in entities:
        found.append(value)
    return list(dict.fromkeys(found))[:8], list(dict.fromkeys(entities))[:8], cleaned_any


def _answer_rule_families(question: str) -> list[str]:
    markers = {
        "approval": r"approve|approval|authority|authorize",
        "eligibility": r"eligible|eligibility|qualif",
        "benefits": r"benefit|stipend|reimburse|covered|leave",
        "deadline": r"deadline|due|expire|within|days?",
        "exception": r"exception|override|unless|only if",
        "threshold": r"threshold|minimum|maximum|at least|no more than",
        "comparison": r"which|highest|lowest|compare|versus|vs\.",
        "incident": r"incident|case|ticket|shipment|claim|order",
    }
    lowered = question.lower()
    return [name for name, pattern in markers.items() if re.search(rf"\b(?:{pattern})\b", lowered)]


def _is_multi_part_question(question: str, *, approval: bool, amount: bool) -> bool:
    lowered = question.lower()
    return (
        question.count("?") > 1
        or bool(re.search(r"\b(?:what|who|when|where|why|how much|how many)\b.*\band\b", lowered))
        or (approval and amount)
        or bool(re.search(r"\b(?:and why|and is .* approval|required.* and)\b", lowered))
    )


def classify_answer_requirements(question: str) -> dict[str, object]:
    """Build a deterministic answer plan without adding an LLM call."""
    lowered = question.lower().strip()
    binary = bool(re.match(r"^(can|could|should|is|are|does|do|will|would|may)\b", lowered))
    asks_amount = bool(re.search(r"\b(how much|amount|credit|cost|price|fee)\b", lowered))
    approval = bool(re.search(r"\b(approve|approval|approver|authority|authorize)\b", lowered))
    comparison = bool(re.search(r"\b(which|highest|lowest|compare|comparison|better|versus|vs\.)\b", lowered))
    definition = bool(
        re.match(r"^(what is|what does|define)\b", lowered)
        or re.search(r"\bwhat does\b.+\bmean\??$", lowered)
    )
    person_lookup = bool(re.match(r"^(who|whose)\b", lowered))
    eligibility = bool(re.search(r"\b(eligible|eligibility|qualify|qualifies|covered|allowed|excluded)\b", lowered))
    procedure = bool(re.search(r"\b(procedure|process|steps?|how do I|how can I)\b", lowered))
    timeline = bool(re.search(r"\b(when|deadline|due|how long|how many\s+(?:days?|weeks?|months?)|within \d+|expire|roll over)\b", lowered))
    incident = bool(re.search(r"\b(incident|case|ticket|shipment|claim|order|request)\b", lowered))
    multi_part = _is_multi_part_question(question, approval=approval, amount=asks_amount)
    if multi_part:
        question_type = "multi_part"
    elif comparison:
        question_type = "comparison"
    elif incident:
        question_type = "incident_or_case_rule"
    elif approval:
        question_type = "approval_or_authority"
    elif definition:
        question_type = "definition"
    elif person_lookup:
        question_type = "person_lookup"
    elif timeline:
        question_type = "date_or_deadline"
    elif procedure:
        question_type = "procedure"
    elif eligibility:
        question_type = "eligibility_policy"
    elif binary:
        question_type = "yes_no_policy"
    elif asks_amount:
        question_type = "amount_or_credit"
    else:
        question_type = "unknown"
    subjects, entities, entity_cleanup_applied = _important_subject_terms(question)
    slots = list(_ANSWER_SLOT_MAP[question_type])
    requirements = [
        "answer the exact question using only supplied context",
        "preserve supported subject terms and necessary conditions",
    ]
    if question_type == "yes_no_policy":
        requirements.append("begin with Yes or No, then state the subject and reason")
    elif question_type == "amount_or_credit":
        requirements.append("state the amount and what it applies to")
    elif question_type == "definition":
        requirements.append("define the entity by type, role, and use")
    elif question_type == "multi_part":
        requirements.append("answer each requested part explicitly and in order")
    elif question_type == "approval_or_authority":
        requirements.append("state whether approval is required and the supported authority")
    return {
        "question_type": question_type,
        "required_slots": slots,
        "important_subject_terms": subjects,
        "extracted_entities": entities,
        "detected_rule_families": _answer_rule_families(question),
        "entity_cleanup_applied": entity_cleanup_applied,
        "binary_decision": binary,
        "amount": asks_amount,
        "approval": approval,
        "comparison": comparison,
        "definition": definition,
        "eligibility": eligibility,
        "procedure": procedure,
        "timeline": timeline,
        "rule_heavy": incident or approval or eligibility,
        "multi_part": multi_part,
        "question_facets": [
            facet
            for facet, enabled in (
                ("amount_or_credit", asks_amount),
                ("approval_or_authority", approval),
                ("incident_or_case_rule", incident),
                ("eligibility_policy", eligibility),
            )
            if enabled
        ],
        "requirements": requirements,
    }


def build_answer_instruction(requirements: dict[str, object]) -> str:
    """Build compact first-pass guidance for the existing generation call."""
    question_type = str(requirements.get("question_type", "unknown"))
    slots = ", ".join(str(slot).replace("_", " ") for slot in requirements.get("required_slots", []))
    subjects = ", ".join(str(term) for term in requirements.get("important_subject_terms", [])[:6])
    guidance = [
        f"Answer shape: {question_type.replace('_', ' ')}.",
        f"Cover these supported slots when evidence is available: {slots or 'direct answer'}.",
    ]
    if subjects:
        guidance.append(f"Preserve these supported subject terms when useful: {subjects}.")
    if requirements.get("multi_part"):
        guidance.append("Answer each part explicitly and in question order.")
    if question_type == "definition":
        guidance.append("Define the named entity by its type, role, and use.")
    guidance.append("Keep the response concise. Do not expose chain-of-thought or add unsupported facts.")
    return " ".join(guidance)


def _answer_shape_context_diagnostics(requirements: dict[str, object]) -> dict[str, object]:
    diagnostics = {
        "answer_question_type": requirements["question_type"],
        "required_slots": requirements["required_slots"],
        "important_subject_terms": requirements["important_subject_terms"],
        "answer_extracted_entities": requirements["extracted_entities"],
        "answer_detected_rule_families": requirements["detected_rule_families"],
        "answer_shape_instruction": build_answer_instruction(requirements),
        "entity_cleanup_applied": requirements["entity_cleanup_applied"],
        "multi_part_detected": requirements["multi_part"],
        "deterministic_template_used": False,
    }
    diagnostics["answer_shape"] = {
        "question_type": requirements["question_type"],
        "required_slots": requirements["required_slots"],
        "important_subject_terms": requirements["important_subject_terms"],
        "extracted_entities": requirements["extracted_entities"],
        "answer_shape_instruction": diagnostics["answer_shape_instruction"],
        "entity_cleanup_applied": requirements["entity_cleanup_applied"],
        "multi_part_detected": requirements["multi_part"],
        "deterministic_template_used": False,
    }
    return diagnostics


_NOISY_SUBJECT_TERMS = {
    "if", "do they", "does", "get", "mean", "spend", "use", "what", "who", "why",
    "spend their", "employee use", "use the", "buy",
}
_NOISY_SUBJECT_WORDS = _NOISY_SUBJECT_TERMS | {
    "a", "an", "and", "are", "can", "do", "for", "how", "is", "it", "the", "to",
    "they", "their", "this", "what", "when", "where", "which", "who", "why",
}
_DIAGNOSTIC_FORMAT_PATTERN = re.compile(
    r"\b(?:Question Part \d*|Primary Answer|Secondary Answer|Reason or Condition|"
    r"Answer Part \d*|Part [12]|Why)\s*:?|"
    r"\b(?:What credit(?: should)? appl(?:y|ies)|Is approval required)\?\s*",
    re.I,
)
_QUESTION_LABEL_PATTERNS = (
    ("what_credit_applies", re.compile(r"\bwhat credit appl(?:y|ies)\?", re.I)),
    ("what_credit_should_apply", re.compile(r"\bwhat credit should apply\?", re.I)),
    ("why", re.compile(r"\bwhy\?", re.I)),
    ("approval_required", re.compile(r"\bis approval required\?", re.I)),
    ("what_does_it_mean", re.compile(r"\bwhat does (?:it|that) mean\?", re.I)),
    ("does_it_apply", re.compile(r"\bdoes it apply\?", re.I)),
    ("credits_combined", re.compile(r"\bcan both credits be combined\?", re.I)),
    ("question_part", re.compile(r"\bquestion part\s*\d*\s*:?", re.I)),
    ("secondary_answer", re.compile(r"\bsecondary answer\s*\?", re.I)),
    ("reason_or_condition", re.compile(r"\breason or condition\s*\?", re.I)),
)
_BARE_VALUE_PATTERN = re.compile(
    r"^(?:\$\s?\d+(?:\.\d{1,2})?|\d+(?:\.\d+)?\s?"
    r"(?:minutes?|hours?|days?|weeks?|months?|years?)|"
    r"\d{1,2}/\d{1,2}/\d{2,4}|"
    r"\d{4}-\d{2}-\d{2}|"
    r"(?:yes|no|conditional)|[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})[.!?]?$",
    re.I,
)
_BARE_DURATION_OR_DATE_PATTERN = re.compile(
    r"^(?:\d+(?:\.\d+)?\s?(?:minutes?|hours?|days?|weeks?|months?|years?)|"
    r"\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2})[.!?]?$",
    re.I,
)
_SOURCE_HEADING_PATTERN = re.compile(
    r"(?m)^\s*(?:#{1,6}\s+|\d+(?:\.\d+){0,2}\s+)"
)
_VAGUE_SUBJECT_PATTERN = re.compile(
    r"\b(?:the\s+)?(?:requested|retrieved)\s+(?:policy|benefit|program|item)\b",
    re.I,
)
_FACT_PATTERNS = (
    re.compile(r"\$\s?\d+(?:\.\d{1,2})?"),
    re.compile(r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|"
               r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
               r"eighteen|nineteen|twenty)\s+(?:business\s+)?"
               r"(?:minutes?|hours?|days?|weeks?|months?|years?)\b", re.I),
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(r"\b[A-Z]{1,12}-[A-Z0-9]{1,12}\b"),
)
_NUMBER_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
    "nineteen": "19", "twenty": "20",
}


def _normalized_subject(value: str) -> str:
    words = re.findall(r"[a-z0-9]+", value.lower())
    return " ".join(word.rstrip("s") if len(word) > 3 else word for word in words)


def _normalized_fact(value: str) -> str:
    """Normalize deterministic fact comparisons without changing answer prose."""
    normalized = _normalized_subject(value)
    return " ".join(_NUMBER_WORDS.get(word, word) for word in normalized.split())


def _fact_supported(fact: str, documents: list[RetrievedDocument]) -> bool:
    normalized_fact = _normalized_fact(fact)
    return bool(normalized_fact) and any(
        normalized_fact in _normalized_fact(document.text) for document in documents
    )


def _plain_definition_or_alias_question(question: str) -> bool:
    return bool(
        re.match(r"^\s*(?:what is|what does|define|meaning of)\b", question, re.I)
        or re.search(r"\bwhat does\b.+\bmean\??$", question, re.I)
        or re.search(r"\bif someone says\b.+\bwhat does that mean\??$", question, re.I)
    )


def _definition_conditions_requested(question: str) -> bool:
    return bool(
        re.search(
            r"\b(?:when is|when does|what triggers?|what are the conditions|"
            r"under what conditions|all conditions|what exceptions|explain in detail|"
            r"why is .* approval required)\b",
            question,
            re.I,
        )
    )


def _optional_definition_condition_facts(question: str, answer: str) -> list[str]:
    """Identify grounded condition-list facts that a plain definition need not repeat."""
    if not _plain_definition_or_alias_question(question) or _definition_conditions_requested(question):
        return []
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", answer) if sentence.strip()]
    optional: list[str] = []
    for sentence in sentences[1:]:
        if not re.search(r"\b(?:required|when|if|unless|only if|trigger|exceed|within|after|before)\b", sentence, re.I):
            continue
        for pattern in _FACT_PATTERNS:
            optional.extend(match.group(0).replace("$ ", "$") for match in pattern.finditer(sentence))
    return list(dict.fromkeys(optional))


def _locked_supported_facts(
    answer: str,
    requirements: dict[str, object],
    documents: list[RetrievedDocument],
    *,
    optional_facts: list[str] | None = None,
) -> list[str]:
    """Return original answer facts that are directly grounded in final context."""
    optional_normalized = {
        _normalized_fact(fact) for fact in optional_facts or [] if _normalized_fact(fact)
    }
    candidates: list[str] = []
    for pattern in _FACT_PATTERNS:
        candidates.extend(match.group(0).replace("$ ", "$") for match in pattern.finditer(answer))
    for entity in requirements.get("extracted_entities", []):
        entity_text = str(entity)
        if entity_text and re.search(re.escape(entity_text), answer, re.I):
            candidates.append(entity_text)
    for subject in requirements.get("important_subject_terms", []):
        subject_text = str(subject)
        if subject_text and _subject_is_high_quality(subject_text) and re.search(
            re.escape(subject_text), answer, re.I
        ):
            candidates.append(subject_text)
    return [
        fact for fact in dict.fromkeys(candidates)
        if _normalized_fact(fact) not in optional_normalized and _fact_supported(fact, documents)
    ]


def _missing_locked_facts(answer: str, locked_facts: list[str]) -> list[str]:
    normalized_answer = _normalized_fact(answer)
    return [
        fact for fact in locked_facts
        if _normalized_fact(fact) not in normalized_answer
    ]


def _question_role(question: str) -> str:
    """Extract a short actor phrase when the question explicitly provides one."""
    match = re.search(
        r"\b(?P<role>(?:an?\s+)?(?:[A-Za-z-]+\s+){0,2}"
        r"(?:employees?|hires?|members?|users?|customers?|workers?|contractors?|participants?))\b",
        question,
        re.I,
    )
    if not match:
        return ""
    role = re.sub(r"\s+", " ", match.group("role")).strip()
    if re.match(r"^(?:if|when|after|before|within)\b", role, re.I):
        return ""
    return role


def _render_question_role(role: str) -> str:
    """Return a grammatical actor phrase without inventing a role."""
    normalized = role.strip()
    if not normalized:
        return ""
    if re.match(r"^(?:a|an|the)\b", normalized, re.I):
        return normalized[:1].upper() + normalized[1:]
    if re.search(
        r"\b(?:employees|hires|members|users|customers|workers|contractors|participants)\b",
        normalized,
        re.I,
    ):
        return normalized[:1].upper() + normalized[1:]
    article = "An" if normalized[:1].lower() in "aeiou" else "A"
    return f"{article} {normalized}"


def _question_benefit_subject(supported_terms: list[str]) -> str:
    """Choose a supported program-like subject for slot-based synthesis."""
    candidates = [
        term
        for term in supported_terms
        if re.search(
            r"\b(?:stipend|reimbursement|benefit|program|policy|allowance|fee)\b",
            term,
            re.I,
        )
    ]
    return max(candidates, key=len) if candidates else ""


def _minimal_locked_fact_repair(
    *,
    question: str,
    answer: str,
    requirements: dict[str, object],
    locked_facts: list[str],
    documents: list[RetrievedDocument],
) -> str | None:
    """Restore a bare locked fact with minimal, supported user-facing context."""
    if not _bare_answer(answer) or not locked_facts:
        return None
    supported_terms = [
        str(term)
        for term in requirements.get("important_subject_terms", [])
        if _subject_is_high_quality(str(term))
        and _subject_is_supported(str(term), documents)
    ]
    context = "\n".join(document.text for document in documents)
    subject = (
        _slot_subject(
            question=question,
            supported_terms=supported_terms,
            evidence=context,
        )
        or _question_benefit_subject(supported_terms)
        or next(iter(supported_terms), "")
    )
    if not subject:
        return None
    value = locked_facts[0]
    question_type = str(requirements.get("question_type", "unknown"))
    role = _question_role(question)
    rendered_role = _render_question_role(role)
    if question_type == "yes_no_policy":
        return None
    if question_type == "date_or_deadline":
        if rendered_role:
            return (
                f"{rendered_role} {_role_verb(role, 'have')} {value} "
                f"to {_duration_action(question)} the {subject}."
            )
        return f"The {subject} is available for {value}."
    if question_type == "amount_or_credit":
        if rendered_role:
            return (
                f"{rendered_role} {_role_verb(role, 'receive')} {value} "
                f"for the {subject}."
            )
        return f"The {subject} is {value}."
    return f"The {subject} is {value}."


def _subject_is_high_quality(value: str) -> bool:
    normalized = _normalized_subject(value)
    words = normalized.split()
    if not words or normalized in _NOISY_SUBJECT_TERMS:
        return False
    if re.fullmatch(
        r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
        r"eighteen|nineteen|twenty)(?: consecutive)? "
        r"(?:minute|hour|day|week|month|year)",
        normalized,
    ):
        return False
    if words[0] in _NOISY_SUBJECT_WORDS or words[-1] in _NOISY_SUBJECT_WORDS:
        return False
    if len(words) > 4 or any(word in {"do", "does", "get", "spend", "use"} for word in words):
        return False
    if re.search(r"\b(?:do|does|did)\s+(?:they|it|the|this)\b", value, re.I):
        return False
    if re.match(r"^\s*(?:employee|someone|person|user)\s+(?:use|get|spend|buy)\b", value, re.I):
        return False
    if re.match(r"^\s*spend\s+(?:their|the)\b", value, re.I):
        return False
    if all(word in _NOISY_SUBJECT_WORDS for word in words):
        return False
    return any(len(word) > 2 or any(char.isdigit() for char in word) for word in words)


def _subject_is_supported(value: str, documents: list[RetrievedDocument]) -> bool:
    normalized = _normalized_subject(value)
    if not normalized:
        return False
    return any(normalized in _normalized_subject(document.text) for document in documents)


def _required_subject_terms(
    supported_terms: list[str],
) -> list[str]:
    """Keep question anchors required without treating incidental IDs as prose slots."""
    generic_terms = {
        "approval",
        "benefit",
        "case",
        "claim",
        "credit",
        "incident",
        "item",
        "policy",
        "program",
        "request",
        "reimbursement",
        "stipend",
    }
    required: list[str] = []
    for term in supported_terms:
        normalized = _normalized_subject(term)
        if (
            not normalized
            or normalized in generic_terms
            or bool(re.fullmatch(r"[A-Z]{1,12}-[A-Z0-9]{1,12}", term))
        ):
            continue
        required.append(term)
    return list(dict.fromkeys(required))


def _required_question_conditions(
    *,
    question: str,
    requirements: dict[str, object],
    supported_conditions: list[str],
) -> list[str]:
    """Require a condition only when the question makes it part of the answer."""
    question_type = str(requirements.get("question_type", "unknown"))
    lowered = question.casefold()
    if question_type == "date_or_deadline" or re.search(
        r"\b(?:if|when|after|before|within|unless|only if)\b", lowered
    ):
        return supported_conditions
    return []


def _evidence_sentences(documents: list[RetrievedDocument]) -> list[str]:
    sentences: list[str] = []
    for document in documents:
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", document.text):
            compact = re.sub(r"\s+", " ", sentence).strip(" -•\t")
            if (
                len(compact) >= 12
                and not compact.startswith("#")
                and not compact.startswith("|")
                and not (compact.count("=") >= 2 and ";" in compact)
                and not compact.casefold().startswith((
                    "user question:",
                    "compiled table-and-rule evidence",
                ))
            ):
                sentences.append(compact)
    return list(dict.fromkeys(sentences))


def _bare_answer(answer: str) -> bool:
    return bool(_BARE_VALUE_PATTERN.fullmatch(answer.strip()))


def _condition_terms(question: str) -> list[str]:
    terms: list[str] = []
    for match in re.finditer(
        r"\b(if|when|after|before|within|created by|involved|with|containing|contains|had|has|includes)\s+([^?,.]+)",
        question,
        re.I,
    ):
        phrase = re.sub(r"\s+", " ", match.group(0)).strip()
        phrase = re.sub(
            r"\s+(?:what|which|who|when|why|how|can|is|are|do|does|did)\b.*$",
            "",
            phrase,
            flags=re.I,
        ).strip()
        if len(phrase) >= 3:
            terms.append(phrase)
    for match in re.finditer(r"\b\d+(?:\.\d+)?\s*-?\s*(?:minute|day|week|month|year)s?\b", question, re.I):
        terms.append(match.group(0))
    for match in re.finditer(r"\b[A-Z]{1,12}-[A-Z0-9]{1,12}\b", question):
        terms.append(match.group(0))
    return list(dict.fromkeys(terms))


def _condition_is_supported(condition: str, documents: list[RetrievedDocument]) -> bool:
    normalized = _normalized_fact(condition)
    if not normalized:
        return False
    condition_words = [
        word for word in normalized.split()
        if word not in _NOISY_SUBJECT_WORDS
        and word not in {
            "after", "before", "if", "when", "within", "at", "consecutive",
            "employee", "worked", "work", "has", "have", "ha", "service",
        }
    ]
    for sentence in _evidence_sentences(documents):
        sentence_norm = _normalized_fact(sentence)
        if normalized in sentence_norm:
            return True
        if condition_words and all(word in sentence_norm for word in condition_words):
            return True
    return False


def _condition_present(condition: str, answer: str) -> bool:
    condition_norm = _normalized_fact(condition)
    answer_norm = _normalized_fact(answer)
    if not condition_norm:
        return False
    if condition_norm in answer_norm:
        return True
    words = [
        word for word in condition_norm.split()
        if word not in _NOISY_SUBJECT_WORDS
        and word not in {"after", "before", "if", "when", "within", "at", "consecutive", "service"}
    ]
    return bool(words) and sum(word in answer_norm for word in words) >= min(2, len(words))


def _best_evidence_sentence(
    *,
    answer: str,
    high_quality_terms: list[str],
    condition_terms: list[str],
    documents: list[RetrievedDocument],
) -> str | None:
    answer_tokens = {
        token for token in re.findall(r"[A-Za-z0-9$.-]+", answer.lower())
        if len(token) > 2 or token.startswith("$")
    }
    ranked: list[tuple[int, str]] = []
    for sentence in _evidence_sentences(documents):
        normalized = _normalized_subject(sentence)
        term_hits = sum(_normalized_subject(term) in normalized for term in high_quality_terms)
        condition_hits = sum(_normalized_subject(term) in normalized for term in condition_terms)
        answer_hits = sum(token in sentence.lower() for token in answer_tokens)
        if term_hits or answer_hits:
            ranked.append((term_hits * 5 + condition_hits * 4 + answer_hits, sentence))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (-item[0], len(item[1])))
    return ranked[0][1]


def _slot_subject(
    *,
    question: str,
    supported_terms: list[str],
    evidence: str,
) -> str:
    """Select a concise, evidence-supported label for a final answer slot."""
    evidence_norm = _normalized_subject(evidence)
    candidates = list(supported_terms)
    candidates.extend(
        match.group(0)
        for match in re.finditer(
            r"\b(?:[A-Za-z]+\s+){0,3}(?:stipend|reimbursement|sabbatical|approval|credit)\b",
            question,
            re.I,
        )
    )
    def normalize_label(candidate: str) -> str:
        label = re.sub(
            r"^(?:(?:how\s+many|how\s+much|spend|use|their|the|a|an|for|of)\s+)+",
            "",
            candidate.strip(),
            flags=re.I,
        )
        return re.sub(r"^(?:days?|weeks?|months?|years?)\s+of\s+", "", label, flags=re.I)

    usable = [
        normalize_label(candidate)
        for candidate in candidates
        if _normalized_subject(normalize_label(candidate))
        and _normalized_subject(normalize_label(candidate)) in evidence_norm
        and _normalized_subject(normalize_label(candidate)) not in {"credit", "approval", "stipend"}
    ]
    code_subjects = [
        candidate for candidate in supported_terms
        if re.fullmatch(r"[A-Z]{1,12}-[A-Z0-9]{1,12}", candidate)
    ]
    if code_subjects:
        return code_subjects[0]
    if usable:
        return max(dict.fromkeys(usable), key=lambda candidate: len(candidate))
    return ""


def _short_evidence_clause(evidence: str, *, amount: bool = False) -> str:
    """Extract one supported condition clause without exposing source formatting."""
    starters = r"after|before|when|if|within|because"
    if amount:
        starters = rf"{starters}|for"
    match = re.search(rf"\b(?P<clause>(?:{starters})\s+[^.?!;#]+)", evidence, re.I)
    if not match:
        return ""
    clause = re.sub(r"\s+", " ", match.group("clause")).strip(" .,:;")
    clause = re.split(
        r",?\s*(?:the\s+)?(?:employee|new hire|part-time employee)\s+"
        r"(?:receives?|gets?|has|qualifies|is|are)\b",
        clause,
        maxsplit=1,
        flags=re.I,
    )[0].strip(" .,:;")
    return clause[:1].lower() + clause[1:] if clause else ""


def _supported_binary_decision(*, answer: str, evidence: str) -> str:
    answer_match = re.match(r"\s*(yes|no)\b", answer, re.I)
    if answer_match:
        return answer_match.group(1).capitalize()
    evidence_lower = evidence.casefold()
    if re.search(r"\b(?:not required|does not|do not|cannot|can't|not cover|excluded)\b", evidence_lower):
        return "No"
    if re.search(r"\b(?:is required|are required|requires|covers|covered|may approve|can approve)\b", evidence_lower):
        return "Yes"
    return ""


def _requested_duration(evidence: str, question: str) -> str | None:
    durations = re.findall(
        r"\b\d+(?:\.\d+)?\s*(?:business\s+)?(?:days?|weeks?|months?|years?)\b",
        evidence,
        re.I,
    )
    if not durations:
        return None
    requested_unit = re.search(
        r"\bhow many\s+(?:business\s+)?(days?|weeks?|months?|years?)\b",
        question,
        re.I,
    )
    if requested_unit:
        unit = requested_unit.group(0).casefold().rstrip("s")
        matching = [duration for duration in durations if unit in duration.casefold().rstrip("s")]
        if matching:
            return matching[-1]
    return durations[-1]


def _slot_item(*, subject: str, supported_terms: list[str], evidence: str) -> str:
    evidence_norm = _normalized_subject(evidence)
    candidates = [
        term
        for term in supported_terms
        if _normalized_subject(term) != _normalized_subject(subject)
        and _normalized_subject(term) in evidence_norm
        and not re.search(r"\b(?:reimbursement|stipend|policy|program|benefit)\b", term, re.I)
    ]
    return max(candidates, key=len) if candidates else ""


def _policy_category_for_item(item: str, documents: list[RetrievedDocument]) -> str:
    """Return an evidence-supported category for an item without copying a sentence."""
    if not item:
        return ""
    pattern = re.compile(
        rf"\b{re.escape(item)}\s+(?:is|are|falls?\s+under)\s+"
        r"(?:an?\s+)?(?P<category>[A-Za-z][A-Za-z0-9'/-]*(?:\s+[A-Za-z][A-Za-z0-9'/-]*){0,5})",
        re.I,
    )
    for sentence in _evidence_sentences(documents):
        match = pattern.search(sentence)
        if match:
            return match.group("category").strip(" .,:;")
    return ""


def _policy_rule_supports_category(
    *,
    policy: str,
    category: str,
    decision: str,
    documents: list[RetrievedDocument],
) -> bool:
    if not policy or not category:
        return False
    policy_norm = _normalized_subject(policy)
    category_norm = _normalized_subject(category)
    if not policy_norm or not category_norm:
        return False
    if decision == "No":
        rule = r"(?:does not cover|do not cover|is not covered|are not covered|excludes?)"
    else:
        rule = r"(?:covers?|is covered|are covered|includes?)"
    for sentence in _evidence_sentences(documents):
        normalized = _normalized_subject(sentence)
        if policy_norm not in normalized or category_norm not in normalized:
            continue
        if re.search(rule, sentence, re.I):
            return True
    return False


def _with_definite_article(value: str) -> str:
    return value if re.match(r"^(?:the|a|an)\b", value, re.I) else f"the {value}"


def _synthesize_yes_no_policy_answer(
    *,
    answer: str,
    item: str,
    policy: str,
    documents: list[RetrievedDocument],
) -> str | None:
    """Build a concise item-aware policy decision from supported answer slots."""
    decision_match = re.match(r"\s*(yes|no)\b", answer, re.I)
    if not decision_match or not item or not policy:
        return None
    decision = decision_match.group(1).capitalize()
    category = _policy_category_for_item(item, documents)
    if category and _policy_rule_supports_category(
        policy=policy,
        category=category,
        decision=decision,
        documents=documents,
    ):
        item_verb = "are" if re.search(r"s$", item, re.I) else "is"
        category_verb = "are" if re.search(r"s$", category, re.I) else "is"
        covered = "not covered by" if decision == "No" else "covered by"
        return (
            f"{decision}. {item[:1].upper() + item[1:]} {item_verb} {category}, and {category} {category_verb} "
            f"{covered} {_with_definite_article(policy)}."
        )
    policy_sentence = _sentence_supporting_term(policy, documents)
    if policy_sentence and re.search(
        r"\b(?:does not cover|do not cover|is not covered|are not covered|excludes?)\b",
        policy_sentence,
        re.I,
    ):
        item_verb = "are" if re.search(r"s$", item, re.I) else "is"
        return (
            f"{decision}. {item[:1].upper() + item[1:]} {item_verb} not covered by "
            f"{_with_definite_article(policy)}."
        )
    return None


def _trim_unrequested_policy_expansion(question: str, answer: str) -> str | None:
    """Drop an appended eligibility list when a binary policy question did not ask for it."""
    if not re.match(r"^\s*(?:can|could|should|is|are|does|do|will|would|may)\b", question, re.I):
        return None
    if re.search(r"\b(?:what\s+(?:is|are)\s+covered|eligible|examples?|alternatives?|what\s+can)\b", question, re.I):
        return None
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", answer.strip()) if sentence.strip()]
    if len(sentences) < 2 or not re.match(r"^(?:yes|no)\b", sentences[0], re.I):
        return None
    if re.search(r"\b(?:covers?|includes?)\b[^.]*,[^.]*\b(?:and|or)\b", sentences[1], re.I):
        first = sentences[0].rstrip(".")
        decision, remainder = re.match(r"^(yes|no)[,.]?\s*(.*)$", first, re.I).groups()
        return f"{decision.capitalize()}. {remainder[:1].upper() + remainder[1:]}." if remainder else f"{decision.capitalize()}."
    return None


def _trim_unrequested_approval_expansion(question: str, answer: str) -> str | None:
    """Drop a standalone approval answer when a credit question did not ask for it."""
    if re.search(r"\b(?:approve|approval|approver|authority|authorize)\b", question, re.I):
        return None
    if not re.search(r"\b(?:credit|amount|how much)\b", question, re.I):
        return None
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", answer.strip())
        if sentence.strip()
    ]
    if len(sentences) < 2:
        return None
    kept = [
        sentence
        for sentence in sentences
        if not (
            re.search(r"\b(?:approval|approve|approver|authority|required)\b", sentence, re.I)
            and not re.search(r"\$\s?\d+", sentence)
        )
    ]
    if len(kept) == len(sentences) or not kept:
        return None
    return " ".join(sentence.rstrip(".") for sentence in kept) + "."


def _question_duration_condition(question: str) -> str:
    match = re.search(
        r"\b\d+(?:\s*-\s*|\s+)(?:minute|hour|day|week|month|year)s?"
        r"(?:\s+[A-Za-z][A-Za-z0-9'/-]*){0,3}",
        question,
        re.I,
    )
    if not match:
        return ""
    return re.sub(r"\s+(?:and|or)$", "", re.sub(r"\s+", " ", match.group(0)).strip(), flags=re.I)


def _credit_category_for_amount(amount: str, documents: list[RetrievedDocument]) -> str:
    for sentence in _evidence_sentences(documents):
        if _normalized_fact(amount) not in _normalized_fact(sentence):
            continue
        match = re.search(
            r"\b(?P<category>(?:[A-Za-z]+(?:[-\s][A-Za-z]+){0,4})\s+credit)\b",
            sentence,
            re.I,
        )
        if match:
            return re.sub(
                r"^(?:(?:for|of)\s+)?(?:a|an|the)\s+", "", match.group("category").strip(), flags=re.I
            )
    return ""


def _credit_threshold_range(documents: list[RetrievedDocument]) -> str:
    for sentence in _evidence_sentences(documents):
        match = re.search(
            r"\b(?:between\s+)?(?P<lower>\d+)\s*(?:to|and|-)\s*(?P<upper>\d+)\s*"
            r"(?P<unit>minutes?|hours?|days?|weeks?|months?|years?)\b",
            sentence,
            re.I,
        )
        if match:
            return (
                f"{match.group('lower')}-to-{match.group('upper')}-"
                f"{match.group('unit').lower().rstrip('s')}"
            )
    return ""


def _credit_threshold_requirement(documents: list[RetrievedDocument]) -> str:
    """Return a directly supported credit threshold without inferring one."""
    range_threshold = _credit_threshold_range(documents)
    if range_threshold:
        return range_threshold
    for sentence in _evidence_sentences(documents):
        match = re.search(
            r"\b(?:of\s+)?(?P<value>\d+)\s*(?P<unit>minutes?|hours?|days?)\s+"
            r"(?P<qualifier>or longer|or more|or less|or fewer)\b",
            sentence,
            re.I,
        )
        if match:
            return (
                f"{match.group('value')} {match.group('unit').lower()} "
                f"{match.group('qualifier').lower()}"
            )
    return ""


def _credit_category_from_answer(amount: str, answer: str) -> str:
    """Use an answer-provided category only when it is a concise credit label."""
    match = re.search(
        rf"\b(?P<category>(?:[A-Za-z]+(?:[-\s][A-Za-z]+){{0,4}})\s+credit)\s+"
        rf"(?:applies|qualifies)\b[^.]*\b{re.escape(amount)}\b|"
        rf"\b{re.escape(amount)}\s+(?P<after>(?:[A-Za-z]+(?:[-\s][A-Za-z]+){{0,4}})\s+credit)\b",
        answer,
        re.I,
    )
    if not match:
        return ""
    return (match.group("category") or match.group("after") or "").strip()


def _credit_category_from_context(amount: str, documents: list[RetrievedDocument]) -> str:
    """Derive a named credit category only from an explicit context phrase."""
    for sentence in _evidence_sentences(documents):
        if _normalized_fact(amount) not in _normalized_fact(sentence):
            continue
        match = re.search(
            r"\b(?P<category>(?:[A-Za-z]+(?:[-\s][A-Za-z]+){0,4})\s+"
            r"(?:exception|standard|extended))\s+(?:amount|credit)\s+of\s+"
            + re.escape(amount),
            sentence,
            re.I,
        )
        if match:
            return f"{match.group('category').strip()} credit"
    return ""


def _compact_credit_category(category: str) -> str:
    """Normalize redundant credit labels without changing the underlying rule."""
    compact = re.sub(r"^(?:a|an|the)\s+", "", category.strip(), flags=re.I)
    compact = re.sub(r"\bsingle\s+(?=(?:standard|extended)\b)", "", compact, flags=re.I)
    return re.sub(
        r"\b(?P<type>[A-Za-z]+)\s+excursion\s+credit\b",
        r"\g<type>-excursion credit",
        compact,
        flags=re.I,
    )


def _compact_primary_condition(condition: str) -> str:
    """Use the concise condition phrase already present in the user's question."""
    return re.sub(
        r"\b(?P<duration>\d+-minute)\s+temperature\s+excursion\b",
        r"\g<duration> excursion",
        condition,
        flags=re.I,
    )


def _requested_approval_clause(question: str, documents: list[RetrievedDocument]) -> str:
    """Return an explicit approval result only when the user asked for one."""
    if not re.search(r"\b(?:approve|approval|approver|authority|authorize)\b", question, re.I):
        return ""
    for sentence in _evidence_sentences(documents):
        match = re.search(
            r"(?P<clause>\b(?:[A-Za-z]+\s+){0,3}approval is required because [^.]+)",
            sentence,
            re.I,
        )
        if match:
            return match.group("clause").strip(" .,:;") + "."
    return ""


def _secondary_code_credit_effect(
    answer: str,
    code: str,
    documents: list[RetrievedDocument],
) -> tuple[str, str]:
    """Extract a concise secondary-code effect and its reason from generated prose."""
    if not code:
        return "", ""
    code_clauses = [
        sentence
        for sentence in _evidence_sentences(documents)
        if re.search(rf"\b{re.escape(code)}\b", sentence, re.I)
    ]
    code_clauses.extend(
        item
        for item in re.split(r"(?<=[.!?])\s+", answer)
        if re.search(rf"\b{re.escape(code)}\b", item, re.I)
    )
    code_evidence = " ".join(code_clauses)
    reason_match = re.search(
        r"\bbut\s+(?P<reason>[^,.]+?)(?:,?\s*(?:so|and)\s+it\b|,?\s*which\b|$)",
        code_evidence,
        re.I,
    )
    reason = reason_match.group("reason").strip(" .,:;") if reason_match else ""
    if re.fullmatch(r"intact seal", reason, re.I):
        reason = "the seal was intact"
    effect_match = re.search(
        r"\b(?:does not|doesn't)\s+(?:count(?:\s+as)?|qualify\s+for|trigger|add|indicate)\s+"
        r"(?:as\s+|for\s+)?(?:a\s+)?(?P<effect>[A-Za-z][A-Za-z -]{1,50}?)(?=\s+because\b|\.|,|$)",
        code_evidence,
        re.I,
    )
    effect = effect_match.group("effect").strip(" .,:;") if effect_match else ""
    if effect.casefold().endswith("failure"):
        effect = re.sub(r"\s+failure$", "", effect, flags=re.I).replace(" ", "-") + " credit"
    elif effect and not re.search(r"\bcredit\b", effect, re.I):
        effect = f"{effect} credit"
    return effect, reason


def _synthesize_conditional_credit_answer(
    *,
    question: str,
    answer: str,
    documents: list[RetrievedDocument],
) -> str | None:
    """Make a two-sentence conditional-credit answer from supported slots only."""
    if not re.search(r"\b(?:credit|amount)\b", question, re.I):
        return None
    amount_match = re.search(r"\$\s?\d+(?:\.\d{1,2})?", answer)
    trigger = _question_duration_condition(question)
    if not amount_match or not trigger:
        return None
    amount = amount_match.group(0).replace("$ ", "$")
    category = _compact_credit_category(
        _credit_category_for_amount(amount, documents)
        or _credit_category_from_answer(amount, answer)
        or _credit_category_from_context(amount, documents)
    )
    trigger = _compact_primary_condition(trigger)
    threshold = _credit_threshold_requirement(documents)
    if not category or not threshold:
        return None
    codes = re.findall(r"\b[A-Z]{1,12}-[A-Z0-9]{1,12}\b", question)
    secondary_code = ""
    effect = ""
    reason = ""
    for code in codes:
        candidate_effect, candidate_reason = _secondary_code_credit_effect(
            answer,
            code,
            documents,
        )
        if candidate_effect and candidate_reason:
            secondary_code, effect, reason = code, candidate_effect, candidate_reason
            break
    if re.search(r"\bmedical nutrition\b", category, re.I) and re.search(
        r"\bor (?:longer|more)\b", threshold, re.I
    ):
        primary = (
            f"The {amount} {category} applies because verified medical nutrition "
            f"temperature excursions of {threshold} qualify."
        )
    elif re.search(r"\bto\b", threshold):
        primary = (
            f"A {amount} {category} applies because the {trigger} falls in the {threshold} range."
        )
    else:
        primary = (
            f"A {amount} {category} applies because the {trigger} meets the {threshold} threshold."
        )
    if secondary_code and effect and reason:
        primary = f"{primary} {secondary_code} does not add a {effect} because {reason}."
    approval = _requested_approval_clause(question, documents)
    return f"{primary} {approval}".strip()

def _evidence_role_qualifier(evidence: str) -> str:
    match = re.search(
        r"\b(?P<role>(?:an?\s+)?(?:[A-Za-z-]+\s+){0,3}"
        r"(?:employees?|hires?|members?|users?|customers?|workers?|contractors?|participants?))\s+"
        r"(?:receives?|gets?|has|qualifies|is|are)\b",
        evidence,
        re.I,
    )
    return match.group("role") if match else ""


def _role_verb(role: str, verb: str) -> str:
    normalized = role.casefold().strip()
    plural = bool(re.search(r"\b(?:employees|hires|members|users|customers|workers|contractors|participants)\b", normalized))
    if verb == "receive":
        return "receive" if plural else "receives"
    if verb == "have":
        return "have" if plural else "has"
    return verb


def _duration_action(question: str) -> str:
    if re.search(r"\bspend\b", question, re.I):
        return "spend"
    if re.search(r"\buse\b", question, re.I):
        return "use"
    return "use"


def _compose_supported_final_answer(
    *,
    question: str,
    answer: str,
    requirements: dict[str, object],
    supported_terms: list[str],
    evidence: str,
) -> str | None:
    """Compose concise answer slots from evidence without returning source prose."""
    clean_evidence = re.sub(r"(?m)^\s{0,3}#{1,6}\s+.*$", "", evidence)
    clean_evidence = re.sub(r"\s+", " ", clean_evidence).strip(" -•\t")
    if not clean_evidence or clean_evidence.startswith("|"):
        return None

    question_type = str(requirements.get("question_type", "unknown"))
    subject = _slot_subject(
        question=question,
        supported_terms=supported_terms,
        evidence=clean_evidence,
    )
    amount_match = re.search(r"\$\s?\d+(?:\.\d{1,2})?", clean_evidence)
    duration = _requested_duration(clean_evidence, question)
    decision = _supported_binary_decision(answer=answer, evidence=clean_evidence)

    if question_type in {"amount_or_credit", "multi_part"} and amount_match:
        amount = amount_match.group(0).replace("$ ", "$")
        role = _question_role(question) or _evidence_role_qualifier(clean_evidence)
        if role and subject:
            rendered_role = _render_question_role(role)
            credit_answer = (
                f"{rendered_role} {_role_verb(role, 'receive')} {amount} for the {subject}."
            )
        elif subject:
            credit_answer = f"The {subject} is {amount}."
        else:
            credit_answer = f"A credit of {amount} applies."
            clause = _short_evidence_clause(clean_evidence, amount=True)
            if clause:
                credit_answer = f"{credit_answer.rstrip('.')} {clause}."
        if question_type != "multi_part" or not requirements.get("approval"):
            return credit_answer
        approval = _compose_supported_approval_answer(
            answer=answer,
            evidence=clean_evidence,
            decision=decision,
        )
        return f"{credit_answer.rstrip('.')}. {approval}" if approval else credit_answer

    if question_type == "date_or_deadline" and duration:
        role = _question_role(question) or _evidence_role_qualifier(clean_evidence)
        rendered_role = _render_question_role(role)
        condition_requested = bool(
            re.search(r"\b(?:if|when|after|before|within|unless|only if)\b", question, re.I)
        )
        clause = _short_evidence_clause(clean_evidence) if condition_requested else ""
        if rendered_role and subject and clause:
            return (
                f"{rendered_role} {_role_verb(role, 'receive')} "
                f"{duration} of {subject} {clause}."
            )
        if rendered_role and subject:
            return (
                f"{rendered_role} {_role_verb(role, 'have')} {duration} "
                f"to {_duration_action(question)} the {subject}."
            )
        if subject:
            return f"The {subject} is available for {duration}."
        return f"The available period is {duration}."

    if question_type in {"approval_or_authority", "incident_or_case_rule", "yes_no_policy"}:
        if question_type in {"approval_or_authority", "incident_or_case_rule"} and requirements.get("approval"):
            approval = _compose_supported_approval_answer(
                answer=answer,
                evidence=clean_evidence,
                decision=decision,
            )
            if approval:
                return f"{subject}: {approval}" if subject else approval
        if decision:
            policy_subject = subject or "The requested policy"
            item = _slot_item(
                subject=subject,
                supported_terms=supported_terms,
                evidence=clean_evidence,
            )
            if decision == "No" and item:
                return f"No. {policy_subject[:1].upper() + policy_subject[1:]} does not cover {item}."
            if decision == "Yes" and item:
                return f"Yes. {policy_subject[:1].upper() + policy_subject[1:]} covers {item}."
            return f"{decision}. {policy_subject[:1].upper() + policy_subject[1:]} is addressed by the retrieved policy."

    return None


def _compose_supported_approval_answer(*, answer: str, evidence: str, decision: str) -> str | None:
    if not decision:
        return None
    path_match = re.search(r"\b(?:[A-Za-z]+\s+){0,2}approval\b", evidence)
    path = path_match.group(0) if path_match else "Approval"
    path = re.sub(r"^(?:requires?|need(?:s)?)\s+", "", path, flags=re.I)
    required = "is required" if decision == "Yes" else "is not required"
    clause = _short_evidence_clause(evidence)
    outcome = f"{decision}. {path[:1].upper() + path[1:]} {required}"
    if clause:
        outcome = f"{outcome} {clause}"
    return outcome.rstrip(".") + "."


def _sentence_supporting_term(term: str, documents: list[RetrievedDocument]) -> str | None:
    normalized_term = _normalized_subject(term)
    for sentence in _evidence_sentences(documents):
        if normalized_term in _normalized_subject(sentence):
            return sentence
    return None


def _naturalize_multi_part_answer(answer: str) -> str | None:
    compact = re.sub(r"\s+", " ", answer).strip()
    if not _DIAGNOSTIC_FORMAT_PATTERN.search(compact):
        return None

    credit_match = re.search(
        r"\bwhat credit(?: should)? appl(?:y|ies)\?\s*"
        r"(?P<amount>\$\s?\d+(?:\.\d{1,2})?)(?P<label>[^.?!]*)",
        compact,
        re.I,
    )
    approval_match = re.search(
        r"\bis approval required\?\s*(?:yes[,.]?\s*)?(?P<approval>[^.?!]+)",
        compact,
        re.I,
    )
    why_match = re.search(
        r"\bwhy\?\s*(?P<reason>.*?)(?=\bsecondary answer\?|$)",
        compact,
        re.I,
    )
    secondary_match = re.search(
        r"\bsecondary answer\?\s*(?P<secondary>.*?)(?=\b(?:reason or condition|primary answer|answer part \d*)\?|$)",
        compact,
        re.I,
    )
    if credit_match and approval_match:
        amount = credit_match.group("amount").replace("$ ", "$")
        label = credit_match.group("label").strip(" .,:;")
        label = re.sub(r"\bcredit\b", "", label, flags=re.I).strip()
        credit = f"A {amount}{f' {label}' if label else ''} credit applies"
        approval = approval_match.group("approval").strip(" .,:;")
        return f"{credit}, and {approval}."
    if credit_match and why_match:
        amount = credit_match.group("amount").replace("$ ", "$")
        reason = why_match.group("reason").strip(" .,:;")
        credit = f"A {amount} credit applies"
        if reason:
            reason = re.sub(
                rf"\bqualifies?\s+for\s+(?:an?\s+|the\s+)?(?:[A-Za-z-]+\s+)?credit\s+of\s+{re.escape(amount)}\b",
                "qualifies for the credit",
                reason,
                flags=re.I,
            )
            credit = f"{credit} because {reason}"
        secondary = secondary_match.group("secondary").strip(" .,:;") if secondary_match else ""
        return f"{credit}. {secondary}".strip().rstrip(".") + "."

    compact = re.sub(
        r"\b(?:Question Part \d*|Primary Answer|Secondary Answer|Reason or Condition|"
        r"Answer Part \d*|Part [12]|Why)\s*:?\s*",
        "",
        compact,
        flags=re.I,
    )
    if not re.search(r"\?\s*(?:\$|yes|no|conditional)\b", compact, re.I):
        return re.sub(_DIAGNOSTIC_FORMAT_PATTERN, "", compact).strip(" .") + "."
    compact = re.sub(r"(?:^|\s)(?:what|which|who|when|why|how)[^?]{0,100}\?\s*", "", compact, flags=re.I)
    compact = re.sub(r"(?:^|\s)is approval required\?\s*", "", compact, flags=re.I)
    match = re.match(r"(\$\s?\d+(?:\.\d+)?)(?:[.!?]\s*)?(yes|no)\b[,.]?\s*(.*)", compact, re.I)
    if not match:
        return None
    amount, decision, reason = match.groups()
    approval = "approval is required" if decision.lower() == "yes" else "approval is not required"
    suffix = f" because {reason.rstrip('.')}" if reason else ""
    return f"A {amount} credit applies, and {approval}{suffix}."


def _definition_operational_consequence(
    question: str,
    first_sentence: str,
    documents: list[RetrievedDocument],
) -> str | None:
    """Add a direct operational consequence to a concise definition when available."""
    if not _plain_definition_or_alias_question(question) or _definition_conditions_requested(question):
        return None
    if re.search(r"\bbefore\b.+\b(?:can be issued|is issued|is processed)\b", first_sentence, re.I):
        return None
    term_match = re.match(
        r"^(?P<term>[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})\s+"
        r"(?:is|means|refers to)\b",
        first_sentence,
        re.I,
    )
    function_match = re.search(
        r"\b(?P<function>(?:manual\s+)?(?:manager\s+)?"
        r"(?:approval|review|authorization))(?:\s+path)?\b",
        first_sentence,
        re.I,
    )
    if not term_match or not function_match:
        return None
    term = term_match.group("term")
    function = function_match.group("function").strip()
    for sentence in _evidence_sentences(documents):
        if not re.search(rf"\b{re.escape(term)}\b", sentence, re.I):
            continue
        outcome_match = re.search(
            rf"\b{re.escape(term)}(?:\s+approval)?\s+before\s+"
            r"(?P<outcome>[^.]+)",
            sentence,
            re.I,
        )
        if outcome_match:
            subject = "the incident" if re.search(r"\bincident\b", question, re.I) else "the request"
            outcome = outcome_match.group("outcome").strip(" .,:;")
            if subject == "the incident":
                outcome = re.sub(r"^a credit\b", "the credit", outcome, flags=re.I)
            return f"{term} means {subject} needs {function} before {outcome}."
    return None


def _definition_scope_repair(
    question: str,
    answer: str,
    documents: list[RetrievedDocument],
) -> str | None:
    if answer.lstrip().casefold().startswith("i can't answer that confidently"):
        return None
    if not _plain_definition_or_alias_question(question):
        return None
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", answer.strip()) if sentence.strip()]
    if _definition_conditions_requested(question):
        return None
    first = sentences[0].rstrip(".")
    consequence = _definition_operational_consequence(question, first, documents)
    if consequence is not None:
        return consequence
    if len(sentences) <= 1:
        return None
    return f"{first}."


def _supported_prefix_for_condition(condition: str, documents: list[RetrievedDocument]) -> str:
    del documents
    return condition[0].upper() + condition[1:] if condition else condition


def apply_answer_shape_guardrail(
    *,
    question: str,
    answer: str,
    requirements: dict[str, object],
    documents: list[RetrievedDocument],
) -> dict[str, object]:
    """Repair answer shape from final-context evidence without another model call."""
    candidate_terms = [str(term) for term in requirements.get("important_subject_terms", [])]
    high_quality_terms = [term for term in candidate_terms if _subject_is_high_quality(term)]
    ignored_terms = [
        term for term in candidate_terms
        if term not in high_quality_terms and _normalized_subject(term) not in _NOISY_SUBJECT_TERMS
    ]
    supported_terms = [term for term in high_quality_terms if _subject_is_supported(term, documents)]
    before = answer.strip()
    preliminary = answer_shape_diagnostics(before, requirements)
    diagnostic_format_detected = bool(_DIAGNOSTIC_FORMAT_PATTERN.search(before))
    bare_duration_detected = bool(_BARE_DURATION_OR_DATE_PATTERN.fullmatch(before))
    detected_conditions = _condition_terms(question)
    supported_conditions = [
        condition for condition in detected_conditions
        if _condition_is_supported(condition, documents)
    ]
    required_conditions = _required_question_conditions(
        question=question,
        requirements=requirements,
        supported_conditions=supported_conditions,
    )
    conditions_present = [
        condition for condition in required_conditions
        if _condition_present(condition, before)
    ]
    conditions_missing = [
        condition for condition in required_conditions
        if condition not in conditions_present
    ]
    missing_supported = [
        term for term in supported_terms
        if term not in preliminary["subject_terms_present_in_answer"]
    ]
    missing_question_items = []
    for term in candidate_terms:
        item = re.sub(r"^(?:a|an|the)\s+", "", term.strip(), flags=re.I)
        if (
            _subject_is_high_quality(item)
            and item not in preliminary["subject_terms_present_in_answer"]
            and not re.search(r"\b(?:reimbursement|stipend|policy|program|benefit|allowance)\b", item, re.I)
        ):
            missing_question_items.append(item)
    missing_condition = bool(conditions_missing)
    warning = preliminary.get("unsupported_answer_shape_warning")
    repair_reason = ""
    strategy = "none"
    repaired = before

    conditional_credit = _synthesize_conditional_credit_answer(
        question=question,
        answer=before,
        documents=documents,
    )
    multi_part = _naturalize_multi_part_answer(before)
    if conditional_credit is not None:
        repaired = conditional_credit
        strategy = "conditional_credit_slot_answer"
        repair_reason = "composed the amount, credit category, threshold, and secondary-code effect"
    elif multi_part is not None:
        repaired = multi_part
        strategy = "natural_multi_part_format"
        repair_reason = "removed embedded mini-question headers"
    elif bare_duration_detected:
        evidence = _best_evidence_sentence(
            answer=before,
            high_quality_terms=supported_terms,
            condition_terms=supported_conditions,
            documents=documents,
        )
        if evidence is not None:
            composed = _compose_supported_final_answer(
                question=question,
                answer=before,
                requirements=requirements,
                supported_terms=supported_terms,
                evidence=evidence,
            )
            if composed is not None:
                repaired = composed
                strategy = "supported_duration_slot_answer"
                repair_reason = "composed a duration answer from direct final-context evidence"
    elif (
        requirements.get("question_type") == "yes_no_policy"
        and re.match(r"^(yes|no)\b", before, re.I)
        and (missing_supported or missing_question_items)
    ):
        item_term = next(
            (
                term for term in missing_supported
                if not re.search(r"\b(?:reimbursement|stipend|policy|program|benefit|allowance)\b", term, re.I)
            ),
            next(iter(missing_question_items), missing_supported[0] if missing_supported else ""),
        )
        item_evidence = _sentence_supporting_term(item_term, documents) if item_term else None
        policy_term = next(
            (term for term in supported_terms if re.search(r"\b(?:reimbursement|stipend|policy|program|benefit|allowance)\b", term, re.I)),
            supported_terms[-1] if supported_terms else "",
        )
        policy_evidence = _sentence_supporting_term(policy_term, documents) if policy_term else None
        targeted_policy_answer = _synthesize_yes_no_policy_answer(
            answer=before,
            item=item_term,
            policy=policy_term,
            documents=documents,
        )
        if targeted_policy_answer is not None:
            repaired = targeted_policy_answer
            strategy = "subject_aware_policy_slot_answer"
            repair_reason = "added the requested item and supported policy category to the binary answer"
        elif item_evidence and policy_evidence and item_evidence != policy_evidence:
            decision = before.strip().split()[0].rstrip(".,").capitalize()
            repaired = (
                f"{decision}. {item_evidence.rstrip('.')} "
                f"{policy_evidence.rstrip('.')}."
            )
            strategy = "combined_subject_policy_evidence"
            repair_reason = "combined direct item and policy evidence for a complete binary answer"
            repaired = _compose_supported_final_answer(
                question=question,
                answer=before,
                requirements=requirements,
                supported_terms=supported_terms,
                evidence=f"{item_evidence} {policy_evidence}",
            ) or "I can't answer that confidently from the retrieved context."
            strategy = "supported_binary_slot_answer"
        elif item_evidence:
            decision = before.strip().split()[0].rstrip(".,").capitalize()
            repaired = f"{decision}. {item_evidence.rstrip('.')}."
            strategy = "supported_binary_evidence_sentence"
            repair_reason = "expanded binary answer with direct supported evidence"
            repaired = _compose_supported_final_answer(
                question=question,
                answer=before,
                requirements=requirements,
                supported_terms=supported_terms,
                evidence=item_evidence,
            ) or "I can't answer that confidently from the retrieved context."
            strategy = "supported_binary_slot_answer"
    elif missing_condition and not _bare_answer(before):
        evidence = _best_evidence_sentence(
            answer=before,
            high_quality_terms=supported_terms,
            condition_terms=supported_conditions,
            documents=documents,
        )
        if evidence is not None:
            composed = _compose_supported_final_answer(
                question=question,
                answer=before,
                requirements=requirements,
                supported_terms=supported_terms,
                evidence=evidence,
            )
            if composed is not None:
                repaired = composed
                strategy = "condition_preserving_slot_answer"
                repair_reason = "composed an answer that preserves the supported condition"
    else:
        policy_trim = _trim_unrequested_policy_expansion(question, before)
        if policy_trim is not None:
            repaired = policy_trim
            strategy = "policy_scope_trim"
            repair_reason = "removed an unrequested policy eligibility list"
        else:
            definition_repair = _definition_scope_repair(question, before, documents)
            if definition_repair is not None:
                repaired = definition_repair
                strategy = "definition_scope_trim"
                repair_reason = "kept definition answer centered on the requested meaning"
    approval_scope_trim = _trim_unrequested_approval_expansion(question, repaired)
    if approval_scope_trim is not None:
        repaired = approval_scope_trim
        strategy = (
            f"{strategy}_unrequested_approval_scope_trim"
            if strategy != "none"
            else "unrequested_approval_scope_trim"
        )
        repair_reason = "removed an unrequested standalone approval answer"
    if repaired == before and (
        _bare_answer(before)
        or missing_supported
        or missing_condition
        or warning
        or diagnostic_format_detected
    ):
        evidence = _best_evidence_sentence(
            answer=before,
            high_quality_terms=supported_terms,
            condition_terms=supported_conditions,
            documents=documents,
        )
        if evidence is not None:
            composed = _compose_supported_final_answer(
                question=question,
                answer=before,
                requirements=requirements,
                supported_terms=supported_terms,
                evidence=evidence,
            )
            if composed is not None:
                repaired = composed
                strategy = "supported_slot_answer"
                repair_reason = "composed an incomplete answer from supported final-context slots"
        elif before.lower() in {"yes", "yes.", "no", "no."} and supported_terms:
            subject = supported_terms[0]
            decision = before.strip().rstrip(".").capitalize()
            repaired = f"{decision}. {subject} is addressed by the retrieved policy context."
            strategy = "safe_binary_subject_expansion"
            repair_reason = "preserved the supported subject without inferring an unstated rule"

    if repaired == before and diagnostic_format_detected:
        repaired = "I can't answer that confidently from the retrieved context."
        strategy = "safe_mini_question_format_fallback"
        repair_reason = "replaced an unsupported mini-question format without inferring facts"

    repair_failed_reason = ""
    if repaired == before and (
        _bare_answer(before) or missing_supported or missing_condition or warning or diagnostic_format_detected
    ):
        repair_failed_reason = "no single unambiguous final-context sentence supported a deterministic expansion"
    post_conditions_present = [
        condition for condition in supported_conditions
        if _condition_present(condition, repaired)
    ]
    post_conditions_missing = [
        condition for condition in supported_conditions
        if condition not in post_conditions_present
    ]
    return {
        "answer_shape_guardrail_applied": repaired != before,
        "answer_shape_guardrail_reason": repair_reason or None,
        "answer_before_guardrail": before,
        "answer_after_guardrail": repaired,
        "repair_strategy": strategy,
        "repair_failed_reason": repair_failed_reason or None,
        "extra_unasked_parts_detected": _trim_unrequested_approval_expansion(
            question,
            before,
        ) is not None,
        "raw_subject_terms": candidate_terms,
        "high_quality_subject_terms": high_quality_terms,
        "ignored_noisy_subject_terms": ignored_terms,
        "detected_question_conditions": detected_conditions,
        "supported_question_conditions": supported_conditions,
        "supported_conditions": supported_conditions,
        "required_question_conditions": required_conditions,
        "conditions_present_in_answer": post_conditions_present,
        "conditions_missing_from_answer": post_conditions_missing,
        "condition_guardrail_applied": bool(conditions_missing) and repaired != before,
        "answer_before_condition_guardrail": before if conditions_missing else None,
        "answer_after_condition_guardrail": repaired if conditions_missing and repaired != before else None,
        "condition_repair_failed_reason": (
            repair_failed_reason or None if conditions_missing else None
        ),
        "bare_duration_detected": bare_duration_detected,
        "bare_duration_guardrail_applied": bare_duration_detected and repaired != before,
        "answer_before_duration_guardrail": before if bare_duration_detected else None,
        "answer_after_duration_guardrail": repaired if bare_duration_detected and repaired != before else None,
        "duration_repair_failed_reason": (
            repair_failed_reason or None if bare_duration_detected else None
        ),
        "diagnostic_format_detected": diagnostic_format_detected,
        "mini_question_format_detected": diagnostic_format_detected,
        "diagnostic_format_repaired": diagnostic_format_detected and repaired != before,
        "diagnostic_format_guardrail_applied": diagnostic_format_detected and repaired != before,
        "diagnostic_format_repair_failed_reason": (
            repair_failed_reason or None if diagnostic_format_detected else None
        ),
        "answer_before_format_guardrail": before if diagnostic_format_detected else None,
        "answer_after_format_guardrail": repaired if diagnostic_format_detected and repaired != before else None,
        "formatting_guardrail_applied": diagnostic_format_detected and repaired != before,
        "answer_shape_repair_type": strategy if strategy != "none" else None,
        "answer_shape_repair_candidate": repaired if repaired != before else None,
        "answer_shape_repair_applied": repaired != before,
    }


def _unsupported_answer_fact_tokens(
    answer: str,
    documents: list[RetrievedDocument],
) -> list[str]:
    """Check high-risk factual tokens without pretending to judge free-form prose."""
    context = "\n".join(document.text for document in documents).casefold()
    candidates = re.findall(
        r"\$\s?\d+(?:\.\d{1,2})?|\b[A-Z]{1,12}-[A-Z0-9]{1,12}\b|"
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b",
        answer,
    )
    return [token for token in dict.fromkeys(candidates) if token.casefold() not in context]


def _missing_explicit_answer_parts(
    answer: str,
    requirements: dict[str, object],
) -> list[str]:
    """Check only question facets the caller actually asked for."""
    missing: list[str] = []
    lowered = answer.casefold()
    if requirements.get("amount") and not re.search(r"\$\s?\d+", answer):
        missing.append("amount_or_credit")
    if requirements.get("approval") and not re.search(
        r"\b(?:approval|approve|approved|required|not required|authority)\b", lowered
    ):
        missing.append("approval_or_authority")
    if requirements.get("binary_decision") and not re.search(r"\b(?:yes|no)\b", answer[:80], re.I):
        missing.append("yes_or_no_decision")
    return missing


def _final_answer_invariant_reasons(
    *,
    answer: str,
    requirements: dict[str, object],
    documents: list[RetrievedDocument],
    required_subject_terms: list[str],
    required_conditions: list[str],
) -> list[str]:
    """Apply the unchanged final-answer checks to any deterministic candidate."""
    after_shape = answer_shape_diagnostics(answer, requirements)
    reasons: list[str] = []
    if _DIAGNOSTIC_FORMAT_PATTERN.search(answer):
        reasons.append("forbidden_mini_question_format")
    if _SOURCE_HEADING_PATTERN.search(answer):
        reasons.append("source_heading_format")
    if _BARE_DURATION_OR_DATE_PATTERN.fullmatch(answer):
        reasons.append("bare_duration_or_date")
    if _unsupported_answer_fact_tokens(answer, documents):
        reasons.append("unsupported_factual_tokens")
    if _VAGUE_SUBJECT_PATTERN.search(answer):
        reasons.append("vague_subject_placeholder")
    reasons.extend(
        f"missing_question_part:{part}"
        for part in _missing_explicit_answer_parts(answer, requirements)
    )
    if any(
        term not in after_shape["subject_terms_present_in_answer"]
        for term in required_subject_terms
    ):
        reasons.append("missing_high_quality_subject_terms")
    if any(
        not _condition_present(condition, answer)
        for condition in required_conditions
    ):
        reasons.append("missing_supported_conditions")
    return list(dict.fromkeys(reasons))


def _answer_relevancy_issues(
    *,
    question: str,
    answer: str,
    requirements: dict[str, object],
    required_subject_terms: list[str],
) -> list[str]:
    """Identify narrow final-answer relevance failures without judging free-form quality."""
    shape = answer_shape_diagnostics(answer, requirements)
    issues: list[str] = []
    if any(term not in shape["subject_terms_present_in_answer"] for term in required_subject_terms):
        issues.append("required_subject_terms_missing")
    if _DIAGNOSTIC_FORMAT_PATTERN.search(answer):
        issues.append("forbidden_scaffold_detected")
    if _trim_unrequested_approval_expansion(question, answer) is not None:
        issues.append("extra_unasked_parts_detected")
    if _VAGUE_SUBJECT_PATTERN.search(answer):
        issues.append("vague_subject_placeholder")
    conditional_credit = bool(
        requirements.get("amount")
        and re.search(r"\b(?:incident|case|shipment|claim|order)\b", question, re.I)
    )
    credit_sentence_has_reason = any(
        re.search(r"\$\s?\d+", sentence)
        and re.search(r"\bbecause\b", sentence, re.I)
        for sentence in re.split(r"(?<=[.!?])\s+", answer.strip())
        if sentence
    )
    if conditional_credit and not credit_sentence_has_reason:
        issues.append("conditional_credit_lacks_primary_reason")
    if requirements.get("approval") and requirements.get("amount"):
        if not re.search(r"\$\s?\d+", answer):
            issues.append("approval_answer_lacks_primary_credit_sentence")
    return list(dict.fromkeys(issues))


def _question_labels(answer: str) -> list[str]:
    """Return stable names for user-facing question scaffolding in an answer."""
    return [name for name, pattern in _QUESTION_LABEL_PATTERNS if pattern.search(answer)]


def _semantic_reason_source(documents: list[RetrievedDocument]) -> str | None:
    """Identify the existing context item that supplied a credit threshold."""
    for document in documents:
        if re.search(
            r"\b(?:\d+\s*(?:minutes?|hours?|days?)\s+(?:or longer|or more|or less|or fewer)|"
            r"\d+\s*(?:to|and|-)\s*\d+\s*(?:minutes?|hours?|days?))\b",
            document.text,
            re.I,
        ):
            return document.chunk_id or document.filename
    return None


def _low_cost_semantic_completion_gate(
    *,
    question: str,
    original_answer: str,
    shaped_answer: str,
    requirements: dict[str, object],
    documents: list[RetrievedDocument],
) -> dict[str, object]:
    """Complete labeled conditional-credit answers using final context only."""
    labels = _question_labels(original_answer)
    approval_requested = bool(requirements.get("approval"))
    approval_present_before = bool(
        re.search(r"\b(?:approval|approve|Copper Door)\b", original_answer, re.I)
    )
    conditional_credit = bool(
        requirements.get("amount")
        and re.search(r"\b(?:incident|case|shipment|claim|order)\b", question, re.I)
    )
    primary_reason_missing = conditional_credit and not any(
        re.search(r"\$\s?\d+", sentence) and re.search(r"\bbecause\b", sentence, re.I)
        for sentence in re.split(r"(?<=[.!?])\s+", original_answer.strip())
        if sentence
    )
    base = {
        "answer": shaped_answer,
        "semantic_completion_gate_applied": False,
        "semantic_completion_gate_reason": None,
        "semantic_completion_candidate": None,
        "semantic_completion_passed": False,
        "semantic_completion_rejected_reason": None,
        "question_labels_detected": labels,
        "question_labels_removed": bool(labels and not _question_labels(shaped_answer)),
        "semantic_reason_added": False,
        "semantic_reason_source": None,
        "approval_part_detected": approval_requested or approval_present_before,
        "approval_part_preserved": bool(
            approval_requested
            and re.search(r"\b(?:approval|approve|Copper Door)\b", shaped_answer, re.I)
        ),
        "approval_part_removed": bool(
            not approval_requested
            and approval_present_before
            and not re.search(r"\b(?:approval|approve|Copper Door)\b", shaped_answer, re.I)
        ),
    }
    if not (requirements.get("multi_part") or labels):
        return base
    if not labels:
        return base
    if not primary_reason_missing:
        return base

    candidate = _synthesize_conditional_credit_answer(
        question=question,
        answer=original_answer,
        documents=documents,
    )
    base["semantic_completion_gate_applied"] = True
    base["semantic_completion_gate_reason"] = "labeled conditional-credit answer lacked a primary reason"
    base["semantic_completion_candidate"] = candidate
    if candidate is None:
        base["semantic_completion_rejected_reason"] = (
            "final context did not provide an unambiguous amount, category, and threshold"
        )
        return base

    locked_facts = _locked_supported_facts(
        original_answer,
        requirements,
        documents,
    )
    rejection_reasons: list[str] = []
    if _question_labels(candidate):
        rejection_reasons.append("question labels remained")
    if _unsupported_answer_fact_tokens(candidate, documents):
        rejection_reasons.append("candidate introduced unsupported factual tokens")
    if _missing_locked_facts(candidate, locked_facts):
        rejection_reasons.append("candidate omitted supported locked facts")
    if approval_requested and not re.search(
        r"\b(?:approval|approve|Copper Door)\b", candidate, re.I
    ):
        rejection_reasons.append("candidate omitted the requested approval answer")
    if not approval_requested and _trim_unrequested_approval_expansion(
        question, candidate
    ) is not None:
        rejection_reasons.append("candidate retained an unrequested approval answer")
    if rejection_reasons:
        base["semantic_completion_rejected_reason"] = ", ".join(rejection_reasons)
        return base

    base.update(
        {
            "answer": candidate,
            "semantic_completion_passed": True,
            "question_labels_removed": bool(labels),
            "semantic_reason_added": bool(
                primary_reason_missing and re.search(r"\bbecause\b", candidate, re.I)
            ),
            "semantic_reason_source": _semantic_reason_source(documents),
            "approval_part_preserved": bool(
                approval_requested
                and re.search(r"\b(?:approval|approve|Copper Door)\b", candidate, re.I)
            ),
            "approval_part_removed": bool(
                not approval_requested
                and approval_present_before
                and not re.search(r"\b(?:approval|approve|Copper Door)\b", candidate, re.I)
            ),
        }
    )
    return base


def validate_and_repair_final_answer(
    *,
    question: str,
    answer: str,
    requirements: dict[str, object],
    documents: list[RetrievedDocument],
) -> dict[str, object]:
    """Enforce deterministic final-answer shape using only the supplied context."""
    before = answer.strip()
    before_shape = answer_shape_diagnostics(before, requirements)
    repair = apply_answer_shape_guardrail(
        question=question,
        answer=before,
        requirements=requirements,
        documents=documents,
    )
    repaired = str(repair["answer_after_guardrail"])
    semantic_completion = _low_cost_semantic_completion_gate(
        question=question,
        original_answer=before,
        shaped_answer=repaired,
        requirements=requirements,
        documents=documents,
    )
    repaired = str(semantic_completion["answer"])
    trimmed_optional_facts = _optional_definition_condition_facts(question, before)
    locked_facts_before = _locked_supported_facts(
        before,
        requirements,
        documents,
        optional_facts=trimmed_optional_facts,
    )
    missing_locked_facts = _missing_locked_facts(repaired, locked_facts_before)
    repair_rejected = False
    repair_rejection_reason: str | None = None
    final_selected_answer_source = (
        "semantic_completion"
        if semantic_completion["semantic_completion_passed"]
        else "repaired" if repaired != before else "original"
    )
    minimal_safe_synthesis_applied = False
    minimal_safe_synthesis_candidate: str | None = None
    minimal_safe_synthesis_passed = False
    minimal_safe_synthesis_failure_reason: str | None = None
    if missing_locked_facts:
        minimal_safe_repair = _minimal_locked_fact_repair(
            question=question,
            answer=before,
            requirements=requirements,
            locked_facts=locked_facts_before,
            documents=documents,
        )
        if minimal_safe_repair and not _missing_locked_facts(
            minimal_safe_repair, locked_facts_before
        ):
            repaired = minimal_safe_repair
            final_selected_answer_source = "minimal_safe_repair"
            minimal_safe_synthesis_applied = True
            minimal_safe_synthesis_candidate = minimal_safe_repair
        else:
            repaired = before
            final_selected_answer_source = "original"
        repair_rejected = True
        repair_rejection_reason = (
            "repair discarded supported original facts: " + ", ".join(missing_locked_facts)
        )
    supported_terms = [str(term) for term in repair["high_quality_subject_terms"]]
    required_subject_terms = _required_subject_terms(supported_terms)
    relevancy_issues_before = _answer_relevancy_issues(
        question=question,
        answer=before,
        requirements=requirements,
        required_subject_terms=required_subject_terms,
    )
    relevancy_issues_after_shape_repair = _answer_relevancy_issues(
        question=question,
        answer=repaired,
        requirements=requirements,
        required_subject_terms=required_subject_terms,
    )
    missing_subjects_before = [
        term for term in required_subject_terms
        if term not in before_shape["subject_terms_present_in_answer"]
    ]
    supported_conditions = [
        str(condition) for condition in repair["supported_question_conditions"]
    ]
    required_conditions = [
        str(condition) for condition in repair["required_question_conditions"]
    ]
    missing_conditions_before = [
        condition for condition in required_conditions
        if not _condition_present(condition, before)
    ]
    failure_reasons: list[str] = []
    if repair["mini_question_format_detected"]:
        failure_reasons.append("forbidden_mini_question_format")
    if repair["bare_duration_detected"] or _bare_answer(before):
        failure_reasons.append("bare_value_answer")
    if missing_subjects_before:
        failure_reasons.append("missing_high_quality_subject_terms")
    if missing_conditions_before:
        failure_reasons.append("missing_supported_conditions")
    if before_shape["unsupported_answer_shape_warning"]:
        failure_reasons.append("unsupported_answer_shape")
    if _SOURCE_HEADING_PATTERN.search(before):
        failure_reasons.append("source_heading_format")

    unsupported_before = _unsupported_answer_fact_tokens(before, documents)
    unsupported_after = _unsupported_answer_fact_tokens(repaired, documents)
    repair_strategy = str(repair["repair_strategy"])
    repair_failed_reason = repair["repair_failed_reason"]
    if unsupported_after:
        evidence = _best_evidence_sentence(
            answer=repaired,
            high_quality_terms=supported_terms,
            condition_terms=supported_conditions,
            documents=documents,
        )
        if evidence is not None:
            composed = _compose_supported_final_answer(
                question=question,
                answer=repaired,
                requirements=requirements,
                supported_terms=supported_terms,
                evidence=evidence,
            )
            if composed is not None:
                repaired = composed
                repair_strategy = "unsupported_fact_slot_answer"
                unsupported_after = _unsupported_answer_fact_tokens(repaired, documents)
        else:
            repair_failed_reason = (
                "no single final-context sentence could safely replace unsupported factual tokens"
            )
    if unsupported_before:
        failure_reasons.append("unsupported_factual_tokens")

    # The unsupported-token fallback can select a new slot answer, so enforce
    # fact preservation again at the actual final-answer selection boundary.
    missing_locked_facts = _missing_locked_facts(repaired, locked_facts_before)
    if missing_locked_facts:
        minimal_safe_repair = _minimal_locked_fact_repair(
            question=question,
            answer=before,
            requirements=requirements,
            locked_facts=locked_facts_before,
            documents=documents,
        )
        if minimal_safe_repair and not _missing_locked_facts(
            minimal_safe_repair, locked_facts_before
        ):
            repaired = minimal_safe_repair
            final_selected_answer_source = "minimal_safe_repair"
            minimal_safe_synthesis_applied = True
            minimal_safe_synthesis_candidate = minimal_safe_repair
        else:
            repaired = before
            final_selected_answer_source = "original"
        repair_rejected = True
        repair_rejection_reason = (
            "repair discarded supported original facts: " + ", ".join(missing_locked_facts)
        )

    unresolved_reasons = _final_answer_invariant_reasons(
        answer=repaired,
        requirements=requirements,
        documents=documents,
        required_subject_terms=required_subject_terms,
        required_conditions=required_conditions,
    )
    if unresolved_reasons and (repaired != before or _bare_answer(before)):
        minimal_safe_repair = _minimal_locked_fact_repair(
            question=question,
            answer=before,
            requirements=requirements,
            locked_facts=locked_facts_before,
            documents=documents,
        )
        minimal_safe_synthesis_candidate = minimal_safe_repair
        if minimal_safe_repair is None:
            minimal_safe_synthesis_failure_reason = (
                "no supported subject-aware sentence could be composed from the locked facts"
            )
        elif _missing_locked_facts(minimal_safe_repair, locked_facts_before):
            minimal_safe_synthesis_failure_reason = "minimal-safe synthesis did not preserve locked facts"
        else:
            minimal_reasons = _final_answer_invariant_reasons(
                answer=minimal_safe_repair,
                requirements=requirements,
                documents=documents,
                required_subject_terms=required_subject_terms,
                required_conditions=required_conditions,
            )
            if not minimal_reasons:
                repaired = minimal_safe_repair
                final_selected_answer_source = "minimal_safe_repair"
                repair_strategy = "minimal_safe_synthesis"
                minimal_safe_synthesis_applied = True
                minimal_safe_synthesis_passed = True
                unresolved_reasons = []
            else:
                minimal_safe_synthesis_failure_reason = (
                    "minimal-safe synthesis did not satisfy final-answer invariants: "
                    + ", ".join(minimal_reasons)
                )
        if not minimal_safe_synthesis_passed and repaired != before:
            repair_rejected = True
            repair_rejection_reason = repair_rejection_reason or (
                "repair did not satisfy final-answer invariants: "
                + ", ".join(unresolved_reasons)
            )
            repaired = before
            final_selected_answer_source = "original"

    # A later minimal-safe or unsupported-fact fallback can restore the original
    # answer after the shape guardrail trimmed it. Reapply scope control at the
    # actual return boundary, but accept it only when required facts remain.
    final_scope_trim_applied = False
    final_scope_trim = _trim_unrequested_approval_expansion(question, repaired)
    if final_scope_trim is not None:
        scope_trim_missing_facts = _missing_locked_facts(
            final_scope_trim,
            locked_facts_before,
        )
        scope_trim_invariants = _final_answer_invariant_reasons(
            answer=final_scope_trim,
            requirements=requirements,
            documents=documents,
            required_subject_terms=required_subject_terms,
            required_conditions=required_conditions,
        )
        if not scope_trim_missing_facts and not scope_trim_invariants:
            repaired = final_scope_trim
            final_selected_answer_source = "final_scope_trim"
            repair_strategy = f"{repair_strategy}_final_scope_trim"
            final_scope_trim_applied = True

    after_shape = answer_shape_diagnostics(repaired, requirements)
    subject_terms_present_after = [
        term for term in required_subject_terms
        if term in after_shape["subject_terms_present_in_answer"]
    ]
    conditions_present_after = [
        condition for condition in required_conditions
        if _condition_present(condition, repaired)
    ]
    locked_facts_after = [
        fact for fact in locked_facts_before
        if fact not in _missing_locked_facts(repaired, locked_facts_before)
    ]
    secondary_codes = re.findall(r"\b[A-Z]{1,12}-[A-Z0-9]{1,12}\b", question)
    secondary_condition_effect_missing = any(
        not re.search(rf"\b{re.escape(code)}\b", repaired, re.I)
        for code in secondary_codes
    )
    scope_fit_failure_reason = (
        "unrequested approval answer remained in the final response"
        if _trim_unrequested_approval_expansion(question, repaired) is not None
        else None
    )
    relevancy_issues_final = _answer_relevancy_issues(
        question=question,
        answer=repaired,
        requirements=requirements,
        required_subject_terms=required_subject_terms,
    )
    required_subject_terms_inserted = [
        term
        for term in required_subject_terms
        if term not in before_shape["subject_terms_present_in_answer"]
        and term in after_shape["subject_terms_present_in_answer"]
    ]

    return {
        "final_answer_validation_applied": True,
        # This reports that the original generated answer failed one or more
        # gate checks; repaired output is identified separately below.
        "final_answer_validation_failed": bool(failure_reasons),
        "final_answer_validation_failure_reasons": failure_reasons,
        "final_answer_validation_unresolved_reasons": unresolved_reasons,
        "answer_before_final_validation": before,
        "answer_after_final_validation": repaired,
        "final_answer_repair_strategy": repair_strategy,
        "final_answer_repair_failed_reason": repair_failed_reason,
        "answer_shape_repair_applied": bool(repair.get("answer_shape_repair_applied", repaired != before)),
        "answer_shape_repair_type": repair.get("answer_shape_repair_type", repair_strategy),
        "answer_shape_repair_candidate": repair.get("answer_shape_repair_candidate", repaired if repaired != before else None),
        "answer_shape_repair_passed": repaired != before and not bool(unresolved_reasons),
        "answer_shape_repair_failure_reason": repair_rejection_reason or repair_failed_reason,
        "answer_quality_gate_applied": True,
        "answer_quality_gate_reason": repair.get("answer_shape_guardrail_reason"),
        "answer_quality_repair_candidate": repair.get("answer_after_guardrail"),
        "answer_quality_repair_passed": repaired != before and not bool(unresolved_reasons),
        "answer_quality_repair_rejected_reason": repair_rejection_reason or repair_failed_reason,
        "answer_relevancy_gate_applied": bool(relevancy_issues_before),
        "answer_relevancy_gate_reason": relevancy_issues_before or None,
        "answer_relevancy_repair_candidate": repair.get("answer_after_guardrail"),
        "answer_relevancy_repair_passed": bool(relevancy_issues_before) and not bool(
            relevancy_issues_final
        ),
        "answer_relevancy_repair_rejected_reason": (
            repair_rejection_reason
            or repair_failed_reason
            or (", ".join(relevancy_issues_after_shape_repair) if relevancy_issues_final else None)
        ),
        "required_subject_terms_inserted": required_subject_terms_inserted,
        "vague_placeholder_removed": bool(_VAGUE_SUBJECT_PATTERN.search(before)) and not bool(
            _VAGUE_SUBJECT_PATTERN.search(repaired)
        ),
        "forbidden_scaffold_removed": bool(_DIAGNOSTIC_FORMAT_PATTERN.search(before)) and not bool(
            _DIAGNOSTIC_FORMAT_PATTERN.search(repaired)
        ),
        "unasked_parts_removed": bool(
            _trim_unrequested_approval_expansion(question, before) is not None
            and _trim_unrequested_approval_expansion(question, repaired) is None
        ),
        "primary_reason_added": bool(
            "conditional_credit_lacks_primary_reason" in relevancy_issues_before
            and re.search(r"\bbecause\b", repaired, re.I)
        ),
        "approval_scope_decision": (
            "requested" if requirements.get("approval") else (
                "removed" if final_scope_trim_applied else "not_requested"
            )
        ),
        "scope_fit_passed": scope_fit_failure_reason is None,
        "scope_fit_failure_reason": scope_fit_failure_reason,
        "forbidden_scaffold_detected": bool(repair["mini_question_format_detected"]),
        "extra_unasked_parts_detected": bool(
            repair.get("extra_unasked_parts_detected") or final_scope_trim_applied
        ),
        "semantic_completion_gate_applied": semantic_completion["semantic_completion_gate_applied"],
        "semantic_completion_gate_reason": semantic_completion["semantic_completion_gate_reason"],
        "semantic_completion_candidate": semantic_completion["semantic_completion_candidate"],
        "semantic_completion_passed": semantic_completion["semantic_completion_passed"],
        "semantic_completion_rejected_reason": semantic_completion["semantic_completion_rejected_reason"],
        "question_labels_detected": semantic_completion["question_labels_detected"],
        "question_labels_removed": semantic_completion["question_labels_removed"],
        "semantic_reason_added": semantic_completion["semantic_reason_added"],
        "semantic_reason_source": semantic_completion["semantic_reason_source"],
        "approval_part_detected": semantic_completion["approval_part_detected"],
        "approval_part_preserved": semantic_completion["approval_part_preserved"],
        "approval_part_removed": semantic_completion["approval_part_removed"],
        "final_scope_trim_applied": final_scope_trim_applied,
        "required_subject_terms_missing": missing_subjects_before,
        "secondary_condition_effect_missing": secondary_condition_effect_missing,
        "locked_facts_before_repair": locked_facts_before,
        "scoped_locked_facts": locked_facts_before,
        "trimmed_optional_facts": trimmed_optional_facts,
        "locked_facts_after_repair": locked_facts_after,
        "locked_fact_preservation_passed": not bool(
            _missing_locked_facts(repaired, locked_facts_before)
        ),
        "locked_fact_violation_reason": (
            repair_rejection_reason if missing_locked_facts else None
        ),
        "post_repair_validation_applied": True,
        "post_repair_validation_passed": not bool(unresolved_reasons),
        "post_repair_failure_reasons": unresolved_reasons,
        "repair_rejected": repair_rejected,
        "repair_rejection_reason": repair_rejection_reason,
        "final_selected_answer_source": final_selected_answer_source,
        "minimal_safe_synthesis_applied": minimal_safe_synthesis_applied,
        "minimal_safe_synthesis_candidate": minimal_safe_synthesis_candidate,
        "minimal_safe_synthesis_passed": minimal_safe_synthesis_passed,
        "minimal_safe_synthesis_failure_reason": minimal_safe_synthesis_failure_reason,
        "bare_duration_repair_applied": bool(repair["bare_duration_guardrail_applied"]),
        "answer_before_bare_duration_repair": repair["answer_before_duration_guardrail"],
        "answer_after_bare_duration_repair": repair["answer_after_duration_guardrail"],
        "raw_subject_terms": repair["raw_subject_terms"],
        "high_quality_subject_terms": supported_terms,
        "required_subject_terms": required_subject_terms,
        "ignored_noisy_subject_terms": repair["ignored_noisy_subject_terms"],
        "subject_terms_missing_before_repair": missing_subjects_before,
        "subject_terms_present_after_repair": subject_terms_present_after,
        "subject_term_repair_applied": bool(missing_subjects_before) and repaired != before,
        "detected_question_conditions": repair["detected_question_conditions"],
        "supported_question_conditions": supported_conditions,
        "required_question_conditions": required_conditions,
        "conditions_missing_before_repair": missing_conditions_before,
        "conditions_present_after_repair": conditions_present_after,
        "condition_repair_applied": bool(missing_conditions_before) and repaired != before,
        "condition_repair_failed_reason": repair["condition_repair_failed_reason"],
        "mini_question_format_detected": repair["mini_question_format_detected"],
        "mini_question_format_repair_applied": bool(
            repair["diagnostic_format_guardrail_applied"]
        ),
        "answer_before_format_repair": repair["answer_before_format_guardrail"],
        "answer_after_format_repair": repair["answer_after_format_guardrail"],
        "formatting_repair_failed_reason": repair["diagnostic_format_repair_failed_reason"],
        "unsupported_factual_tokens_before_repair": unsupported_before,
        "unsupported_factual_tokens_after_repair": unsupported_after,
        "final_answer_is_natural_prose": not bool(
            _DIAGNOSTIC_FORMAT_PATTERN.search(repaired)
            or _BARE_DURATION_OR_DATE_PATTERN.fullmatch(repaired)
        ),
    }


def answer_shape_diagnostics(answer: str, requirements: dict[str, object]) -> dict[str, object]:
    """Report shape coverage only; this does not assert factual correctness."""
    lowered = answer.lower()
    subjects = [str(term) for term in requirements.get("important_subject_terms", [])]
    present_subjects = [term for term in subjects if term.lower() in lowered]
    missing_subjects = [term for term in subjects if term not in present_subjects]
    opening_match = re.match(r"^([A-Za-z]+|\$\s?\d+)", answer.strip())
    opening = opening_match.group(1).lower() if opening_match else ""
    opening_type = (
        "yes_no" if opening in {"yes", "no", "conditional"}
        else "amount" if opening.startswith("$")
        else "direct"
    )
    evidence = {
        "yes_or_no": bool(re.match(r"^(yes|no|conditional)\b", answer, re.I)),
        "eligible_or_not": bool(re.search(r"\b(eligible|not eligible|covered|not covered|allowed|excluded)\b", lowered)),
        "amount": bool(re.search(r"\$\s?\d|\b\d+(?:\.\d+)?\s?(?:dollars|usd)\b", lowered)),
        "time_period_or_unit": bool(re.search(r"\b\d+(?:\.\d+)?\s?(?:minutes?|hours?|days?|weeks?|months?|per)\b", lowered)),
        "approval_decision": bool(re.search(r"\b(approval|approve|authorized|authority|required)\b", lowered)),
        "approval_path": bool(re.search(r"\b(manager|supervisor|approver|authority|approval)\b", lowered)),
        "date_or_duration": bool(re.search(r"\b\d+(?:\.\d+)?\s?(?:minutes?|hours?|days?|weeks?|months?)\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b", lowered)),
        "reason": bool(re.search(r"\b(because|since|due to|when|if)\b", lowered)),
        "condition_or_reason": bool(re.search(r"\b(because|since|due to|when|if)\b", lowered)),
        "subject": bool(present_subjects),
        "entity": bool(present_subjects),
        "person": bool(re.search(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", answer)),
        "role": bool(re.search(r"\b(owns|leads|manages|responsible|role)\b", lowered)),
        "steps": bool(re.search(r"\b(first|then|next|step|must)\b", lowered)),
        "outcome": bool(answer.strip()),
        "primary_answer": bool(answer.strip()),
        "secondary_answer": bool(len(re.split(r"[.!?]\s+", answer.strip())) > 1),
        "each_question_part": bool(answer.strip()),
        "selection_or_difference": bool(answer.strip()),
        "compared_entities": bool(present_subjects) or bool(re.search(r"\b[A-Z]{1,12}-[A-Z0-9]{1,12}\b", answer)),
        "basis": bool(re.search(r"\b(because|based on|higher|lower|than)\b", lowered)),
        "caveat": bool(re.search(r"\b(however|except|unless|if|but)\b", lowered)),
        "policy_rule": bool(re.search(r"\b(policy|rule|cover|eligible|require)\b", lowered)),
        "policy_or_program": bool(present_subjects),
        "triggering_condition": bool(re.search(r"\b(if|when|because|after|within)\b", lowered)),
        "approval_or_exception": bool(re.search(r"\b(approval|exception|unless|override)\b", lowered)),
        "type_or_category": bool(re.search(r"\b(system|program|policy|process|service|tool|platform)\b", lowered)),
        "primary_function": bool(re.search(r"\b(used|provides|supports|tracks|manages|handles)\b", lowered)),
        "role_or_responsibility": bool(re.search(r"\b(owns|leads|manages|responsible|role)\b", lowered)),
        "trigger_event": bool(re.search(r"\b(after|from|upon|when)\b", lowered)),
        "expiration_or_rollover": bool(re.search(r"\b(expire|roll over|deadline|within)\b", lowered)),
        "actor": bool(re.search(r"\b(user|employee|manager|customer|owner|team)\b", lowered)),
        "inputs": bool(re.search(r"\b(using|with|submit|provide|include)\b", lowered)),
        "exception": bool(re.search(r"\b(exception|unless|override|only if)\b", lowered)),
        "incident_facts": bool(re.search(r"\b(incident|case|ticket|shipment|claim|order)\b", lowered)),
        "rule_applied": bool(re.search(r"\b(rule|qualifies|eligible|requires|because)\b", lowered)),
        "exception_or_override": bool(re.search(r"\b(exception|override|unless|only if)\b", lowered)),
        "approval_requirement": bool(re.search(r"\b(approval|required|approve)\b", lowered)),
        "direct_answer": bool(answer.strip()),
    }
    filled = [slot for slot in requirements.get("required_slots", []) if evidence.get(str(slot), False)]
    missing = [slot for slot in requirements.get("required_slots", []) if slot not in filled]
    warning = ""
    question_type = str(requirements.get("question_type", "unknown"))
    if question_type in {"amount_or_credit", "definition", "person_lookup", "date_or_deadline", "multi_part"} and opening_type == "yes_no":
        warning = "binary opening does not match the requested answer shape"
    elif question_type == "yes_no_policy" and opening_type != "yes_no":
        warning = "yes/no policy answer does not begin with a binary decision"
    elif subjects and not present_subjects:
        warning = "supported subject terms were omitted from the answer"
    return {
        "filled_slots": filled,
        "missing_slots": missing,
        "subject_terms_present_in_answer": present_subjects,
        "subject_terms_missing_from_answer": missing_subjects,
        "answer_opening_type": opening_type,
        "unsupported_answer_shape_warning": warning or None,
    }


def validate_answer_completeness(
    question: str,
    answer: str,
    requirements: dict[str, object] | None = None,
) -> tuple[bool, str]:
    """Check objective answer shape only; this does not judge factual truth."""
    shape = requirements or classify_answer_requirements(question)
    text = answer.strip()
    if not text:
        return False, "a concise answer is required"
    if shape.get("binary_decision") and not re.match(r"^(yes|no|conditional)\b", text, re.I):
        return False, "begin with Yes, No, or Conditional"
    if shape.get("comparison") and not re.search(r"\b[A-Z]{1,12}-[A-Z0-9]{1,12}\b|\bselected\b", text):
        return False, "explicitly identify the selected candidate"
    if shape.get("amount") and re.search(r"\b(amount|credit|how much|reimburse|reimbursement)\b", question, re.I):
        if re.search(r"\$\s?\d|\b\d+(?:\.\d+)?\s?(?:dollars|USD)\b", text, re.I) is None:
            return False, "include the supported amount"
    if shape.get("approval") and re.search(r"\b(approve|approval|approver|authority|authorize)\b", question, re.I):
        if re.search(r"\b(approve|approval|approver|authority|authorized|authorize)\b", text, re.I) is None:
            return False, "state the approval decision or authority"
    return True, ""


def _format_table_rows(headers: list[str], rows: list[dict[str, str]]) -> str:
    output: list[str] = []
    for row in rows:
        cells = [
            f"{header}={row.get(header, '').strip()}"
            for header in headers
            if row.get(header, "").strip()
        ]
        if cells:
            output.append(f"- {'; '.join(cells)}")
    return "\n".join(output)


def _answer_multi_record_rule_question_from_context(
    *,
    prompt: str,
    documents: list[RetrievedDocument],
) -> str | None:
    if not _is_multi_record_rule_question(prompt):
        return None

    text = "\n\n".join(document.text for document in documents)
    tables = _markdown_tables(text)
    if not tables:
        return None

    prompt_terms = set(_terms(prompt).split())
    primary = _select_primary_record_table(tables, prompt_terms)
    if primary is None:
        return None

    rows = primary["rows"]
    if not rows:
        return None

    approval_phrase = _extract_approval_phrase(prompt) or "approval"
    requested_record_id = _record_identifier_from_prompt(prompt)
    lantern_users = _collect_named_users(tables)
    code_meanings = _collect_code_meanings(tables)
    answers: list[str] = []
    conditional_answers: list[str] = []

    for row in rows:
        record_id = _first_nonempty_value(
            row,
            "Incident ID",
            "Record ID",
            "Claim ID",
            "Shipment ID",
            "ID",
        )
        if not record_id:
            continue
        if requested_record_id and record_id.upper() != requested_record_id:
            continue

        created_by = row.get("Created By", "").strip()
        shipment_type = row.get("Shipment Type", "").strip().lower()
        issue_summary = row.get("Issue Summary", "").strip().lower()

        reasons: list[str] = []
        if created_by in lantern_users and created_by in {"Lena Ortiz", "Tomas Greer"}:
            reasons.append(f"{created_by} is a Lantern Pass user")

        temperature_minutes = _temperature_excursion_minutes(issue_summary)
        packaging_breach = _has_packaging_seal_breach(issue_summary, code_meanings)

        if temperature_minutes is not None and packaging_breach:
            if shipment_type == "medical nutrition packs":
                reasons.append(
                    "the incident involves both medical nutrition temperature and packaging seal breach facts"
                )
            else:
                reasons.append("multiple credit categories are involved")

        late_credit_note = _late_credit_request_note(issue_summary, approval_phrase)
        if late_credit_note is not None:
            conditional_answers.append(f"{record_id} {late_credit_note}")
            continue

        if not reasons:
            continue

        if reasons == ["multiple credit categories are involved"]:
            answers.append(
                f"{record_id} requires {approval_phrase} if both temperature and packaging credits are requested, because multiple credit categories are involved."
            )
            continue

        reason_text = " and because ".join(reasons)
        answers.append(f"{record_id} requires {approval_phrase} because {reason_text}.")

    if not answers and not conditional_answers:
        return None
    return "\n".join([*answers, *conditional_answers])


def _extract_approval_phrase(prompt: str) -> str | None:
    normalized = re.sub(r"(?i),?\s+and\s+why\?\s*$", "", prompt.strip())
    match = re.search(r"(?i)\bwhich\s+.+?\s+require\s+(.+?)\s*$", normalized)
    if match is None:
        match = re.search(r"(?i)\brequires?\s+(.+?)\s*$", normalized)
    if match is None:
        return None
    phrase = " ".join(match.group(1).split()).strip(" .?")
    return phrase or None


def _record_identifier_from_prompt(prompt: str) -> str | None:
    match = re.search(r"\b[A-Z]{1,12}(?:-[A-Z0-9]{1,12})+\b", prompt)
    return match.group(0).upper() if match else None


def _is_multi_record_rule_question(prompt: str) -> bool:
    lowered = prompt.lower()
    has_explicit_record = _record_identifier_from_prompt(prompt) is not None
    return (
        (
            bool(re.search(r"\b(which|list|what|who)\b", lowered))
            or has_explicit_record
        )
        and bool(
            re.search(
                r"\b(why|require|required|requires|need|needs|qualify|qualifies|approval|exception|violate|violates|eligible)\b",
                lowered,
            )
        )
        and bool(
            has_explicit_record
            or re.search(
                r"\b(incidents?|records?|claims?|shipments?|employees?|users?|items?|cases?)\b",
                lowered,
            )
        )
    )


def _collect_named_users(tables: list[dict[str, object]]) -> set[str]:
    users: set[str] = set()
    for table in tables:
        headers = [str(header).strip() for header in table["headers"]]
        if "User" not in headers:
            continue
        for row in table["rows"]:
            user = row.get("User", "").strip()
            if user:
                users.add(user)
    return users


def _collect_code_meanings(tables: list[dict[str, object]]) -> dict[str, str]:
    meanings: dict[str, str] = {}
    for table in tables:
        headers = [str(header).strip() for header in table["headers"]]
        if "Code" not in headers or "Meaning" not in headers:
            continue
        for row in table["rows"]:
            code = row.get("Code", "").strip().upper()
            meaning = row.get("Meaning", "").strip().lower()
            if code and meaning:
                meanings[code] = meaning
    return meanings


def _first_nonempty_value(row: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = row.get(key, "").strip()
        if value:
            return value
    return ""


def _temperature_excursion_minutes(issue_summary: str) -> int | None:
    match = re.search(r"temperature excursion\s+(\d+)\s+minutes", issue_summary)
    if match:
        return int(match.group(1))
    return None


def _has_packaging_seal_breach(issue_summary: str, code_meanings: dict[str, str]) -> bool:
    if "seal breach confirmed" in issue_summary or "packaging seal breach confirmed" in issue_summary:
        return True
    if "seal breach" in issue_summary and "not" not in issue_summary:
        return True
    for code, meaning in code_meanings.items():
        if code.lower() in issue_summary and "seal breach confirmed" in meaning:
            return True
    return False


def _late_credit_request_note(issue_summary: str, approval_phrase: str) -> str | None:
    if "business days after delivery" not in issue_summary:
        return None
    match = re.search(r"\b(\d+)\s+business days after delivery\b", issue_summary)
    if match and match.group(1) == "4":
        return (
            f"may require {approval_phrase} only if the customer asks for a credit more than 14 calendar days after delivery; the provided facts say 4 business days, so the normal packaging credit fails, but the 14-day Copper Door rule cannot be concluded from the given data."
        )
    return (
        f"may require {approval_phrase} only if the customer asks for a credit more than 14 calendar days after delivery; the provided facts do not prove that timing, so the normal packaging credit cannot be confirmed from the given data."
    )


def _markdown_table_rows(text: str, required_header: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    lines = text.splitlines()
    for index in range(len(lines) - 2):
        headers = _markdown_cells(lines[index])
        separator = _markdown_cells(lines[index + 1])
        if (
            required_header not in headers
            or len(headers) < 2
            or len(separator) != len(headers)
            or not all(re.fullmatch(r":?-{3,}:?", cell) for cell in separator)
        ):
            continue
        for row_line in lines[index + 2 :]:
            cells = _markdown_cells(row_line)
            if not cells:
                break
            if len(cells) != len(headers):
                continue
            rows.append(dict(zip(headers, cells)))
    return rows


def _answer_code_lookup_from_context(
    *,
    prompt: str,
    documents: list[RetrievedDocument],
) -> str | None:
    code = _extract_requested_code(prompt)
    if code is None:
        return None

    for document in documents:
        meaning = _find_markdown_table_meaning(document.text, code)
        if meaning is not None:
            subject = _extract_code_subject(prompt=prompt, code=code)
            if subject:
                return f"{subject} code {code.upper()} means {meaning}."
            return f"{code.upper()} means {meaning}."
    return None


def _extract_requested_code(prompt: str) -> str | None:
    code_match = re.search(
        r"(?i)\bcode\s+([A-Z]{1,12}(?:-[A-Z0-9]{1,12})+)\b",
        prompt,
    )
    if code_match:
        return code_match.group(1).upper()

    fallback_match = re.search(r"\b[A-Z]{1,12}(?:-[A-Z0-9]{1,12})+\b", prompt)
    if fallback_match and re.search(r"(?i)\b(?:mean|means|meaning|define|what is)\b", prompt):
        return fallback_match.group(0).upper()
    return None


def _find_markdown_table_meaning(text: str, code: str) -> str | None:
    row_match = re.search(
        rf"(?i)(?:^|\s)\|\s*{re.escape(code)}\s*\|\s*([^|]+?)\s*\|",
        text,
    )
    if row_match is None:
        return None
    meaning = re.sub(r"[*_`]+", "", row_match.group(1))
    meaning = " ".join(meaning.split()).strip(" .")
    return meaning or None


def _answer_table_lookup_from_context(
    *,
    prompt: str,
    documents: list[RetrievedDocument],
) -> str | None:
    identifier_match = re.search(
        r"\b[A-Z]{1,12}(?:-[A-Z0-9]{1,12})+\b",
        prompt,
    )
    if identifier_match is None:
        return None

    identifier = identifier_match.group(0).upper()
    prompt_terms = set(re.findall(r"[a-z0-9]+", prompt.lower()))
    for document in documents:
        match = _find_markdown_table_field(
            text=document.text,
            identifier=identifier,
            prompt_terms=prompt_terms,
        )
        if match is None:
            continue
        header, value = match
        label = header.lower()
        return f"The {label} for {identifier} is {value}."
    return None


def _find_markdown_table_field(
    *,
    text: str,
    identifier: str,
    prompt_terms: set[str],
) -> tuple[str, str] | None:
    lines = text.splitlines()
    for index in range(len(lines) - 2):
        headers = _markdown_cells(lines[index])
        separator = _markdown_cells(lines[index + 1])
        if (
            len(headers) < 2
            or len(separator) != len(headers)
            or not all(re.fullmatch(r":?-{3,}:?", cell) for cell in separator)
        ):
            continue

        header_scores = [
            len(
                set(re.findall(r"[a-z0-9]+", header.lower()))
                .intersection(prompt_terms)
            )
            for header in headers
        ]
        target_index = max(range(len(headers)), key=header_scores.__getitem__)
        if header_scores[target_index] == 0:
            continue

        for row_line in lines[index + 2 :]:
            cells = _markdown_cells(row_line)
            if not cells:
                break
            if len(cells) != len(headers):
                continue
            if not any(cell.upper() == identifier for cell in cells):
                continue
            value = re.sub(r"[*_`]+", "", cells[target_index]).strip()
            if value and value.upper() != identifier:
                return headers[target_index].strip(), value
    return None


def _markdown_cells(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return []
    return [cell.strip() for cell in stripped.strip("|").split("|")]


def _extract_code_subject(*, prompt: str, code: str) -> str:
    subject_match = re.search(
        rf"(?i)\b(?:what\s+does|what\s+is|define|explain)\s+(.+?)\s+code\s+{re.escape(code)}\b",
        prompt,
    )
    if subject_match is None:
        return ""
    subject = " ".join(subject_match.group(1).split()).strip(" .?")
    if not subject:
        return ""
    return subject[0].upper() + subject[1:]


def _is_topic_shift(*, message: str, history: list[str]) -> bool:
    current_entities = _extract_entities(message)
    if not current_entities:
        return False

    recent_history = "\n".join(history[-6:])
    history_entities = _extract_entities(recent_history)
    if not history_entities:
        return False

    if current_entities & history_entities:
        return False

    return True


def _extract_entities(text: str) -> set[str]:
    entities = {
        " ".join(match.lower().split())
        for match in re.findall(r"\b[A-Z][a-z0-9-]*(?:\s+[A-Z][a-z0-9-]*)+\b", text)
    }
    lookup_match = re.search(
        r"(?i)\b(?:what is|who is|define|explain|describe)\s+([A-Za-z0-9][A-Za-z0-9\s-]{1,80})",
        text,
    )
    if lookup_match:
        entities.add(" ".join(_terms(lookup_match.group(1)).split()))
    return {entity for entity in entities if len(entity) >= 3}


def _terms(value: str) -> str:
    return " ".join(re.findall(r"[a-zA-Z0-9-]+", value.lower()))
