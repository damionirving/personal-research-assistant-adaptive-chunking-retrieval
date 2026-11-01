import pytest

from pra_core.agentcore.base import AgentRequest
from pra_core.agentcore.local import LocalAgentMemoryGateway, LocalAgentRuntime
from pra_core.evaluation.framework import (
    EvaluationCase,
    EvaluationRunner,
    ExpectedTermsEvaluator,
)
from pra_core.graphstores.base import GraphEdge, GraphNode
from pra_core.graphstores.local import LocalGraphStore
from pra_core.graphstores.neptune import NeptuneGraphStore
from pra_core.guardrails.local import GuardrailChain, LocalPromptGuardrail
from pra_core.memory.store import LocalMemoryStore
from pra_core.tools.registry import ToolDefinition, ToolRegistry


def test_local_agentcore_runtime_is_placeholder() -> None:
    runtime = LocalAgentRuntime()

    response = runtime.invoke(
        AgentRequest(message="hello", session_id="session-1")
    )

    assert "Local AgentCore placeholder" in response.answer
    assert response.metadata["agentcore_production"] is False


def test_local_agent_memory_gateway_uses_memory_store() -> None:
    memory_store = LocalMemoryStore()
    gateway = LocalAgentMemoryGateway(memory_store)

    gateway.save("session-1", "user", "hello")

    assert gateway.load("session-1") == ["user: hello"]


def test_local_graph_store_returns_neighbors() -> None:
    graph_store = LocalGraphStore()
    source = GraphNode(id="doc-1", label="Document", properties={"title": "Doc"})
    target = GraphNode(id="topic-1", label="Topic", properties={"name": "AI"})
    graph_store.add_node(source)
    graph_store.add_node(target)
    graph_store.add_edge(
        GraphEdge(
            source_id=source.id,
            target_id=target.id,
            relationship="MENTIONS",
            properties={},
        )
    )

    assert graph_store.neighbors("doc-1") == [target]


def test_neptune_graph_store_is_placeholder_only() -> None:
    graph_store = NeptuneGraphStore()

    with pytest.raises(NotImplementedError, match="deferred"):
        graph_store.neighbors("doc-1")


def test_tool_registry_registers_and_invokes_tools() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="echo",
            description="Echo payload.",
            handler=lambda payload: {"echo": payload["value"]},
        )
    )

    assert registry.invoke("echo", {"value": "ok"}) == {"echo": "ok"}
    assert registry.get("echo") is not None
    with pytest.raises(ValueError):
        registry.register(
            ToolDefinition(
                name="echo",
                description="Duplicate.",
                handler=lambda payload: payload,
            )
        )


def test_evaluation_runner_scores_expected_terms() -> None:
    runner = EvaluationRunner([ExpectedTermsEvaluator()])

    results = runner.run(
        [
            EvaluationCase(
                name="answer-quality",
                question="What is used?",
                answer="The local vector store is used.",
                expected_terms=("local", "vector"),
            )
        ]
    )

    assert len(results) == 1
    assert results[0].passed is True
    assert results[0].score == 1.0


def test_guardrail_chain_blocks_configured_terms() -> None:
    guardrail = GuardrailChain(
        [LocalPromptGuardrail(blocked_terms=("forbidden",))]
    )

    blocked = guardrail.validate("this contains forbidden content")
    allowed = guardrail.validate("ordinary research question")

    assert blocked.allowed is False
    assert blocked.reason == "Prompt contains blocked term: forbidden"
    assert allowed.allowed is True
