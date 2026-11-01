from pra_core.agent.workflow import ResearchAssistantWorkflow
from pra_core.config import Settings
from pra_core.documents.models import Document, DocumentChunk, RetrievedDocument
from pra_core.retrieval.hybrid import HybridRetriever
from pra_core.llm.base import LLMProvider
from pra_core.memory.store import LocalMemoryStore
from pra_core.pricing import load_bedrock_pricing_catalog
from pra_core.routing.model import DeterministicModelRouter
from pra_core.tracing.langsmith import (
    HostedLangSmithTracer,
    NoOpLangSmithTracer,
    build_langsmith_tracer,
)


class FakeLangSmithClient:
    def __init__(self) -> None:
        self.created_runs: list[dict] = []
        self.updated_runs: list[dict] = []

    def create_run(self, **kwargs) -> None:
        self.created_runs.append(kwargs)

    def update_run(self, run_id, **kwargs) -> None:
        self.updated_runs.append({"run_id": run_id, **kwargs})


class StaticRetriever:
    def __init__(self, documents: list[RetrievedDocument]) -> None:
        self._documents = documents

    def search(self, query: str, k: int = 3) -> list[RetrievedDocument]:
        return self._documents[:k]


class StaticLLMProvider(LLMProvider):
    def generate(
        self,
        prompt: str,
        context: list[RetrievedDocument],
        history: list[str],
    ) -> str:
        return "Short answer."


class StubEmbeddingProvider:
    def embed(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]


class StubVectorStore:
    def search(self, embedding: list[float], k: int = 25):
        return [
            type(
                "Result",
                (),
                {
                    "chunk": DocumentChunk(
                        id="chunk-vector",
                        document_id="doc-vector",
                        filename="vector.txt",
                        text="vector match",
                        index=0,
                    ),
                    "score": 0.9,
                },
            )(),
        ][:k]


class StubRepository:
    def list(self) -> list[Document]:
        return [
            Document(
                id="doc-bm25",
                filename="policy.txt",
                content_type="text/plain",
                text="alpha alpha beta",
            )
        ]


class FakeSecretsClient:
    def __init__(self, secret_string: str) -> None:
        self._secret_string = secret_string
        self.secret_ids: list[str] = []

    def get_secret_value(self, SecretId: str) -> dict[str, str]:
        self.secret_ids.append(SecretId)
        return {"SecretString": self._secret_string}


def test_build_langsmith_tracer_returns_local_safe_tracer_when_api_key_missing() -> None:
    tracer = build_langsmith_tracer(Settings(environment="test"))

    assert isinstance(tracer, NoOpLangSmithTracer)


def test_build_langsmith_tracer_returns_hosted_tracer_when_api_key_present() -> None:
    client = FakeLangSmithClient()
    tracer = build_langsmith_tracer(
        Settings(
            environment="test",
            langsmith_project="trace-project",
        ),
        client=client,
        secrets_client=FakeSecretsClient(
            '{"LANGSMITH_API_KEY":"test-api-key","LANGSMITH_ENDPOINT":"https://api.smith.langchain.com"}'
        ),
    )

    assert isinstance(tracer, HostedLangSmithTracer)


def test_build_langsmith_tracer_disables_reporting_for_ragas_worker() -> None:
    tracer = build_langsmith_tracer(
        Settings(
            environment="test",
            ragas_evaluation_worker=True,
        ),
        client=FakeLangSmithClient(),
        secrets_client=FakeSecretsClient(
            '{"LANGSMITH_API_KEY":"test-api-key","LANGSMITH_ENDPOINT":"https://api.smith.langchain.com"}'
        ),
    )

    assert isinstance(tracer, NoOpLangSmithTracer)


def test_build_langsmith_tracer_reads_endpoint_from_secret_payload() -> None:
    tracer = build_langsmith_tracer(
        Settings(
            environment="test",
            langsmith_project="trace-project",
        ),
        client=FakeLangSmithClient(),
        secrets_client=FakeSecretsClient(
            '{"LANGSMITH_API_KEY":"test-api-key-from-secret","LANGSMITH_ENDPOINT":"https://custom.langsmith.local"}'
        ),
    )

    assert isinstance(tracer, HostedLangSmithTracer)


def test_build_langsmith_tracer_returns_noop_for_placeholder_secret() -> None:
    tracer = build_langsmith_tracer(
        Settings(
            environment="test",
        ),
        secrets_client=FakeSecretsClient('{"LANGSMITH_API_KEY":"replace-me"}'),
    )

    assert isinstance(tracer, NoOpLangSmithTracer)


def test_noop_tracer_accepts_run_lifecycle() -> None:
    tracer = NoOpLangSmithTracer(project_name="test-project")

    run = tracer.start_run(
        name="workflow",
        trace_id="trace-1",
        inputs={"message_length": 5},
    )
    tracer.end_run(run, outputs={"answer_length": 2})

    assert run.trace_id == "trace-1"
    assert run.name == "workflow"
    assert run.run_type == "chain"


def test_hosted_tracer_creates_and_updates_runs() -> None:
    client = FakeLangSmithClient()
    tracer = HostedLangSmithTracer(
        project_name="test-project",
        service_name="personal-research-assistant",
        environment="test",
        client=client,
    )

    parent = tracer.start_run(
        name="workflow",
        trace_id="11111111-1111-1111-1111-111111111111",
        inputs={"message_length": 5},
        metadata={"workflow": "research_assistant_workflow"},
    )
    child = tracer.start_run(
        name="retrieve",
        trace_id="11111111-1111-1111-1111-111111111111",
        inputs={"top_k": 3},
        run_type="retriever",
        parent=parent,
        metadata={"workflow_step": "retrieve"},
        tags=["workflow_step:retrieve"],
    )

    tracer.end_run(child, outputs={"retrieved_document_count": 2})
    tracer.end_run(parent, outputs={"selected_model": "google.gemma-3-12b-it"})

    assert len(client.created_runs) == 2
    assert client.created_runs[0]["id"] == "11111111-1111-1111-1111-111111111111"
    assert client.created_runs[0]["trace_id"] == "11111111-1111-1111-1111-111111111111"
    assert client.created_runs[0]["dotted_order"].endswith(
        "11111111-1111-1111-1111-111111111111"
    )
    assert client.created_runs[1]["parent_run_id"] == parent.run_id
    assert client.created_runs[1]["trace_id"] == parent.langsmith_trace_id
    assert client.created_runs[1]["dotted_order"].startswith(
        client.created_runs[0]["dotted_order"] + "."
    )
    assert client.created_runs[1]["run_type"] == "retriever"
    assert len(client.updated_runs) == 2
    assert client.updated_runs[0]["run_id"] == child.run_id
    assert client.updated_runs[0]["trace_id"] == parent.langsmith_trace_id
    assert client.updated_runs[0]["dotted_order"] == child.dotted_order
    assert client.updated_runs[1]["run_id"] == parent.run_id


def test_workflow_emits_langsmith_runs_for_each_major_step() -> None:
    client = FakeLangSmithClient()
    tracer = HostedLangSmithTracer(
        project_name="test-project",
        service_name="personal-research-assistant",
        environment="test",
        client=client,
    )
    workflow = ResearchAssistantWorkflow(
        retriever=StaticRetriever(
            [
                RetrievedDocument(
                    document_id="doc-1",
                    filename="policy.md",
                    text="Remote stipend expires after 90 days.",
                    score=0.84,
                )
            ]
        ),
        memory_store=LocalMemoryStore(),
        tracer=tracer,
        llm_provider=StaticLLMProvider(),
        llm_providers={"google.gemma-3-12b-it": StaticLLMProvider()},
        model_router=DeterministicModelRouter(
            simple_model_id="google.gemma-3-12b-it",
            reasoning_model_id="mistral.mistral-large-3-675b-instruct",
        ),
    )

    workflow.run(
        message="When does the stipend expire?",
        session_id="trace-session",
        top_k=3,
    )

    created_names = [item["name"] for item in client.created_runs]
    assert "research_assistant_workflow" in created_names
    assert "load_memory" in created_names
    assert "retrieve" in created_names
    assert "route_model" in created_names
    assert "generate" in created_names
    assert "generate_llm" in created_names
    assert "score_grounding" in created_names
    assert "apply_grounding_policy" in created_names
    assert "save_memory" in created_names

    workflow_updates = [
        item for item in client.updated_runs if item["run_id"] == client.created_runs[0]["id"]
    ]
    assert workflow_updates
    assert workflow_updates[-1]["outputs"]["llm_calls"] >= 1
    assert workflow_updates[-1]["outputs"]["llm_call_count"] >= 1
    assert "input_tokens" in workflow_updates[-1]["outputs"]
    assert workflow_updates[-1]["outputs"]["input_token_count"] > 0
    assert "output_tokens" in workflow_updates[-1]["outputs"]
    assert workflow_updates[-1]["outputs"]["output_token_count"] > 0
    assert "cost_per_call_usd" in workflow_updates[-1]["outputs"]
    assert "total_cost_usd" in workflow_updates[-1]["outputs"]
    assert workflow_updates[-1]["outputs"]["throughput_requests_per_minute"] > 0
    assert workflow_updates[-1]["outputs"]["uptime_seconds"] > 0
    assert workflow_updates[-1]["outputs"]["cpu_time_seconds"] >= 0
    assert "usage_metadata" in workflow_updates[-1]["outputs"]
    assert workflow_updates[-1]["outputs"]["usage_metadata"]["total_tokens"] > 0

def test_bedrock_pricing_catalog_loads_snapshot() -> None:
    catalog = load_bedrock_pricing_catalog()

    assert catalog.version == "2026-07-07"
    assert catalog.get("mistral.mistral-large-3-675b-instruct") is not None


def test_workflow_limits_llm_context_to_top_five_documents() -> None:
    tracer = NoOpLangSmithTracer(project_name="test-project")
    documents = [
        RetrievedDocument(
            document_id=f"doc-{index}",
            filename=f"doc-{index}.md",
            text=f"Document {index}",
            score=1.0 - (index * 0.01),
        )
        for index in range(8)
    ]
    workflow = ResearchAssistantWorkflow(
        retriever=StaticRetriever(documents),
        memory_store=LocalMemoryStore(),
        tracer=tracer,
        llm_provider=StaticLLMProvider(),
        llm_providers={"google.gemma-3-12b-it": StaticLLMProvider()},
        model_router=DeterministicModelRouter(
            simple_model_id="google.gemma-3-12b-it",
            reasoning_model_id="mistral.mistral-large-3-675b-instruct",
        ),
        llm_context_top_k=5,
    )

    result = workflow.run(
        message="Tell me about the documents",
        session_id="top-five-session",
        top_k=25,
    )

    assert len(result["citations"]) == 5
    assert len(result["retrieval_scores"]) == 5


def test_workflow_emits_hybrid_retrieval_stage_latencies() -> None:
    client = FakeLangSmithClient()
    tracer = HostedLangSmithTracer(
        project_name="test-project",
        service_name="personal-research-assistant",
        environment="test",
        client=client,
    )
    workflow = ResearchAssistantWorkflow(
        retriever=HybridRetriever(
            repository=StubRepository(),
            embedding_provider=StubEmbeddingProvider(),
            vector_store=StubVectorStore(),
            bm25_top_k=25,
            vector_top_k=25,
            rrf_k=60,
            rerank_top_k=25,
        ),
        memory_store=LocalMemoryStore(),
        tracer=tracer,
        llm_provider=StaticLLMProvider(),
        llm_providers={"google.gemma-3-12b-it": StaticLLMProvider()},
        model_router=DeterministicModelRouter(
            simple_model_id="google.gemma-3-12b-it",
            reasoning_model_id="mistral.mistral-large-3-675b-instruct",
        ),
    )

    workflow.run(
        message="alpha match",
        session_id="hybrid-trace-session",
        top_k=25,
    )

    retrieve_run = next(
        item
        for item in client.updated_runs
        if item["outputs"].get("retrieved_document_count") is not None
        and item["outputs"].get("citation_count") is not None
    )
    assert "bm25_latency_ms" in retrieve_run["outputs"]
    assert "vector_latency_ms" in retrieve_run["outputs"]
    assert "rrf_merge_latency_ms" in retrieve_run["outputs"]
    assert "reranker_latency_ms" in retrieve_run["outputs"]
    assert "retrieval_latency_ms" in retrieve_run["outputs"]
