from types import SimpleNamespace
from datetime import datetime, timedelta, timezone
from io import BytesIO
import json
import math
import os
from pathlib import Path

import pytest

from fastapi.testclient import TestClient

from pra_api.main import _build_llm_routing, create_app
from pra_core.auth.cognito import CognitoAuthProvider
from pra_core.config import Settings
from pra_core.documents.chunking import TextChunker
from pra_core.documents.models import RetrievedDocument
from pra_core.embeddings.local import LocalHashEmbeddingProvider
from pra_core.evaluation.ragas import (
    FULL_RAGAS_METRICS,
    STANDARD_RAGAS_METRICS,
    RagasEvaluationSample,
    RagasEvaluationSummary,
    RagasQuestionResult,
    _result_rows,
    ragas_completion,
    sanitize_json_response,
    evaluate_ragas,
)
from pra_core.evaluation.rag_retrieval import (
    RagEvaluationDataset,
    RagEvaluationSourceDocument,
)
from pra_core.evaluation.ragas_jobs import (
    MemoryRagasEvaluationJobStore,
    RagasEvaluationJobCoordinator,
    RagasEvaluationJobRecord,
    RagasEvaluationRequest,
)
from pra_core.llm.base import LLMProvider, RetryableLLMProviderError
from pra_core.llm.fallback import RetryableFallbackLLMProvider
from pra_core.routing.model import DeterministicModelRouter


class FakeVerifier:
    def verify(self, token: str) -> dict[str, object]:
        assert token == "valid-cognito-token"
        return {"sub": "cognito-user-1", "cognito:groups": ["researcher"]}


AUTH_HEADERS = {"Authorization": "Bearer valid-cognito-token"}


@pytest.fixture(autouse=True)
def restore_ragas_worker_environment():
    worker_keys = (
        "PRA_RAGAS_EVALUATION_WORKER",
        "PRA_RAGAS_MODEL_ONLY_ID",
        "PRA_BEDROCK_SIMPLE_MODEL_ID",
        "PRA_BEDROCK_REASONING_MODEL_ID",
        "PRA_BEDROCK_REASONING_FALLBACK_MODEL_ID",
        "PRA_RAGAS_EVALUATOR_MODEL_ID",
    )
    previous = {key: os.environ.get(key) for key in worker_keys}
    yield
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def test_ragas_evaluator_returns_metric_schema_without_library_call() -> None:
    samples = [
        RagasEvaluationSample(
            id="q1",
            user_input="What is BF-C?",
            response="BF-C means seal breach confirmed.",
            retrieved_contexts=["BF-C means seal breach confirmed."],
            reference="BF-C means seal breach confirmed.",
            citations=["doc-1"],
            retrieval_details=[{"document_id": "doc-1"}],
        )
    ]

    report = evaluate_ragas(
        samples,
        metric_runner=lambda samples, config: [
            {
                "faithfulness": 1.0,
                "context_precision": 1.0,
                "context_recall": 1.0,
                "answer_relevancy": 0.9,
                "answer_correctness": 1.0,
            }
        ],
    )

    assert isinstance(report, RagasEvaluationSummary)
    assert report.completed_questions == 1
    assert report.metrics["faithfulness"] == 1.0
    assert report.metrics["citation_correctness"] == 1.0
    assert report.results[0].context_count == 1


def test_ragas_bedrock_models_use_litellm_provider(monkeypatch) -> None:
    import sys
    from types import ModuleType

    calls: dict[str, object] = {}
    litellm = ModuleType("litellm")
    litellm.completion = object()
    monkeypatch.setitem(sys.modules, "litellm", litellm)

    embeddings = ModuleType("ragas.embeddings.base")
    embeddings.embedding_factory = lambda *args, **kwargs: calls.update(
        embeddings=(args, kwargs)
    ) or "embeddings"
    embeddings.LangchainEmbeddingsWrapper = lambda value: ("wrapped", value)
    llms = ModuleType("ragas.llms")
    llms.llm_factory = lambda *args, **kwargs: calls.update(
        llm=(args, kwargs)
    ) or "llm"
    monkeypatch.setitem(sys.modules, "ragas.embeddings.base", embeddings)
    monkeypatch.setitem(sys.modules, "ragas.llms", llms)

    from pra_core.evaluation.ragas import RagasRunConfig, _build_models

    result = _build_models(
        RagasRunConfig(
            evaluator_model_id="bedrock/google.gemma-3-12b-it",
            embedding_model_id="bedrock/amazon.titan-embed-text-v2:0",
        )
    )

    assert result[0] == "llm"
    assert result[1][0] == "wrapped"
    assert result[1][1].embeddings == "embeddings"
    assert calls["llm"][0] == ("bedrock/google.gemma-3-12b-it",)
    assert calls["llm"][1]["provider"] == "litellm"
    assert calls["llm"][1]["client"] is not litellm.completion
    assert callable(calls["llm"][1]["client"])
    assert calls["llm"][1]["max_tokens"] == 768
    assert calls["llm"][1]["max_retries"] == 1
    assert calls["embeddings"] == (
        ("litellm",),
        {"model": "bedrock/amazon.titan-embed-text-v2:0"},
    )


def test_ragas_evaluator_callback_passes_provider_timeout(monkeypatch) -> None:
    import sys
    from types import ModuleType

    calls: dict[str, object] = {}
    litellm = ModuleType("litellm")
    litellm.completion = lambda *args, **kwargs: calls.update(kwargs=kwargs) or {}
    monkeypatch.setitem(sys.modules, "litellm", litellm)

    embeddings = ModuleType("ragas.embeddings.base")
    embeddings.embedding_factory = lambda *args, **kwargs: object()
    llms = ModuleType("ragas.llms")
    llms.llm_factory = lambda *args, **kwargs: kwargs["client"]
    monkeypatch.setitem(sys.modules, "ragas.embeddings.base", embeddings)
    monkeypatch.setitem(sys.modules, "ragas.llms", llms)

    from pra_core.evaluation.ragas import RagasRunConfig, _build_models

    llm, _ = _build_models(
        RagasRunConfig(
            evaluator_model_id="bedrock/google.gemma-3-12b-it",
            metric_timeout_seconds=45,
        )
    )
    llm("prompt", model="bedrock/google.gemma-3-12b-it")

    assert calls["kwargs"]["timeout"] == 45


def test_ragas_worker_routes_all_llm_calls_to_gemma() -> None:
    settings = Settings(
        environment="test",
        llm_provider="bedrock",
        ragas_evaluation_worker=True,
        bedrock_simple_model_id="google.gemma-3-12b-it",
        bedrock_reasoning_model_id="mistral.mistral-large-3-675b-instruct",
        bedrock_reasoning_fallback_model_id="arn:aws:bedrock:us-east-1::inference-profile/"
        "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        ragas_model_only_id="google.gemma-3-12b-it",
    )

    provider, providers, router = _build_llm_routing(settings)

    assert provider._model_id == "google.gemma-3-12b-it"
    assert set(providers) == {"google.gemma-3-12b-it"}
    assert router.simple_model_id == "google.gemma-3-12b-it"
    assert router.reasoning_model_id == "google.gemma-3-12b-it"

    normal_settings = Settings(
        environment="test",
        llm_provider="bedrock",
        ragas_evaluation_worker=False,
    )
    _, normal_providers, normal_router = _build_llm_routing(normal_settings)

    assert normal_router.simple_model_id == "google.gemma-3-12b-it"
    assert normal_router.reasoning_model_id == "google.gemma-4-31b"
    assert set(normal_providers) == {
        "google.gemma-3-12b-it",
        "amazon.nova-lite-v1:0",
        "google.gemma-4-31b",
        "mistral.ministral-3-14b-instruct",
    }


def test_ragas_opt_in_reuses_normal_model_routing() -> None:
    settings = Settings(
        environment="test",
        llm_provider="bedrock",
        ragas_evaluation_worker=True,
        ragas_use_normal_model_routing=True,
    )

    provider, providers, router = _build_llm_routing(settings)

    assert router.simple_model_id == settings.bedrock_simple_model_id
    assert router.reasoning_model_id == settings.bedrock_reasoning_model_id
    assert set(providers) == {
        settings.bedrock_simple_model_id,
        settings.bedrock_simple_fallback_model_id,
        settings.bedrock_reasoning_model_id,
        settings.bedrock_reasoning_fallback_model_id,
    }
    assert provider.resolved_model_id() == settings.bedrock_reasoning_model_id


class CountingRagasProvider(LLMProvider):
    def __init__(self, model_id: str, *, fail_retryably: bool = False) -> None:
        self.model_id = model_id
        self.fail_retryably = fail_retryably
        self.calls = 0

    def generate(self, prompt, context, history) -> str:
        self.calls += 1
        if self.fail_retryably:
            raise RetryableLLMProviderError("transient provider failure")
        return f"answer from {self.model_id}"

    def resolved_model_id(self) -> str:
        return self.model_id


def test_ragas_opt_in_routes_simple_and_complex_questions_without_cross_route_calls() -> None:
    gemma = CountingRagasProvider("google.gemma-3-12b-it")
    gemma4 = CountingRagasProvider("google.gemma-4-31b")
    router = DeterministicModelRouter(
        simple_model_id=gemma.model_id,
        reasoning_model_id=gemma4.model_id,
    )

    for _ in range(100):
        simple_decision = router.select_model("hello", [], [])
        complex_decision = router.select_model(
            "Which incidents require approval, and why?", [], []
        )
        assert simple_decision.selected_model == gemma.model_id
        assert complex_decision.selected_model == gemma4.model_id
        gemma.generate("question", [], [])
        gemma4.generate("question", [], [])

    assert gemma.calls == 100
    assert gemma4.calls == 100


def test_ragas_opt_in_retryable_reasoning_failure_uses_only_its_fallback() -> None:
    primary = CountingRagasProvider("google.gemma-4-31b", fail_retryably=True)
    fallback = CountingRagasProvider("mistral.ministral-3-14b-instruct")
    provider = RetryableFallbackLLMProvider(primary=primary, fallback=fallback)

    assert provider.generate("question", [], []) == "answer from mistral.ministral-3-14b-instruct"
    assert primary.calls == 1
    assert fallback.calls == 1
    assert provider.resolved_model_id() == fallback.model_id


def test_ragas_library_uses_native_dataset_without_huggingface_conversion(monkeypatch) -> None:
    import sys
    from types import ModuleType

    captured: dict[str, object] = {}

    class FakeEvaluationDataset:
        @classmethod
        def from_list(cls, rows):
            captured["dataset"] = cls()
            captured["rows"] = rows
            return captured["dataset"]

    ragas = ModuleType("ragas")

    def fake_evaluate(**kwargs):
        captured["evaluate_kwargs"] = kwargs
        return {"faithfulness": 1.0}

    ragas.evaluate = fake_evaluate
    dataset_schema = ModuleType("ragas.dataset_schema")
    dataset_schema.EvaluationDataset = FakeEvaluationDataset
    run_config = ModuleType("ragas.run_config")

    class FakeRunConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    run_config.RunConfig = FakeRunConfig
    monkeypatch.setitem(sys.modules, "ragas", ragas)
    monkeypatch.setitem(sys.modules, "ragas.dataset_schema", dataset_schema)
    monkeypatch.setitem(sys.modules, "ragas.run_config", run_config)
    monkeypatch.setattr(
        "pra_core.evaluation.ragas._build_metrics",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        "pra_core.evaluation.ragas._build_models",
        lambda config: (None, None),
    )

    from pra_core.evaluation.ragas import (
        RagasEvaluationSample,
        RagasRunConfig,
        _run_ragas_library,
    )

    scores = _run_ragas_library(
        [
            RagasEvaluationSample(
                id="q1",
                user_input="What is BF-C?",
                response="Seal breach confirmed.",
                retrieved_contexts=["Seal breach confirmed."],
            )
        ],
        config=RagasRunConfig(
            evaluator_model_id="model",
            embedding_model_id="embedding",
        ),
    )

    assert isinstance(captured["dataset"], FakeEvaluationDataset)
    assert captured["rows"] == [
        {
            "user_input": "What is BF-C?",
            "response": "Seal breach confirmed.",
            "retrieved_contexts": ["Seal breach confirmed."],
            "reference": "",
        }
    ]
    assert captured["evaluate_kwargs"]["dataset"] is captured["dataset"]
    assert captured["evaluate_kwargs"]["run_config"].max_retries == 1
    assert captured["evaluate_kwargs"]["run_config"].timeout == 120
    assert scores == [{"faithfulness": 1.0}]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ('{"statements":["a"]}', '{"statements":["a"]}'),
        ('```json\n{"statements":["a"]}\n```', '{"statements":["a"]}'),
        ('```\n{"statements":["a"]}\n```', '{"statements":["a"]}'),
        ('  \n```json\n{\n  "statements": ["a"]\n}\n```  ', '{\n  "statements": ["a"]\n}'),
        ('not JSON', 'not JSON'),
        ('```json\n{"statements":["a"]}', '{"statements":["a"]}'),
    ],
)
def test_sanitize_json_response_is_deterministic(value: str, expected: str) -> None:
    assert sanitize_json_response(value) == expected


def test_ragas_completion_sanitizes_fenced_json_without_losing_metadata(monkeypatch) -> None:
    import sys
    from types import ModuleType

    response = {
        "id": "run-1",
        "model": "google.gemma-3-12b-it",
        "usage": {"prompt_tokens": 10, "completion_tokens": 4},
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": '```json\n{"statements":["example"]}\n```',
                }
            }
        ],
    }
    litellm = ModuleType("litellm")
    litellm.completion = lambda *args, **kwargs: response
    monkeypatch.setitem(sys.modules, "litellm", litellm)

    result = ragas_completion(
        model="google.gemma-3-12b-it",
        messages=[{"role": "user", "content": "judge"}],
    )

    assert result["choices"][0]["message"]["content"] == '{"statements":["example"]}'
    assert result["usage"] == {"prompt_tokens": 10, "completion_tokens": 4}
    assert result["model"] == "google.gemma-3-12b-it"


def test_ragas_metric_completion_requires_all_finite_requested_scores() -> None:
    sample = [
        RagasEvaluationSample(
            id="q1",
            user_input="Question",
            response="Answer",
            retrieved_contexts=["Context"],
        )
    ]

    completed = evaluate_ragas(
        sample,
        metric_runner=lambda samples, config: [{
            "faithfulness": 0.8,
            "context_precision": 0.7,
            "answer_correctness": 0.6,
        }],
    )
    empty = evaluate_ragas(
        sample,
        metric_runner=lambda samples, config: [{}],
    )
    none_scores = evaluate_ragas(
        sample,
        metric_runner=lambda samples, config: [
            {"faithfulness": None, "context_precision": None, "answer_correctness": None}
        ],
    )
    nan_score = evaluate_ragas(
        sample,
        metric_runner=lambda samples, config: [{"faithfulness": math.nan}],
    )

    assert completed.completed_questions == 1
    assert empty.failed_questions == 1
    assert empty.results[0].error_category == "missing_metric_scores"
    assert none_scores.failed_questions == 1
    assert nan_score.failed_questions == 1
    assert nan_score.results[0].metrics["faithfulness"] is None


def test_ragas_missing_metric_retries_only_affected_question_and_preserves_scores() -> None:
    calls: list[list[str]] = []

    def runner(samples, config):
        calls.append([sample.id for sample in samples])
        if len(calls) == 1:
            return [
                {"faithfulness": 0.8, "answer_correctness": 0.6},
                {"faithfulness": 0.9, "context_precision": 0.8, "answer_correctness": 0.7},
            ]
        return [{"context_precision": 0.75}]

    samples = [
        RagasEvaluationSample(id="q1", user_input="q1", response="a", retrieved_contexts=["c"]),
        RagasEvaluationSample(id="q2", user_input="q2", response="a", retrieved_contexts=["c"]),
    ]
    report = evaluate_ragas(samples, metric_runner=runner)

    assert calls == [["q1", "q2"], ["q1"]]
    assert report.completed_questions == 2
    assert report.results[0].metrics["faithfulness"] == 0.8
    assert report.results[0].metrics["context_precision"] == 0.75


def test_ragas_missing_metric_after_one_recovery_fails_only_that_question() -> None:
    calls: list[list[str]] = []

    def runner(samples, config):
        calls.append([sample.id for sample in samples])
        return [{"faithfulness": 0.8, "answer_correctness": 0.6} for _ in samples]

    sample = RagasEvaluationSample(
        id="q1",
        user_input="q1",
        response="a",
        retrieved_contexts=["c"],
        citations=["doc-1"],
        retrieval_details=[{"document_id": "doc-1"}],
    )
    report = evaluate_ragas([sample], metric_runner=runner)

    assert calls == [["q1"], ["q1"]]
    assert report.completed_questions == 0
    assert report.results[0].error_category == "missing_metric_scores"
    assert "context_precision" in (report.results[0].error or "")
    assert report.results[0].metrics["faithfulness"] == 0.8
    assert report.results[0].metrics["citation_correctness"] == 1.0
    assert report.failed_metric_counts == {"context_precision": 1}


def test_ragas_low_valid_score_is_not_retried() -> None:
    calls = 0

    def runner(samples, config):
        nonlocal calls
        calls += 1
        return [{"faithfulness": 0.1, "context_precision": 0.2, "answer_correctness": 0.1}]

    report = evaluate_ragas(
        [RagasEvaluationSample(id="q1", user_input="q", response="a", retrieved_contexts=["c"])],
        metric_runner=runner,
    )

    assert calls == 1
    assert report.completed_questions == 1


def test_ragas_result_rows_rejects_missing_requested_columns() -> None:
    class FakeDataFrame:
        columns = ["user_input", "response", "reference"]

        def to_dict(self, orient):
            assert orient == "records"
            return [{"user_input": "q", "response": "a", "reference": "r"}]

    class FakeEvaluation:
        def to_pandas(self):
            return FakeDataFrame()

    with pytest.raises(ValueError, match="none of the requested metric columns"):
        _result_rows(
            FakeEvaluation(),
            1,
            metric_names=("faithfulness", "context_precision", "answer_correctness"),
        )


def test_ragas_route_runs_as_separate_async_job(monkeypatch) -> None:
    dataset_path = "data/rag_eval_dataset_v2.json"
    app = create_app(
        Settings(
            environment="test",
            cognito_issuer="https://issuer.example.test",
            cognito_client_id="client-id",
            ragas_enabled=True,
            ragas_eval_dataset_path=str(dataset_path),
        )
    )
    app.state.auth_provider = CognitoAuthProvider(
        app.state.settings,
        token_verifier=FakeVerifier(),
    )
    production_document_count = len(app.state.document_repository.list())
    production_vector_count = app.state.vector_store.count()
    app.state.ragas_coordinator._dataset_exists_checker = lambda path: True
    app.state.ragas_coordinator._dataset_loader = lambda path: RagEvaluationDataset.model_validate(
        {
            "name": "ragas-test",
            "items": [
                {
                    "id": "q1",
                    "question": "What is Orchid Ledger?",
                    "source_files": ["scenario-northstar-meridian.md"],
                    "expected_answer_summary": (
                        "Orchid Ledger is the main incident and route analytics system."
                    ),
                }
            ],
        }
    )
    app.state.ragas_coordinator._evaluation_workflow_factory = lambda retriever: SimpleNamespace(
        run=lambda **kwargs: {
            "answer": "BF-C means seal breach confirmed.",
            "ragas_contexts": ["Orchid Ledger is the main incident and route analytics system."],
            "citations": ["doc-1"],
            "retrieval_details": [{"document_id": "doc-1"}],
        }
    )
    captured_samples = []

    def fake_evaluate(samples, config):
        captured_samples.extend(samples)
        metric_name = config.metric_names[0]
        return RagasEvaluationSummary(
            total_questions=len(samples),
            completed_questions=len(samples),
            failed_questions=0,
            metrics={metric_name: 1.0},
            results=[
                RagasQuestionResult(
                    id=sample.id,
                    question=sample.user_input,
                    answer=sample.response,
                    metrics={metric_name: 1.0},
                )
                for sample in samples
            ],
        )

    monkeypatch.setattr(
        "pra_core.evaluation.ragas_jobs.evaluate_ragas",
        fake_evaluate,
    )

    response = TestClient(app).post(
        "/rag/evaluate/ragas",
        headers=AUTH_HEADERS,
        json={"dataset": "default", "top_k": 25, "max_questions": 1},
    )

    assert response.status_code == 202
    payload = response.json()
    assert payload["status"] == "running"

    for _ in range(len(FULL_RAGAS_METRICS)):
        record = app.state.ragas_coordinator.execute(payload["job_id"])

    assert record.status == "completed"
    assert record.summary.metrics == {name: 1.0 for name in FULL_RAGAS_METRICS}
    assert len(app.state.document_repository.list()) == production_document_count
    assert app.state.vector_store.count() == production_vector_count
    assert captured_samples[0].retrieved_contexts
    assert captured_samples[0].retrieval_details[0]["document_id"] == "doc-1"
    assert isinstance(
        app.state.ragas_coordinator._evaluation_embedding_provider,
        LocalHashEmbeddingProvider,
    )


def test_ragas_missing_source_fails_before_evaluation_workflow(tmp_path: Path) -> None:
    store = MemoryRagasEvaluationJobStore()
    dataset = RagEvaluationDataset.model_validate(
        {
            "name": "missing-source",
            "items": [
                {
                    "id": "q1",
                    "question": "Question",
                    "source_files": ["missing.md"],
                }
            ],
        }
    )
    workflow_called = False

    def workflow_factory(_retriever):
        nonlocal workflow_called
        workflow_called = True
        raise AssertionError("workflow must not be built when source files are missing")

    coordinator = RagasEvaluationJobCoordinator(
        job_store=store,
        workflow=SimpleNamespace(),
        chunker=TextChunker(chunk_size=120, overlap=20),
        evaluation_embedding_provider=LocalHashEmbeddingProvider(dimensions=8),
        evaluation_docs_root=tmp_path,
        evaluation_workflow_factory=workflow_factory,
        dataset_loader=lambda _path: dataset,
        dataset_exists_checker=lambda _path: True,
    )
    store.put(
        RagasEvaluationJobRecord(
            job_id="missing-source-job",
            status="queued",
            request=RagasEvaluationRequest(max_questions=1),
            principal="user-1",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            dataset_path="missing.json",
        )
    )

    result = coordinator.execute("missing-source-job")

    assert result.status == "failed"
    assert "source documents" in (result.error or "")
    assert workflow_called is False


def test_ragas_route_can_be_disabled() -> None:
    app = create_app(
        Settings(
            environment="test",
            cognito_issuer="https://issuer.example.test",
            cognito_client_id="client-id",
            ragas_enabled=False,
            ragas_eval_dataset_path="data/rag_eval_dataset_v2.json",
        )
    )
    app.state.auth_provider = CognitoAuthProvider(
        app.state.settings,
        token_verifier=FakeVerifier(),
    )

    response = TestClient(app).post(
        "/rag/evaluate/ragas",
        headers=AUTH_HEADERS,
        json={"dataset": "default"},
    )

    assert response.status_code == 503
    assert "disabled" in response.json()["error"]


def test_ragas_job_heartbeat_tracks_progress() -> None:
    store = MemoryRagasEvaluationJobStore()
    record = RagasEvaluationJobRecord(
        job_id="job-1",
        status="running",
        request=RagasEvaluationRequest(max_questions=2),
        principal="user-1",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        dataset_path="data/ragas.json",
    )
    store.put(record)
    coordinator = RagasEvaluationJobCoordinator(job_store=store, workflow=SimpleNamespace())

    coordinator._heartbeat("job-1", completed=1, total=2)

    updated = store.get("job-1")
    assert updated is not None
    assert updated.completed_questions == 1
    assert updated.total_questions == 2
    assert updated.updated_at >= record.updated_at


def test_ragas_cancellation_is_idempotent_and_worker_skips_work() -> None:
    store = MemoryRagasEvaluationJobStore()
    record = RagasEvaluationJobRecord(
        job_id="cancel-job",
        status="queued",
        request=RagasEvaluationRequest(max_questions=2),
        principal="user-1",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        dataset_path="data/ragas.json",
    )
    store.put(record)
    coordinator = RagasEvaluationJobCoordinator(job_store=store, workflow=SimpleNamespace())

    requested = coordinator.cancel("cancel-job")
    repeated = coordinator.cancel("cancel-job")
    result = coordinator.execute("cancel-job")

    assert requested is not None
    assert requested.status == "cancel_requested"
    assert repeated is not None
    assert repeated.status == "cancel_requested"
    assert result.status == "cancelled"
    assert store.get("cancel-job").status == "cancelled"


def test_ragas_cancel_route_checks_owner_and_returns_cancelled_state() -> None:
    app = create_app(
        Settings(
            environment="test",
            cognito_issuer="https://issuer.example.test",
            cognito_client_id="client-id",
            ragas_enabled=True,
            ragas_eval_dataset_path="data/rag_eval_dataset_v2.json",
        )
    )
    app.state.auth_provider = CognitoAuthProvider(app.state.settings, token_verifier=FakeVerifier())
    record = RagasEvaluationJobRecord(
        job_id="route-cancel-job",
        status="queued",
        request=RagasEvaluationRequest(),
        principal="cognito-user-1",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        dataset_path="data/ragas.json",
    )
    app.state.ragas_coordinator._job_store.put(record)

    response = TestClient(app).post(
        "/rag/evaluate/ragas/route-cancel-job/cancel",
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200
    assert response.json()["status"] == "cancel_requested"


def test_ragas_stale_running_job_is_recovered_as_failed() -> None:
    store = MemoryRagasEvaluationJobStore()
    record = RagasEvaluationJobRecord(
        job_id="job-2",
        status="running",
        request=RagasEvaluationRequest(),
        principal="user-1",
        created_at=datetime.now(timezone.utc) - timedelta(minutes=40),
        updated_at=datetime.now(timezone.utc) - timedelta(minutes=40),
        dataset_path="data/ragas.json",
    )
    store.put(record)
    coordinator = RagasEvaluationJobCoordinator(job_store=store, workflow=SimpleNamespace())

    recovered = coordinator.get("job-2")

    assert recovered is not None
    assert recovered.status == "failed"
    assert "no persisted job checkpoint" in (recovered.error or "")
    assert "progress: 0" in (recovered.error or "")
    assert "another Lambda log stream" in (recovered.error or "")


def test_ragas_stale_queued_job_reports_orchestration_paths() -> None:
    store = MemoryRagasEvaluationJobStore()
    record = RagasEvaluationJobRecord(
        job_id="queued-job",
        status="queued",
        request=RagasEvaluationRequest(),
        principal="user-1",
        created_at=datetime.now(timezone.utc) - timedelta(minutes=40),
        updated_at=datetime.now(timezone.utc) - timedelta(minutes=40),
        dataset_path="data/ragas.json",
    )
    store.put(record)
    coordinator = RagasEvaluationJobCoordinator(job_store=store, workflow=SimpleNamespace())

    recovered = coordinator.get("queued-job")

    assert recovered is not None
    assert recovered.status == "failed"
    assert "worker-capacity retries" in (recovered.error or "")
    assert "Step Functions, starter, and worker logs" in (recovered.error or "")


def test_ragas_running_job_survives_sqs_recovery_window() -> None:
    store = MemoryRagasEvaluationJobStore()
    record = RagasEvaluationJobRecord(
        job_id="job-recovery-window",
        status="running",
        request=RagasEvaluationRequest(),
        principal="user-1",
        created_at=datetime.now(timezone.utc) - timedelta(minutes=20),
        updated_at=datetime.now(timezone.utc) - timedelta(minutes=20),
        dataset_path="data/ragas.json",
    )
    store.put(record)
    coordinator = RagasEvaluationJobCoordinator(job_store=store, workflow=SimpleNamespace())

    current = coordinator.get("job-recovery-window")

    assert current is not None
    assert current.status == "running"


def test_ragas_job_processes_small_sqs_batches(monkeypatch) -> None:
    store = MemoryRagasEvaluationJobStore()
    calls: list[int] = []
    queued_messages: list[dict[str, str]] = []
    dataset = RagEvaluationDataset.model_validate(
        {
            "name": "batch-test",
            "items": [
                {"id": "q1", "question": "Q1", "expected_answer_summary": "A1"},
                {"id": "q2", "question": "Q2", "expected_answer_summary": "A2"},
            ],
        }
    )

    def fake_evaluate(samples, config):
        calls.append(len(samples))
        return RagasEvaluationSummary(
            total_questions=len(samples),
            completed_questions=len(samples),
            failed_questions=0,
            metrics={"faithfulness": 1.0},
        )

    class FakeSqsClient:
        def send_message(self, **kwargs):
            queued_messages.append(kwargs)

    retrieved_document = RetrievedDocument(
        document_id="doc-1",
        filename="batch.md",
        text="context",
        score=1.0,
        chunk_id="batch.md#chunk-0",
        chunk_index=0,
    )

    class FakeEvaluationRetriever:
        def search(self, query: str, k: int = 25):
            return [retrieved_document]

    monkeypatch.setattr(
        "pra_core.evaluation.ragas_jobs.build_isolated_evaluation_retriever",
        lambda **kwargs: (
            FakeEvaluationRetriever(),
            [
                RagEvaluationSourceDocument(
                    requested_name="batch.md",
                    resolved_name="batch.md",
                    path="batch.md",
                )
            ],
        ),
    )

    monkeypatch.setattr("pra_core.evaluation.ragas_jobs.evaluate_ragas", fake_evaluate)
    store.put(
        RagasEvaluationJobRecord(
            job_id="batch-job",
            status="queued",
            request=RagasEvaluationRequest(max_questions=2, mode="standard"),
            principal="user-1",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            dataset_path="data/rag_eval_dataset_v2.json",
        )
    )
    coordinator = RagasEvaluationJobCoordinator(
        job_store=store,
        workflow=SimpleNamespace(),
        chunker=TextChunker(chunk_size=120, overlap=20),
        evaluation_embedding_provider=LocalHashEmbeddingProvider(dimensions=8),
        evaluation_workflow_factory=lambda retriever: SimpleNamespace(
            run=lambda **kwargs: {
                "answer": "answer",
                "retrieval_details": [{"document_id": "doc-1"}],
                "citations": ["doc-1"],
            }
        ),
        dataset_loader=lambda path: dataset,
        dataset_exists_checker=lambda path: True,
        queue_url="https://sqs.us-east-1.amazonaws.com/123/ragas",
        sqs_client=FakeSqsClient(),
        batch_size=1,
    )

    pending = coordinator.execute("batch-job")

    assert pending.status == "running"
    assert pending.completed_questions == 1
    assert pending.total_questions == 2
    assert queued_messages == [
        {
            "QueueUrl": "https://sqs.us-east-1.amazonaws.com/123/ragas",
            "MessageBody": '{"job_id": "batch-job"}',
        }
    ]

    completed = coordinator.execute("batch-job")

    assert completed.status == "completed"
    assert completed.completed_questions == 2
    assert completed.total_questions == 2
    assert completed.summary is not None
    assert completed.summary.metrics == {"faithfulness": 1.0}
    assert calls == [1, 1]


def test_ragas_worker_no_records_is_explicit_skip() -> None:
    from pra_api import ragas_worker

    class UnexpectedCoordinator:
        def get(self, job_id):
            raise AssertionError("no-record events must not load a job")

    original = ragas_worker.WORKER_APP
    ragas_worker.WORKER_APP = SimpleNamespace(state=SimpleNamespace(ragas_coordinator=UnexpectedCoordinator()))
    try:
        assert ragas_worker.handler({}, SimpleNamespace(aws_request_id="request-1")) == {"processed": []}
    finally:
        ragas_worker.WORKER_APP = original


def test_full_mode_persists_one_sample_and_runs_one_metric_per_continuation(monkeypatch) -> None:
    store = MemoryRagasEvaluationJobStore()
    dataset = RagEvaluationDataset.model_validate(
        {
            "name": "metric-step-test",
            "items": [
                {
                    "id": "q1",
                    "question": "What is BF-C?",
                    "expected_answer_summary": "Seal breach confirmed.",
                }
            ],
        }
    )
    corpus_builds = 0
    workflow_calls = 0
    metric_calls: list[tuple[str, ...]] = []
    messages: list[dict[str, str]] = []

    class FakeRetriever:
        pass

    def build_retriever(**_kwargs):
        nonlocal corpus_builds
        corpus_builds += 1
        return FakeRetriever(), []

    def workflow_factory(_retriever):
        def run(**_kwargs):
            nonlocal workflow_calls
            workflow_calls += 1
            return {
                "answer": "BF-C means seal breach confirmed.",
                "ragas_contexts": ["BF-C means seal breach confirmed."],
                "citations": ["doc-1"],
                "retrieval_details": [{"document_id": "doc-1"}],
                "context_compression": {"final_context_count": 1},
            }

        return SimpleNamespace(run=run)

    def fake_evaluate(samples, config):
        metric_name = config.metric_names[0]
        metric_calls.append(config.metric_names)
        sample = samples[0]
        return RagasEvaluationSummary(
            total_questions=1,
            completed_questions=1,
            failed_questions=0,
            metrics={metric_name: 1.0},
            results=[
                RagasQuestionResult(
                    id=sample.id,
                    question=sample.user_input,
                    answer=sample.response,
                    metrics={metric_name: 1.0},
                )
            ],
        )

    monkeypatch.setattr(
        "pra_core.evaluation.ragas_jobs.build_isolated_evaluation_retriever",
        build_retriever,
    )
    monkeypatch.setattr("pra_core.evaluation.ragas_jobs.evaluate_ragas", fake_evaluate)
    store.put(
        RagasEvaluationJobRecord(
            job_id="full-metric-job",
            status="queued",
            request=RagasEvaluationRequest(max_questions=1, mode="full"),
            principal="user-1",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            dataset_path="dataset.json",
        )
    )
    coordinator = RagasEvaluationJobCoordinator(
        job_store=store,
        workflow=SimpleNamespace(),
        chunker=TextChunker(chunk_size=120, overlap=20),
        evaluation_embedding_provider=LocalHashEmbeddingProvider(dimensions=8),
        evaluation_workflow_factory=workflow_factory,
        dataset_loader=lambda _path: dataset,
        dataset_exists_checker=lambda _path: True,
        queue_url="queue-url",
        sqs_client=SimpleNamespace(send_message=lambda **kwargs: messages.append(kwargs)),
        batch_size=1,
    )

    for _ in range(6):
        record = coordinator.execute("full-metric-job")

    assert record.status == "completed"
    assert corpus_builds == 1
    assert workflow_calls == 1
    assert metric_calls == [(name,) for name in FULL_RAGAS_METRICS]
    assert record.question_states[0].retrieval_count == 1
    assert record.question_states[0].answer_generation_count == 1
    assert record.summary.metrics == {name: 1.0 for name in FULL_RAGAS_METRICS}
    assert len(messages) == 5

    coordinator.execute("full-metric-job")
    assert workflow_calls == 1
    assert metric_calls == [(name,) for name in FULL_RAGAS_METRICS]


def test_full_mode_deadline_checkpoints_before_the_next_paid_operation() -> None:
    store = MemoryRagasEvaluationJobStore()
    sent: list[dict[str, str]] = []
    dataset = RagEvaluationDataset.model_validate(
        {"name": "deadline-test", "items": [{"id": "q1", "question": "Q1"}]}
    )
    store.put(
        RagasEvaluationJobRecord(
            job_id="full-deadline-job",
            status="queued",
            request=RagasEvaluationRequest(max_questions=1, mode="full"),
            principal="user-1",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            dataset_path="dataset.json",
        )
    )
    coordinator = RagasEvaluationJobCoordinator(
        job_store=store,
        workflow=SimpleNamespace(),
        dataset_loader=lambda _path: dataset,
        dataset_exists_checker=lambda _path: True,
        queue_url="queue-url",
        sqs_client=SimpleNamespace(send_message=lambda **kwargs: sent.append(kwargs)),
    )

    record = coordinator.execute("full-deadline-job", remaining_time_ms=1)

    assert record.status == "running"
    assert record.question_states[0].sample_generation_complete is False
    assert sent[0]["MessageBody"] == (
        '{"job_id": "full-deadline-job", "question_index": 0, '
        '"phase": "sample", "metric": null}'
    )


def test_step_functions_owned_full_step_never_enqueues_a_continuation() -> None:
    store = MemoryRagasEvaluationJobStore()
    sent: list[dict[str, str]] = []
    dataset = RagEvaluationDataset.model_validate(
        {"name": "step-test", "items": [{"id": "q1", "question": "Q1"}]}
    )
    store.put(
        RagasEvaluationJobRecord(
            job_id="step-functions-job",
            status="queued",
            request=RagasEvaluationRequest(max_questions=1, mode="full"),
            principal="user-1",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            dataset_path="dataset.json",
        )
    )
    coordinator = RagasEvaluationJobCoordinator(
        job_store=store,
        workflow=SimpleNamespace(),
        dataset_loader=lambda _path: dataset,
        dataset_exists_checker=lambda _path: True,
        queue_url="queue-url",
        sqs_client=SimpleNamespace(send_message=lambda **kwargs: sent.append(kwargs)),
    )

    record = coordinator.execute(
        "step-functions-job",
        remaining_time_ms=1,
        external_orchestrator=True,
    )

    assert record.status == "running"
    assert record.question_states[0].sample_generation_complete is False
    assert sent == []


def test_ragas_worker_dispatches_valid_queued_job_once() -> None:
    from pra_api import ragas_worker

    job = RagasEvaluationJobRecord(
        job_id="dispatch-job",
        status="queued",
        request=RagasEvaluationRequest(max_questions=1),
        principal="user-1",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        dataset_path="data/ragas.json",
    )
    calls: list[tuple[str, int | None, str | None]] = []

    class Coordinator:
        def get(self, job_id):
            assert job_id == job.job_id
            return job

        def execute(
            self,
            job_id,
            *,
            remaining_time_ms=None,
            worker_request_id=None,
            external_orchestrator=False,
        ):
            calls.append((job_id, remaining_time_ms, worker_request_id, external_orchestrator))
            return job.model_copy(update={"status": "completed"})

    original = ragas_worker.WORKER_APP
    ragas_worker.WORKER_APP = SimpleNamespace(state=SimpleNamespace(ragas_coordinator=Coordinator()))
    try:
        result = ragas_worker.handler(
            {
                "Records": [
                    {
                        "messageId": "message-1",
                        "eventSource": "aws:sqs",
                        "attributes": {"ApproximateReceiveCount": "1"},
                        "body": '{"job_id":"dispatch-job","mode":"standard","max_questions":1}',
                    }
                ]
            },
            SimpleNamespace(aws_request_id="request-1", get_remaining_time_in_millis=lambda: 840000),
        )
    finally:
        ragas_worker.WORKER_APP = original

    assert result == {"processed": [{"job_id": "dispatch-job", "status": "completed"}]}
    assert calls == [("dispatch-job", 840000, "request-1", False)]


def test_ragas_worker_step_functions_invocation_owns_continuation() -> None:
    from pra_api import ragas_worker

    job = RagasEvaluationJobRecord(
        job_id="step-dispatch-job",
        status="queued",
        request=RagasEvaluationRequest(max_questions=1),
        principal="user-1",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        dataset_path="data/ragas.json",
    )
    calls: list[bool] = []

    class Coordinator:
        def get(self, job_id):
            assert job_id == job.job_id
            return job

        def execute(self, job_id, *, external_orchestrator=False, **_kwargs):
            assert job_id == job.job_id
            calls.append(external_orchestrator)
            return job.model_copy(update={"status": "running", "completed_questions": 1})

    original = ragas_worker.WORKER_APP
    ragas_worker.WORKER_APP = SimpleNamespace(state=SimpleNamespace(ragas_coordinator=Coordinator()))
    try:
        result = ragas_worker.handler(
            {"job_id": "step-dispatch-job"},
            SimpleNamespace(aws_request_id="step-request", get_remaining_time_in_millis=lambda: 840000),
        )
    finally:
        ragas_worker.WORKER_APP = original

    assert calls == [True]
    assert result == {
        "job_id": "step-dispatch-job",
        "status": "running",
        "completed_questions": 1,
        "total_questions": None,
    }


def test_ragas_starter_starts_one_idempotent_state_machine_execution(monkeypatch) -> None:
    from pra_api import ragas_starter

    calls: list[dict[str, str]] = []
    monkeypatch.setenv("PRA_RAGAS_STATE_MACHINE_ARN", "arn:aws:states:us-east-1:123:stateMachine:ragas")
    monkeypatch.setattr(
        ragas_starter,
        "_step_functions_client",
        lambda: SimpleNamespace(
            start_execution=lambda **kwargs: calls.append(kwargs) or {"executionArn": "execution-arn"}
        ),
    )

    result = ragas_starter.handler(
        {"Records": [{"body": '{"job_id": "starter-job"}'}]},
        SimpleNamespace(),
    )

    assert result == {"job_id": "starter-job", "status": "started"}
    assert calls == [{
        "stateMachineArn": "arn:aws:states:us-east-1:123:stateMachine:ragas",
        "name": "ragas-starter-job",
        "input": '{"job_id": "starter-job"}',
    }]


def test_ragas_status_checkpoint_records_capacity_wait_without_model_work(monkeypatch) -> None:
    from pra_api import ragas_status_checkpoint

    payload = {"job_id": "capacity-job", "status": "queued", "capacity_wait_count": 0}
    writes: list[dict[str, object]] = []

    class S3Client:
        def get_object(self, **kwargs):
            assert kwargs == {
                "Bucket": "test-bucket",
                "Key": "pra-prod/ragas-evaluation/jobs/capacity-job.json",
            }
            return {"Body": BytesIO(json.dumps(payload).encode("utf-8"))}

        def put_object(self, **kwargs):
            writes.append(kwargs)

    monkeypatch.setenv("PRA_DOCUMENT_BUCKET_NAME", "test-bucket")
    monkeypatch.setenv("PRA_PERSISTENCE_S3_PREFIX", "pra-prod")
    monkeypatch.setattr(ragas_status_checkpoint, "_s3_client", S3Client)

    result = ragas_status_checkpoint.handler(
        {"job_id": "capacity-job", "checkpoint": "waiting_for_capacity"},
        SimpleNamespace(),
    )

    assert result == {
        "job_id": "capacity-job",
        "status": "queued",
        "capacity_wait_count": 1,
        "checkpoint": "waiting_for_capacity",
    }
    saved = json.loads(writes[0]["Body"].decode("utf-8"))
    assert saved["status"] == "queued"
    assert saved["capacity_wait_count"] == 1
    assert saved["orchestration_state"] == "waiting_for_worker_capacity"
    assert "single cost-capped worker" in saved["orchestration_message"]


def test_ragas_worker_defers_application_construction_until_invocation(monkeypatch) -> None:
    from pra_api import ragas_worker

    fake_app = SimpleNamespace(state=SimpleNamespace(ragas_coordinator=object()))
    original = ragas_worker.WORKER_APP
    ragas_worker.WORKER_APP = None
    monkeypatch.setattr(ragas_worker, "_get_worker_app", lambda: fake_app)

    try:
        assert ragas_worker.WORKER_APP is None
        assert ragas_worker._get_worker_app() is fake_app
    finally:
        ragas_worker.WORKER_APP = original


def test_ragas_status_checkpoint_marks_exhausted_capacity_as_failed(monkeypatch) -> None:
    from pra_api import ragas_status_checkpoint

    payload = {"job_id": "capacity-timeout-job", "status": "queued", "capacity_wait_count": 60}
    writes: list[dict[str, object]] = []

    class S3Client:
        def get_object(self, **_kwargs):
            return {"Body": BytesIO(json.dumps(payload).encode("utf-8"))}

        def put_object(self, **kwargs):
            writes.append(kwargs)

    monkeypatch.setenv("PRA_DOCUMENT_BUCKET_NAME", "test-bucket")
    monkeypatch.setattr(ragas_status_checkpoint, "_s3_client", S3Client)

    result = ragas_status_checkpoint.handler(
        {"job_id": "capacity-timeout-job", "checkpoint": "capacity_exhausted"},
        SimpleNamespace(),
    )

    assert result["status"] == "failed"
    saved = json.loads(writes[0]["Body"].decode("utf-8"))
    assert saved["orchestration_state"] == "capacity_timeout"
    assert "could not acquire worker capacity" in saved["error"]


def test_ragas_worker_claim_replaces_capacity_wait_message() -> None:
    store = MemoryRagasEvaluationJobStore()
    record = RagasEvaluationJobRecord(
        job_id="claimed-job",
        status="queued",
        request=RagasEvaluationRequest(max_questions=1, mode="full"),
        principal="user-1",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        dataset_path="data/ragas.json",
        orchestration_state="waiting_for_worker_capacity",
        orchestration_message="RAGAS evaluation is waiting for the single cost-capped worker.",
    )
    store.put(record)
    coordinator = RagasEvaluationJobCoordinator(
        job_store=store,
        workflow=SimpleNamespace(),
        dataset_exists_checker=lambda _path: True,
        dataset_loader=lambda _path: RagEvaluationDataset.model_validate(
            {"name": "empty", "items": []}
        ),
    )

    claimed = coordinator.execute("claimed-job", external_orchestrator=True)

    assert claimed.status == "completed"
    assert claimed.orchestration_state == "worker_running"
    assert claimed.orchestration_message == "RAGAS worker capacity acquired; evaluation is running."


def test_ragas_worker_confirms_cancel_requested_job() -> None:
    from pra_api import ragas_worker

    job = RagasEvaluationJobRecord(
        job_id="cancel-dispatch-job",
        status="cancel_requested",
        request=RagasEvaluationRequest(max_questions=1),
        principal="user-1",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        dataset_path="data/ragas.json",
    )

    class Coordinator:
        def get(self, job_id):
            assert job_id == job.job_id
            return job

        def execute(
            self,
            job_id,
            *,
            remaining_time_ms=None,
            worker_request_id=None,
            external_orchestrator=False,
        ):
            assert job_id == job.job_id
            return job.model_copy(
                update={
                    "status": "cancelled",
                    "updated_at": datetime.now(timezone.utc),
                }
            )

    original = ragas_worker.WORKER_APP
    ragas_worker.WORKER_APP = SimpleNamespace(
        state=SimpleNamespace(ragas_coordinator=Coordinator())
    )
    try:
        result = ragas_worker.handler(
            {
                "Records": [
                    {
                        "messageId": "cancel-message",
                        "body": '{"job_id": "cancel-dispatch-job"}',
                        "eventSource": "aws:sqs",
                        "attributes": {},
                    }
                ]
            },
            SimpleNamespace(
                aws_request_id="request-cancel",
                get_remaining_time_in_millis=lambda: 100_000,
            ),
        )
    finally:
        ragas_worker.WORKER_APP = original

    assert result == {"processed": [
        {"job_id": "cancel-dispatch-job", "status": "cancelled"}
    ]}


def test_ragas_worker_malformed_message_raises_for_sqs_retry() -> None:
    from pra_api import ragas_worker

    original = ragas_worker.WORKER_APP
    ragas_worker.WORKER_APP = SimpleNamespace(state=SimpleNamespace(ragas_coordinator=object()))
    try:
        with pytest.raises(ValueError, match="Malformed RAGAS SQS message body"):
            ragas_worker.handler(
                {
                    "Records": [
                        {
                            "messageId": "message-1",
                            "eventSource": "aws:sqs",
                            "body": "not-json",
                        }
                    ]
                },
                SimpleNamespace(aws_request_id="request-1"),
            )
    finally:
        ragas_worker.WORKER_APP = original


def test_worker_cannot_inline_continuation_without_queue() -> None:
    coordinator = RagasEvaluationJobCoordinator(
        job_store=MemoryRagasEvaluationJobStore(),
        workflow=None,
        allow_inline_continuation=False,
    )

    with pytest.raises(RuntimeError, match="continuation queue is not configured"):
        coordinator._enqueue_or_continue(
            "job-1",
            next_index=1,
            remaining_question_count=1,
        )


def test_ragas_deadline_guard_stops_before_next_batch(caplog) -> None:
    coordinator = RagasEvaluationJobCoordinator(
        job_store=MemoryRagasEvaluationJobStore(),
        workflow=None,
    )

    with pytest.raises(TimeoutError, match="execution deadline"):
        coordinator._check_deadline(0, "deadline-job")

    assert "ragas_worker_deadline_approaching" in caplog.text


def test_ragas_standard_and_full_modes_select_expected_metrics(monkeypatch) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_evaluate(samples, config):
        calls.append(config.metric_names)
        return RagasEvaluationSummary(
            total_questions=len(samples),
            completed_questions=len(samples),
            failed_questions=0,
        )

    monkeypatch.setattr("pra_core.evaluation.ragas_jobs.evaluate_ragas", fake_evaluate)
    coordinator = RagasEvaluationJobCoordinator(
        job_store=MemoryRagasEvaluationJobStore(),
        workflow=SimpleNamespace(),
    )
    sample = RagasEvaluationSample(
        id="q1",
        user_input="Question",
        response="Answer",
        retrieved_contexts=["Context"],
    )

    coordinator._evaluate_batch([sample], mode="standard")
    coordinator._evaluate_batch([sample], mode="full")

    assert calls == [STANDARD_RAGAS_METRICS, FULL_RAGAS_METRICS]


def test_ragas_request_defaults_to_standard_mode() -> None:
    request = RagasEvaluationRequest()

    assert request.mode == "full"
    assert request.top_k == 15
    assert request.max_questions == 12


def test_ragas_skips_judge_calls_when_required_context_is_missing(monkeypatch) -> None:
    called = False

    def fake_evaluate(samples, config):
        nonlocal called
        called = True
        raise AssertionError("judge should not be called")

    monkeypatch.setattr("pra_core.evaluation.ragas_jobs.evaluate_ragas", fake_evaluate)
    coordinator = RagasEvaluationJobCoordinator(
        job_store=MemoryRagasEvaluationJobStore(),
        workflow=SimpleNamespace(),
    )

    summary = coordinator._evaluate_batch(
        [
            RagasEvaluationSample(
                id="q1",
                user_input="Question",
                response="Unsupported answer",
                expected_behavior="answer",
            )
        ],
        mode="standard",
    )

    assert called is False
    assert summary.failed_questions == 1
    assert "no context" in (summary.results[0].error or "")


def test_ragas_still_scores_empty_context_refusals(monkeypatch) -> None:
    metric_calls: list[tuple[str, ...]] = []

    def fake_evaluate(samples, config):
        metric_calls.append(config.metric_names)
        return RagasEvaluationSummary(
            total_questions=1,
            completed_questions=1,
            failed_questions=0,
        )

    monkeypatch.setattr("pra_core.evaluation.ragas_jobs.evaluate_ragas", fake_evaluate)
    coordinator = RagasEvaluationJobCoordinator(
        job_store=MemoryRagasEvaluationJobStore(),
        workflow=SimpleNamespace(),
    )

    summary = coordinator._evaluate_batch(
        [
            RagasEvaluationSample(
                id="q1",
                user_input="Unknown question",
                response="There is not enough evidence.",
                expected_behavior="insufficient_evidence",
            )
        ],
        mode="standard",
    )

    assert summary.completed_questions == 1
    assert metric_calls == [("answer_correctness",)]


def test_ragas_standard_sampling_covers_simple_and_complex_questions() -> None:
    dataset = RagEvaluationDataset.model_validate(
        {
            "name": "representative",
            "items": [
                {"id": "s1", "question": "S1", "intent": "basic_lookup"},
                {"id": "s2", "question": "S2", "intent": "table_code_lookup"},
                {"id": "s3", "question": "S3", "intent": "direct_fact_retrieval"},
                {"id": "c1", "question": "C1", "intent": "multi_hop_exception"},
                {"id": "c2", "question": "C2", "intent": "comparison_reasoning"},
                {"id": "c3", "question": "C3", "intent": "global_scan"},
            ],
        }
    )

    selected = RagasEvaluationJobCoordinator._select_items(
        dataset.items,
        RagasEvaluationRequest(mode="standard", max_questions=4),
    )

    assert len(selected) == 4
    assert any(item.id.startswith("s") for item in selected)
    assert any(item.id.startswith("c") for item in selected)
    assert selected[-1].id == "c3"


def test_ragas_duplicate_delivery_does_not_reprocess_terminal_job() -> None:
    store = MemoryRagasEvaluationJobStore()
    record = RagasEvaluationJobRecord(
        job_id="finished-job",
        status="completed",
        request=RagasEvaluationRequest(),
        principal="user-1",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        dataset_path="data/rag_eval_dataset_v2.json",
        completed_questions=12,
        total_questions=12,
    )
    store.put(record)
    coordinator = RagasEvaluationJobCoordinator(
        job_store=store,
        workflow=SimpleNamespace(
            run=lambda **kwargs: (_ for _ in ()).throw(
                AssertionError("terminal job must not run")
            )
        ),
    )

    result = coordinator.execute("finished-job")

    assert result == record


def _job_record(job_id: str, *, completed: int = 0, total: int = 3) -> RagasEvaluationJobRecord:
    return RagasEvaluationJobRecord(
        job_id=job_id,
        status="queued",
        request=RagasEvaluationRequest(max_questions=total, mode="standard"),
        principal="user-1",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        dataset_path="data/rag_eval_dataset_v2.json",
        completed_questions=completed,
        total_questions=total,
    )


def _summary(completed: int, total: int = 3) -> RagasEvaluationSummary:
    return RagasEvaluationSummary(
        total_questions=total,
        completed_questions=completed,
        failed_questions=0,
        results=[],
    )


def test_ragas_post_batch_deadline_yields_after_checkpoint(monkeypatch) -> None:
    store = MemoryRagasEvaluationJobStore()
    store.put(_job_record("deadline-yield-job"))
    sent: list[dict[str, str]] = []
    coordinator = RagasEvaluationJobCoordinator(
        job_store=store,
        workflow=SimpleNamespace(),
        queue_url="queue-url",
        sqs_client=SimpleNamespace(send_message=lambda **kwargs: sent.append(kwargs)),
        batch_size=1,
    )
    monkeypatch.setattr(coordinator, "_run_request", lambda *args, **kwargs: (_summary(1), False, 1))

    result = coordinator.execute("deadline-yield-job", remaining_time_ms=1)

    assert result.status == "running"
    assert result.completed_questions == 1
    assert len(sent) == 1


def test_ragas_exception_preserves_latest_checkpoint(monkeypatch) -> None:
    store = MemoryRagasEvaluationJobStore()
    store.put(_job_record("checkpoint-job"))
    coordinator = RagasEvaluationJobCoordinator(job_store=store, workflow=SimpleNamespace())

    def fail_after_checkpoint(*args, **kwargs):
        latest = _job_record("checkpoint-job", completed=1)
        store.put(latest.model_copy(update={"status": "running", "summary": _summary(1)}))
        raise RuntimeError("continuation bookkeeping failed")

    monkeypatch.setattr(coordinator, "_run_request", fail_after_checkpoint)

    result = coordinator.execute("checkpoint-job")

    assert result.status == "failed"
    assert result.completed_questions == 1
    assert result.summary is not None
    assert result.summary.completed_questions == 1


def test_ragas_duplicate_delivery_starts_from_durable_next_index(monkeypatch) -> None:
    store = MemoryRagasEvaluationJobStore()
    store.put(_job_record("duplicate-job", completed=1))
    submitted: list[tuple[object, str]] = []
    coordinator = RagasEvaluationJobCoordinator(
        job_store=store,
        workflow=SimpleNamespace(),
        executor=SimpleNamespace(
            submit=lambda function, job_id: submitted.append((function, job_id))
        ),
    )
    observed: list[int] = []

    def resume_at_durable_index(*args, **kwargs):
        observed.append(kwargs["start_index"])
        return _summary(2), False, 2

    monkeypatch.setattr(coordinator, "_run_request", resume_at_durable_index)
    result = coordinator.execute("duplicate-job")

    assert observed == [1]
    assert result.status == "running"
    assert len(submitted) == 1


def test_ragas_cancel_race_suppresses_continuation(monkeypatch) -> None:
    store = MemoryRagasEvaluationJobStore()
    store.put(_job_record("cancel-race-job"))
    sent: list[dict[str, str]] = []
    coordinator = RagasEvaluationJobCoordinator(
        job_store=store,
        workflow=SimpleNamespace(),
        queue_url="queue-url",
        sqs_client=SimpleNamespace(send_message=lambda **kwargs: sent.append(kwargs)),
    )

    def checkpoint_then_cancel(*args, **kwargs):
        current = store.get("cancel-race-job")
        store.put(current.model_copy(update={"status": "cancel_requested", "completed_questions": 1}))
        return _summary(1), False, 1

    monkeypatch.setattr(coordinator, "_run_request", checkpoint_then_cancel)

    result = coordinator.execute("cancel-race-job")

    assert result.status == "cancelled"
    assert sent == []


def test_ragas_final_question_near_deadline_does_not_schedule_continuation(monkeypatch) -> None:
    store = MemoryRagasEvaluationJobStore()
    store.put(_job_record("final-job", total=1))
    sent: list[dict[str, str]] = []
    coordinator = RagasEvaluationJobCoordinator(
        job_store=store,
        workflow=SimpleNamespace(),
        queue_url="queue-url",
        sqs_client=SimpleNamespace(send_message=lambda **kwargs: sent.append(kwargs)),
    )
    monkeypatch.setattr(coordinator, "_run_request", lambda *args, **kwargs: (_summary(1, total=1), True, 1))

    result = coordinator.execute("final-job", remaining_time_ms=1)

    assert result.status == "completed"
    assert sent == []
