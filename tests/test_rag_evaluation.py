import json
import sys
import types
from pathlib import Path

from fastapi.testclient import TestClient

from pra_api.main import create_app
from pra_core.auth.cognito import CognitoAuthProvider
from pra_core.config import Settings
from pra_core.documents.chunking import TextChunker
from pra_core.documents.models import ParsedDocument, RetrievedDocument
from pra_core.embeddings.base import EmbeddingProvider
from pra_core.evaluation.rag_retrieval import (
    _EvaluationChunkEntry,
    RagEvaluationDataset,
    build_isolated_evaluation_retriever,
    evaluate_retrieval_only,
    load_rag_evaluation_dataset,
    s3_object_exists,
)
from pra_core.evaluation.jobs import (
    MemoryRagEvaluationJobStore,
    RagEvaluationJobCoordinator,
    RagEvaluationRequest,
)


class FakeCognitoVerifier:
    def verify(self, token: str) -> dict[str, object]:
        assert token == "valid-cognito-token"
        return {"sub": "cognito-user-1", "cognito:groups": ["researcher"]}


class StaticRetriever:
    def __init__(
        self,
        results_by_query: dict[str, list[RetrievedDocument]],
        chunk_entries: list[_EvaluationChunkEntry] | None = None,
    ) -> None:
        self._results_by_query = results_by_query
        self._chunk_entries = chunk_entries or []

    def search(self, query: str, k: int = 25) -> list[RetrievedDocument]:
        return self._results_by_query.get(query, [])[:k]


class CountingEmbeddingProvider(EmbeddingProvider):
    provider_id = "counting-test"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return [float(len(text)), 1.0]


AUTH_HEADERS = {"Authorization": "Bearer valid-cognito-token"}


def write_eval_docs(docs_root: Path, filename: str = "northstar.md") -> Path:
    docs_root.mkdir(exist_ok=True)
    path = docs_root / filename
    path.write_text(
        "# Systems\n\n"
        "Orchid Ledger is the main analytics system.\n\n"
        "# Codes\n\n"
        "Blue Finch code BF-C means cold-chain carrier hold.",
        encoding="utf-8",
    )
    return path


def chunk_entry(
    chunk_id: str,
    *,
    filename: str = "a.md",
    text: str = "a",
    heading: str = "Section",
    index: int = 0,
) -> _EvaluationChunkEntry:
    return _EvaluationChunkEntry(
        document_id=f"doc-{chunk_id}",
        filename=filename,
        text=text,
        chunk_id=chunk_id,
        chunk_index=index,
        tokens=[],
        metadata={"heading": heading},
    )


def authenticated_app(*, dataset_path: str | None = None):
    settings_kwargs = {
        "environment": "test",
        "cognito_issuer": "https://issuer.example.test",
        "cognito_client_id": "client-id",
        "rag_evaluation_enabled": True,
    }
    if dataset_path is not None:
        settings_kwargs["rag_eval_dataset_path"] = dataset_path
    app = create_app(Settings(**settings_kwargs))
    app.state.auth_provider = CognitoAuthProvider(
        app.state.settings,
        token_verifier=FakeCognitoVerifier(),
    )
    return app


def test_rag_evaluation_metrics_perfect_precision_and_recall() -> None:
    dataset = RagEvaluationDataset.model_validate(
        {
            "name": "test",
            "items": [
                {
                    "id": "q1",
                    "question": "question 1",
                    "source_files": ["a.md"],
                    "expected_sections": ["Section"],
                    "relevant_chunk_ids": ["stale-id"],
                }
            ],
        }
    )
    retriever = StaticRetriever(
        {
            "question 1": [
                RetrievedDocument(
                    document_id="doc-1",
                    filename="a.md",
                    text="a",
                    score=1.0,
                    chunk_id="chunk-1",
                ),
                RetrievedDocument(
                    document_id="doc-2",
                    filename="b.md",
                    text="b",
                    score=0.9,
                    chunk_id="chunk-2",
                ),
            ]
        },
        chunk_entries=[
            chunk_entry("chunk-1", text="first", heading="Section", index=0),
            chunk_entry("chunk-2", text="second", heading="Section", index=1),
        ],
    )

    result = evaluate_retrieval_only(
        retriever=retriever,
        dataset=dataset,
        top_k=5,
        eval_k=[5],
    )

    assert result.metrics["recall_at_5"] == 1.0
    assert result.metrics["precision_at_5"] == 0.4


def test_rag_evaluation_metrics_partial_match() -> None:
    dataset = RagEvaluationDataset.model_validate(
        {
            "name": "test",
            "items": [
                {
                    "id": "q1",
                    "question": "question 1",
                    "source_files": ["a.md"],
                    "expected_sections": ["Section"],
                    "relevant_chunk_ids": ["stale-id"],
                }
            ],
        }
    )
    retriever = StaticRetriever(
        {
            "question 1": [
                RetrievedDocument(
                    document_id="doc-1",
                    filename="a.md",
                    text="a",
                    score=1.0,
                    chunk_id="chunk-1",
                ),
                RetrievedDocument(
                    document_id="doc-4",
                    filename="d.md",
                    text="d",
                    score=0.9,
                    chunk_id="chunk-4",
                ),
            ]
        },
        chunk_entries=[
            chunk_entry("chunk-1", text="first", heading="Section", index=0),
            chunk_entry("chunk-2", text="second", heading="Section", index=1),
            chunk_entry("chunk-3", text="third", heading="Section", index=2),
        ],
    )

    result = evaluate_retrieval_only(
        retriever=retriever,
        dataset=dataset,
        top_k=5,
        eval_k=[5],
    )

    assert result.metrics["recall_at_5"] == 0.3333
    assert result.metrics["precision_at_5"] == 0.2


def test_rag_evaluation_metrics_with_no_retrieved_documents() -> None:
    dataset = RagEvaluationDataset.model_validate(
        {
            "name": "test",
            "items": [
                {
                    "id": "q1",
                    "question": "question 1",
                    "source_files": ["a.md"],
                    "expected_sections": ["Section"],
                    "relevant_chunk_ids": ["stale-id"],
                }
            ],
        }
    )
    retriever = StaticRetriever(
        {"question 1": []},
        chunk_entries=[chunk_entry("chunk-1")],
    )

    result = evaluate_retrieval_only(
        retriever=retriever,
        dataset=dataset,
        top_k=5,
        eval_k=[5],
    )

    assert result.metrics["recall_at_5"] == 0.0
    assert result.metrics["precision_at_5"] == 0.0


def test_rag_evaluation_metrics_with_unlabeled_question() -> None:
    dataset = RagEvaluationDataset.model_validate(
        {
            "name": "test",
            "items": [{"id": "q1", "question": "question 1", "relevant_chunk_ids": []}],
        }
    )
    retriever = StaticRetriever({"question 1": []})

    result = evaluate_retrieval_only(
        retriever=retriever,
        dataset=dataset,
        top_k=5,
        eval_k=[5],
    )

    assert result.labeled_questions == 0
    assert result.unlabeled_questions == 1
    assert result.metrics["recall_at_5"] is None
    assert result.metrics["precision_at_5"] is None
    assert result.results[0].labeled is False


def test_rag_evaluation_route_returns_retrieval_summary(tmp_path: Path) -> None:
    docs_root = tmp_path / "docs"
    write_eval_docs(docs_root)
    dataset_path = tmp_path / "rag_eval_dataset_v1.json"
    dataset_path.write_text(
        json.dumps(
            {
                "name": "personal_research_assistant_rag_eval_v1",
                "default_top_k": 25,
                "eval_k": [5, 10, 25],
                "items": [
                    {
                        "id": "rag-eval-001",
                        "question": "What is Orchid Ledger?",
                        "source_files": ["northstar.md"],
                        "expected_sections": ["Systems"],
                        "relevant_chunk_ids": ["stale-id"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    app = authenticated_app(dataset_path=str(dataset_path))
    app.state.retriever = StaticRetriever(
        {
            "What is Orchid Ledger?": [
                RetrievedDocument(
                    document_id="doc-orchid",
                    filename="northstar.md",
                    text="Orchid Ledger is the main analytics system.",
                    score=1.0,
                    chunk_id="chunk-orchid",
                    chunk_index=0,
                )
            ]
        },
        chunk_entries=[
            chunk_entry(
                "chunk-orchid",
                filename="northstar.md",
                text="Orchid Ledger is the main analytics system.",
                heading="Systems",
            )
        ],
    )
    app.state.rag_evaluation_coordinator._retriever = app.state.retriever
    app.state.rag_evaluation_coordinator._dataset_exists_checker = lambda path: True
    app.state.rag_evaluation_coordinator._evaluation_docs_root = docs_root
    app.state.ensure_vector_index = lambda: None
    client = TestClient(app)

    response = client.post(
        "/rag/evaluate",
        headers=AUTH_HEADERS,
        json={"dataset": "default", "top_k": 25, "eval_k": [5, 10, 25]},
    )

    assert response.status_code == 202
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["summary"]["dataset"] == "personal_research_assistant_rag_eval_v1"
    assert payload["summary"]["total_questions"] == 1
    assert payload["summary"]["labeled_questions"] == 1
    assert payload["summary"]["unlabeled_questions"] == 0
    assert payload["summary"]["failed_question_count"] == 0
    assert payload["summary"]["metrics"]["recall_at_5"] == 1.0
    assert payload["summary"]["results"][0]["retrieved_ids"] == ["northstar.md#chunk-0"]


def test_rag_evaluation_route_returns_missing_dataset_error(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.json"
    app = authenticated_app(dataset_path=str(missing_path))
    app.state.ensure_vector_index = lambda: None
    client = TestClient(app)

    response = client.post(
        "/rag/evaluate",
        headers=AUTH_HEADERS,
        json={"dataset": "default", "top_k": 25, "eval_k": [5, 10, 25]},
    )

    assert response.status_code == 404
    assert "PRA_RAG_EVAL_DATASET_PATH" in response.json()["error"]


def test_rag_evaluation_dataset_can_load_from_s3_uri(monkeypatch) -> None:
    payload = {
        "name": "personal_research_assistant_rag_eval_v1",
        "default_top_k": 25,
        "eval_k": [5, 10, 25],
        "items": [
            {
                "id": "rag-eval-001",
                "question": "What is Orchid Ledger?",
                "source_files": ["northstar.md"],
                "expected_sections": ["Systems"],
                "relevant_chunk_ids": ["stale-id"],
            }
        ],
    }

    monkeypatch.setattr(
        "pra_core.evaluation.rag_retrieval._read_s3_text",
        lambda uri: json.dumps(payload),
    )

    dataset = load_rag_evaluation_dataset(
        "s3://example-bucket/pra-prod/testing/rag-eval/rag_eval_dataset_v1.json"
    )

    assert dataset.name == "personal_research_assistant_rag_eval_v1"
    assert dataset.items[0].id == "rag-eval-001"


def test_s3_dataset_object_exists_returns_true(monkeypatch) -> None:
    class FakeS3Client:
        def head_object(self, *, Bucket: str, Key: str) -> None:
            assert Bucket == "example-bucket"
            assert Key == "pra-prod/testing/rag-eval/rag_eval_dataset_v1.json"

    fake_boto3 = types.ModuleType("boto3")
    fake_boto3.client = lambda service_name: FakeS3Client()
    fake_botocore = types.ModuleType("botocore")
    fake_botocore_exceptions = types.ModuleType("botocore.exceptions")

    class FakeClientError(Exception):
        def __init__(self, response: dict[str, object], operation_name: str) -> None:
            super().__init__(operation_name)
            self.response = response

    fake_botocore_exceptions.ClientError = FakeClientError
    fake_botocore.exceptions = fake_botocore_exceptions
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    monkeypatch.setitem(sys.modules, "botocore", fake_botocore)
    monkeypatch.setitem(sys.modules, "botocore.exceptions", fake_botocore_exceptions)

    assert s3_object_exists(
        "s3://example-bucket/pra-prod/testing/rag-eval/rag_eval_dataset_v1.json"
    )


def test_s3_dataset_object_exists_returns_false_for_missing_key(monkeypatch) -> None:
    class FakeClientError(Exception):
        def __init__(self, response: dict[str, object], operation_name: str) -> None:
            super().__init__(operation_name)
            self.response = response

    class FakeS3Client:
        def head_object(self, *, Bucket: str, Key: str) -> None:
            raise FakeClientError(
                {"Error": {"Code": "404", "Message": "Not Found"}},
                "HeadObject",
            )

    fake_boto3 = types.ModuleType("boto3")
    fake_boto3.client = lambda service_name: FakeS3Client()
    fake_botocore = types.ModuleType("botocore")
    fake_botocore_exceptions = types.ModuleType("botocore.exceptions")
    fake_botocore_exceptions.ClientError = FakeClientError
    fake_botocore.exceptions = fake_botocore_exceptions
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    monkeypatch.setitem(sys.modules, "botocore", fake_botocore)
    monkeypatch.setitem(sys.modules, "botocore.exceptions", fake_botocore_exceptions)

    assert (
        s3_object_exists(
            "s3://example-bucket/pra-prod/testing/rag-eval/rag_eval_dataset_v1.json"
        )
        is False
    )


def test_rag_evaluation_route_accepts_s3_dataset_path(monkeypatch, tmp_path: Path) -> None:
    docs_root = tmp_path / "docs"
    write_eval_docs(docs_root)
    payload = {
        "name": "personal_research_assistant_rag_eval_v1",
        "default_top_k": 25,
        "eval_k": [5, 10, 25],
        "items": [
            {
                "id": "rag-eval-001",
                "question": "What is Orchid Ledger?",
                "source_files": ["northstar.md"],
                "expected_sections": ["Systems"],
                "relevant_chunk_ids": ["stale-id"],
            }
        ],
    }
    monkeypatch.setattr(
        "pra_core.evaluation.rag_retrieval._read_s3_text",
        lambda uri: json.dumps(payload),
    )
    monkeypatch.setattr(
        "pra_api.routes.rag.s3_object_exists",
        lambda uri: True,
    )

    app = authenticated_app(
        dataset_path="s3://example-bucket/pra-prod/testing/rag-eval/rag_eval_dataset_v1.json"
    )
    app.state.retriever = StaticRetriever(
        {
            "What is Orchid Ledger?": [
                RetrievedDocument(
                    document_id="doc-orchid",
                    filename="northstar.md",
                    text="Orchid Ledger is the main analytics system.",
                    score=1.0,
                    chunk_id="chunk-orchid",
                    chunk_index=0,
                )
            ]
        },
        chunk_entries=[
            chunk_entry(
                "chunk-orchid",
                filename="northstar.md",
                text="Orchid Ledger is the main analytics system.",
                heading="Systems",
            )
        ],
    )
    app.state.rag_evaluation_coordinator._retriever = app.state.retriever
    app.state.rag_evaluation_coordinator._dataset_exists_checker = lambda path: True
    app.state.rag_evaluation_coordinator._evaluation_docs_root = docs_root
    app.state.ensure_vector_index = lambda: None
    client = TestClient(app)

    response = client.post(
        "/rag/evaluate",
        headers=AUTH_HEADERS,
        json={"dataset": "default", "top_k": 25, "eval_k": [5, 10, 25]},
    )

    assert response.status_code == 202
    assert response.json()["status"] == "completed"
    assert response.json()["summary"]["dataset"] == "personal_research_assistant_rag_eval_v1"


def test_rag_evaluation_route_returns_404_for_missing_s3_dataset(monkeypatch) -> None:
    monkeypatch.setattr(
        "pra_api.routes.rag.s3_object_exists",
        lambda uri: False,
    )

    app = authenticated_app(
        dataset_path="s3://example-bucket/pra-prod/testing/rag-eval/rag_eval_dataset_v1.json"
    )
    app.state.ensure_vector_index = lambda: None
    client = TestClient(app)

    response = client.post(
        "/rag/evaluate",
        headers=AUTH_HEADERS,
        json={"dataset": "default", "top_k": 25, "eval_k": [5, 10, 25]},
    )

    assert response.status_code == 404
    assert "PRA_RAG_EVAL_DATASET_PATH" in response.json()["error"]


def test_rag_evaluation_route_gets_job_status(tmp_path: Path) -> None:
    docs_root = tmp_path / "docs"
    write_eval_docs(docs_root)
    dataset_path = tmp_path / "rag_eval_dataset_v1.json"
    dataset_path.write_text(
        json.dumps(
            {
                "name": "personal_research_assistant_rag_eval_v1",
                "default_top_k": 25,
                "eval_k": [5, 10, 25],
                "items": [
                    {
                        "id": "rag-eval-001",
                        "question": "What is Orchid Ledger?",
                        "source_files": ["northstar.md"],
                        "expected_sections": ["Systems"],
                        "relevant_chunk_ids": ["stale-id"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    app = authenticated_app(dataset_path=str(dataset_path))
    app.state.retriever = StaticRetriever(
        {
            "What is Orchid Ledger?": [
                RetrievedDocument(
                    document_id="doc-orchid",
                    filename="northstar.md",
                    text="Orchid Ledger is the main analytics system.",
                    score=1.0,
                    chunk_id="chunk-orchid",
                    chunk_index=0,
                )
            ]
        },
        chunk_entries=[
            chunk_entry(
                "chunk-orchid",
                filename="northstar.md",
                text="Orchid Ledger is the main analytics system.",
                heading="Systems",
            )
        ],
    )
    app.state.rag_evaluation_coordinator._retriever = app.state.retriever
    app.state.rag_evaluation_coordinator._evaluation_docs_root = docs_root
    app.state.ensure_vector_index = lambda: None
    client = TestClient(app)

    submitted = client.post(
        "/rag/evaluate",
        headers=AUTH_HEADERS,
        json={"dataset": "default", "top_k": 25, "eval_k": [5, 10, 25]},
    ).json()

    response = client.get(f"/rag/evaluate/{submitted['job_id']}", headers=AUTH_HEADERS)

    assert response.status_code == 200
    assert response.json()["status"] == "completed"


def test_rag_evaluation_route_returns_isolated_diagnostics_with_docs_corpus(
    tmp_path: Path,
) -> None:
    app = authenticated_app()
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    (docs_root / "northstar.md").write_text(
        "# Systems\n\n"
        "Orchid Ledger is the main incident and route analytics system.\n\n"
        "# Codes\n\n"
        "Blue Finch code BF-C means cold-chain carrier hold.",
        encoding="utf-8",
    )

    dataset_path = tmp_path / "rag_eval_dataset_v1.json"
    dataset_path.write_text(
        json.dumps(
            {
                "name": "personal_research_assistant_rag_eval_v1",
                "default_top_k": 25,
                "eval_k": [5, 10, 25],
                "items": [
                    {
                        "id": "rag-eval-001",
                        "question": "What is Orchid Ledger?",
                        "source_files": ["northstar.md"],
                        "expected_sections": ["Systems"],
                        "relevant_chunk_ids": ["stale-id"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    app.state.settings.rag_eval_dataset_path = str(dataset_path)
    app.state.rag_evaluation_coordinator._dataset_exists_checker = lambda path: True
    app.state.ensure_vector_index = lambda: None
    app.state.rag_evaluation_coordinator._ensure_vector_index = lambda: None
    app.state.rag_evaluation_coordinator._evaluation_docs_root = docs_root
    client = TestClient(app)

    response = client.post(
        "/rag/evaluate",
        headers=AUTH_HEADERS,
        json={"dataset": "default", "top_k": 25, "eval_k": [5, 10, 25]},
    )

    assert response.status_code == 202
    payload = response.json()
    assert payload["diagnostics"]["name"] == "isolated_docs"
    assert payload["summary"]["labeled_questions"] == 1
    assert payload["summary"]["unlabeled_questions"] == 0
    assert payload["summary"]["metrics"]["recall_at_5"] is not None
    assert payload["diagnostics"]["loss_summary"]["present_in_index"] == 1
    assert payload["diagnostics"]["loss_summary"]["missing_from_index"] == 0
    assert payload["diagnostics"]["questions"][0]["label_integrity"]["present_ids"] == [
        "northstar.md#chunk-0"
    ]
    assert "bm25" in payload["diagnostics"]["stage_metric_means"]


def test_isolated_evaluation_retriever_reuses_prebuilt_corpus_for_repeat_queries(
    tmp_path: Path,
) -> None:
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    (docs_root / "alpha.md").write_text(
        "# Alpha\n\nOrchid Ledger is the main analytics system.",
        encoding="utf-8",
    )
    dataset = RagEvaluationDataset.model_validate(
        {
            "name": "personal_research_assistant_rag_eval_v1",
            "items": [
                {
                    "id": "rag-eval-001",
                    "question": "What is Orchid Ledger?",
                    "source_files": ["alpha.md"],
                    "expected_sections": ["Alpha"],
                    "relevant_chunk_ids": ["stale-id"],
                }
            ],
        }
    )
    embedding_provider = CountingEmbeddingProvider()

    retriever, source_documents = build_isolated_evaluation_retriever(
        dataset=dataset,
        chunker=TextChunker(chunk_size=120, overlap=20),
        embedding_provider=embedding_provider,
        docs_root=docs_root,
    )

    first = retriever.search("What is Orchid Ledger?", k=5)
    second = retriever.search("What is Orchid Ledger?", k=5)

    assert len(source_documents) == 1
    assert first
    assert second
    assert first[0].chunk_id == second[0].chunk_id
    assert embedding_provider.calls.count("What is Orchid Ledger?") == 1


def test_evaluation_resolves_labels_from_live_index_after_chunk_size_change(
    tmp_path: Path,
) -> None:
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    (docs_root / "alpha.md").write_text(
        "# Alpha\n\n"
        "Orchid Ledger is the main analytics system for incident routing. "
        "It tracks exceptions, credits, carrier holds, and recovery timing.",
        encoding="utf-8",
    )
    dataset = RagEvaluationDataset.model_validate(
        {
            "name": "personal_research_assistant_rag_eval_v1",
            "items": [
                {
                    "id": "rag-eval-001",
                    "question": "What is Orchid Ledger?",
                    "source_files": ["alpha.md"],
                    "expected_sections": ["Alpha"],
                    "relevant_chunk_ids": ["old-index#chunk-99"],
                }
            ],
        }
    )

    for chunk_size in (12, 120):
        retriever, _source_documents = build_isolated_evaluation_retriever(
            dataset=dataset,
            chunker=TextChunker(chunk_size=chunk_size, overlap=0),
            embedding_provider=CountingEmbeddingProvider(),
            docs_root=docs_root,
        )

        result = evaluate_retrieval_only(
            retriever=retriever,
            dataset=dataset,
            top_k=5,
            eval_k=[5],
        )

        assert result.labeled_questions == 1
        assert result.unlabeled_questions == 0
        assert result.results[0].relevant_chunk_ids
        assert "old-index#chunk-99" not in result.results[0].relevant_chunk_ids
        assert result.results[0].ground_truth_warnings == []


def test_rag_evaluation_job_submit_enqueues_sqs_message() -> None:
    messages: list[dict[str, str]] = []

    class FakeSqsClient:
        def send_message(self, *, QueueUrl: str, MessageBody: str) -> None:
            messages.append({"QueueUrl": QueueUrl, "MessageBody": MessageBody})

    coordinator = RagEvaluationJobCoordinator(
        job_store=MemoryRagEvaluationJobStore(),
        retriever=StaticRetriever({}),
        chunker=TextChunker(chunk_size=120, overlap=20),
        embedding_provider=CountingEmbeddingProvider(),
        ensure_vector_index=lambda: None,
        queue_url="https://sqs.us-east-1.amazonaws.com/123456789012/rag-evaluation",
        sqs_client=FakeSqsClient(),
    )

    record = coordinator.submit(
        request=RagEvaluationRequest(),
        principal="cognito-user-1",
        trace_id="trace-rag-queue",
        dataset_path="s3://example-bucket/pra-prod/testing/rag-eval/rag_eval_dataset_v1.json",
    )

    assert record.status == "queued"
    assert len(messages) == 1
    assert messages[0]["QueueUrl"].endswith("/rag-evaluation")
    assert json.loads(messages[0]["MessageBody"]) == {"job_id": record.job_id}


def test_rag_evaluation_job_uses_dedicated_evaluation_embedding_provider(
    tmp_path: Path,
) -> None:
    docs_root = tmp_path / "docs"
    write_eval_docs(docs_root)
    dataset_path = tmp_path / "rag_eval_dataset.json"
    dataset_path.write_text(
        json.dumps(
            {
                "name": "rag-eval-test",
                "items": [
                    {
                        "id": "rag-eval-001",
                        "question": "What is Orchid Ledger?",
                        "source_files": ["northstar.md"],
                        "expected_sections": ["Systems"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    primary_embedding_provider = CountingEmbeddingProvider()
    evaluation_embedding_provider = CountingEmbeddingProvider()
    coordinator = RagEvaluationJobCoordinator(
        job_store=MemoryRagEvaluationJobStore(),
        retriever=StaticRetriever({}),
        chunker=TextChunker(chunk_size=120, overlap=20),
        embedding_provider=primary_embedding_provider,
        evaluation_embedding_provider=evaluation_embedding_provider,
        ensure_vector_index=lambda: None,
        evaluation_docs_root=docs_root,
    )

    record = coordinator.submit(
        request=RagEvaluationRequest(),
        principal="cognito-user-1",
        dataset_path=str(dataset_path),
    )

    assert record.status == "completed"
    assert primary_embedding_provider.calls == []
    assert evaluation_embedding_provider.calls
