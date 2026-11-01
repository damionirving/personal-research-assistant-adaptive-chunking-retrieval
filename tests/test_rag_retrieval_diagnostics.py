from pathlib import Path

from pra_core.embeddings.local import LocalHashEmbeddingProvider
from pra_core.evaluation.rag_retrieval import RagEvaluationDataset
from pra_core.evaluation.retrieval_diagnostics import (
    ChunkDiagnosticConfig,
    RetrievalDiagnosticConfig,
    run_retrieval_diagnostics,
)
from pra_core.evaluation.rag_retrieval import load_rag_evaluation_dataset


def test_rebuilt_v2_dataset_uses_current_source_corpus() -> None:
    dataset = load_rag_evaluation_dataset("data/rag_eval_dataset_v2.json")

    assert len(dataset.items) == 22
    assert {
        source
        for item in dataset.items
        for source in item.source_files
    } == {
        "acmecorp-benefits-policy.md",
        "scenario-northstar-meridian.md",
    }
    assert all(item.relevant_chunk_ids for item in dataset.items)


def _dataset() -> RagEvaluationDataset:
    return RagEvaluationDataset.model_validate(
        {
            "name": "diagnostic-dataset",
            "eval_k": [5, 10, 25],
            "items": [
                {
                    "id": "rag-eval-001",
                    "question": "What is Orchid Ledger?",
                    "source_files": ["alpha.md"],
                    "expected_sections": ["Orchid Ledger"],
                    "relevant_chunk_ids": ["stale-id"],
                },
                {
                    "id": "rag-eval-002",
                    "question": "What does Blue Finch BF-C mean?",
                    "source_files": ["alpha.md"],
                    "expected_sections": ["Blue Finch"],
                    "relevant_chunk_ids": ["stale-id"],
                },
                {
                    "id": "rag-eval-003",
                    "question": "What does Blue Finch BF-C mean?",
                    "source_files": ["beta.md"],
                    "expected_sections": ["Blue Finch"],
                    "relevant_chunk_ids": ["stale-id"],
                },
                {
                    "id": "rag-eval-004",
                    "question": "Define Cedar Route.",
                    "source_files": ["missing.md"],
                    "expected_sections": ["Cedar Route"],
                    "relevant_chunk_ids": ["stale-id"],
                },
                {
                    "id": "rag-eval-005",
                    "question": "Define Cedar Route.",
                    "source_files": ["beta.md"],
                    "expected_sections": ["Cedar Route"],
                    "relevant_chunk_ids": ["stale-id"],
                },
            ],
        }
    )


def _docs_root(tmp_path: Path) -> Path:
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    (docs_root / "alpha.md").write_text(
        "\n\n".join(
            [
                "# Orchid Ledger\nOrchid Ledger is the main analytics system.",
                "# Blue Finch\nBlue Finch code BF-C means cold-chain carrier hold.",
            ]
        ),
        encoding="utf-8",
    )
    (docs_root / "beta.md").write_text(
        "# Cedar Route\nCedar Route is a recovery lane.",
        encoding="utf-8",
    )
    return docs_root


def test_retrieval_diagnostics_report_label_integrity_and_stage_metrics(
    tmp_path: Path,
) -> None:
    report = run_retrieval_diagnostics(
        dataset=_dataset(),
        embedding_provider=LocalHashEmbeddingProvider(dimensions=1024),
        docs_root=_docs_root(tmp_path),
        base_config=RetrievalDiagnosticConfig(
            bm25_top_k=75,
            vector_top_k=100,
            rrf_k=30,
            rerank_top_k=50,
            llm_context_top_k=5,
            eval_k=[5, 10, 25],
        ),
        base_chunk_config=ChunkDiagnosticConfig(chunk_size=9, chunk_overlap=0),
        rrf_k_values=[],
        bm25_top_k_values=[],
        vector_top_k_values=[],
        chunk_configs=[],
    )

    issue_types = {issue.issue for issue in report.base_run.label_integrity_issues}
    assert "expected_section_not_found" in issue_types
    assert "source_file_not_found" in issue_types
    assert report.base_run.stage_metric_means["bm25"]["recall_at_5"] is not None
    assert report.base_run.stage_metric_means["bm25"]["mrr_at_10"] is not None
    assert report.base_run.stage_metric_means["bm25"]["ndcg_at_10"] is not None
    assert report.base_run.stage_metric_means["vector"]["precision_at_10"] is not None
    assert report.base_run.stage_metric_means["hybrid_pre_rerank"]["recall_at_25"] is not None
    assert report.base_run.stage_metric_means["hybrid_post_rerank"]["precision_at_25"] is not None
    assert report.recommended_run_name


def test_retrieval_diagnostics_tracks_relevant_chunk_survival_and_latency(
    tmp_path: Path,
) -> None:
    report = run_retrieval_diagnostics(
        dataset=_dataset(),
        embedding_provider=LocalHashEmbeddingProvider(dimensions=1024),
        docs_root=_docs_root(tmp_path),
        base_config=RetrievalDiagnosticConfig(eval_k=[5, 10, 25]),
        base_chunk_config=ChunkDiagnosticConfig(chunk_size=9, chunk_overlap=0),
        rrf_k_values=[],
        bm25_top_k_values=[],
        vector_top_k_values=[],
        chunk_configs=[],
    )

    first_question = report.base_run.questions[0]
    survival = first_question.relevant_chunk_survival[0]

    assert survival.relevant_chunk_id == "alpha.md#chunk-0"
    assert survival.bm25_rank is not None
    assert survival.rrf_rank is not None
    assert survival.reranked_rank is not None
    assert survival.included_in_final_llm_context is True
    assert first_question.latency.bm25_latency_ms >= 0
    assert first_question.latency.vector_latency_ms >= 0
    assert first_question.latency.rrf_merge_latency_ms >= 0
    assert first_question.latency.reranker_latency_ms >= 0
    assert first_question.latency.total_retrieval_latency_ms >= 0
    assert report.base_run.final_context_coverage_rate >= 0
    assert report.base_run.contribution_summary.shared_relevant_hits >= 0


def test_retrieval_diagnostics_runs_configurable_ablations(tmp_path: Path) -> None:
    report = run_retrieval_diagnostics(
        dataset=_dataset(),
        embedding_provider=LocalHashEmbeddingProvider(dimensions=1024),
        docs_root=_docs_root(tmp_path),
        base_config=RetrievalDiagnosticConfig(eval_k=[5, 10, 25]),
        base_chunk_config=ChunkDiagnosticConfig(chunk_size=9, chunk_overlap=0),
        rrf_k_values=[20, 90],
        bm25_top_k_values=[25],
        vector_top_k_values=[50],
        chunk_configs=[
            ChunkDiagnosticConfig(chunk_size=9, chunk_overlap=0),
            ChunkDiagnosticConfig(chunk_size=12, chunk_overlap=0),
        ],
    )

    names = {run.name for run in report.ablations}
    assert {"rrf_k=20", "rrf_k=90", "bm25_top_k=25", "vector_top_k=50"} <= names
    assert "chunk=12/0" in names
    assert "bm25_only" in names
    assert "vector_only" in names
    assert "weighted_rrf_0.8_0.2_preserve_bm25" in names


def test_weighted_bm25_preserving_hybrid_clears_rrf_regression_threshold(
    tmp_path: Path,
) -> None:
    docs_root = _docs_root(tmp_path)
    dataset = _dataset()
    report = run_retrieval_diagnostics(
        dataset=dataset,
        embedding_provider=LocalHashEmbeddingProvider(dimensions=1024),
        docs_root=docs_root,
        base_config=RetrievalDiagnosticConfig(
            fusion_strategy="weighted_rrf",
            bm25_fusion_weight=0.8,
            vector_fusion_weight=0.2,
            preserve_bm25_candidates=True,
            enable_reranking=False,
            eval_k=[5, 10, 25],
        ),
        base_chunk_config=ChunkDiagnosticConfig(chunk_size=9, chunk_overlap=0),
        rrf_k_values=[],
        bm25_top_k_values=[],
        vector_top_k_values=[],
        chunk_configs=[],
    )

    weighted_ndcg = report.base_run.stage_metric_means["hybrid_post_rerank"]["ndcg_at_10"]
    bm25_ndcg = report.base_run.stage_metric_means["bm25"]["ndcg_at_10"]

    assert weighted_ndcg is not None
    assert bm25_ndcg is not None
    assert weighted_ndcg >= bm25_ndcg
