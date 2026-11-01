from pathlib import Path

import pytest

from pra_core.documents.chunking import TextChunker
from pra_core.documents.filenames import normalize_filename
from pra_core.documents.models import Document
from pra_core.evaluation.dataset_relabel import (
    assert_dataset_labels_exist,
    assert_dataset_labels_match_current_resolution,
    find_duplicate_normalized_filenames,
    rebuild_rag_evaluation_dataset,
)
from pra_core.evaluation.rag_retrieval import RagEvaluationDataset


ROOT = Path(__file__).resolve().parents[1]


def test_normalize_filename_removes_duplicate_suffix_and_preserves_extension() -> None:
    assert normalize_filename("architecture-overview(3).md") == "architecture-overview.md"
    assert normalize_filename("guardrails(1).md") == "guardrails.md"
    assert normalize_filename("report(12).txt") == "report.txt"


def test_text_chunker_builds_deterministic_chunk_ids_from_normalized_filename() -> None:
    chunks = TextChunker(chunk_size=4, overlap=2).chunk(
        Document(
            id="doc-1",
            filename="notes(2).txt",
            content_type="text/plain",
            text="one two three four five six",
        )
    )

    assert [chunk.id for chunk in chunks] == [
        "notes.txt#chunk-0",
        "notes.txt#chunk-1",
    ]


def test_find_duplicate_normalized_filenames_reports_collisions(tmp_path: Path) -> None:
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    first = docs_root / "guardrails.md"
    second = docs_root / "guardrails(1).md"
    first.write_text("# A\n", encoding="utf-8")
    second.write_text("# B\n", encoding="utf-8")

    duplicates = find_duplicate_normalized_filenames([first, second])

    assert sorted(duplicates) == ["guardrails.md"]


def test_repo_docs_do_not_contain_duplicate_normalized_filenames() -> None:
    duplicates = find_duplicate_normalized_filenames((ROOT / "docs").rglob("*.md"))

    assert duplicates == {}


def test_rebuilt_dataset_labels_exist_in_rebuilt_index(tmp_path: Path) -> None:
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    (docs_root / "guardrails.md").write_text(
        "# Current Stage\n\nA deterministic prompt guardrail runs before retrieval.\n\n"
        "# Implemented Guardrails\n\nA grounding-risk policy runs after generation.",
        encoding="utf-8",
    )
    (docs_root / "architecture-overview.md").write_text(
        "# Runtime Flow\n\nBrowser calls API Gateway and Lambda.",
        encoding="utf-8",
    )
    dataset = RagEvaluationDataset.model_validate(
        {
            "name": "personal_research_assistant_rag_eval_v1",
            "eval_k": [5, 10, 25],
            "items": [
                {
                    "id": "rag-eval-001",
                    "question": "What guardrails are implemented?",
                    "intent": "guardrails",
                    "source_files": ["guardrails(1).md"],
                    "expected_sections": ["Implemented Guardrails"],
                    "relevant_chunk_ids": [],
                    "expected_answer_summary": "A grounding-risk policy runs after generation.",
                },
                {
                    "id": "rag-eval-002",
                    "question": "What is the runtime flow?",
                    "source_files": ["architecture-overview(3).md"],
                    "expected_sections": ["Runtime Flow"],
                    "relevant_chunk_ids": [],
                    "expected_answer_summary": "Browser calls API Gateway and Lambda.",
                },
            ],
        }
    )

    rebuilt = rebuild_rag_evaluation_dataset(
        dataset=dataset,
        docs_root=docs_root,
        chunker=TextChunker(chunk_size=120, overlap=20),
    )

    assert rebuilt.name == "personal_research_assistant_rag_eval_v2"
    assert rebuilt.items[0].relevant_chunk_ids == ["guardrails.md#chunk-1"]
    assert rebuilt.items[1].relevant_chunk_ids == ["architecture-overview.md#chunk-0"]
    assert_dataset_labels_exist(
        dataset=rebuilt,
        docs_root=docs_root,
        chunker=TextChunker(chunk_size=120, overlap=20),
    )


def test_dataset_label_validation_fails_clearly_for_stale_labels(tmp_path: Path) -> None:
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    (docs_root / "guardrails.md").write_text(
        "# Current Stage\n\nA deterministic prompt guardrail runs before retrieval.",
        encoding="utf-8",
    )
    dataset = RagEvaluationDataset.model_validate(
        {
            "name": "personal_research_assistant_rag_eval_v1",
            "items": [
                {
                    "id": "rag-eval-001",
                    "question": "What guardrails are implemented?",
                    "source_files": ["guardrails.md"],
                    "expected_sections": ["Current Stage"],
                    "relevant_chunk_ids": ["guardrails.md#chunk-99"],
                }
            ],
        }
    )

    with pytest.raises(ValueError, match="Stale relevant_chunk_ids detected"):
        assert_dataset_labels_exist(
            dataset=dataset,
            docs_root=docs_root,
            chunker=TextChunker(chunk_size=120, overlap=20),
        )


def test_dataset_label_resolution_match_fails_clearly_for_drifted_materialized_ids(
    tmp_path: Path,
) -> None:
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    (docs_root / "guardrails.md").write_text(
        "# Current Stage\n\nA deterministic prompt guardrail runs before retrieval.",
        encoding="utf-8",
    )
    dataset = RagEvaluationDataset.model_validate(
        {
            "name": "personal_research_assistant_rag_eval_v2",
            "items": [
                {
                    "id": "rag-eval-001",
                    "question": "What guardrails are implemented?",
                    "source_files": ["guardrails.md"],
                    "expected_sections": ["Current Stage"],
                    "relevant_chunk_ids": ["guardrails.md#chunk-99"],
                }
            ],
        }
    )

    with pytest.raises(ValueError, match="Materialized relevant_chunk_ids drift detected"):
        assert_dataset_labels_match_current_resolution(
            dataset=dataset,
            docs_root=docs_root,
            chunker=TextChunker(chunk_size=120, overlap=20),
        )


def test_checked_in_v2_dataset_labels_exist_in_repo_docs() -> None:
    dataset = RagEvaluationDataset.model_validate_json(
        (ROOT / "data" / "rag_eval_dataset_v2.json").read_text(encoding="utf-8")
    )

    assert_dataset_labels_match_current_resolution(
        dataset=dataset,
        docs_root=ROOT / "docs",
        chunker=TextChunker(chunk_size=600, overlap=100),
    )
    assert_dataset_labels_exist(
        dataset=dataset,
        docs_root=ROOT / "docs",
        chunker=TextChunker(chunk_size=600, overlap=100),
    )
