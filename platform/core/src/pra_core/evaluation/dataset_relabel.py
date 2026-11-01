from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable
from pathlib import Path

from pra_core.documents.chunking import TextChunker
from pra_core.documents.filenames import normalize_filename
from pra_core.documents.models import Document
from pra_core.evaluation.rag_retrieval import (
    RagEvaluationDataset,
    RagEvaluationItem,
    load_evaluation_source_documents,
)


@dataclass(frozen=True)
class RebuiltChunk:
    chunk_id: str
    filename: str
    text: str
    index: int
    metadata: dict


def rebuild_rag_evaluation_dataset(
    *,
    dataset: RagEvaluationDataset,
    docs_root: Path,
    chunker: TextChunker,
) -> RagEvaluationDataset:
    duplicate_names = find_duplicate_normalized_filenames(docs_root.rglob("*.md"))
    if duplicate_names:
        rendered = ", ".join(
            f"{name}: {', '.join(paths)}"
            for name, paths in sorted(duplicate_names.items())
        )
        raise ValueError(f"Duplicate normalized filenames found in evaluation corpus: {rendered}")

    corpus = _build_chunk_corpus(dataset=dataset, docs_root=docs_root, chunker=chunker)
    rebuilt_items = [
        _relabel_item(item, corpus)
        for item in dataset.items
    ]
    rebuilt_dataset = dataset.model_copy(
        update={
            "name": "personal_research_assistant_rag_eval_v2",
            "items": rebuilt_items,
        }
    )
    assert_dataset_labels_exist(
        dataset=rebuilt_dataset,
        docs_root=docs_root,
        chunker=chunker,
    )
    return rebuilt_dataset


def write_rag_evaluation_dataset(dataset: RagEvaluationDataset, output_path: Path) -> None:
    output_path.write_text(
        json.dumps(dataset.model_dump(), indent=2),
        encoding="utf-8",
    )


def assert_dataset_labels_exist(
    *,
    dataset: RagEvaluationDataset,
    docs_root: Path,
    chunker: TextChunker,
) -> None:
    corpus = _build_chunk_corpus(dataset=dataset, docs_root=docs_root, chunker=chunker)
    available = {
        chunk.chunk_id
        for chunks in corpus.values()
        for chunk in chunks
    }
    missing_by_question: dict[str, list[str]] = {}
    for item in dataset.items:
        missing = [
            label
            for label in item.relevant_chunk_ids
            if label not in available
        ]
        if missing:
            missing_by_question[item.id] = missing
    if missing_by_question:
        rendered = "; ".join(
            f"{question_id}: {', '.join(labels)}"
            for question_id, labels in sorted(missing_by_question.items())
        )
        raise ValueError(f"Stale relevant_chunk_ids detected: {rendered}")


def assert_dataset_labels_match_current_resolution(
    *,
    dataset: RagEvaluationDataset,
    docs_root: Path,
    chunker: TextChunker,
) -> None:
    rebuilt = rebuild_rag_evaluation_dataset(
        dataset=dataset,
        docs_root=docs_root,
        chunker=chunker,
    )
    drifted: dict[str, tuple[list[str], list[str]]] = {}
    rebuilt_items = {item.id: item for item in rebuilt.items}
    for item in dataset.items:
        current_ids = item.relevant_chunk_ids
        resolved_ids = rebuilt_items[item.id].relevant_chunk_ids
        if current_ids != resolved_ids:
            drifted[item.id] = (current_ids, resolved_ids)
    if drifted:
        rendered = "; ".join(
            (
                f"{question_id}: checked_in={current_ids} "
                f"resolved_now={resolved_ids}"
            )
            for question_id, (current_ids, resolved_ids) in sorted(drifted.items())
        )
        raise ValueError(f"Materialized relevant_chunk_ids drift detected: {rendered}")


def find_duplicate_normalized_filenames(paths: Iterable[Path | str]) -> dict[str, list[str]]:
    counter: Counter[str] = Counter()
    grouped: dict[str, list[str]] = {}
    for raw_path in paths:
        path = Path(raw_path)
        normalized = normalize_filename(path.name)
        counter[normalized] += 1
        grouped.setdefault(normalized, []).append(str(path))
    return {
        name: values
        for name, values in grouped.items()
        if counter[name] > 1
    }


def _build_chunk_corpus(
    *,
    dataset: RagEvaluationDataset,
    docs_root: Path,
    chunker: TextChunker,
) -> dict[str, list[RebuiltChunk]]:
    source_documents = load_evaluation_source_documents(dataset, docs_root=docs_root)
    corpus: dict[str, list[RebuiltChunk]] = {}
    for source_document in source_documents:
        normalized_filename = normalize_filename(source_document.resolved_name)
        text = Path(source_document.path).read_text(encoding="utf-8")
        chunks = chunker.chunk(
            Document(
                id=normalized_filename,
                filename=normalized_filename,
                content_type="text/markdown",
                text=text,
            )
        )
        corpus[normalized_filename] = [
            RebuiltChunk(
                chunk_id=chunk.id,
                filename=chunk.filename,
                text=chunk.text,
                index=chunk.index,
                metadata=chunk.metadata,
            )
            for chunk in chunks
        ]
    return corpus


def _relabel_item(
    item: RagEvaluationItem,
    corpus: dict[str, list[RebuiltChunk]],
) -> RagEvaluationItem:
    source_files = [normalize_filename(name) for name in item.source_files]
    candidates = [
        chunk
        for source_file in source_files
        for chunk in corpus.get(source_file, [])
    ]
    selected_ids = _select_chunk_ids(
        item=item,
        candidates=candidates,
    )
    return item.model_copy(update={"source_files": source_files, "relevant_chunk_ids": selected_ids})


def _select_chunk_ids(
    *,
    item: RagEvaluationItem,
    candidates: list[RebuiltChunk],
) -> list[str]:
    if not candidates:
        return []

    selected: list[str] = []
    selected_set: set[str] = set()

    for section in item.expected_sections:
        ranked = sorted(
            candidates,
            key=lambda chunk: _section_match_score(section, item, chunk),
            reverse=True,
        )
        if not ranked:
            continue
        top = ranked[0]
        if _section_match_score(section, item, top) <= 0:
            continue
        if top.chunk_id not in selected_set:
            selected.append(top.chunk_id)
            selected_set.add(top.chunk_id)

    ranked_global = sorted(
        candidates,
        key=lambda chunk: _global_match_score(item, chunk),
        reverse=True,
    )
    max_chunks = min(max(len(item.expected_sections), 1), 4)
    for chunk in ranked_global:
        if len(selected) >= max_chunks:
            break
        if chunk.chunk_id in selected_set:
            continue
        if _global_match_score(item, chunk) <= 0:
            continue
        selected.append(chunk.chunk_id)
        selected_set.add(chunk.chunk_id)

    return selected[:max_chunks]


def _section_match_score(section: str, item: RagEvaluationItem, chunk: RebuiltChunk) -> float:
    heading = str(chunk.metadata.get("heading", ""))
    heading_tokens = set(_terms(heading))
    section_tokens = set(_terms(section))
    text_tokens = set(_terms(chunk.text))

    score = 0.0
    if heading.strip().casefold() == section.strip().casefold():
        score += 100.0
    if section.strip() and section.strip().casefold() in chunk.text.casefold():
        score += 25.0
    score += len(heading_tokens & section_tokens) * 15.0
    score += len(text_tokens & section_tokens) * 3.0
    score += _global_match_score(item, chunk)
    return score


def _global_match_score(item: RagEvaluationItem, chunk: RebuiltChunk) -> float:
    chunk_tokens = set(_terms(chunk.text))
    score = 0.0
    score += len(chunk_tokens & set(_terms(item.question))) * 2.0
    score += len(chunk_tokens & set(_terms(item.expected_answer_summary))) * 4.0
    score += len(chunk_tokens & set(_terms(" ".join(item.expected_sections)))) * 3.0
    if item.intent:
        score += len(chunk_tokens & set(_terms(item.intent))) * 1.5
    return score


def _terms(value: str) -> list[str]:
    return [term.lower() for term in re.findall(r"[a-zA-Z0-9]+", value)]
