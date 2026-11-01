from pra_core.documents.chunking import TextChunker
from pra_core.documents.models import Document


def _document(text: str, filename: str = "notes.md") -> Document:
    return Document(
        id="doc-1",
        filename=filename,
        content_type="text/markdown",
        text=text,
    )


def test_chunker_uses_whole_document_for_small_files() -> None:
    chunks = TextChunker(chunk_size=120, overlap=20).chunk(
        _document("Tiny note for review.")
    )

    assert len(chunks) == 1
    assert chunks[0].metadata["strategy"] == "whole_document"


def test_chunker_uses_section_aware_chunks_when_headings_fit() -> None:
    text = """
# Overview
System summary and scope notes.

## Risks
Security review items and deployment cautions.
""".strip()

    chunks = TextChunker(chunk_size=20, overlap=4).chunk(_document(text))

    assert [chunk.metadata["strategy"] for chunk in chunks] == [
        "section_aware",
        "section_aware",
    ]
    assert chunks[0].metadata["heading"] == "Overview"
    assert chunks[1].metadata["heading"] == "Risks"


def test_chunker_uses_paragraph_fallback_when_no_headings_exist() -> None:
    text = """
This first paragraph explains the upload flow in practical terms.

This second paragraph explains retrieval behavior for users.
""".strip()

    chunks = TextChunker(chunk_size=20, overlap=4).chunk(_document(text, "notes.txt"))

    assert [chunk.metadata["strategy"] for chunk in chunks] == [
        "paragraph_aware",
        "paragraph_aware",
    ]
    assert chunks[0].metadata["paragraph_index"] == 0
    assert chunks[1].metadata["paragraph_index"] == 1


def test_chunker_uses_sentence_fallback_when_sentences_fit() -> None:
    text = (
        "Upload indexing stores the parsed file first. "
        "Retrieval runs only after embeddings are ready. "
        "Users can chat again after indexing completes."
    )

    chunks = TextChunker(chunk_size=8, overlap=2).chunk(_document(text, "notes.txt"))

    assert [chunk.metadata["strategy"] for chunk in chunks] == [
        "sentence_aware",
        "sentence_aware",
        "sentence_aware",
    ]
    assert all(chunk.metadata["sentence_count"] == 1 for chunk in chunks)


def test_chunker_uses_semantic_fallback_when_sentence_grouping_is_too_large() -> None:
    text = (
        "Alpha platform compliance policy remains aligned with audit records today. "
        "Alpha platform compliance review confirms the same control posture today. "
        "Banana orchard harvest schedule changes after severe overnight weather damage. "
        "Banana orchard field crews reschedule packing operations after the storm."
    )

    chunks = TextChunker(chunk_size=16, overlap=2).chunk(_document(text, "notes.txt"))

    assert len(chunks) >= 2
    assert all(chunk.metadata["strategy"] == "semantic_deterministic" for chunk in chunks)
    assert chunks[0].metadata["semantic_embedding_provider"] == "local-hash:32"
    assert chunks[0].metadata["semantic_similarity_threshold"] == 0.72


def test_chunker_uses_sliding_window_as_final_fallback() -> None:
    text = " ".join(f"token{i}" for i in range(1, 13))

    chunks = TextChunker(chunk_size=4, overlap=2).chunk(_document(text, "notes.txt"))

    assert [chunk.text for chunk in chunks] == [
        "token1 token2 token3 token4",
        "token3 token4 token5 token6",
        "token5 token6 token7 token8",
        "token7 token8 token9 token10",
        "token9 token10 token11 token12",
    ]
    assert all(chunk.metadata["strategy"] == "sliding_window" for chunk in chunks)
    assert chunks[0].metadata["start_word_index"] == 0
    assert chunks[1].metadata["window_overlap"] == 2
