import math
import re

from pra_core.documents.filenames import normalize_filename
from pra_core.documents.models import Document, DocumentChunk
from pra_core.embeddings.local import LocalHashEmbeddingProvider


class TextChunker:
    """Deterministic chunker with a fixed non-LLM decision chain."""

    _SMALL_DOCUMENT_WORD_THRESHOLD = 5
    _SEMANTIC_EMBEDDING_DIMENSIONS = 32
    _SEMANTIC_SIMILARITY_THRESHOLD = 0.72
    _SEMANTIC_PREPROCESSING_NAME = "lowercase-alnum-collapse-whitespace"

    def __init__(self, chunk_size: int = 120, overlap: int = 20) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be greater than or equal to 0 and less than chunk_size.")
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._semantic_embedder = LocalHashEmbeddingProvider(
            dimensions=self._SEMANTIC_EMBEDDING_DIMENSIONS
        )

    def chunk(self, document: Document) -> list[DocumentChunk]:
        normalized_text = document.text.strip()
        if not normalized_text:
            return []

        total_words = _word_count(normalized_text)
        if total_words <= self._SMALL_DOCUMENT_WORD_THRESHOLD:
            return [
                self._build_chunk(
                    document=document,
                    index=0,
                    text=normalized_text,
                    strategy="whole_document",
                    metadata={"word_count": total_words},
                )
            ]

        sections = self._section_blocks(normalized_text)
        if sections and all(_word_count(section["text"]) <= self._chunk_size for section in sections):
            return [
                self._build_chunk(
                    document=document,
                    index=index,
                    text=section["text"],
                    strategy="section_aware",
                    metadata={
                        "heading": section["heading"],
                        "section_index": index,
                        "word_count": _word_count(section["text"]),
                    },
                )
                for index, section in enumerate(sections)
            ]

        paragraphs = self._paragraph_blocks(normalized_text)
        if len(paragraphs) >= 2 and all(_word_count(paragraph) <= self._chunk_size for paragraph in paragraphs):
            return [
                self._build_chunk(
                    document=document,
                    index=index,
                    text=paragraph,
                    strategy="paragraph_aware",
                    metadata={
                        "paragraph_index": index,
                        "word_count": _word_count(paragraph),
                    },
                )
                for index, paragraph in enumerate(paragraphs)
            ]

        sentences = self._sentence_blocks(normalized_text)
        sentence_chunks = self._sentence_chunks(document, sentences)
        if sentence_chunks:
            return sentence_chunks

        semantic_chunks = self._semantic_chunks(document, sentences)
        if semantic_chunks:
            return semantic_chunks

        return self._sliding_window_chunks(document, normalized_text)

    def _build_chunk(
        self,
        document: Document,
        index: int,
        text: str,
        strategy: str,
        metadata: dict,
    ) -> DocumentChunk:
        cleaned = " ".join(text.split())
        base_metadata = {
            "strategy": strategy,
            "word_count": _word_count(cleaned),
            "character_count": len(cleaned),
        }
        base_metadata.update(metadata)
        normalized_filename = normalize_filename(document.filename)
        return DocumentChunk(
            id=f"{normalized_filename}#chunk-{index}",
            document_id=document.id,
            filename=normalized_filename,
            text=cleaned,
            index=index,
            metadata=base_metadata,
        )

    def _section_blocks(self, text: str) -> list[dict[str, str]]:
        lines = text.splitlines()
        if not any(re.match(r"^\s{0,3}#{1,6}\s+\S", line) for line in lines):
            return []

        sections: list[dict[str, str]] = []
        current_heading = ""
        current_lines: list[str] = []

        for line in lines:
            if re.match(r"^\s{0,3}#{1,6}\s+\S", line):
                if current_lines:
                    sections.append(
                        {
                            "heading": current_heading,
                            "text": "\n".join(current_lines).strip(),
                        }
                    )
                current_heading = re.sub(r"^\s{0,3}#{1,6}\s+", "", line).strip()
                current_lines = [line.strip()]
                continue
            current_lines.append(line)

        if current_lines:
            sections.append(
                {
                    "heading": current_heading,
                    "text": "\n".join(current_lines).strip(),
                }
            )

        return [section for section in sections if section["text"]]

    def _paragraph_blocks(self, text: str) -> list[str]:
        return [
            " ".join(block.split())
            for block in re.split(r"\n\s*\n+", text)
            if block.strip()
        ]

    def _sentence_blocks(self, text: str) -> list[str]:
        parts = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
        return [part.strip() for part in parts if part.strip()]

    def _sentence_chunks(
        self,
        document: Document,
        sentences: list[str],
    ) -> list[DocumentChunk]:
        if (
            len(sentences) < 2
            or len(sentences) > 3
            or any(_word_count(sentence) > self._chunk_size for sentence in sentences)
        ):
            return []

        chunks: list[DocumentChunk] = []
        current_sentences: list[str] = []
        current_words = 0

        for sentence in sentences:
            sentence_words = _word_count(sentence)
            if current_sentences and current_words + sentence_words > self._chunk_size:
                chunks.append(
                    self._build_chunk(
                        document=document,
                        index=len(chunks),
                        text=" ".join(current_sentences),
                        strategy="sentence_aware",
                        metadata={"sentence_count": len(current_sentences)},
                    )
                )
                current_sentences = []
                current_words = 0
            current_sentences.append(sentence)
            current_words += sentence_words

        if current_sentences:
            chunks.append(
                self._build_chunk(
                    document=document,
                    index=len(chunks),
                    text=" ".join(current_sentences),
                    strategy="sentence_aware",
                    metadata={"sentence_count": len(current_sentences)},
                )
            )

        return chunks if chunks else []

    def _semantic_chunks(
        self,
        document: Document,
        sentences: list[str],
    ) -> list[DocumentChunk]:
        if len(sentences) < 2:
            return []

        prepared = [
            {
                "raw": sentence,
                "normalized": _semantic_normalize(sentence),
                "word_count": _word_count(sentence),
            }
            for sentence in sentences
        ]
        if any(item["word_count"] > self._chunk_size for item in prepared):
            return []

        embeddings = [
            self._semantic_embedder.embed(item["normalized"])
            for item in prepared
        ]

        chunks: list[DocumentChunk] = []
        current_sentences = [prepared[0]["raw"]]
        current_words = int(prepared[0]["word_count"])
        current_embedding = embeddings[0]

        for item, embedding in zip(prepared[1:], embeddings[1:]):
            similarity = _cosine_similarity(current_embedding, embedding)
            next_words = int(item["word_count"])
            if (
                current_sentences
                and (
                    current_words + next_words > self._chunk_size
                    or similarity < self._SEMANTIC_SIMILARITY_THRESHOLD
                )
            ):
                chunks.append(
                    self._build_chunk(
                        document=document,
                        index=len(chunks),
                        text=" ".join(current_sentences),
                        strategy="semantic_deterministic",
                        metadata={
                            "sentence_count": len(current_sentences),
                            "semantic_similarity_threshold": self._SEMANTIC_SIMILARITY_THRESHOLD,
                            "semantic_preprocessing": self._SEMANTIC_PREPROCESSING_NAME,
                            "semantic_embedding_provider": self._semantic_embedder.provider_id,
                        },
                    )
                )
                current_sentences = []
                current_words = 0
                current_embedding = embedding

            current_sentences.append(item["raw"])
            current_words += next_words
            current_embedding = _average_embedding(current_embedding, embedding)

        if current_sentences:
            chunks.append(
                self._build_chunk(
                    document=document,
                    index=len(chunks),
                    text=" ".join(current_sentences),
                    strategy="semantic_deterministic",
                    metadata={
                        "sentence_count": len(current_sentences),
                        "semantic_similarity_threshold": self._SEMANTIC_SIMILARITY_THRESHOLD,
                        "semantic_preprocessing": self._SEMANTIC_PREPROCESSING_NAME,
                        "semantic_embedding_provider": self._semantic_embedder.provider_id,
                    },
                )
            )

        return chunks if len(chunks) > 1 else []

    def _sliding_window_chunks(
        self,
        document: Document,
        text: str,
    ) -> list[DocumentChunk]:
        words = text.split()
        chunks: list[DocumentChunk] = []
        step = self._chunk_size - self._overlap
        start = 0
        index = 0

        while start < len(words):
            end = min(start + self._chunk_size, len(words))
            chunk_text = " ".join(words[start:end])
            chunks.append(
                self._build_chunk(
                    document=document,
                    index=index,
                    text=chunk_text,
                    strategy="sliding_window",
                    metadata={
                        "start_word_index": start,
                        "end_word_index": end - 1,
                        "window_size": self._chunk_size,
                        "window_overlap": self._overlap,
                    },
                )
            )
            if end == len(words):
                break
            start += step
            index += 1

        return chunks


def _word_count(text: str) -> int:
    return len(text.split())


def _semantic_normalize(text: str) -> str:
    lowered = text.lower()
    alnum = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return " ".join(alnum.split())


def _cosine_similarity(first: list[float], second: list[float]) -> float:
    numerator = sum(left * right for left, right in zip(first, second))
    left_norm = math.sqrt(sum(value * value for value in first))
    right_norm = math.sqrt(sum(value * value for value in second))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _average_embedding(first: list[float], second: list[float]) -> list[float]:
    averaged = [(left + right) / 2.0 for left, right in zip(first, second)]
    norm = math.sqrt(sum(value * value for value in averaged))
    if norm == 0:
        return averaged
    return [value / norm for value in averaged]
