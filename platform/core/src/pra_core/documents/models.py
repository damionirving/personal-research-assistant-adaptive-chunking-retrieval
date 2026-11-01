from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class ParsedDocument:
    filename: str
    content_type: str
    text: str


@dataclass(frozen=True)
class Document:
    id: str
    filename: str
    content_type: str
    text: str
    content_hash: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedDocument:
    document_id: str
    filename: str
    text: str
    score: float
    chunk_id: str | None = None
    chunk_index: int | None = None
    bm25_score: float | None = None
    vector_score: float | None = None
    rrf_score: float | None = None
    entity_anchor_score: float | None = None
    knowledge_graph_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentChunk:
    id: str
    document_id: str
    filename: str
    text: str
    index: int
    metadata: dict[str, Any] = field(default_factory=dict)
