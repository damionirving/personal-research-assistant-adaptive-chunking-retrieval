from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any

SCHEMA_VERSION = "adaptive-rag/v1"


def stable_id(*parts: str) -> str:
    return sha256("|".join(parts).encode("utf-8")).hexdigest()[:24]


@dataclass(frozen=True)
class Classification:
    primary: str = "general"
    secondary: tuple[str, ...] = ()
    confidence: float = 0.0


@dataclass(frozen=True)
class ChunkDecision:
    strategy: str
    reasons: tuple[str, ...]
    confidence: float
    deterministic_signals: dict[str, int | bool | str]
    bedrock_used: bool = False
    fallback_strategy: str = "fixed_size"


@dataclass(frozen=True)
class ParentArtifact:
    id: str
    document_id: str
    title: str
    section_path: tuple[str, ...]
    text: str
    ordinal: int
    domain: str
    document_type: str


@dataclass(frozen=True)
class ChildArtifact:
    id: str
    document_id: str
    parent_id: str
    filename: str
    text: str
    ordinal: int
    chunk_type: str
    section_path: tuple[str, ...]
    token_count: int
    domain: str
    document_type: str
    strategy: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RelationshipArtifact:
    source_id: str
    relationship_type: str
    target_id: str
    evidence: str
    confidence: float


@dataclass(frozen=True)
class TableArtifact:
    id: str
    parent_id: str
    title: str
    raw_text: str
    rows: tuple[tuple[str, ...], ...]
    confidence: float


@dataclass(frozen=True)
class AdaptiveManifest:
    document_id: str
    document_checksum: str
    decision: ChunkDecision
    domain: Classification
    document_type: Classification
    counts: dict[str, int]
    warnings: tuple[str, ...] = ()
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QueryPlan:
    query: str
    intent: str
    domain: str
    entities: tuple[str, ...]
    exact_identifiers: tuple[str, ...]
    expanded_terms: tuple[str, ...]
    metadata_filters: dict[str, str]
    use_bm25: bool
    use_vector: bool
    use_relationships: bool
    parent_expansion: bool
    weights: dict[str, float]
    confidence: float
    final_context_top_k: int = 5
    bm25_first: bool = True


@dataclass(frozen=True)
class RetrievalCandidate:
    child: ChildArtifact
    score: float
    sources: tuple[str, ...]
    bm25_score: float = 0.0
    vector_score: float = 0.0
    relationship_score: float = 0.0
    fusion_score: float = 0.0
    relationship_evidence: tuple[str, ...] = ()
