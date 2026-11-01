from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from pra_core.documents.models import RetrievedDocument


FusionStrategy = Literal["rrf", "weighted_rrf", "normalized_weighted"]


@dataclass(frozen=True)
class FusionConfig:
    strategy: FusionStrategy = "rrf"
    rrf_k: int = 60
    bm25_weight: float = 0.7
    vector_weight: float = 0.3
    preserve_bm25_candidates: bool = False


def merge_retrieval_hits(
    *,
    bm25_hits: list[RetrievedDocument],
    vector_hits: list[RetrievedDocument],
    config: FusionConfig,
    key_fn: Callable[[RetrievedDocument], str],
    prefer_vector_chunk: bool = True,
) -> dict[str, RetrievedDocument]:
    merged: dict[str, RetrievedDocument] = {}
    bm25_rank_scores = _rank_scores(
        bm25_hits,
        weight=1.0 if config.strategy == "rrf" else config.bm25_weight,
        rrf_k=config.rrf_k,
        key_fn=key_fn,
    )
    vector_rank_scores = _rank_scores(
        vector_hits,
        weight=1.0 if config.strategy == "rrf" else config.vector_weight,
        rrf_k=config.rrf_k,
        key_fn=key_fn,
    )
    bm25_norm_scores = _normalized_raw_scores(
        bm25_hits,
        lambda item: item.bm25_score or 0.0,
        key_fn=key_fn,
    )
    vector_norm_scores = _normalized_raw_scores(
        vector_hits,
        lambda item: item.vector_score or 0.0,
        key_fn=key_fn,
    )

    for hit in [*bm25_hits, *vector_hits]:
        key = key_fn(hit)
        bm25_rank_score = bm25_rank_scores.get(key, 0.0)
        vector_rank_score = vector_rank_scores.get(key, 0.0)
        bm25_norm_score = bm25_norm_scores.get(key, 0.0)
        vector_norm_score = vector_norm_scores.get(key, 0.0)
        fused_score = _fused_score(
            config=config,
            bm25_rank_score=bm25_rank_score,
            vector_rank_score=vector_rank_score,
            bm25_norm_score=bm25_norm_score,
            vector_norm_score=vector_norm_score,
        )
        if config.preserve_bm25_candidates:
            fused_score = max(fused_score, bm25_rank_score, bm25_norm_score)

        existing = merged.get(key)
        if existing is None:
            merged[key] = RetrievedDocument(
                document_id=hit.document_id,
                filename=hit.filename,
                text=hit.text,
                score=fused_score,
                chunk_id=hit.chunk_id,
                chunk_index=hit.chunk_index,
                bm25_score=hit.bm25_score or 0.0,
                vector_score=hit.vector_score or 0.0,
                rrf_score=fused_score,
                entity_anchor_score=hit.entity_anchor_score or 0.0,
            )
            continue

        prefer_new = prefer_vector_chunk and (hit.vector_score or 0.0) > (existing.vector_score or 0.0)
        merged[key] = RetrievedDocument(
            document_id=existing.document_id,
            filename=existing.filename,
            text=(
                hit.text
                if prefer_new
                else (existing.text if len(existing.text) >= len(hit.text) else hit.text)
            ),
            score=fused_score,
            chunk_id=hit.chunk_id if prefer_new else (existing.chunk_id or hit.chunk_id),
            chunk_index=(
                hit.chunk_index
                if prefer_new
                else (
                    existing.chunk_index
                    if existing.chunk_index is not None
                    else hit.chunk_index
                )
            ),
            bm25_score=max(existing.bm25_score or 0.0, hit.bm25_score or 0.0),
            vector_score=max(existing.vector_score or 0.0, hit.vector_score or 0.0),
            rrf_score=fused_score,
            entity_anchor_score=max(
                existing.entity_anchor_score or 0.0,
                hit.entity_anchor_score or 0.0,
            ),
        )
    return merged


def rerank_key(item: RetrievedDocument) -> tuple[float, float, float, float]:
    return (
        item.entity_anchor_score or 0.0,
        item.score,
        item.vector_score or 0.0,
        item.bm25_score or 0.0,
    )


def no_rerank_key(item: RetrievedDocument) -> tuple[float, float, float]:
    return (
        item.score,
        item.vector_score or 0.0,
        item.bm25_score or 0.0,
    )


def _rank_scores(
    hits: list[RetrievedDocument],
    *,
    weight: float,
    rrf_k: int,
    key_fn: Callable[[RetrievedDocument], str],
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for rank, hit in enumerate(hits, start=1):
        key = key_fn(hit)
        scores[key] = scores.get(key, 0.0) + (weight / (rrf_k + rank))
    return scores


def _normalized_raw_scores(
    hits: list[RetrievedDocument],
    score_fn: Callable[[RetrievedDocument], float],
    *,
    key_fn: Callable[[RetrievedDocument], str],
) -> dict[str, float]:
    if not hits:
        return {}
    max_score = max(score_fn(hit) for hit in hits)
    if max_score <= 0:
        return {}
    scores: dict[str, float] = {}
    for hit in hits:
        key = key_fn(hit)
        scores[key] = max(scores.get(key, 0.0), score_fn(hit) / max_score)
    return scores


def _fused_score(
    *,
    config: FusionConfig,
    bm25_rank_score: float,
    vector_rank_score: float,
    bm25_norm_score: float,
    vector_norm_score: float,
) -> float:
    if config.strategy == "normalized_weighted":
        return (config.bm25_weight * bm25_norm_score) + (config.vector_weight * vector_norm_score)
    return bm25_rank_score + vector_rank_score
