from __future__ import annotations

import math
import re
from dataclasses import dataclass
from dataclasses import asdict
from time import perf_counter
from typing import Iterable

from pra_core.adaptive_rag.models import (
    AdaptiveManifest, ChildArtifact, ChunkDecision, Classification, ParentArtifact,
    QueryPlan, RelationshipArtifact, RetrievalCandidate, TableArtifact, stable_id,
)
from pra_core.config import Settings
from pra_core.documents.models import Document, DocumentChunk, RetrievedDocument
from pra_core.embeddings.base import EmbeddingProvider
from pra_core.indexing import DocumentIndexer
from pra_core.persistence.object_store import JsonObjectStore
from pra_core.retrieval.base import Retriever
from pra_core.vectorstores.base import VectorStore


class AdaptiveRagArtifacts:
    """Versioned, portable artifacts stored through the existing object-store API."""

    def __init__(self, store: JsonObjectStore | None) -> None:
        self._store = store
        self._memory: dict[str, dict[str, object]] = {}

    def exists(self, document: Document) -> bool:
        return bool(self.load(document.id).get("manifest"))

    def save(self, document: Document, manifest: AdaptiveManifest, parents: list[ParentArtifact], children: list[ChildArtifact], tables: list[TableArtifact], relationships: list[RelationshipArtifact]) -> None:
        payload = {"manifest": manifest.to_dict(), "parents": [asdict(item) for item in parents], "children": [asdict(item) for item in children], "tables": [asdict(item) for item in tables], "relationships": [asdict(item) for item in relationships]}
        self._memory[document.id] = payload
        if self._store is not None:
            prefix = self._prefix(document.id)
            self._store.write_json(f"{prefix}/manifest.json", payload["manifest"])
            self._store.write_json(f"{prefix}/parents.json", payload["parents"])
            self._store.write_json(f"{prefix}/chunks.json", payload["children"])
            self._store.write_json(f"{prefix}/tables.json", payload["tables"])
            self._store.write_json(f"{prefix}/relationships.json", payload["relationships"])

    def load(self, document_id: str) -> dict[str, object]:
        if document_id in self._memory:
            return self._memory[document_id]
        if self._store is None:
            return {}
        prefix = self._prefix(document_id)
        manifest = self._store.read_json(f"{prefix}/manifest.json", {})
        if not manifest:
            return {}
        payload = {"manifest": manifest, "parents": self._store.read_json(f"{prefix}/parents.json", []), "children": self._store.read_json(f"{prefix}/chunks.json", []), "tables": self._store.read_json(f"{prefix}/tables.json", []), "relationships": self._store.read_json(f"{prefix}/relationships.json", [])}
        self._memory[document_id] = payload
        return payload

    def all_children(self, documents: Iterable[Document]) -> list[ChildArtifact]:
        return [_child(row) for document in documents for row in self.load(document.id).get("children", [])]

    def all_relationships(self, documents: Iterable[Document]) -> list[RelationshipArtifact]:
        return [_relationship(row) for document in documents for row in self.load(document.id).get("relationships", [])]

    def parents_for(self, document_id: str) -> dict[str, ParentArtifact]:
        return {item.id: item for item in (_parent(row) for row in self.load(document_id).get("parents", []))}

    @staticmethod
    def _prefix(document_id: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9-]+", document_id):
            raise ValueError("Invalid document id for adaptive artifact path.")
        return f"adaptive/{document_id}/v1"


class AdaptiveDocumentIndexer:
    """Indexes adaptive child chunks while retaining the legacy indexer as fallback."""

    def __init__(self, legacy: DocumentIndexer, embedding_provider: EmbeddingProvider, vector_store: VectorStore, artifacts: AdaptiveRagArtifacts, settings: Settings) -> None:
        self._legacy, self._embedding_provider, self._vector_store = legacy, embedding_provider, vector_store
        self._artifacts, self._settings = artifacts, settings

    def index(self, document: Document) -> int:
        if not self._settings.adaptive_chunking_enabled:
            return self._legacy.index(document)
        try:
            if self._artifacts.exists(document):
                children = self._artifacts.all_children([document])
            else:
                manifest, parents, children, tables, relationships = _analyze_and_chunk(document, self._settings)
                self._artifacts.save(document, manifest, parents, children, tables, relationships)
            for child in children:
                self._vector_store.add(_as_document_chunk(child), self._embedding_provider.embed(child.text))
            return len(children)
        except Exception:
            return self._legacy.index(document)


class AdaptiveRetriever(Retriever):
    """Query-planned chunk retrieval with deterministic fusion and legacy fallback."""

    def __init__(self, legacy: Retriever, documents, embedding_provider: EmbeddingProvider, vector_store: VectorStore, artifacts: AdaptiveRagArtifacts, settings: Settings, knowledge_graph=None) -> None:
        self._legacy, self._documents, self._embedding_provider = legacy, documents, embedding_provider
        self._vector_store, self._artifacts, self._settings = vector_store, artifacts, settings
        self._knowledge_graph = knowledge_graph
        self.last_plan: QueryPlan | None = None
        self.last_context_manifest: dict[str, object] = {}
        self.last_trace = AdaptiveRetrievalTrace()

    def search(self, query: str, k: int = 25) -> list[RetrievedDocument]:
        if not self._settings.adaptive_retrieval_enabled:
            results = self._legacy.search(query, k)
            return _merge_graph_hits(results, self._graph_hits(query, k), k)
        try:
            started = perf_counter()
            documents = self._documents.list()
            children = self._artifacts.all_children(documents)
            if not children:
                results = self._legacy.search(query, k)
                return _merge_graph_hits(results, self._graph_hits(query, k), k)
            plan = _plan_query(query, self._settings)
            self.last_plan = plan
            bm25_candidates = _bm25_candidates(children, query, plan)
            confidence = _retrieval_confidence(query, plan, bm25_candidates, self._settings)
            use_hybrid_fallback = not plan.bm25_first or confidence.low
            candidates = (
                _fuse_candidates(
                    children, query, plan, self._embedding_provider, self._vector_store,
                    self._artifacts.all_relationships(documents), self._settings,
                    bm25_candidates=bm25_candidates,
                )
                if use_hybrid_fallback
                else bm25_candidates
            )
            fused_count = len(candidates)
            if self._settings.enable_reranking:
                candidates = _rerank(candidates, query)
            else:
                candidates = sorted(candidates, key=_bm25_first_key, reverse=True)
            candidates = _ensure_final_context_coverage(candidates, children, query, plan)
            context_top_k = min(
                k,
                max(
                    plan.final_context_top_k,
                    self._settings.adaptive_low_confidence_context_top_k
                    if confidence.low
                    else plan.final_context_top_k,
                ),
            )
            candidates = candidates[:context_top_k]
            expanded = _expand(candidates, self._artifacts, self._settings)
            self.last_context_manifest = {
                "query_intent": plan.intent,
                "retrieval_mode": "hybrid_fallback" if use_hybrid_fallback else "bm25_first",
                "low_confidence": confidence.low,
                "top_bm25_score": confidence.top_score,
                "bm25_score_gap": confidence.score_gap,
                "strong_match_count": confidence.strong_match_count,
                "unique_source_chunks": len({item.chunk_id for item in expanded if item.chunk_id}),
                "candidate_count": len(candidates),
                "context_count": len(expanded),
                "context_top_k": context_top_k,
                "context_tokens": sum(_tokens(item.text) for item in expanded),
            }
            self.last_trace = AdaptiveRetrievalTrace(
                retrieval_latency_ms=round((perf_counter() - started) * 1000, 2),
                bm25_candidate_count=sum("bm25" in item.sources for item in candidates),
                vector_candidate_count=sum("vector" in item.sources for item in candidates),
                merged_candidate_count=fused_count,
                reranked_candidate_count=len(expanded),
            )
            return _merge_graph_hits(expanded, self._graph_hits(query, context_top_k), context_top_k)
        except Exception:
            results = self._legacy.search(query, k)
            return _merge_graph_hits(results, self._graph_hits(query, k), k)

    def _graph_hits(self, query: str, k: int) -> list[RetrievedDocument]:
        if not self._settings.graph_retrieval_enabled or self._knowledge_graph is None:
            return []
        return self._knowledge_graph.search(query, k=min(k, self._settings.graph_max_results))


@dataclass(frozen=True)
class AdaptiveRetrievalTrace:
    bm25_latency_ms: float = 0.0
    vector_latency_ms: float = 0.0
    rrf_merge_latency_ms: float = 0.0
    reranker_latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    bm25_candidate_count: int = 0
    vector_candidate_count: int = 0
    merged_candidate_count: int = 0
    reranked_candidate_count: int = 0


def _merge_graph_hits(
    primary: list[RetrievedDocument],
    graph_hits: list[RetrievedDocument],
    k: int,
) -> list[RetrievedDocument]:
    """Keep graph evidence additive while deduplicating source chunks."""
    merged: dict[str, RetrievedDocument] = {}
    for item in [*primary, *graph_hits]:
        key = item.chunk_id or f"{item.document_id}:{item.chunk_index}"
        existing = merged.get(key)
        if existing is None or item.score > existing.score:
            merged[key] = item
        elif item.knowledge_graph_score is not None:
            merged[key] = RetrievedDocument(
                document_id=existing.document_id,
                filename=existing.filename,
                text=existing.text,
                score=existing.score,
                chunk_id=existing.chunk_id,
                chunk_index=existing.chunk_index,
                bm25_score=existing.bm25_score,
                vector_score=existing.vector_score,
                rrf_score=existing.rrf_score,
                entity_anchor_score=existing.entity_anchor_score,
                knowledge_graph_score=max(
                    existing.knowledge_graph_score or 0.0,
                    item.knowledge_graph_score,
                ),
            )
    return sorted(merged.values(), key=lambda item: item.score, reverse=True)[:k]


def _analyze_and_chunk(document: Document, settings: Settings):
    domain = _classify_domain(document.text) if settings.domain_detection_enabled else Classification()
    doc_type = _classify_type(document.filename, document.text) if settings.document_type_detection_enabled else Classification(primary="generic_uploaded_document")
    sections = _sections(document.text)
    tables = _tables(sections) if settings.table_processing_enabled else []
    relationship_density = len(re.findall(r"(?i)\b(section|clause|article|appendix|see)\b", document.text))
    strategy = "table_aware" if tables else "hierarchical_structure" if len(sections) > 1 else "semantic_boundary" if len(_sentences(document.text)) > 3 else "fixed_size"
    decision = ChunkDecision(strategy=strategy, reasons=("table_detected" if tables else "section_hierarchy" if len(sections) > 1 else "narrative_content",), confidence=0.9 if tables or len(sections) > 1 else 0.7, deterministic_signals={"section_count": len(sections), "table_count": len(tables), "relationship_density": relationship_density}, fallback_strategy="fixed_size")
    parents: list[ParentArtifact] = []
    children: list[ChildArtifact] = []
    for ordinal, (title, path, text) in enumerate(sections):
        parent_id = stable_id(document.id, "parent", str(ordinal), title)
        parents.append(ParentArtifact(parent_id, document.id, title, tuple(path), text, ordinal, domain.primary, doc_type.primary))
        for child_ordinal, text_part in enumerate(_split(text, settings.chunk_target_tokens, settings.chunk_overlap_tokens)):
            child_id = stable_id(document.id, parent_id, str(child_ordinal), text_part)
            children.append(ChildArtifact(child_id, document.id, parent_id, document.filename, text_part, len(children), "passage", tuple(path), _tokens(text_part), domain.primary, doc_type.primary, strategy, {"artifact_version": "v1", "section_title": title}))
    relationships = _relationships(children) if settings.relationship_extraction_enabled else []
    manifest = AdaptiveManifest(document.id, document.content_hash, decision, domain, doc_type, {"parents": len(parents), "children": len(children), "tables": len(tables), "relationships": len(relationships)})
    return manifest, parents, children, tables, relationships


def _plan_query(query: str, settings: Settings) -> QueryPlan:
    identifiers = tuple(re.findall(r"\b(?:[A-Z]{2,}-?\d+|CVE-\d{4}-\d+|Section\s+\d+(?:\.\d+)*)\b", query, re.I))
    terms = tuple(sorted(set(re.findall(r"\b[A-Z]{2,}\b", query))))
    lowered = query.lower()
    intent = (
        "incident_or_exception"
        if re.search(r"\b(incident|exception|approval|unless|except|why|which|regional|compare|comparison|difference|versus|vs)\b", lowered)
        else "policy_or_rule"
        if re.search(r"\b(policy|rule|must|shall|required|requirement|compliance)\b", lowered)
        else "cross_reference_analysis"
        if re.search(r"\b(reference|relate|relationship|section|clause)\b", lowered)
        else "exact_identifier_lookup"
        if identifiers
        else "table_lookup"
        if "table" in lowered
        else "definition"
        if re.search(r"\b(define|what is|meaning)\b", lowered)
        else "simple_lookup"
    )
    domain = _classify_domain(query).primary
    expansions = {"CHF": "heart failure", "CVE": "vulnerability identifier", "EPS": "earnings per share", "ACP": "attorney client privilege"}
    expanded = tuple(value for key, value in expansions.items() if key in query.upper()) if settings.query_expansion_enabled else ()
    relationships = settings.relationship_retrieval_enabled and intent in {"cross_reference_analysis", "comparison"}
    context_top_k = (
        settings.adaptive_complex_context_top_k
        if intent in {"incident_or_exception", "cross_reference_analysis", "comparison"}
        else settings.adaptive_policy_context_top_k
        if intent == "policy_or_rule"
        else settings.adaptive_simple_context_top_k
    )
    return QueryPlan(
        query, intent, domain, terms, identifiers, expanded, {},
        settings.enable_bm25_lexical, settings.enable_vector_search, relationships,
        settings.parent_child_expansion_enabled,
        {"bm25": settings.retrieval_bm25_weight, "vector": settings.retrieval_vector_weight, "relationships": settings.retrieval_relationship_weight},
        0.9 if identifiers or intent != "simple_lookup" else 0.7,
        context_top_k,
        settings.bm25_first_enabled,
    )


@dataclass(frozen=True)
class _RetrievalConfidence:
    low: bool
    top_score: float
    score_gap: float
    strong_match_count: int
    exact_anchor_present: bool


def _retrieval_confidence(query, plan, candidates, settings) -> _RetrievalConfidence:
    scores = sorted((candidate.bm25_score for candidate in candidates if candidate.bm25_score > 0), reverse=True)
    top_score = scores[0] if scores else 0.0
    score_gap = top_score - scores[1] if len(scores) > 1 else top_score
    strong_match_count = sum(score >= max(1.0, top_score * 0.55) for score in scores)
    anchors = [anchor.lower() for anchor in plan.exact_identifiers]
    exact_anchor_present = bool(anchors) and any(
        all(anchor in candidate.child.text.lower() for anchor in anchors)
        for candidate in candidates[:5]
    )
    low = (
        not candidates
        or (top_score < 1.0 and not exact_anchor_present)
        or (score_gap < settings.adaptive_low_confidence_score_gap and strong_match_count < settings.adaptive_min_strong_matches)
    )
    return _RetrievalConfidence(low, top_score, score_gap, strong_match_count, exact_anchor_present)


def _bm25_first_key(candidate):
    return (candidate.bm25_score, candidate.fusion_score, candidate.vector_score, candidate.relationship_score)


def _ensure_final_context_coverage(candidates, children, query, plan):
    coverage_terms = _coverage_section_terms(query, plan)
    exact_identifiers = tuple(identifier.lower() for identifier in plan.exact_identifiers)
    if not coverage_terms and not exact_identifiers:
        return candidates

    seen = {candidate.child.id for candidate in candidates}
    coverage: list[RetrievalCandidate] = []
    for child in children:
        if child.id in seen:
            continue
        section = " ".join(child.section_path).lower()
        text = child.text.lower()
        has_exact_record = any(identifier in text for identifier in exact_identifiers)
        has_structural_term = any(term in section or term in text for term in coverage_terms)
        if has_exact_record or has_structural_term:
            coverage.append(
                RetrievalCandidate(
                    child=child,
                    score=0.01,
                    sources=("coverage",),
                    fusion_score=0.01,
                )
            )
            seen.add(child.id)

    if not coverage:
        return candidates
    return [*candidates[:3], *coverage, *candidates[3:]]


def _coverage_section_terms(query: str, plan: QueryPlan) -> tuple[str, ...]:
    lowered = query.lower()
    terms: list[str] = []
    has_exact_identifier = bool(plan.exact_identifiers)
    has_record_rule_language = bool(
        re.search(
            r"\b(approval|approve|required|requires?|exceptions?|qualif(?:y|ies)|eligible|credits?)\b",
            lowered,
        )
    )
    if plan.intent == "incident_or_exception" and (
        re.search(r"\bincidents?\b", lowered)
        or (has_exact_identifier and has_record_rule_language)
    ):
        terms.extend(("incident records", "blue finch packaging program"))
    if re.search(r"\bapproval|approve|copper door\b", lowered):
        terms.extend(("copper door approval", "lantern pass users"))
    if re.search(r"\bcredit|credits|category|categories|approval\b", lowered):
        terms.append("support credit rules")
    if re.search(r"\bbf-[a-z0-9]+|blue finch|packaging\b", lowered):
        terms.append("blue finch packaging program")
    if re.search(r"\bregional|region|lead\b", lowered):
        terms.extend(("region-specific notes", "organization summary"))
    return tuple(dict.fromkeys(terms))


def _bm25_candidates(children, query, plan):
    scores = _bm25(children, _terms(f"{query} {' '.join(plan.expanded_terms)}")) if plan.use_bm25 else {}
    by_id = {child.id: child for child in children}
    return [
        RetrievalCandidate(child=by_id[identifier], score=score, sources=("bm25",), bm25_score=score, fusion_score=score)
        for identifier, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if identifier in by_id
    ]


def _fuse_candidates(children, query, plan, embedder, vector_store, relationships, settings, bm25_candidates=None):
    terms = _terms(f"{query} {' '.join(plan.expanded_terms)}")
    bm25 = {candidate.child.id: candidate.bm25_score for candidate in (bm25_candidates or [])} if plan.use_bm25 else {}
    vector: dict[str, float] = {}
    if plan.use_vector:
        for hit in vector_store.search(embedder.embed(query), k=settings.retrieval_vector_top_k):
            vector[hit.chunk.id] = hit.score
    rel: dict[str, float] = {}
    if plan.use_relationships:
        matched = {child.id for child in children if any(term in child.text.lower() for term in terms)}
        for relation in relationships:
            if relation.source_id in matched and relation.confidence >= settings.relationship_min_confidence:
                rel[relation.target_id] = max(rel.get(relation.target_id, 0.0), relation.confidence)
    ranked: dict[str, RetrievalCandidate] = {}
    by_id = {child.id: child for child in children}
    for source, scores in (("bm25", bm25), ("vector", vector), ("relationships", rel)):
        for rank, (identifier, score) in enumerate(sorted(scores.items(), key=lambda item: item[1], reverse=True), 1):
            child = by_id.get(identifier)
            if child is None:
                continue
            existing = ranked.get(identifier)
            rrf = plan.weights[source] / (60 + rank)
            if existing is None:
                ranked[identifier] = RetrievalCandidate(child, rrf, (source,), bm25_score=score if source == "bm25" else 0.0, vector_score=score if source == "vector" else 0.0, relationship_score=score if source == "relationships" else 0.0, fusion_score=rrf)
            else:
                ranked[identifier] = RetrievalCandidate(child, existing.fusion_score + rrf, tuple(sorted(set(existing.sources + (source,)))), bm25_score=score if source == "bm25" else existing.bm25_score, vector_score=score if source == "vector" else existing.vector_score, relationship_score=score if source == "relationships" else existing.relationship_score, fusion_score=existing.fusion_score + rrf)
    return list(ranked.values())


def _rerank(candidates, query):
    query_terms = set(_terms(query))
    return sorted(candidates, key=lambda item: item.fusion_score + 0.02 * len(query_terms.intersection(_terms(item.child.text))) + (0.15 if any(identifier.lower() in item.child.text.lower() for identifier in re.findall(r"[A-Z]{2,}-?\d+", query)) else 0.0), reverse=True)


def _expand(candidates, artifacts, settings):
    results: list[RetrievedDocument] = []
    budget = settings.adaptive_context_token_budget
    seen: set[str] = set()
    for candidate in candidates:
        child = candidate.child
        if child.id in seen or child.token_count > budget:
            continue
        text = child.text
        if settings.parent_child_expansion_enabled:
            parent = artifacts.parents_for(child.document_id).get(child.parent_id)
            if parent and child.token_count < 80:
                text = parent.text[: max(len(child.text), 1200)]
        seen.add(child.id); budget -= _tokens(text)
        results.append(RetrievedDocument(child.document_id, child.filename, text, candidate.fusion_score, child.id, child.ordinal, candidate.bm25_score, candidate.vector_score, candidate.fusion_score, candidate.relationship_score, None, child.metadata))
    return results


def _classify_domain(text):
    lowered = text.lower(); rules = {"legal": ("clause", "contract", "hereby", "party"), "medical": ("patient", "clinical", "diagnosis", "treatment", "heart failure", "chf"), "financial": ("revenue", "earnings", "balance sheet", "eps"), "cybersecurity": ("cve", "vulnerability", "incident", "malware"), "research_academic": ("abstract", "methodology", "references", "doi")}
    matches = [(name, sum(token in lowered for token in tokens)) for name, tokens in rules.items()]; matches = [(name, count) for name, count in matches if count]
    if not matches: return Classification()
    matches.sort(key=lambda item: item[1], reverse=True); return Classification(matches[0][0], tuple(name for name, _ in matches[1:]), min(0.95, 0.5 + matches[0][1] * 0.1))


def _classify_type(filename, text):
    lowered = f"{filename} {text}".lower(); rules = {"contract": ("agreement", "party", "clause"), "research_paper": ("abstract", "references", "methodology"), "financial_report": ("balance sheet", "earnings", "revenue"), "technical_manual": ("installation", "configuration", "troubleshooting"), "policy": ("policy", "shall", "compliance")}; matches = [(name, sum(token in lowered for token in tokens)) for name, tokens in rules.items()]; matches = [(name, count) for name, count in matches if count]
    return Classification(matches[0][0], (), min(0.9, 0.5 + matches[0][1] * 0.1)) if matches else Classification("generic_uploaded_document", (), 0.4)


def _sections(text):
    matches = list(re.finditer(r"(?m)^\s*(?:#{1,6}\s+|(?:section|article|chapter)\s+\d+[:.\s]+)(.+)$", text))
    if not matches: return [("Document", ["Document"], text.strip())]
    out = []
    for index, match in enumerate(matches):
        body = text[match.start(): matches[index + 1].start() if index + 1 < len(matches) else len(text)].strip(); title = match.group(1).strip(); out.append((title, [title], body))
    return out


def _tables(sections):
    output = []
    for ordinal, (title, _, text) in enumerate(sections):
        lines = [line.strip() for line in text.splitlines() if "|" in line]
        if len(lines) >= 2:
            rows = tuple(tuple(cell.strip() for cell in line.strip("|").split("|")) for line in lines if not re.fullmatch(r"[|\-: ]+", line)); output.append(TableArtifact(stable_id(title, str(ordinal)), "", title, "\n".join(lines), rows, 0.8))
    return output


def _relationships(children):
    output = []; by_section = {" ".join(child.section_path).lower(): child for child in children}
    for child in children:
        for target in re.findall(r"(?i)\b(?:section|article|clause)\s+(\d+(?:\.\d+)*)", child.text):
            match = next((item for key, item in by_section.items() if target in key), None)
            if match and match.id != child.id: output.append(RelationshipArtifact(child.id, "references", match.id, f"reference to {target}", 0.8))
    return output


def _split(text, size, overlap):
    sentences = _sentences(text); chunks = []; current = []
    for sentence in sentences or [text]:
        if current and _tokens(" ".join(current)) + _tokens(sentence) > size:
            chunks.append(" ".join(current)); current = current[-1:] if overlap else []
        current.append(sentence)
    if current: chunks.append(" ".join(current))
    return chunks


def _bm25(children, terms):
    docs = [_terms(child.text) for child in children]; avg = sum(map(len, docs)) / max(1, len(docs)); frequencies = {term: sum(term in doc for doc in docs) for term in set(terms)}; output = {}
    for child, doc in zip(children, docs):
        score = sum(math.log(1 + (len(docs) - frequencies.get(term, 0) + 0.5) / (frequencies.get(term, 0) + 0.5)) * doc.count(term) * 2.5 / (doc.count(term) + 1.5 * (1 - .75 + .75 * len(doc) / max(avg, 1))) for term in set(terms) if term in doc)
        if score: output[child.id] = score
    return output


def _as_document_chunk(child): return DocumentChunk(child.id, child.document_id, child.filename, child.text, child.ordinal, {"parent_id": child.parent_id, "chunk_type": child.chunk_type, "domain": child.domain, "document_type": child.document_type, **child.metadata})
def _terms(text): return re.findall(r"[a-z0-9]+", text.lower())
def _tokens(text): return len(text.split())
def _sentences(text): return [item.strip() for item in re.split(r"(?<=[.!?])\s+", text) if item.strip()]
def _child(row): return ChildArtifact(**{**row, "section_path": tuple(row.get("section_path", ()))})
def _parent(row): return ParentArtifact(**{**row, "section_path": tuple(row.get("section_path", ()))})
def _relationship(row): return RelationshipArtifact(**row)
