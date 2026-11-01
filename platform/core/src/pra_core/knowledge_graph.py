"""Deterministic-first ontology and knowledge-graph enhancement.

The service is deliberately additive. It stores portable JSON records through
the existing object-store boundary and returns ordinary ``RetrievedDocument``
values so callers can use the current context and citation pipeline unchanged.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from threading import Lock
from typing import Any

from pra_core.documents.models import Document, DocumentChunk, RetrievedDocument
from pra_core.graphstores.base import GraphEdge, GraphNode, GraphStore
from pra_core.persistence.object_store import JsonObjectStore


@dataclass(frozen=True)
class KGBenefitDecision:
    score: float
    eligible: bool
    signals: dict[str, float]
    explanation: tuple[str, ...]


@dataclass(frozen=True)
class OntologyRecord:
    ontology_id: str
    name: str
    domain: str
    version: int
    classes: tuple[str, ...]
    relations: tuple[str, ...]
    aliases: tuple[str, ...] = ()
    change_history: tuple[str, ...] = ()


@dataclass(frozen=True)
class _StoredChunk:
    id: str
    document_id: str
    filename: str
    text: str
    index: int


class _MemoryStore(JsonObjectStore):
    def __init__(self) -> None:
        self._values: dict[str, Any] = {}

    def read_json(self, key: str, default: Any) -> Any:
        return self._values.get(key, default)

    def write_json(self, key: str, value: Any) -> None:
        self._values[key] = value


class KnowledgeGraphService:
    """Build and query a bounded document graph with explainable decisions."""

    _STATE_KEY = "knowledge-graph/state.json"

    def __init__(
        self,
        object_store: JsonObjectStore | None = None,
        graph_store: GraphStore | None = None,
        *,
        enabled: bool = False,
        benefit_threshold: float = 60.0,
        reuse_threshold: float = 0.90,
        extension_threshold: float = 0.70,
        max_hops: int = 2,
        max_results: int = 20,
        llm_proposer: Any | None = None,
        llm_enabled: bool = False,
    ) -> None:
        self._store = object_store or _MemoryStore()
        self._graph_store = graph_store
        self._enabled = enabled
        self._benefit_threshold = benefit_threshold
        self._reuse_threshold = reuse_threshold
        self._extension_threshold = extension_threshold
        self._max_hops = max(0, max_hops)
        self._max_results = max(0, max_results)
        self._llm_proposer = llm_proposer
        self._llm_enabled = llm_enabled
        self._lock = Lock()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def assess(self, document: Document) -> KGBenefitDecision:
        text = document.text
        words = max(1, len(re.findall(r"\w+", text)))
        headings = len(re.findall(r"(?m)^\s{0,3}#{1,6}\s+", text))
        tables = len(re.findall(r"(?m)^\s*\|.*\|\s*$", text)) // 2
        entities = len(_entities(text))
        relation_terms = len(re.findall(r"(?i)\b(requires?|belongs to|related to|causes?|depends on|associated with)\b", text))
        cross_refs = len(re.findall(r"(?i)\b(see|section|appendix|refer(?:ence|red)?)\b", text))
        signals = {
            "section_structure": min(25.0, headings * 5.0),
            "table_structure": min(20.0, tables * 4.0),
            "entity_density": min(25.0, entities / max(words, 1) * 1000.0),
            "relationship_indicators": min(20.0, relation_terms * 4.0),
            "cross_references": min(10.0, cross_refs * 2.0),
        }
        score = round(min(100.0, sum(signals.values())), 2)
        explanation = tuple(
            label
            for label, value in (
                ("structured sections detected", signals["section_structure"]),
                ("tabular facts detected", signals["table_structure"]),
                ("named entities detected", signals["entity_density"]),
                ("relationship language detected", signals["relationship_indicators"]),
                ("cross references detected", signals["cross_references"]),
            )
            if value > 0
        )
        return KGBenefitDecision(score, score >= self._benefit_threshold, signals, explanation)

    def ingest(self, document: Document, chunks: list[DocumentChunk]) -> KGBenefitDecision:
        decision = self.assess(document)
        if not self._enabled or not decision.eligible:
            return decision

        entities = _entities(document.text)
        relations = _relations(document.text, entities)
        ontology, action = self._select_ontology(document, entities, relations)
        ontology = self._validated_llm_extension(document, ontology)
        state = self._state()
        stored_chunks = {
            item["id"]: item for item in state.get("chunks", [])
        }
        nodes = state.setdefault("nodes", {})
        edges = state.setdefault("edges", [])
        document_node = f"document:{document.id}"
        nodes[document_node] = {"id": document_node, "label": "Document", "properties": {"document_id": document.id, "filename": document.filename}}
        self._add_node(GraphNode(document_node, "Document", {"document_id": document.id, "filename": document.filename}))
        ontology_node = f"ontology:{ontology.ontology_id}"
        nodes[ontology_node] = {"id": ontology_node, "label": "Ontology", "properties": {"name": ontology.name, "version": str(ontology.version)}}
        self._add_node(GraphNode(ontology_node, "Ontology", {"name": ontology.name, "version": str(ontology.version)}))
        self._append_edge(edges, GraphEdge(document_node, ontology_node, "USES_ONTOLOGY", {"action": action}))
        for chunk in chunks:
            stored_chunks[chunk.id] = asdict(_StoredChunk(chunk.id, chunk.document_id, chunk.filename, chunk.text, chunk.index))
            chunk_node = f"chunk:{chunk.id}"
            nodes[chunk_node] = {"id": chunk_node, "label": "Chunk", "properties": {"chunk_id": chunk.id, "document_id": document.id, "text": chunk.text}}
            self._add_node(GraphNode(chunk_node, "Chunk", {"chunk_id": chunk.id, "document_id": document.id, "text": chunk.text}))
            self._append_edge(edges, GraphEdge(document_node, chunk_node, "CONTAINS", {"source_chunk_id": chunk.id}))
        for name, entity_type in entities.items():
            entity_node = f"entity:{_stable_id(name.lower(), entity_type)}"
            nodes[entity_node] = {"id": entity_node, "label": "Entity", "properties": {"name": name, "entity_type": entity_type}}
            self._add_node(GraphNode(entity_node, "Entity", {"name": name, "entity_type": entity_type}))
            for chunk in chunks:
                if name.lower() in chunk.text.lower():
                    self._append_edge(edges, GraphEdge(chunk_node_id(chunk), entity_node, "MENTIONS", {"source_chunk_id": chunk.id}))
        for source, relationship, target in relations:
            source_id = f"entity:{_stable_id(source.lower(), 'entity')}"
            target_id = f"entity:{_stable_id(target.lower(), 'entity')}"
            self._append_edge(edges, GraphEdge(source_id, target_id, relationship, {"document_id": document.id}))
        state["chunks"] = list(stored_chunks.values())
        ontologies = self._ontologies_from_state(state)
        existing = {item.ontology_id: item for item in ontologies}
        existing[ontology.ontology_id] = ontology
        state["ontologies"] = [asdict(item) for item in existing.values()]
        state["last_decisions"] = {document.id: asdict(decision)}
        with self._lock:
            self._store.write_json(self._STATE_KEY, state)
        return decision

    def search(self, query: str, *, k: int | None = None) -> list[RetrievedDocument]:
        if not self._enabled:
            return []
        state = self._state()
        chunks = {item["id"]: item for item in state.get("chunks", [])}
        entities = [item for item in state.get("nodes", {}).values() if item.get("label") == "Entity"]
        terms = set(_terms(query))
        matched_chunk_ids: set[str] = set()
        scores: dict[str, float] = {}
        matched_entity_ids: set[str] = set()
        for entity in entities:
            name_terms = set(_terms(entity.get("properties", {}).get("name", "")))
            overlap = len(terms & name_terms)
            if overlap == 0:
                continue
            entity_id = entity["id"]
            matched_entity_ids.add(entity_id)
            for edge in state.get("edges", []):
                if edge.get("target_id") != entity_id or edge.get("relationship") != "MENTIONS":
                    continue
                chunk_id = edge.get("properties", {}).get("source_chunk_id")
                if chunk_id in chunks:
                    matched_chunk_ids.add(chunk_id)
                    scores[chunk_id] = max(scores.get(chunk_id, 0.0), min(0.95, 0.5 + overlap * 0.15))
        frontier = matched_entity_ids
        for hop in range(self._max_hops):
            next_frontier: set[str] = set()
            for edge in state.get("edges", []):
                if edge.get("source_id") not in frontier and edge.get("target_id") not in frontier:
                    continue
                neighbor = edge.get("target_id") if edge.get("source_id") in frontier else edge.get("source_id")
                if neighbor in next_frontier:
                    continue
                next_frontier.add(neighbor)
                if str(neighbor).startswith("entity:"):
                    for mention in state.get("edges", []):
                        if mention.get("target_id") != neighbor or mention.get("relationship") != "MENTIONS":
                            continue
                        chunk_id = mention.get("properties", {}).get("source_chunk_id")
                        if chunk_id in chunks:
                            matched_chunk_ids.add(chunk_id)
                            scores[chunk_id] = max(scores.get(chunk_id, 0.0), max(0.35, 0.65 - hop * 0.1))
            frontier = next_frontier
            if not frontier:
                break
        limit = self._max_results if k is None else min(self._max_results, k)
        return [
            RetrievedDocument(
                document_id=chunks[chunk_id]["document_id"],
                filename=chunks[chunk_id]["filename"],
                text=chunks[chunk_id]["text"],
                score=scores[chunk_id],
                chunk_id=chunk_id,
                chunk_index=chunks[chunk_id]["index"],
                rrf_score=scores[chunk_id],
                knowledge_graph_score=scores[chunk_id],
            )
            for chunk_id in sorted(matched_chunk_ids, key=lambda item: scores[item], reverse=True)[:limit]
        ]

    def _select_ontology(self, document: Document, entities: dict[str, str], relations: list[tuple[str, str, str]]) -> tuple[OntologyRecord, str]:
        state = self._state()
        current = self._ontologies_from_state(state)
        domain = _domain(document.text)
        vocabulary = set(_terms(" ".join(entities)))
        best: tuple[float, OntologyRecord] | None = None
        for item in current:
            reference = set(_terms(f"{item.name} {item.domain} {' '.join(item.classes)} {' '.join(item.aliases)}"))
            score = len(vocabulary & reference) / max(1, len(vocabulary | reference))
            if best is None or score > best[0]:
                best = (score, item)
        if best and best[0] >= self._reuse_threshold:
            return best[1], "reuse"
        if best and best[0] >= self._extension_threshold:
            item = best[1]
            updated = OntologyRecord(item.ontology_id, item.name, item.domain, item.version + 1, tuple(sorted(set(item.classes) | set(entities.values()))), tuple(sorted(set(item.relations) | {relation for _, relation, _ in relations})), item.aliases, item.change_history + (f"extended from {document.id}",))
            return updated, "extend"
        ontology_id = _stable_id(domain, document.filename)
        return OntologyRecord(ontology_id, f"{domain} ontology", domain, 1, tuple(sorted(set(entities.values()))), tuple(sorted({relation for _, relation, _ in relations})), (), (f"created from {document.id}",)), "create"

    def _validated_llm_extension(self, document: Document, ontology: OntologyRecord) -> OntologyRecord:
        """Optionally accept an LLM proposal, after deterministic validation."""
        if not self._llm_enabled or not callable(self._llm_proposer):
            return ontology
        proposal = self._llm_proposer(document.text, ontology)
        if not isinstance(proposal, dict):
            return ontology
        classes = tuple(sorted({str(item).strip() for item in proposal.get("classes", ()) if str(item).strip()}))
        relations = tuple(sorted({str(item).strip().upper() for item in proposal.get("relations", ()) if str(item).strip()}))
        if not classes and not relations:
            return ontology
        allowed_relations = {"REQUIRES", "RELATED_TO", "BELONGS_TO", "CAUSES", "ASSOCIATED_WITH"}
        if not set(relations).issubset(allowed_relations):
            return ontology
        return OntologyRecord(
            ontology.ontology_id,
            ontology.name,
            ontology.domain,
            ontology.version + 1,
            tuple(sorted(set(ontology.classes) | set(classes))),
            tuple(sorted(set(ontology.relations) | set(relations))),
            ontology.aliases,
            ontology.change_history + ("validated LLM proposal",),
        )

    def _ontologies_from_state(self, state: dict[str, Any]) -> list[OntologyRecord]:
        return [OntologyRecord(**item) for item in state.get("ontologies", [])]

    def _state(self) -> dict[str, Any]:
        return self._store.read_json(self._STATE_KEY, {"nodes": {}, "edges": [], "chunks": [], "ontologies": [], "last_decisions": {}})

    def _add_node(self, node: GraphNode) -> None:
        if self._graph_store is not None:
            self._graph_store.add_node(node)

    def _append_edge(self, edges: list[dict[str, Any]], edge: GraphEdge) -> None:
        record = {"source_id": edge.source_id, "target_id": edge.target_id, "relationship": edge.relationship, "properties": edge.properties}
        if record not in edges:
            edges.append(record)
        if self._graph_store is not None:
            self._graph_store.add_edge(edge)


def chunk_node_id(chunk: DocumentChunk) -> str:
    return f"chunk:{chunk.id}"


def _stable_id(*parts: str) -> str:
    return sha256("|".join(parts).encode("utf-8")).hexdigest()[:24]


def _terms(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.lower())


def _domain(text: str) -> str:
    lowered = text.lower()
    for name, words in (("policy", ("policy", "rule", "approval")), ("incident", ("incident", "exception", "severity")), ("operations", ("shipment", "regional", "dispatch"))):
        if any(word in lowered for word in words):
            return name
    return "general"


def _entities(text: str) -> dict[str, str]:
    found: dict[str, str] = {}
    for identifier in re.findall(r"\b[A-Z]{1,12}(?:-[A-Z0-9]{1,12})+\b", text):
        found[identifier] = "Identifier"
    for phrase in re.findall(r"\b[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+){1,3}\b", text):
        found[phrase] = "NamedEntity"
    for word in re.findall(r"\b(?:approval|exception|temperature|packaging|regional|shipment|incident|credit)\b", text, re.I):
        found[word.lower()] = "Concept"
    return found


def _relations(text: str, entities: dict[str, str]) -> list[tuple[str, str, str]]:
    names = sorted(entities, key=len, reverse=True)
    relations: list[tuple[str, str, str]] = []
    for line in text.splitlines():
        lowered = line.lower()
        relation = next((word for word in ("REQUIRES", "RELATED_TO", "BELONGS_TO", "CAUSES", "ASSOCIATED_WITH") if word.replace("_", " ").lower() in lowered), None)
        if relation is None:
            continue
        present = [name for name in names if name.lower() in lowered]
        if len(present) >= 2:
            relations.append((present[0], relation, present[1]))
    return relations
