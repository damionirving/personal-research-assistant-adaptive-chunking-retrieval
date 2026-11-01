from pra_core.documents.models import Document, DocumentChunk
from pra_core.graphstores.local import LocalGraphStore
from pra_core.knowledge_graph import KnowledgeGraphService
from pra_core.persistence.object_store import FileJsonObjectStore


def _document() -> Document:
    return Document(
        id="doc-1",
        filename="scenario.md",
        content_type="text/markdown",
        text=(
            "# Incident Rules\n\n"
            "NM-1045 requires Copper Door approval.\n"
            "Blue Finch is associated with packaging exceptions."
        ),
    )


def _chunks(document: Document) -> list[DocumentChunk]:
    return [
        DocumentChunk(
            id="scenario.md#chunk-0",
            document_id=document.id,
            filename=document.filename,
            text=document.text,
            index=0,
        )
    ]


def test_kg_is_disabled_without_changing_normal_behavior(tmp_path) -> None:
    service = KnowledgeGraphService(FileJsonObjectStore(tmp_path), enabled=False)

    decision = service.ingest(_document(), _chunks(_document()))

    assert decision.eligible is False
    assert service.search("NM-1045", k=5) == []


def test_benefit_decision_is_deterministic_and_explainable(tmp_path) -> None:
    service = KnowledgeGraphService(
        FileJsonObjectStore(tmp_path),
        enabled=True,
        benefit_threshold=0,
    )

    first = service.assess(_document())
    second = service.assess(_document())

    assert first == second
    assert first.score >= 0
    assert first.signals
    assert first.explanation


def test_graph_ingestion_persists_provenance_and_retrieves_source_chunks(tmp_path) -> None:
    document = _document()
    service = KnowledgeGraphService(
        FileJsonObjectStore(tmp_path),
        LocalGraphStore(),
        enabled=True,
        benefit_threshold=0,
        max_hops=2,
    )

    service.ingest(document, _chunks(document))
    recreated = KnowledgeGraphService(
        FileJsonObjectStore(tmp_path),
        enabled=True,
        benefit_threshold=0,
        max_hops=2,
    )

    results = recreated.search("What does NM-1045 require?", k=5)

    assert results
    assert results[0].chunk_id == "scenario.md#chunk-0"
    assert results[0].document_id == document.id
    assert "NM-1045" in results[0].text


def test_ontology_reuse_and_versioned_extension_are_persisted(tmp_path) -> None:
    store = FileJsonObjectStore(tmp_path)
    service = KnowledgeGraphService(
        store,
        enabled=True,
        benefit_threshold=0,
        reuse_threshold=0.5,
        extension_threshold=0.1,
    )
    first = _document()
    service.ingest(first, _chunks(first))
    second = Document(
        id="doc-2",
        filename="scenario-2.md",
        content_type="text/markdown",
        text="# Incident Rules\n\nNM-1046 requires Copper Door approval and is related to packaging exceptions.",
    )

    service.ingest(second, _chunks(second))
    state = store.read_json("knowledge-graph/state.json", {})

    assert state["ontologies"]
    assert max(item["version"] for item in state["ontologies"]) >= 1
    assert state["nodes"]
    assert state["edges"]


def test_invalid_llm_proposal_is_rejected_by_deterministic_validation(tmp_path) -> None:
    calls = []

    def proposer(text, ontology):
        calls.append(text)
        return {"classes": ["ApprovedClass"], "relations": ["DROP_DATABASE"]}

    service = KnowledgeGraphService(
        FileJsonObjectStore(tmp_path),
        enabled=True,
        benefit_threshold=0,
        llm_enabled=True,
        llm_proposer=proposer,
    )
    document = _document()
    service.ingest(document, _chunks(document))
    state = FileJsonObjectStore(tmp_path).read_json("knowledge-graph/state.json", {})

    assert calls
    assert all("DROP_DATABASE" not in item["relations"] for item in state["ontologies"])
