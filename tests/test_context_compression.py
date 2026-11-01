import re

from pra_core.documents.models import RetrievedDocument
from pra_core.retrieval.context_compression import compress_context, classify_context_question


def _document(index: int, text: str, score: float = 1.0, filename: str = "scenario.md", metadata=None) -> RetrievedDocument:
    return RetrievedDocument(
        document_id=f"doc-{filename}", filename=filename, text=text, score=score,
        chunk_id=f"{filename}#chunk-{index}", chunk_index=index, metadata=metadata or {},
    )


def test_simple_lookup_is_compressed_to_five_chunks() -> None:
    documents = [_document(index, f"General note {index}", 10 - index) for index in range(9)]
    result = compress_context("Who is the regional lead?", documents)
    assert classify_context_question("Who is the regional lead?") == "simple_lookup"
    assert len(result.documents) == 5
    assert result.question_type == "simple_lookup"
    assert len(result.dropped_reasons) == 4


def test_incident_question_keeps_entity_and_rule_family_evidence() -> None:
    documents = [
        _document(0, "ZX-2043 was created by Mira Chen.", 20),
        _document(1, "Executive approval is required for Passport Tier users.", 19),
        _document(2, "Clinical temperature excursion qualifies for a $210 credit.", 18),
        _document(3, "Packaging seal breach has a separate credit rule.", 17),
        _document(4, "Unrelated route delay note.", 16),
    ]
    result = compress_context(
        "What credit applies to ZX-2043 and is executive approval required?", documents
    )
    assert result.question_type == "approval"
    assert "ZX-2043" in result.detected_entities
    assert "approval" in result.detected_rule_families
    assert {document.chunk_id for document in result.documents} >= {
        "scenario.md#chunk-0", "scenario.md#chunk-1", "scenario.md#chunk-2",
    }


def test_comparison_keeps_candidate_evidence_and_deduplicates() -> None:
    documents = [
        _document(0, "ZX-2043 credit is $210.", 10),
        _document(1, "ZX-2044 credit is $90.", 9),
        _document(2, "ZX-2043 credit is $210.", 8),
        _document(3, "ZX-2045 has a regional exception.", 7),
    ]
    result = compress_context("Which incident has the highest credit?", documents)
    assert result.question_type == "comparison_which_incident"
    assert len(result.documents) == 3
    assert "scenario.md#chunk-0" in result.kept_chunk_ids
    assert "scenario.md#chunk-1" in result.kept_chunk_ids
    assert result.dropped_reasons["scenario.md#chunk-2"] == ["near-duplicate of a kept chunk"]


def test_spaced_entities_are_reconstructed_without_an_llm_call() -> None:
    result = compress_context(
        "What does P a s s p o r t   T i e r approval mean?",
        [_document(0, "Passport Tier approval is required for multiple credits.")],
    )
    assert "Passport Tier" in result.detected_entities
    assert "scenario.md#chunk-0" in result.kept_chunk_ids


def test_metadata_affinity_works_for_unseen_document_families() -> None:
    documents = [
        _document(
            0, "The laboratory protocol requires a control sample.", 20,
            filename="lab-protocol.md",
            metadata={"source_family": "scientific", "document_type": "procedure", "domain": "research"},
        ),
        _document(
            1, "The contract requires written notice before termination.", 19,
            filename="vendor-agreement.md",
            metadata={"source_family": "legal", "document_type": "contract", "domain": "legal"},
        ),
    ]
    result = compress_context("What procedure requires a control sample?", documents)
    assert result.inferred_source_family == "procedure"
    assert "lab-protocol.md#chunk-0" in result.kept_chunk_ids
    assert "vendor-agreement.md#chunk-1" in result.dropped_reasons


def test_missing_metadata_degrades_to_lexical_signals() -> None:
    result = compress_context(
        "How do I restart the gateway service?",
        [
            _document(0, "Gateway service restart steps are listed here.", 1, filename="ops-guide.md"),
            _document(1, "The quarterly budget review is complete.", 10, filename="finance.md"),
        ],
    )
    assert result.metadata_known is False
    assert result.kept_chunk_ids == ["ops-guide.md#chunk-0"]


def test_diagnostics_capture_generic_selection_details() -> None:
    result = compress_context(
        "Is ZX-7 eligible under the regional exception?",
        [_document(0, "ZX-7 is eligible under the regional exception.", metadata={"domain": "operations"})],
    )
    assert result.original_retrieved_context_count == 1
    assert result.kept_chunk_ids == ["scenario.md#chunk-0"]
    assert result.inferred_domain == "operations"
    assert result.fallback_path_used == "none"


def test_weak_domain_match_does_not_displace_direct_lexical_evidence() -> None:
    documents = [
        _document(
            0,
            "Unrelated platform migration notes.",
            100,
            filename="platform.md",
            metadata={"domain": "technical"},
        ),
        *[
            _document(
                index,
                f"The API gateway owner is team {index}.",
                10 - index,
                filename=f"gateway-{index}.md",
            )
            for index in range(1, 6)
        ],
    ]

    result = compress_context("Who owns the API gateway?", documents)

    assert len(result.documents) == 5
    assert "platform.md#chunk-0" in result.dropped_reasons
    assert result.weak_domain_only_kept_count == 0


def test_specific_lookup_drops_weak_domain_fill_when_stronger_chunks_exist() -> None:
    documents = [
        _document(
            0,
            "Benefits policy glossary and unrelated onboarding notes.",
            50,
            filename="benefits-general.md",
            metadata={"domain": "human_resources"},
        ),
        _document(
            1,
            "The remote work stipend gives new hires 90 days to spend the allowance.",
            20,
            filename="remote-work.md",
            metadata={"domain": "human_resources"},
        ),
        _document(
            2,
            "Remote work stipend purchases must be submitted through the benefits portal.",
            19,
            filename="remote-work.md",
            metadata={"domain": "human_resources"},
        ),
    ]

    result = compress_context(
        "How many days does an employee new hire have to spend the remote work stipend?",
        documents,
    )

    assert "remote-work.md#chunk-1" in result.kept_chunk_ids
    assert "remote-work.md#chunk-2" in result.kept_chunk_ids
    assert "benefits-general.md#chunk-0" in result.dropped_reasons
    assert result.weak_domain_only_kept_count == 0
    assert result.weak_domain_dropped_count >= 1
    assert result.strong_signal_kept_count == 2
    assert result.context_cap_reason.startswith("timeline context cap")
    assert result.final_context_count_before_precision_filter == 3
    assert result.final_context_count_after_precision_filter == 2
    assert "benefits-general.md#chunk-0" in result.dropped_extra_context_chunk_ids
