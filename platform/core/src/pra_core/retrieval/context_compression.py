from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from pra_core.documents.models import RetrievedDocument


QUESTION_TYPES = (
    "simple_lookup", "definition", "yes_no_policy", "amount_credit", "approval",
    "eligibility", "procedure", "timeline", "exception", "incident_rule",
    "comparison_which_incident", "multi_document_comparison",
)

_RULE_FAMILY_MARKERS = {
    "eligibility": ("eligible", "eligibility", "qualify", "qualifies"),
    "approval": ("approval", "approve", "approver", "authority", "authorize"),
    "exception": ("exception", "override", "special case", "waiver", "waived"),
    "deadline": ("deadline", "due", "expires", "timely", "within", "calendar days"),
    "pricing": ("price", "pricing", "cost", "rate", "fee", "amount"),
    "reimbursement": ("reimburse", "reimbursement", "stipend", "allowance", "refund"),
    "safety": ("safety", "hazard", "risk", "exposure", "breach"),
    "regional_rule": ("regional", "region", "jurisdiction", "territory", "local"),
    "escalation": ("escalate", "escalation", "supervisor", "senior", "manager"),
    "threshold": ("threshold", "minimum", "maximum", "at least", "no more than", "exceeds"),
    "combination_rule": ("combine", "combined", "multiple categories", "cannot be combined"),
    "precedence": ("precedence", "priority", "takes precedence", "supersedes"),
}

_DOMAIN_MARKERS = {
    "legal": ("contract", "clause", "party", "agreement", "hereby"),
    "medical": ("clinical", "patient", "diagnosis", "treatment", "medication"),
    "financial": ("revenue", "earnings", "balance sheet", "invoice", "tax"),
    "technical": ("api", "configuration", "deployment", "error", "version"),
    "operations": ("shipment", "delivery", "route", "incident", "warehouse"),
    "research": ("abstract", "methodology", "hypothesis", "references", "study"),
    "human_resources": ("employee", "benefits", "handbook", "leave", "payroll"),
}

_DOCUMENT_TYPE_MARKERS = {
    "policy": ("policy", "policies", "standard", "guideline"),
    "procedure": ("procedure", "runbook", "instructions", "process", "steps"),
    "handbook": ("handbook", "employee guide", "manual"),
    "contract": ("contract", "agreement", "terms"),
    "technical_reference": ("technical", "reference", "api", "configuration"),
    "incident_guide": ("incident", "incident guide", "postmortem"),
    "clinical_reference": ("clinical", "medical", "treatment", "drug"),
    "scientific_procedure": ("experiment", "methodology", "laboratory", "protocol"),
}

_ENTITY_PATTERNS = (
    re.compile(r"\b[A-Z]{1,12}-\d{2,}\b"),
    re.compile(r"\b[A-Z]{1,8}-[A-Z0-9]{1,8}\b"),
    re.compile(r"\$\s?\d+(?:\.\d{1,2})?\b"),
    re.compile(r"\b\d+(?:\.\d+)?\s?(?:minutes?|hours?|days?|business days?)\b", re.I),
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
)

_STOPWORDS = {
    "a", "an", "and", "are", "be", "can", "does", "for", "how", "is", "it",
    "of", "on", "or", "the", "to", "what", "which", "who", "why",
}


@dataclass(frozen=True)
class ContextCompressionResult:
    documents: list[RetrievedDocument]
    question_type: str
    detected_entities: list[str]
    detected_rule_families: list[str]
    kept_reasons: dict[str, list[str]]
    dropped_reasons: dict[str, list[str]]
    inferred_domain: str | None = None
    inferred_source_family: str | None = None
    metadata_known: bool = False
    fallback_path_used: str = "none"
    final_context_selection_strategy: str = "strong_evidence_then_weak_domain"
    weak_domain_only_kept_count: int = 0
    strong_signal_kept_count: int = 0
    weak_domain_dropped_count: int = 0
    context_cap_reason: str = "question_type_limit"
    recall_protection_reason: str | None = None
    required_evidence_chunk_ids: list[str] | None = None
    final_context_count_before_precision_filter: int = 0
    final_context_count_after_precision_filter: int = 0
    dropped_extra_context_chunk_ids: list[str] | None = None

    @property
    def kept_chunk_ids(self) -> list[str]:
        return [_chunk_key(document) for document in self.documents]

    @property
    def original_retrieved_context_count(self) -> int:
        return len(self.documents) + len(self.dropped_reasons)


def classify_context_question(question: str) -> str:
    lowered = question.lower().strip()
    if re.search(r"\b(which|highest|lowest|compare|comparison|versus|vs\.)\b", lowered):
        if re.search(r"\b(documents?|sources?|policies?)\b", lowered):
            return "multi_document_comparison"
        return "comparison_which_incident"
    if re.search(r"\b(procedure|process|steps?|how do I|how can I)\b", lowered):
        return "procedure"
    if re.search(r"\b(when|deadline|due|how long|how many\s+(?:days?|weeks?|months?|years?)|within \d+)\b", lowered):
        return "timeline"
    if re.search(r"\b(exception|override|special case|unless|only if)\b", lowered):
        return "exception"
    if re.search(r"\b(incident|rule|multi-condition|qualif)\b", lowered):
        return "incident_rule"
    if re.search(r"\b(approve|approval|approver|authority|authorize)\b", lowered):
        return "approval"
    if re.search(r"\b(eligible|eligibility|qualify|qualifies)\b", lowered):
        return "eligibility"
    if re.search(r"\b(how much|amount|credit|reimburse|reimbursement|cost|price)\b", lowered):
        return "amount_credit"
    if re.match(r"^(can|could|should|is|are|does|do|will|would)\b", lowered):
        return "yes_no_policy"
    if re.match(r"^(what is|what does|define)\b", lowered):
        return "definition"
    return "simple_lookup"


def _limit_for(question_type: str, available: int) -> int:
    limits = {
        "simple_lookup": 5, "definition": 5, "yes_no_policy": 8, "amount_credit": 10,
        "approval": 10, "eligibility": 8, "procedure": 10, "timeline": 8,
        "exception": 10, "incident_rule": 12, "comparison_which_incident": 15,
        "multi_document_comparison": 15,
    }
    return min(available, limits[question_type])


def _chunk_key(document: RetrievedDocument) -> str:
    return document.chunk_id or f"{document.filename}#chunk-{document.chunk_index or 0}"


def _normalize_spaced_entity(value: str) -> str:
    words = []
    for part in (part for part in re.split(r"\s{2,}", value.strip()) if part):
        letters = "".join(re.findall(r"[A-Za-z]", part))
        if letters:
            words.append(letters)
    return " ".join(words)


def _entities(question: str) -> list[str]:
    found: list[str] = []
    for pattern in _ENTITY_PATTERNS:
        found.extend(match.group(0) for match in pattern.finditer(question))
    found.extend(
        " ".join(match.split())
        for match in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b", question)
    )
    for spaced in re.findall(
        r"(?<!\w)(?:[A-Za-z]\s+){2,}[A-Za-z](?:\s{2,}(?:[A-Za-z]\s+){2,}[A-Za-z])?(?=\s|[?.!,]|$)",
        question,
    ):
        normalized = _normalize_spaced_entity(spaced)
        if normalized:
            found.append(normalized)
    return list(dict.fromkeys(found))


def _normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower()).strip()


def _entity_in_text(entity: str, text: str) -> bool:
    normalized_entity = _normalized_text(entity)
    normalized_text = _normalized_text(text)
    if normalized_entity in normalized_text:
        return True
    compact_entity = re.sub(r"\s+", "", normalized_entity)
    compact_text = re.sub(r"\s+", "", normalized_text)
    return len(compact_entity) >= 4 and compact_entity in compact_text


def _families(text: str) -> list[str]:
    lowered = text.lower()
    families = [
        family for family, markers in _RULE_FAMILY_MARKERS.items()
        if any(marker in lowered for marker in markers)
    ]
    for phrase, suffix in re.findall(
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\s+(approval|exception|policy|rule)\b", text
    ):
        families.append(f"{re.sub(r'[^a-z0-9]+', '_', phrase.lower()).strip('_')}_{suffix}")
    return list(dict.fromkeys(families))


def _metadata(document: RetrievedDocument) -> dict[str, object]:
    metadata = dict(getattr(document, "metadata", {}) or {})
    text = f"{document.filename}\n{document.text}"
    lowered = text.lower()
    for key, markers in _DOCUMENT_TYPE_MARKERS.items():
        if not metadata.get("document_type") and any(marker in lowered for marker in markers):
            metadata["document_type"] = key
            break
    for key, markers in _DOMAIN_MARKERS.items():
        if not metadata.get("domain") and sum(marker in lowered for marker in markers) >= 2:
            metadata["domain"] = key
            break
    if not metadata.get("source_family"):
        source = re.sub(r"[_-]+", " ", document.filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1])
        metadata["source_family"] = re.sub(r"\.[A-Za-z0-9]+$", "", source).strip().lower()
    return metadata


def _query_metadata(question: str, families: list[str], question_type: str) -> tuple[str | None, str | None]:
    lowered = question.lower()
    domain = next(
        (name for name, markers in _DOMAIN_MARKERS.items() if any(marker in lowered for marker in markers)),
        None,
    )
    source_family = domain or (families[0] if families else None)
    if source_family is None and question_type in {"procedure", "definition", "timeline"}:
        source_family = question_type
    return domain, source_family


def _metadata_match(document: RetrievedDocument, query_domain: str | None, query_family: str | None) -> tuple[int, int, bool]:
    metadata = _metadata(document)
    source_values = " ".join(
        str(metadata.get(key, "")).lower()
        for key in ("source_family", "document_type", "source_file", "sections")
    )
    return (
        int(bool(query_domain and query_domain == str(metadata.get("domain", "")).lower())),
        int(bool(query_family and query_family in source_values)),
        bool(getattr(document, "metadata", {}) or {}),
    )


def _lexical_overlap(question: str, document: RetrievedDocument) -> int:
    terms = {
        term.lower() for term in re.findall(r"[A-Za-z0-9][A-Za-z0-9'/-]+", question)
        if term.lower() not in _STOPWORDS and len(term) > 2
    }
    text_terms = {term.lower() for term in re.findall(r"[A-Za-z0-9][A-Za-z0-9'/-]+", document.text)}
    return len(terms & text_terms)


def _matches(question: str, document: RetrievedDocument, entities: list[str], families: list[str], query_domain: str | None, query_family: str | None) -> tuple[int, int, int, int, int, int, bool]:
    entity_matches = sum(1 for entity in entities if _entity_in_text(entity, document.text))
    document_families = _families(document.text)
    family_matches = sum(1 for family in families if family in document_families)
    domain_match, source_match, metadata_known = _metadata_match(document, query_domain, query_family)
    lexical_matches = _lexical_overlap(question, document)
    signal = int(bool(entity_matches or family_matches or domain_match or source_match or lexical_matches))
    return entity_matches, family_matches, domain_match, source_match, lexical_matches, signal, metadata_known


def _redundant(document: RetrievedDocument, kept: list[RetrievedDocument]) -> bool:
    normalized = " ".join(document.text.lower().split())
    for prior in kept:
        if document.document_id != prior.document_id:
            continue
        prior_normalized = " ".join(prior.text.lower().split())
        if normalized == prior_normalized or SequenceMatcher(None, normalized, prior_normalized).ratio() >= 0.94:
            return True
    return False


def _weak_domain_only_match(item: tuple[object, ...]) -> bool:
    """A broad domain match is useful fallback evidence, not primary evidence."""
    entity_matches, family_matches, domain_match, source_match, lexical_matches = item[:5]
    return bool(domain_match) and not any(
        (entity_matches, family_matches, source_match, lexical_matches)
    )


def _specific_question(question_type: str, entities: list[str], families: list[str]) -> bool:
    return bool(
        entities
        or families
        or question_type in {
            "simple_lookup", "definition", "yes_no_policy", "amount_credit",
            "approval", "eligibility", "timeline", "incident_rule",
        }
    )


def compress_context(question: str, documents: list[RetrievedDocument]) -> ContextCompressionResult:
    question_type = classify_context_question(question)
    limit = _limit_for(question_type, len(documents))
    entities = _entities(question)
    families = _families(question)
    query_domain, query_family = _query_metadata(question, families, question_type)
    scored: list[tuple[int, int, int, int, int, int, float, int, RetrievedDocument]] = []
    for index, document in enumerate(documents):
        signals = _matches(question, document, entities, families, query_domain, query_family)
        scored.append((*signals[:-1], document.score, -index, document))

    selected: list[RetrievedDocument] = []
    kept_reasons: dict[str, list[str]] = {}

    def keep(document: RetrievedDocument, reason: str) -> None:
        if document in selected or _redundant(document, selected):
            return
        selected.append(document)
        kept_reasons.setdefault(_chunk_key(document), []).append(reason)

    # Preserve retrieval candidates, but prefer direct evidence before a broad
    # document-domain affinity when choosing the smaller generation context.
    ordered = sorted(
        scored,
        reverse=True,
        key=lambda item: (item[0], item[1], item[3], item[4], item[2], item[6], item[7]),
    )
    source_family_available = bool(query_family and any(item[3] for item in scored))
    for entity in entities:
        for item in ordered:
            document = item[-1]
            if _entity_in_text(entity, document.text):
                keep(document, f"exact entity: {entity}")
                break
    for family in families:
        for item in ordered:
            document = item[-1]
            if family in _families(document.text):
                keep(document, f"rule family: {family}")
                break

    for item in ordered:
        entity_matches, family_matches, domain_match, source_match, lexical_matches, signal, _, _, document = item
        if len(selected) >= limit:
            break
        if document in selected or _redundant(document, selected) or not signal:
            continue
        if _weak_domain_only_match(item):
            continue
        if source_family_available and source_match == 0 and not entity_matches and not family_matches:
            continue
        reason = "deterministic relevance signal"
        if domain_match:
            reason = "document domain match"
        elif source_match:
            reason = "source-family match"
        elif lexical_matches:
            reason = "direct lexical overlap"
        keep(document, reason)

    # Use broad domain affinity only as a fallback. Specific questions should
    # not retain weak same-domain chunks once stronger evidence is available.
    allow_weak_domain_fill = not selected or not _specific_question(question_type, entities, families)
    for item in ordered:
        if len(selected) >= limit:
            break
        document = item[-1]
        if document in selected or _redundant(document, selected):
            continue
        if _weak_domain_only_match(item) and allow_weak_domain_fill:
            keep(document, "weak document-domain match")

    fallback_path = "none"
    if not selected and documents:
        fallback_path = "metadata_and_signal_fallback"
        for document in documents:
            keep(document, "fallback because no deterministic relevance signal was available")
            if len(selected) >= limit:
                break

    kept_keys = {_chunk_key(document) for document in selected}
    weak_domain_dropped_count = 0
    dropped_reasons: dict[str, list[str]] = {}
    for item in scored:
        entity_matches, family_matches, domain_match, source_match, lexical_matches, signal, _, _, document = item
        key = _chunk_key(document)
        if key in kept_keys:
            continue
        reasons = ["outside question-type context limit"]
        if entity_matches == 0 and entities:
            reasons.append("no exact question entity")
        if family_matches == 0 and families:
            reasons.append("no detected rule-family match")
        if domain_match == 0 and query_domain:
            reasons.append("no document-domain match")
        if source_match == 0 and query_family:
            reasons.append("no source-family match")
        if lexical_matches == 0:
            reasons.append("no direct lexical overlap")
        if _weak_domain_only_match(item):
            reasons.append("weak document-domain match")
            weak_domain_dropped_count += 1
        if not signal:
            reasons.append("no deterministic relevance signal")
        if _redundant(document, selected):
            reasons = ["near-duplicate of a kept chunk"]
        dropped_reasons[key] = reasons

    inferred_domain = query_domain
    if inferred_domain is None:
        candidate_domains = {
            str(_metadata(document).get("domain"))
            for document in selected
            if _metadata(document).get("domain")
        }
        if len(candidate_domains) == 1:
            inferred_domain = next(iter(candidate_domains))
    inferred_source_family = query_family
    if inferred_source_family is None:
        candidate_families = {
            str(_metadata(document).get("source_family"))
            for document in selected
            if _metadata(document).get("source_family")
        }
        if len(candidate_families) == 1:
            inferred_source_family = next(iter(candidate_families))

    return ContextCompressionResult(
        documents=selected,
        question_type=question_type,
        detected_entities=entities,
        detected_rule_families=families,
        kept_reasons=kept_reasons,
        dropped_reasons=dropped_reasons,
        inferred_domain=inferred_domain,
        inferred_source_family=inferred_source_family,
        metadata_known=any(getattr(item, "metadata", {}) for item in documents),
        fallback_path_used=fallback_path,
        weak_domain_only_kept_count=sum(
            _weak_domain_only_match(item)
            for item in scored
            if item[-1] in selected
        ),
        strong_signal_kept_count=sum(
            not _weak_domain_only_match(item)
            for item in scored
            if item[-1] in selected
        ),
        weak_domain_dropped_count=weak_domain_dropped_count,
        context_cap_reason=f"{question_type} context cap {limit}",
        recall_protection_reason=(
            "weak-domain fallback retained because no stronger selected evidence was available"
            if allow_weak_domain_fill and any(
                _weak_domain_only_match(item) for item in scored if item[-1] in selected
            )
            else None
        ),
        required_evidence_chunk_ids=[
            _chunk_key(item[-1])
            for item in scored
            if item[0] or item[1]
        ],
        final_context_count_before_precision_filter=len(documents),
        final_context_count_after_precision_filter=len(selected),
        dropped_extra_context_chunk_ids=list(dropped_reasons),
    )
