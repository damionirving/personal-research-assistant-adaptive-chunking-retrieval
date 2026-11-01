from __future__ import annotations

import re


_DOMAIN_MARKERS = {
    "legal": ("contract", "clause", "party", "agreement", "hereby"),
    "medical": ("clinical", "patient", "diagnosis", "treatment", "medication"),
    "financial": ("revenue", "earnings", "invoice", "tax", "balance sheet"),
    "technical": ("api", "configuration", "deployment", "error", "version"),
    "operations": ("shipment", "delivery", "route", "incident", "warehouse"),
    "research": ("abstract", "methodology", "hypothesis", "references", "study"),
    "human_resources": ("employee", "benefits", "handbook", "leave", "payroll"),
}

_DOCUMENT_TYPES = {
    "policy": ("policy", "policies", "standard", "guideline"),
    "procedure": ("procedure", "runbook", "instructions", "process", "steps"),
    "handbook": ("handbook", "employee guide", "manual"),
    "contract": ("contract", "agreement", "terms"),
    "technical_reference": ("technical", "reference", "api", "configuration"),
    "incident_guide": ("incident", "incident guide", "postmortem"),
    "clinical_reference": ("clinical", "medical", "treatment", "drug"),
    "scientific_procedure": ("experiment", "methodology", "laboratory", "protocol"),
}

_RULE_MARKERS = {
    "eligibility": ("eligible", "eligibility", "qualify", "qualifies"),
    "approval": ("approval", "approve", "approver", "authority", "authorize"),
    "exception": ("exception", "override", "special case", "waiver", "waived"),
    "deadline": ("deadline", "due", "expires", "timely", "within"),
    "threshold": ("threshold", "minimum", "maximum", "at least", "exceeds"),
    "precedence": ("precedence", "priority", "supersedes"),
}


def infer_document_metadata(filename: str, content_type: str, text: str) -> dict[str, object]:
    """Infer conservative metadata using filename and document structure only."""
    lowered = f"{filename}\n{text}".lower()
    metadata: dict[str, object] = {
        "source_file": filename,
        "content_type": content_type,
        "source_family": re.sub(r"\.[A-Za-z0-9]+$", "", re.sub(r"[_-]+", " ", filename)).strip().lower(),
    }
    type_matches = [
        name for name, markers in _DOCUMENT_TYPES.items()
        if any(marker in lowered for marker in markers)
    ]
    if len(type_matches) == 1:
        metadata["document_type"] = type_matches[0]
    domain_matches = [
        name for name, markers in _DOMAIN_MARKERS.items()
        if sum(marker in lowered for marker in markers) >= 2
    ]
    if len(domain_matches) == 1:
        metadata["domain"] = domain_matches[0]
    rule_families = [
        name for name, markers in _RULE_MARKERS.items()
        if any(marker in lowered for marker in markers)
    ]
    if rule_families:
        metadata["rule_families"] = rule_families
    headings = re.findall(r"(?m)^\s{0,3}#{1,6}\s+(.+?)\s*$", text)
    if headings:
        metadata["sections"] = headings[:32]
        metadata["topic_tags"] = [re.sub(r"[^a-z0-9]+", "_", heading.lower()).strip("_") for heading in headings[:16]]
    return metadata
