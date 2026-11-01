# Adaptive Artifact Schemas

Artifacts are JSON under `adaptive/<document-id>/v1`. The manifest records schema
version, checksum, classification, strategy decision, counts, warnings, and timestamp.

```json
{"document_id":"uuid","schema_version":"adaptive-rag/v1","decision":{"strategy":"hierarchical_structure","confidence":0.9}}
```

Child: `{"id":"stable-id","parent_id":"stable-parent","chunk_type":"passage","section_path":["Section 1"],"token_count":128}`.
Parent: `{"id":"stable-parent","title":"Section 1","ordinal":0,"text":"source text"}`.
Table: `{"id":"table-id","title":"Results","raw_text":"| A | B |","rows":[["A","B"]],"confidence":0.8}`.
Relationship: `{"source_id":"chunk-a","relationship_type":"references","target_id":"chunk-b","confidence":0.8}`.
Query plans, candidates, and context manifests are typed in process and retained in
LangGraph tracing metadata without raw source content.
