# Adaptive RAG Operations

Enable each feature independently through its `PRA_` flag. Roll back immediately by
setting both adaptive flags to `false`; legacy ingestion and `HybridRetriever` continue
to operate. Do not modify Cognito or replace buckets for this rollout.

Inspect artifact JSON below the configured persistence prefix, then
`adaptive/<document-id>/v1/`. To rebuild a document, invalidate only that document's
adaptive v1 files and re-index it; retain document and vector artifacts until reindexing
is complete. Watch structured retrieval fields for candidate counts, total latency, and
fallbacks. LangSmith retains plan/context count metadata without document text.

Common fallbacks: no artifacts -> legacy retrieval; malformed table -> raw text only;
analysis/indexing exception -> legacy indexer; vector/BM25/relationship exception ->
legacy retriever. CDK deployment is unchanged, so synth/diff should not show Cognito,
bucket, or API replacement.
