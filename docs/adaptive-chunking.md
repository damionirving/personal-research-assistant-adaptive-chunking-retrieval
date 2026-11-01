# Adaptive Chunking

Adaptive chunking is deterministic by default. It inspects headings, sentence
boundaries, table-like pipe rows, reference language, filename, and domain vocabulary.
The selector chooses `table_aware`, `hierarchical_structure`, `semantic_boundary`, or
`fixed_size`, returning a typed `ChunkDecision` containing reasons, confidence, and
fallback strategy. Optional model IDs are configuration hooks only; no Bedrock call is
made by default for analysis.

The chunking decision architecture is rendered in
[chunking-decision-chain.mmd](chunking-decision-chain.mmd). If
`PRA_ADAPTIVE_CHUNKING_ENABLED=false`, or if adaptive analysis/artifact persistence
fails, indexing falls back to the legacy `TextChunker` so uploads can still be indexed.

Each section becomes a parent artifact and sentence-bounded passages become retrievable
children. Children carry stable IDs, document and parent IDs, section path, ordinal,
token count, domain, type, strategy, and artifact version. Tables preserve raw text and
only structured rows for recognizable pipe tables; unreliable parsing remains raw text.

References such as `See Section 1` become confidence-scored portable relationships.
Artifacts are idempotent for a document ID and stored at
`adaptive/<document-id>/v1/{manifest,parents,chunks,tables,relationships}.json` beneath
the existing object-store prefix. Reprocess by deleting that versioned artifact set or
using a new artifact version after a schema/config/model change.
