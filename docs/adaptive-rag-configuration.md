# Adaptive RAG Configuration

All settings use the `PRA_` prefix and are validated at startup. Application config
defaults keep adaptive chunking and retrieval opt-in with
`ADAPTIVE_CHUNKING_ENABLED=false` and `ADAPTIVE_RETRIEVAL_ENABLED=false`; table
artifact preservation defaults to `TABLE_PROCESSING_ENABLED=true`. The deployed CDK
stack explicitly enables all three settings.

| Setting | Default | Effect |
| --- | --- | --- |
| `DOMAIN_DETECTION_ENABLED` / `DOCUMENT_TYPE_DETECTION_ENABLED` | `true` | Deterministic metadata classification; no model cost. |
| `RELATIONSHIP_EXTRACTION_ENABLED` / `RELATIONSHIP_RETRIEVAL_ENABLED` | `true` | Create/use bounded lightweight references. |
| `HIERARCHICAL_CHUNKING_ENABLED` / `PARENT_CHILD_EXPANSION_ENABLED` | `true` | Preserve parents and add selective context. |
| `QUERY_EXPANSION_ENABLED` | `true` | Enables conservative acronym expansion. |
| `CHUNK_*_TOKENS` | `80/300/600/60` | Min/target/max/overlap; min <= target <= max required. |
| `RETRIEVAL_*_TOP_K` | `30/30/30/10` | Vector/BM25/fused/final candidate limits. |
| `RETRIEVAL_*_WEIGHT` | `.40/.60/.00` | Vector/BM25/relationship RRF weights; the deployed policy is BM25-first. |
| `RERANKER_MODE` | `deterministic` | Local reranking; `bedrock` is reserved for a configured implementation. |
| `*_MODEL_ID` | empty | Optional override records; defaults preserve MVP model IDs. |

Profiles accept `auto`, plus `structure`/`semantic`/`fixed` for chunking and
`lexical`/`semantic`/`relationship` for retrieval. They are intentionally advisory in
v1; use feature flags for operational rollout. Larger limits increase latency and
context cost. No default option adds Bedrock requests.
