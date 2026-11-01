# Model Cost and Usage

This inventory distinguishes active application models from planning candidates. The
supplied US East estimates are per 1 million tokens and are ordered by output price;
rows without a verified output price appear last. Actual Bedrock charges can vary by
region, endpoint, inference profile, service tier, and model version. Verify current
prices before making a production routing decision.

| Model | Input / 1M | Output / 1M | Currently used? | Pipeline role | Recommendation |
|---|---:|---:|---|---|---|
| `google.gemma-3-4b-it` | $0.04* | $0.08* | No | Candidate lightweight answer or evaluation model | Benchmark before use for multi-hop evaluation |
| `meta.llama3-2-1b-instruct-v1:0` | $0.10 | $0.10 | No | Lightweight classification, extraction, and short summarization | Limited reason to adopt for this architecture without a benchmark |
| `amazon.nova-micro-v1:0` | $0.035 | $0.14 | No | Ultra-low-cost text chat, classification, summarization, and routing | Candidate for simple chat or a future LLM router |
| `meta.llama3-2-3b-instruct-v1:0` | $0.15 | $0.15 | No | Lightweight chat, extraction, and simple RAG | Reasonable open-weight candidate; benchmark first |
| `mistral.ministral-3-14b-instruct` (Ministral 14B 3.0) | $0.20 | $0.20 | **Yes, fallback** | Request-scoped Bedrock Runtime fallback after retryable Gemma 4 Mantle failures | Keep as reasoning resilience path; monitor invocation rate |
| `amazon.nova-lite-v1:0` | $0.06 | $0.24 | **Yes, fallback** | Request-scoped fallback after retryable Gemma 3 simple-answer failures | Keep as simple-answer resilience path; monitor invocation rate |
| `meta.llama3-2-11b-instruct-v1:0` | $0.20 | $0.27 | No | Multimodal and document reasoning | Legacy candidate; benchmark before use |
| `google.gemma-3-12b-it` | $0.09 | $0.29 | **Yes** | Simple-chat primary and the only RAGAS generation/evaluator model | Current low-cost default |
| `Gemma 3 27B` | $0.23 | $0.38 | No | Candidate complex-reasoning model | Evaluate against the active reasoning route before use |
| `meta.llama4-scout-17b-instruct-v1:0` | $0.17 | $0.66 | No | Long-context RAG, multimodal retrieval, and document reasoning | High-priority candidate to benchmark |
| `meta.llama3-3-70b-instruct-v1:0` | $0.72 | $0.72 | No | General reasoning, RAG, coding, and tool use | Benchmark before use |
| `meta.llama4-maverick-17b-instruct-v1:0` | $0.24 | $0.97 | No | Multimodal reasoning, RAG, and agent workflows | Candidate for complex RAG |
| `mistral.mistral-large-3-675b-instruct` | $0.50 | $1.50 | No | Previous primary reasoning model | Historical comparison only |
| `amazon.nova-2-lite-v1:0` | $0.30 | $2.50 | No | Current-generation multimodal document and agent model | Benchmark only where its capabilities justify output cost |
| `amazon.nova-pro-v1:0` | $0.80 | $3.20 | No | Multimodal reasoning, complex RAG, and agent workflows | Benchmark only for higher-value workloads |
| `amazon.nova-premier-v1:0` | $2.50 | $12.50 | No | Highest-capability Nova model | Reserve for difficult, high-value requests |
| `google.gemma-4-31b` | Verify current Mantle pricing | Verify current Mantle pricing | **Yes** | Primary normal-chat reasoning model through Bedrock Mantle | Keep IAM token access scoped to the API Lambda |
| `Claude Opus 4.1` | Not verified for the legacy configured ID | Not verified for the legacy configured ID | No normal runtime use | Legacy `PRA_BEDROCK_MODEL_ID` compatibility reference | Not selected by the normal router |

An asterisk marks a supplied planning estimate that must be verified before budgeting.
Candidate rows are not active merely because they appear in this table.

## Non-LLM Embedding Model

| Model | Pricing | Currently used? | Pipeline role |
|---|---|---|---|
| `amazon.titan-embed-text-v2:0` | Separate embedding pricing | Yes when the Bedrock embedding provider is enabled | Creates vector embeddings for semantic retrieval and RAGAS context metrics; it does not generate answers |

## Preprocessing and Routing Model Usage

These steps prepare queries and documents or choose an answer model before generation.
They use deterministic rules, regular expressions, retrieval scores, and configured
thresholds rather than an LLM, so they do not add routing-model token cost.

| Function | Model used | Current implementation |
|---|---|---|
| Intent classification | No LLM | Deterministic keyword and pattern rules classify simple lookup, policy/rule, incident/exception, and comparison questions. |
| Query rewriting | No LLM | No general rewriting model is active; only deterministic expansions are applied when enabled. |
| Metadata extraction | No LLM in the active path | Domain, document type, sections, tables, relationships, and identifiers use deterministic parsing. |
| Knowledge-graph proposal extraction | No active LLM call | An optional proposal hook exists, but the API does not wire an LLM proposer into the knowledge-graph service. |
| Routing decisions | No LLM for clear cases; Gemma 3 12B only for ambiguous cases | `DeterministicModelRouter` first classifies SIMPLE, COMPLEX, or AMBIGUOUS. The existing Gemma 3 model is used as a tiny query-only fallback classifier only for AMBIGUOUS decisions. Invalid classifier output fails upward to the complex route without repeated routing calls. |
| Simple answer generation | `google.gemma-3-12b-it`, then `amazon.nova-lite-v1:0` after exhausted retryable failures | Nova Lite is request-scoped and does not replace the configured primary. |
| Complex answer generation | `google.gemma-4-31b` through Mantle, then `mistral.ministral-3-14b-instruct` after exhausted retryable failures | The Mantle provider obtains a short-lived IAM bearer token; the fallback uses Bedrock Runtime. |
| RAGAS answer generation and evaluation | `google.gemma-3-12b-it` by default; production routing is opt-in | `PRA_RAGAS_USE_NORMAL_MODEL_ROUTING=false` keeps answer generation on the configured Gemma-only model. The RAGAS judge remains Gemma 3; setting the flag to `true` can add normal-chat Gemma 4, Nova Lite, or Ministral answer calls. |

## Application Verification

- Normal simple chat uses Gemma 3 12B, with Nova Lite as a request-scoped fallback
  after exhausted retryable failures.
- Normal complex chat uses Gemma 4 31B through Bedrock Mantle, with Ministral 14B 3.0
  as a request-scoped fallback after exhausted retryable failures.
- Clear simple and complex queries make zero fallback-router calls; only ambiguous
  queries may add one small Gemma 3 classification call.
- RAGAS forces Gemma 3 12B for answer generation and evaluator paths, so it does not
  use Nova Lite, Gemma 4, or Ministral.
- The legacy `PRA_BEDROCK_MODEL_ID` remains for compatibility, but normal routing uses
  the explicit simple and reasoning settings.

## Gemma 4 Mantle Authentication

The API Lambda invokes Gemma 4 through the OpenAI-compatible Amazon Bedrock Mantle
endpoint. It obtains a short-lived bearer token from the
`aws-bedrock-token-generator` package using its IAM execution role, creates a client
for the request, and never stores or logs that token. CDK grants Mantle inference
creation on project resources plus bearer-token use. RAGAS does not use Mantle.

## Sources

- [Amazon Bedrock pricing](https://aws.amazon.com/bedrock/pricing/)
- [Supported foundation models in Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html)
- [Amazon Nova model cards](https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards-amazon.html)
- [Google Gemma model cards](https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards-google.html)
- [Meta Llama model cards](https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards-meta.html)
- [Mistral AI model cards](https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards-mistral-ai.html)
