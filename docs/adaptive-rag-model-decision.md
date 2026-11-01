# Adaptive RAG Model Decision Record

The current MVP uses `google.gemma-3-12b-it` for simple generation, with
`amazon.nova-lite-v1:0` as its request-scoped fallback after exhausted retryable
failures. Higher-value reasoning uses `google.gemma-4-31b` through the Bedrock Mantle
OpenAI-compatible endpoint, with `mistral.ministral-3-14b-instruct` as its retryable
Bedrock Runtime fallback. Bedrock embeddings use `amazon.titan-embed-text-v2:0`.
RAGAS is deliberately isolated on Gemma 3 12B and cannot escalate into the normal-chat
reasoning path. Adaptive v1 uses deterministic analysis, planning, and reranking, so
it adds no model call by default.

## Cost-Aware Normal-Chat Routing

Normal chat keeps deterministic routing as the first decision point. The router
can return `SIMPLE`, `COMPLEX`, or `AMBIGUOUS`:

- Clear simple requests use Gemma 3 12B and retain the existing Nova Lite
  request-scoped fallback for retryable provider failures.
- Clear complex or multi-hop requests use Gemma 4 31B through Bedrock Mantle
  and retain Ministral 14B 3.0 as the retryable reasoning fallback.
- Only an ambiguous decision may invoke one small Gemma 3 12B classifier call.
  The classifier receives the query only, returns `SIMPLE` or `COMPLEX`, and
  uses a tiny output budget. It never answers the question or receives the
  retrieved context.
- Empty, malformed, unexpected, timed-out, or failed classifier output routes
  directly to the complex answer chain. It does not trigger another classifier
  call.

The classifier is a routing decision aid, not an answer fallback. This keeps
the existing Gemma 3-to-Nova Lite and Gemma 4-to-Ministral provider chains
unchanged while limiting additional paid calls to genuinely uncertain queries.
The NM-1047 multi-attribute incident query is an explicit regression case: it
must be recognized deterministically as complex and make zero classifier calls.

`PRA_CHUNK_ANALYSIS_MODEL_ID`, `PRA_QUERY_PLANNER_MODEL_ID`, and
`PRA_RERANKER_MODEL_ID` reserve configurable overrides for a future structured-output
Bedrock adapter. A stronger model may improve ambiguous domains or multi-hop plans, but
will add request latency and token cost. Any routing or model change should be validated
with the fixed retrieval dataset and answer-quality evaluation before deployment.
