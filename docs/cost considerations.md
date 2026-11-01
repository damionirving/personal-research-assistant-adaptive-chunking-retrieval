# Cost Considerations

This document describes the current MVP cost design. AWS and model-provider prices
vary by region and can change, so verify the live pricing pages and the repository's
pricing snapshot before approving a production budget.

## Current Model Paths

| Workload | Primary | Request-scoped fallback | Cost behavior |
| --- | --- | --- | --- |
| Simple chat and strong retrieval | `google.gemma-3-12b-it` | `amazon.nova-lite-v1:0` | Keeps ordinary answers on lower-cost models. |
| Complex, conditional, exception, or weak-retrieval reasoning | `google.gemma-4-31b` through Bedrock Mantle | `mistral.ministral-3-14b-instruct` through Bedrock Runtime | Reserves the larger model for requests selected by deterministic routing. |
| RAGAS answer-quality evaluation | `google.gemma-3-12b-it` | None | Prevents evaluation jobs from escalating into normal-chat fallback or reasoning models. |
| Vector embeddings | `amazon.titan-embed-text-v2:0` | None | Charged during indexing, re-indexing, and RAGAS metrics that require embeddings. |

Both chat fallbacks are used only after the primary exhausts retryable timeout,
throttling, internal-service, or temporary-unavailability retries. Access, validation,
missing-model, and credential errors fail fast. A fallback does not replace the
primary for later requests.

Gemma 4 uses the Bedrock Mantle OpenAI-compatible endpoint. The Lambda execution role
obtains a short-lived bearer token; the application does not store a Mantle API key.

## Pricing References

- [Amazon Bedrock pricing](https://aws.amazon.com/bedrock/pricing/)
- [AWS Lambda pricing](https://aws.amazon.com/lambda/pricing/)
- [Amazon API Gateway pricing](https://aws.amazon.com/api-gateway/pricing/)
- [Amazon Cognito pricing](https://aws.amazon.com/cognito/pricing/)
- [Amazon S3 pricing](https://aws.amazon.com/s3/pricing/)
- [Amazon CloudWatch pricing](https://aws.amazon.com/cloudwatch/pricing/)

The application pricing snapshot is
`platform/core/src/pra_core/data/bedrock_pricing.json`. Refresh it before deployment
when reliable per-call estimates are required. Gemma 4 Mantle pricing must be checked
against the active region and service terms rather than inferred from an older Bedrock
Runtime model.

## Cost Drivers

| Layer | Main driver | Practical control |
| --- | --- | --- |
| Bedrock generation | Input context, output length, route selection, retries, and fallbacks | Keep answers short, use adaptive context limits, and monitor route/fallback rates. |
| Bedrock embeddings | Uploaded text and full re-index operations | Avoid unnecessary re-indexing and keep the fixed 600/100 chunk baseline unless evaluation supports a change. |
| RAGAS | Answer generation plus multiple judge metrics | The default full mode maximizes coverage; use small batches, Gemma-only routing, bounded tokens, and submission limits. |
| Lambda and Step Functions | Invocation duration, state transitions, retries, and concurrent workers | Use one bounded worker operation per state-machine step, durable S3 checkpoints, and controlled concurrency. |
| API Gateway | Request count, including browser polling and retries | Use bounded polling and avoid duplicate submissions. |
| S3 | Documents, memory, FAISS artifacts, datasets, and evaluation job records | Apply lifecycle and retention policies appropriate to the environment. |
| CloudWatch | Log ingestion, retention, dashboards, and queries | Keep structured logs concise and set deliberate retention periods. |
| Cognito | Monthly active users and optional message delivery | Watch SMS MFA and verification traffic separately. |

The S3-backed FAISS design avoids an always-on managed search collection for the MVP.
That keeps idle retrieval infrastructure inexpensive, while re-indexing remains a
variable embedding cost.

## Retry Amplification

Each model provider currently permits two retries after the initial request, for at
most three attempts on that provider. If a retryable primary failure remains, the
request-scoped fallback can also make up to three provider attempts. The browser can
silently retry one failed chat request.

In the worst retryable-failure case, one user action can therefore cause up to 12
model attempts: six in the first backend request and six in the browser retry. This is
an upper bound, not the normal case, and failed attempts do not all have identical
billable token usage. Monitor retryable errors and fallback invocation rates because a
degraded service window can cost much more than a normal successful request.

## RAGAS Cost Controls

RAGAS costs more than the retrieval-only performance test because it generates
answers and runs answer-quality judges. The deployed design limits that spend:

- Full mode is the browser and API default. The browser requests 12 questions with
  `top_k=15`; explicit API requests can increase full mode to 50 questions and
  `top_k=25`. Standard mode remains the lower-cost three-native-metric option.
- The worker processes one bounded sample or metric operation per invocation and
  stores partial progress in S3 between steps. The implemented Step Functions
  orchestration is described in [RAGAS Step Functions Recovery Design](ragas-step-functions-recovery.md).
- The worker retains reserved concurrency of one. Capacity throttles are checkpointed,
  followed by a two-minute wait and retry; after 60 waits the job records an explicit
  capacity-timeout failure without making additional model calls.
- The worker uses only Gemma 3 12B and does not use Nova Lite, Gemma 4, or Ministral.
- Full retrieval uses `top_k=25`; answer output is capped at 400 tokens and judge
  output at 768 tokens.
- The worker checks remaining Lambda time, bounds retries, uses reserved concurrency
  of one, and skips judge work when no useful context exists.
- Each authenticated user is limited to two RAGAS submissions per hour per warm API
  process.
- LangSmith reporting is disabled for RAGAS jobs.

These are workload-based budgets, not a live dollar-balance cutoff. See
[rag-master.md](rag-master.md) for the complete RAGAS workflow and configuration.

## Budgeting Method

Use observed route counts and token telemetry rather than a fixed historical monthly
estimate:

```text
generation_cost = sum(
  input_tokens / 1_000_000 * model_input_price
  + output_tokens / 1_000_000 * model_output_price
)

embedding_cost = embedded_input_tokens / 1_000_000 * embedding_input_price

monthly_total = generation_cost + embedding_cost + lambda + api_gateway
                + s3 + cloudwatch + cognito + external_observability
```

Separate primary, retry, and fallback calls in the calculation. Also separate ordinary
chat, uploads/re-indexing, retrieval evaluation, and RAGAS so a test run cannot be
mistaken for normal customer traffic.

## Recommended Guardrails

1. Keep Gemma 3 as the simple primary and reserve Gemma 4 for deterministic reasoning
   routes.
2. Treat Nova Lite and Ministral as resilience paths, not load-balancing destinations.
3. Alert on retry, fallback, throttling, and RAGAS submission spikes.
4. Keep output-token limits explicit for every provider and evaluator.
5. Run retrieval-only evaluation before paid answer-quality evaluation.
6. Require controlled retrieval and answer-quality comparisons before changing model,
   routing, chunking, or context defaults.
7. Revalidate live AWS pricing and model availability before every budget commitment.

In simple terms: the design keeps ordinary requests on the cheaper route, escalates
only when the question requires it, uses fallback models only for temporary failures,
and tightly limits evaluation work.
