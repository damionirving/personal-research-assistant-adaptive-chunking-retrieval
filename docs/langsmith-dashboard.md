# LangSmith Dashboard

This document describes how the MVP uses LangSmith as an agent observability layer, how it is implemented in AWS, and why this level of visibility helps operationally.

## What It Is Used For

LangSmith is not the chat application itself. It is the tracing and workflow-observability layer around the application.

In this MVP, LangSmith is useful for:

- seeing which workflow steps ran for a chat request
- understanding which model was selected
- tracking retrieval quality signals
- monitoring hallucination-risk trends
- understanding whether reasoning fallback is happening more often
- spotting slow or unstable workflow steps before users report them

## AWS Implementation

The deployed chat tracing path is:

```text
Browser -> API Gateway -> API Worker Lambda -> LangGraph workflow -> LangSmith tracer
```

The implementation is AWS-first:

- you create a Secrets Manager secret before deployment
- the secret name must be `LANGSMITH_API_KEY`
- the secret value should be JSON with the keys `LANGSMITH_API_KEY` and `LANGSMITH_ENDPOINT`
- CDK references that existing secret by name
- access to the secret should be limited to the AWS owner/admin and the Lambda execution role only
- CDK grants the Lambda permission to read it
- Lambda receives `LANGSMITH_PROJECT` as a normal environment variable
- the tracer reads `LANGSMITH_API_KEY` and `LANGSMITH_ENDPOINT` from Secrets Manager during startup and only enables LangSmith if it gets a usable API key

If the secret does not exist, cannot be read, or contains an unusable API key, the Lambda falls back to the no-op tracer and does not attempt to send metadata to LangSmith.

The MVP has three principal Lambda workloads plus two lightweight RAGAS orchestration helpers:

1. the API Worker Lambda for the live product
2. the Async Retrieval Evaluation Lambda for retrieval-only diagnostics
3. the Async RAGAS Evaluation Lambda for answer-quality evaluation batches

The RAGAS starter Lambda begins the Step Functions execution, and the RAGAS
status-checkpoint Lambda records orchestration and capacity-wait state. Neither
helper performs retrieval, answer generation, judge calls, or LangSmith reporting.

The API Worker Lambda currently sends state and workflow-step metadata to LangSmith.

The evaluation workers use no-op tracing for their evaluation paths. In particular, the RAGAS worker does not report generated answers, contexts, or metric calls to LangSmith. This keeps evaluation traffic and cost separate from normal chat observability.

The current rendered system diagram in `docs/mvp-architecture.png` reflects that split explicitly so the chat tracing path, separate evaluation workers, and disabled RAGAS reporting are visible at a glance.

## Secret Requirement Before Deploy

The secret must be created before `cdk deploy`.

Required secret name:

```text
LANGSMITH_API_KEY
```

Required JSON shape:

```json
{
  "LANGSMITH_API_KEY": "your-real-langsmith-api-key",
  "LANGSMITH_ENDPOINT": "https://api.smith.langchain.com"
}
```

## What Metadata The MVP Sends

The MVP uses metadata-first tracing rather than full-content tracing by default.

At the LangSmith API level, the app sends:

- `project_name`
- `trace_id`
- `run_id`
- `parent_run_id`
- `dotted_order`
- `name`
- `run_type`
- `start_time`
- `end_time`
- `inputs`
- `outputs`
- `tags`
- `extra.metadata`
- `error` when a run fails

For the main workflow run, the `inputs` and `outputs` are mostly metadata fields rather than raw content.

Top-level workflow run inputs:

- `message_length`
- `session_id`
- `top_k`
- `llm_top_k`

Top-level workflow run metadata:

- `workflow`
- `uptime_seconds`
- `throughput_requests_per_minute`
- `cpu_time_seconds`
- `memory_rss_mb`
- `app_trace_id`
- `service_name`
- `environment`

Top-level workflow run outputs:

- `session_id`
- `citation_count`
- `retrieved_document_count`
- `max_retrieval_score`
- `avg_retrieval_score`
- `selected_model`
- `routing_reason`
- `routing_confidence`
- `hallucination_score`
- `hallucination_action`
- `answer_length`
- `llm_calls`
- `llm_call_count`
- `llm_count`
- `input_tokens`
- `input_token_count`
- `output_tokens`
- `output_token_count`
- `cost_per_call_usd`
- `total_cost_usd`
- `uptime_seconds`
- `throughput_requests_per_minute`
- `cpu_time_seconds`
- `memory_rss_mb`
- `usage_metadata`

Per-step trace inputs are also sent for:

- `load_memory`
- `retrieve`
- `route_model`
- `generate`
- `score_grounding`
- `apply_grounding_policy`
- `save_memory`

Those step inputs include fields such as:

- `session_id`
- `query_length`
- `history_message_count`
- `retrieved_document_count`
- `llm_retrieved_document_count`
- `selected_model`
- `answer_length`
- `hallucination_score`

Per-step trace outputs include fields such as:

- `history_message_count`
- `retrieved_document_count`
- `llm_retrieved_document_count`
- `citation_count`
- `min_retrieval_score`
- `max_retrieval_score`
- `avg_retrieval_score`
- `bm25_latency_ms`
- `vector_latency_ms`
- `rrf_merge_latency_ms`
- `reranker_latency_ms`
- `retrieval_latency_ms`
- `bm25_candidate_count`
- `vector_candidate_count`
- `merged_candidate_count`
- `reranked_candidate_count`
- `selected_model`
- `routing_reason`
- `routing_confidence`
- `answer_length`
- `hallucination_score`
- `hallucination_reason`
- `hallucination_action`
- `saved_message_count`

Tags sent include:

- `workflow:research_assistant`
- `workflow_step:<step-name>`
- the service name
- the environment tag such as `environment:prod`

Typical metadata includes:

- message length
- history count
- retrieved document count
- min, max, and average retrieval score
- selected model
- routing reason
- routing confidence
- hallucination score
- grounding action
- answer length
- llm call count
- input token estimate
- output token estimate
- usage metadata with total tokens and estimated cost
- process uptime, approximate throughput, and process CPU/memory telemetry
- app trace ID

This means the Lambda can send useful agent telemetry to LangSmith when `/aws/lambda/PersonalResearchAssistantStack-api` is invoked for chat workflow execution, without automatically sending full proprietary document text.

## Requested Dashboard Metrics

The following metrics are the right operational targets for the MVP LangSmith view:

- cost per call
- total cost
- output tokens
- input tokens
- LLM calls
- trace latency
- uptime
- throughput
- CPU and memory

In practice, these split into three buckets:

1. Direct LangSmith trace metrics

- `LLM Calls`: one or more LLM spans per workflow run, including fallback or grounding-escalation calls
- `LLM Count`: number of LLM calls over time, emitted deterministically before cost estimation
- `LLM Count`: number of LLM calls over time, emitted deterministically before cost estimation and mirrored as `llm_count`
- `LLM Count`: number of LLM calls over time, emitted deterministically before cost estimation
- `Trace Latency`: total workflow latency and step latency by run name

2. Derived or estimated LangSmith metrics

- `Cost per call`: estimated from model selection plus estimated token usage and the current pricing assumptions for the models this repo can use
- `Total Cost`: sum of estimated per-call costs over a time window
- `Input Tokens`: estimated from prompt size, retrieved context, and history size unless the model provider returns exact token counts
- `Output Tokens`: estimated from answer size unless the model provider returns exact token counts
- `Throughput`: emitted as a deterministic process-level requests-per-minute estimate and can also be derived from trace counts per minute or per hour in LangSmith views

3. Process-level runtime telemetry exposed in LangSmith for quick diagnosis

- `Uptime`: deterministic process uptime for the current Lambda container
- `CPU and memory`: deterministic process CPU time and resident memory from the current Lambda container

4. AWS operational metrics that should still stay in CloudWatch as the source of truth

- fleet-wide uptime and availability
- Lambda concurrency and throttles
- API Gateway integration latency and 4XX/5XX rates
- memory-pressure and timeout behavior across the whole service

This separation matters because LangSmith is strongest at workflow traces and LLM-step analysis, while CloudWatch remains the source of truth for service-wide runtime health and infrastructure behavior.

## What It Does Not Send By Default

The current MVP avoids automatically tracing:

- raw uploaded document text
- full retrieved chunk text
- full prompt bodies
- full assistant answer bodies

## Benefits Of This Approach

Using Secrets Manager plus Lambda-side tracing has a few strong advantages:

- the LangSmith API key stays out of GitHub
- GitHub Actions does not need the LangSmith secret
- the raw API key is not stored as a plain Lambda environment variable
- IAM keeps the secret readable by the AWS owner/admin and the Lambda execution role only
- the app degrades safely when the secret is unavailable

Operationally, this is a very practical middle ground between no tracing and overly invasive tracing.

## Operational Value

This level of observability helps operationally because it lets you answer questions such as:

- Did the workflow fail in retrieval, routing, generation, or grounding?
- Is the simple model still handling most low-complexity traffic?
- Are weak retrieval cases increasing?
- Is fallback to the reasoning backup model becoming more frequent?
- Are latency spikes caused by Bedrock generation or something earlier in the workflow?
- Are grounding-risk scores drifting upward over time?

CloudWatch can tell you that the Lambda ran, errored, or slowed down.

LangSmith adds the workflow view:

- what the agent did
- which decisions it made
- how retrieval looked
- how the final answer path was chosen

That is what makes debugging and production support much faster.

## Recommended Dashboard Views

Useful LangSmith views for this MVP are:

1. Workflow latency by run name `research_assistant_workflow`
2. Step latency for `retrieve`, `route_model`, `generate_llm`, and `score_grounding`
3. Model selection trends by `selected_model`
4. Retrieval quality trends using `max_retrieval_score` and `avg_retrieval_score`
5. Grounding-risk trends using `hallucination_score`
6. Fallback pressure by watching how often reasoning fallback metadata appears
7. Error rate grouped by workflow step
8. Estimated input tokens by workflow run
9. Estimated output tokens by workflow run
10. Estimated cost per call and total estimated cost by selected model
11. `LLM Count` over time using `llm_count` or `llm_call_count`
12. Approximate throughput by trace count over time or `throughput_requests_per_minute`
13. Process uptime, CPU time, and memory RSS for current Lambda containers

## Interpretation Notes

- `Cost per call` is an estimate, not a billing-grade source of truth.
- `Total Cost` in LangSmith should be treated as a model-usage estimate, not the full AWS application cost.
- `Input Tokens` and `Output Tokens` are most accurate when the provider returns exact token usage. Until then, they should be treated as operational approximations.
- `Uptime`, `CPU`, and `memory` in LangSmith are process-level snapshots from the active Lambda container, not fleet-wide AWS service metrics.
- `llm_call_count` in LangSmith and `GET /metrics` both reflect the current warm Lambda container, so they are useful for comparing request-level traces against the process currently serving them.
- CloudWatch should still be treated as the source of truth for operational alarms and whole-service health.

## Current Limitation

This MVP currently traces the chat workflow only. Upload indexing is not yet traced into LangSmith.
