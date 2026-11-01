# LangSmith In AWS

This document describes the current MVP path for sending workflow metadata from the deployed AWS Lambda to LangSmith without storing the LangSmith API key in GitHub or in plain Lambda environment variables.

## Current Design

The deployed path for the current chat workflow is:

```text
Browser -> API Gateway -> API Worker Lambda -> LangGraph workflow -> LangSmith tracer
```

The Lambda does not keep the LangSmith API key directly in code or in the repository.

Instead:

- you create the Secrets Manager secret before deployment
- CDK references that existing secret by name
- Lambda receives `LANGSMITH_PROJECT` as a normal environment variable
- Lambda reads the secret from Secrets Manager during tracer setup
- the tracer sends metadata-first workflow and step runs to LangSmith

## Why This Design

This is the safer MVP pattern because:

- the GitHub repository does not need the LangSmith API key
- GitHub Actions does not need the LangSmith API key
- the Lambda environment does not need the raw key value
- IAM controls which Lambda can read the secret

## Create The Secret First

Before `cdk deploy`, create a Secrets Manager secret named:

```text
LANGSMITH_API_KEY
```

This pre-creation step matters because the stack references an existing secret rather than creating one for you.

## Secret Shape

The secret should use a JSON payload like:

```json
{
  "LANGSMITH_API_KEY": "replace-me",
  "LANGSMITH_ENDPOINT": "https://api.smith.langchain.com"
}
```

The key names inside the JSON must include both `LANGSMITH_API_KEY` and `LANGSMITH_ENDPOINT`.

## Runtime Settings

The deployed Lambda uses this environment variable:

```text
LANGSMITH_PROJECT=personal-research-assistant
```

The project name is assigned by this repo and currently matches the repository name.

The deployed Lambda reads these values from the existing AWS Secrets Manager secret named `LANGSMITH_API_KEY`:

| Secret JSON field | Required | Purpose |
| --- | --- | --- |
| `LANGSMITH_API_KEY` | Yes | LangSmith API key. |
| `LANGSMITH_ENDPOINT` | Yes | LangSmith API endpoint, for example `https://api.smith.langchain.com`. |

If tracing is enabled but the secret does not exist, cannot be read, or does not contain a usable `LANGSMITH_API_KEY` value, the app safely falls back to the no-op tracer and continues serving requests. In that case the Lambda does not attempt to send metadata to LangSmith.

## What Is Sent

The MVP sends metadata-first trace payloads rather than full proprietary content by default.

Typical metadata includes:

- message length
- history count
- retrieved document count
- retrieval scores
- selected model
- routing reason
- hallucination score
- grounding action
- answer length
- llm call count
- input and output token estimates
- usage metadata with total tokens and estimated cost
- process uptime, approximate throughput, and process CPU/memory telemetry
- app trace ID

For dashboard planning, the MVP should think about LangSmith fields in three categories:

- direct trace metrics such as `LLM Calls` and `Trace Latency`
- derived metrics such as estimated `Input Tokens`, `Output Tokens`, `Cost per call`, `Total Cost`, and `Throughput`
- process-level runtime telemetry emitted by the active Lambda container such as `Uptime`, `CPU`, and `memory`
- `llm_call_count`, which is also exposed through `GET /metrics` as a warm-container counter

That split keeps LangSmith focused on workflow and model observability while CloudWatch remains the source of truth for fleet-wide Lambda runtime health.

## What Is Not Sent By Default

The current MVP avoids automatically shipping:

- raw uploaded document text
- full retrieved chunk text
- full prompt bodies
- full assistant answer bodies

## CDK Responsibilities

The stack now does four things for LangSmith:

- references the existing Secrets Manager secret by name
- grants the Lambda execution role permission to read that secret
- injects `LANGSMITH_PROJECT` into Lambda
- keeps `LANGSMITH_API_KEY` and `LANGSMITH_ENDPOINT` in Secrets Manager instead of in plain environment variables

## How To Enable It After Deploy

1. Create the `LANGSMITH_API_KEY` secret in AWS Secrets Manager.
2. Store the real LangSmith API key under the JSON key `LANGSMITH_API_KEY`.
3. Store the LangSmith endpoint under the JSON key `LANGSMITH_ENDPOINT`.
4. Deploy the CDK stack.

If the secret value is valid, new chat requests will start emitting LangSmith runs from inside AWS.

## GitHub Actions Impact

The current GitHub Actions deployment does not need a LangSmith secret for this design.

That is because:

- GitHub Actions uses AWS OIDC for deployment
- the secret is stored in AWS Secrets Manager
- Lambda reads it at runtime inside AWS

## Current Limitation

This MVP still traces chat workflow metadata only. Upload indexing is not yet traced into LangSmith.

The deployment now has two separate Lambda processes:

1. `PersonalResearchAssistantStack-api` for the interactive app.
2. The async retrieval evaluation worker Lambda for retrieval-only diagnostics.
3. The async RAGAS evaluation worker Lambda for answer-quality batches; it uses no-op tracing and does not report to LangSmith.

The API Worker Lambda currently sends workflow state and step traces to LangSmith.

The async RAG evaluation worker Lambda can read the same LangSmith configuration path because it is built from the same app assembly, but the retrieval-evaluation execution path is not currently instrumented to emit that detailed LangSmith trace stream.

That distinction is also shown in the rendered architecture diagram at `docs/mvp-architecture.png` and in the source at `docs/mvp-architecture.mmd`.
