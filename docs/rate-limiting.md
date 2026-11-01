# Rate Limiting

This document describes the current MVP rate-limiting design.

## Current Behavior

The application now applies a fixed-window rate limiter in app code for four request types:

| Route | Key | Default Limit |
| --- | --- | --- |
| `/chat` | Cognito `sub` | 30 requests per 60 seconds |
| `/documents/upload` | Cognito `sub` | 5 requests per 3600 seconds |
| `/auth/config` | Client IP or `X-Forwarded-For` | 60 requests per 60 seconds |
| `/rag/evaluate/ragas` | Cognito `sub` | 2 requests per 3600 seconds |

Protected routes use the authenticated Cognito subject as the limiter key. The public auth configuration route uses the first `X-Forwarded-For` address when present, then falls back to the direct client IP.

## Why This Exists

The limiter is meant to reduce:

- accidental Bedrock cost spikes from repeated chat requests
- document-upload abuse that increases parsing, embedding, and indexing work
- noisy retries from multiple browser tabs or refresh loops
- basic request storms against public configuration endpoints
- repeated RAGAS submissions that trigger answer generation and judge-model calls

This is an MVP control, not a complete production quota system.

## Response Behavior

When a request exceeds its limit, the API returns:

```text
HTTP 429 Too Many Requests
```

It also includes:

- a JSON error body
- a `Retry-After` header with the number of seconds left in the current window

The limiter logs a structured `rate_limit_exceeded` event with the policy name, route, subject key, and retry window.

## Configuration

The current settings are:

```text
PRA_RATE_LIMIT_CHAT_REQUESTS
PRA_RATE_LIMIT_CHAT_WINDOW_SECONDS
PRA_RATE_LIMIT_UPLOAD_REQUESTS
PRA_RATE_LIMIT_UPLOAD_WINDOW_SECONDS
PRA_RATE_LIMIT_PUBLIC_REQUESTS
PRA_RATE_LIMIT_PUBLIC_WINDOW_SECONDS
PRA_RATE_LIMIT_RAGAS_REQUESTS
PRA_RATE_LIMIT_RAGAS_WINDOW_SECONDS
```

These values control simple fixed-window counters only.

## Current Implementation Choice

The limiter is process-local and in-memory. This was chosen because it is:

- easy to test
- easy to reason about
- fast to add without new infrastructure
- good enough for local development and low-scale MVP use

It also fits the current repo shape without forcing DynamoDB, Redis, or API Gateway quota configuration into the same phase.

## Current Limitation

This implementation is not a distributed global limit.

In deployed AWS, separate Lambda execution environments keep separate counters. That means the limiter is helpful, but it does not guarantee a single shared quota across all concurrent Lambda containers.

## Future Direction

For a stronger production design, move the counters to a shared backend such as DynamoDB. That would allow:

- consistent limits across Lambda instances
- tenant-aware quotas
- plan-based usage tiers
- better operational reporting
- more reliable enforcement during bursts

Possible later enhancements:

- separate burst and sustained limits
- daily usage caps
- admin bypass
- per-tenant throttling
- UI messaging for remaining quota
