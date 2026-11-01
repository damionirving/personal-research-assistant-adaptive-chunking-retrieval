# Web Access Guardrails

This document describes the product and architecture considerations around adding web access to the Personal Research Assistant while keeping the tool focused, understandable, and cost-conscious.

## Why This Needs Guardrails

The current assistant is designed as a focused research tool centered on:

- uploaded/private documents
- saved conversation context
- deterministic retrieval behavior
- predictable model routing

Web access can add value, especially for current events, public regulations, recent product changes, and other freshness-sensitive questions. But it also introduces real product risk if it is not tightly governed.

Without guardrails, the assistant can start to feel less like a personal research assistant and more like a generic internet chatbot.

## Main Risks

### User Confusion

If the app can answer from both private sources and the public web, users may not understand where an answer came from.

That can weaken trust because the source of truth becomes unclear:

- uploaded documents
- prior chat memory
- public web sources

### Loss of Product Focus

If web access is always available, the tool can drift away from being a grounded private research assistant and become a general-purpose assistant with search.

### Higher Latency

Web access adds extra stages beyond local retrieval:

```text
query planning -> search -> fetch -> extract -> compress -> answer
```

That is usually slower than private retrieval alone.

### Higher Token Use

Fetched web content can expand prompt size quickly, especially if multiple pages are summarized together.

### More Hallucination Surface

Public web material can be:

- noisy
- stale
- contradictory
- low quality

This increases the chance of weak synthesis or overconfident answers.

### Privacy and Compliance Concerns

Some users may expect the assistant to operate entirely inside their own uploaded corpus. If prompts or query fragments are sent into web retrieval flows, that expectation may be broken unless the behavior is clearly disclosed.

## Recommended Product Principle

Web access should be treated as a **controlled fallback mode**, not the default behavior.

Recommended order:

```text
1. Try uploaded/private sources first
2. Decide whether web access is justified
3. Use only a small, bounded web context
4. Clearly label the answer origin
```

That keeps the product focused while still allowing a path for freshness-sensitive questions.

It is also recommended to use the web only as a fallback when the question is closely related to the knowledge base already being used, and only when that knowledge base is not proprietary. If the relevant knowledge base is proprietary, internal, or user-confidential, the assistant should stay in private-only mode unless the user explicitly separates the public-web question from the private corpus.

## Recommended Retrieval Modes

The cleanest future product shape is to support explicit retrieval modes:

### 1. Private-only

Use:

- uploaded documents
- stored conversation memory

Do not use web access.

Best for:

- focused private research
- sensitive/internal material
- predictable latency and costs

### 2. Private-first with web fallback

Try private retrieval first. Only use web access when support is weak and policy allows it.

Best for:

- general assistant usage
- mixed private/public workflows
- maintaining a focused product identity

### 3. Web-assisted

Allow web retrieval when the user explicitly asks for it or when the request is clearly freshness-sensitive.

Best for:

- latest news
- regulatory changes
- current product/pricing information
- time-sensitive market or policy questions

## Guardrails That Reduce Confusion

### Source Transparency Guardrail

Every answer should make its source basis clear.

Recommended answer-origin labels:

```text
Based on your uploaded materials
Based on web sources
Based on both your materials and web sources
```

This is one of the best ways to preserve trust.

### Scope Guardrail

Default to uploaded/private materials first. Do not invoke web retrieval unless:

- no relevant private support exists
- the query is clearly current or time-sensitive
- the user explicitly asks for web/current/latest information

### Intent Guardrail

Examples:

- `What does my document say?` -> private only
- `What is the latest FDA rule?` -> web allowed
- `Compare my notes to current public guidance` -> mixed mode allowed

This keeps the assistant from using web access when the user expects a private-corpus answer.

## Guardrails That Reduce Token Use and Latency

### Web Budget Guardrail

Apply hard limits such as:

- max 1-3 fetched pages per request
- max extracted tokens per page
- max total web-context tokens
- no recursive browsing unless explicitly requested

This directly controls both cost and latency.

### Trigger Threshold Guardrail

Only allow web access when one or more of these conditions are met:

- retrieval similarity is below a threshold
- no relevant private documents are found
- the query appears freshness-sensitive
- the user explicitly requests latest/current/web information

### Answer-Shape Guardrail

When web retrieval is used:

- answer concisely
- cite only the few sources actually used
- avoid dumping long extracted passages into the prompt or final answer

This helps keep the interaction fast and readable.

## How Guardrails Help Performance

Good guardrails do more than improve safety. They also improve runtime behavior.

Benefits:

- fewer unnecessary web requests
- smaller prompts
- lower token spend
- faster responses
- less noisy context for the model

So yes: guardrails can absolutely reduce user confusion, token usage, and latency at the same time.

## Recommended Architecture Direction

If web retrieval is added in a future phase, the clean architecture would likely include:

- a retrieval mode setting
- a policy layer that decides whether web access is allowed
- bounded web search/fetch budgets
- explicit source-origin metadata in responses
- separate confidence signals for private support and web support

Possible future setting:

```text
retrieval_mode = private_only | private_first | web_assisted
```

## Recommended Product Position

The assistant should not feel like “a chatbot with search.”

It should feel like:

- a private research assistant by default
- capable of controlled web escalation when needed
- transparent about when and why it used public sources

That balance preserves product focus while still allowing useful public-web augmentation.

## Summary Recommendation

For this project, web access should eventually be:

- off by default for private-corpus questions
- enabled by policy for freshness-sensitive requests
- clearly disclosed in the UI and response metadata
- tightly limited in page count and token budget
- treated as a controlled escalation path rather than a default answer source
