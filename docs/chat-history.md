# Chat History

This document describes how chat history works in the current MVP, what is persisted today, and how the project can evolve toward a more production-ready conversation history system.

## Current Behavior

The application already saves conversational history, but in a lightweight MVP form.

Current flow:

```text
POST /chat
  -> load prior messages by session_id
  -> retrieve supporting documents
  -> route model
  -> generate answer
  -> save user message
  -> save assistant message
```

The save/load behavior is handled by the LangGraph workflow in:

```text
platform/core/src/pra_core/agent/workflow.py
```

The workflow loads prior messages through the configured `MemoryStore`, then appends:

- the new user message
- the new assistant answer

The stored history is keyed by `session_id`.

## What Is Stored Today

In the current MVP, chat history is primarily conversational memory, not a fully modeled product feature.

Persisted today:

- user message text
- assistant message text
- message order within a `session_id`

Depending on configuration, that memory can be stored through:

```text
memory -> process-local in-memory store
file   -> local JSON persistence
s3     -> deployed JSON persistence in S3
```

## What Is Not Yet First-Class History

The current design does not yet treat every chat turn as a rich structured conversation record.

Not yet modeled as first-class persisted chat-history fields:

- conversation title
- structured turn timestamps
- citations per turn
- retrieval scores per turn
- hallucination score per turn
- hallucination action per turn
- selected model per turn
- searchable conversation index
- rename/archive/delete conversation operations
- explicit conversation ownership and listing UX

That means the system can continue a conversation, but it is not yet a polished “chat history product” with a browsable conversation library.

## Why This Matters

There are really two different needs hiding under the phrase "chat history":

1. Conversational memory for inference
2. User-visible conversation history for product UX

Those overlap, but they are not the same thing.

Inference memory is about helping the model answer the next message. Product history is about helping the user browse, revisit, organize, and audit prior conversations.

The current MVP mostly solves the first need.

## Current Strengths

The current approach is good for:

- continuing a session with prior context
- local development
- low-complexity persistence
- simple deployment wiring
- deterministic testing

It also keeps the LangGraph workflow and memory abstraction backend-agnostic.

## Current Limitations

S3-backed JSON memory is acceptable for MVP persistence, but it is awkward for a mature chat-history feature.

Limitations of the current approach:

- difficult to list conversations efficiently by user
- awkward to sort by most recent activity
- weak support for partial updates
- not ideal for concurrent writes
- limited querying and analytics
- hard to power a polished sidebar/history browser

In short: the current memory store is fine for “remember the thread,” but not ideal for “manage a full conversation product.”

## Recommended Evolution

The clean next step is to separate:

- memory used for inference
- durable chat history used for product UX and auditability

Recommended direction:

1. Keep the current `MemoryStore` abstraction for prompt-time memory.
2. Add a separate `ConversationStore` abstraction for durable conversation history.
3. Store structured conversations in a relational database for production.
4. Expose conversation-management APIs for list/load/rename/delete/archive behavior.
5. Keep only a bounded recent window, or a summary plus recent turns, in model context.

## Suggested Production Data Model

Example shape:

```text
Conversation
  conversation_id
  user_id
  title
  created_at
  updated_at

ConversationMessage
  message_id
  conversation_id
  role
  content
  created_at
  selected_model
  citations
  retrieval_scores
  hallucination_score
  hallucination_action
```

This structure would support both user-facing history and operational auditing much better than raw JSON memory blobs.

## Recommended Storage Direction

For the current MVP:

- keep the existing `MemoryStore`
- keep local/file/S3 compatibility

For a stronger production implementation:

- move durable chat history into Postgres or another relational system
- reserve S3 for document/vector persistence and large-object storage

This gives the project a better foundation for:

- user conversation lists
- thread resume
- sorting/filtering/search
- analytics
- billing/account controls
- compliance and audit needs

## Front-End Implications

Once durable structured history exists, the browser UI can add:

- a conversation sidebar
- recent chats
- rename conversation
- start new chat
- reopen prior thread
- archive/delete thread

At that point, `session_id` can evolve from being just a transient client identifier into a stable persisted conversation identifier.

## Recommended Architecture Principle

The important design rule is:

```text
Do not treat inference memory and product history as the same storage concern.
```

They can share data, but they should not be forced into the same abstraction forever.

That separation will make the app easier to scale, easier to query, and easier to evolve into a more complete research assistant product.
