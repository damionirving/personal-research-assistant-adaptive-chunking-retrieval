# Vector Stores

This document compares the vector-store options relevant to the current MVP and the enterprise roadmap.

## Current State

The deployed MVP currently uses:

```text
Bedrock Titan embeddings
  -> FaissVectorStore
  -> FAISS index persisted in S3
```

OpenSearch Serverless is no longer deployed by CDK. It was removed to stop ongoing OpenSearch Serverless collection charges during MVP testing. The OpenSearch adapter remains in code so it can be re-enabled later if the retrieval workload justifies the cost.

## Comparison

| Option | Deployment Shape | Strengths | Weaknesses | Best Fit |
| --- | --- | --- | --- | --- |
| Local / persistent JSON vectors | In-process or S3/file-backed JSON snapshots | Lowest cost, simple, deterministic, easy to inspect | Not suited for large corpora, concurrent writes, advanced filtering, or enterprise search | Local development, controlled demos |
| FAISS | Lambda-friendly in-memory index persisted to S3 | Fast similarity search, no extra managed vector service baseline, compatible with the current abstraction | Not a managed database, weak multi-user story, persistence and metadata require extra work, Lambda memory becomes a ceiling | Current deployed MVP |
| OpenSearch Serverless | Managed AWS vector/search service | Managed scaling, vector search, metadata filtering, hybrid search potential | Baseline cost can be high for small MVPs, provider-specific behavior, more infrastructure | Larger AWS-native document search workloads |
| pgvector | PostgreSQL extension | SQL plus vectors together, good metadata joins, often cheaper than OpenSearch | Requires Postgres operations/tuning, less specialized than search-first systems | SaaS apps with users, billing, metadata, and moderate vector search |

## How Similarity Search Works

The embedding does not contain the words. It is only a list of numbers. The app stores the original chunk text and metadata alongside the embedding or stores enough identifiers to map back to the original text.

The retrieval flow is:

```text
User question
  -> embedding provider creates a question vector
  -> vector store compares question vector to stored chunk vectors
  -> vector store returns top matching chunks plus similarity scores
  -> LLM receives the original chunk text as context
  -> LLM generates a grounded answer
```

In the current deployed MVP, those vectors are loaded into a FAISS-style in-memory index during Lambda execution and persisted back to S3 after updates. If OpenSearch Serverless is re-enabled later, the same high-level flow stays the same, but OpenSearch stores the vector records and returns search hits from its vector index.

The raw vector similarity score is cosine similarity, so it is theoretically bounded from `-1.0` to `1.0`. BM25 is a separate lexical ranking score with no fixed universal upper bound. When the hybrid retriever merges those two signals, the combined score should be treated as an internal ranking value rather than a probability or a normalized confidence score.

The deployed JSON persistence objects use the neutral `pra-prod/` S3 prefix rather than the older phase-specific label.

To keep repeated uploads from inflating the deployed index, the document layer now stores a SHA-256 `content_hash` for each parsed document. Before a new upload is added, the API checks whether a stored document already has that hash. If so, it reuses the existing document record and usually skips re-indexing. If the stored document exists but its vectors are missing, the duplicate upload path retries indexing so a previously failed upload can recover cleanly.

## Recommendation

For now, keep the local JSON vector path for testing and the FAISS-backed S3-persisted index for deployed AWS. This minimizes idle infrastructure cost while keeping deployed retrieval fast enough for the MVP.

Move to a managed vector database when one of these becomes true:

- Document volume grows beyond comfortable FAISS-in-Lambda performance.
- Multiple users or organizations need stronger isolation and metadata filtering.
- Upload/indexing becomes asynchronous and high-volume.
- Search quality needs hybrid keyword plus semantic retrieval.
- Production usage justifies the baseline cost of a managed vector service.

Future AWS paths to document, but not implement yet:

| Candidate | When To Prefer It |
| --- | --- |
| ECS Fargate + FAISS | Your FAISS index or document corpus becomes too large for Lambda memory or cold-start constraints. |
| pgvector | You want user data, permissions, metadata, billing, and vectors in one relational database. |
| OpenSearch Serverless | You need AWS-native managed search, large-scale retrieval, filtering, or hybrid search. |
