# Adaptive Retrieval

Queries are deterministically classified into definition, exact identifier, table,
comparison, cross-reference, or explanation intent. The typed `QueryPlan` captures
domain, identifiers, acronym expansions, selected retrievers, weights, and confidence.
Cross-reference questions prioritize relationship search; exact identifiers retain
lexical retrieval. Known conservative expansions include CHF, CVE, EPS, and ACP.

Chunk-level BM25 and existing vector-store results are fused using weighted reciprocal
rank fusion, so incomparable raw scores are never mixed directly. A deterministic
reranker rewards term overlap and exact identifiers. Selected children optionally gain
only limited parent context under a token budget; output keeps original document and
chunk citation identifiers. Missing BM25/artifacts/vector hits, or any exception,
returns the legacy `HybridRetriever` result.
