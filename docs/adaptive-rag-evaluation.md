# Adaptive RAG Evaluation

Use the existing test suite and RAG evaluation harness to compare legacy and adaptive
modes against the same fixtures. Track boundary/structure/table/relationship retention
at ingestion and Recall@k, Precision@k, MRR, nDCG, duplicate rate, parent-expansion
usefulness, reranker lift, and P50/P95/P99 latency at retrieval.

Suggested release gates: no legacy regression when flags are false, no table claim when
parsing confidence is low, equal-or-better Recall@10 on representative questions, and
bounded latency growth under the configured candidate limits. Interpret lower precision
with higher recall as a retrieval-plan tuning signal before increasing context budgets.
