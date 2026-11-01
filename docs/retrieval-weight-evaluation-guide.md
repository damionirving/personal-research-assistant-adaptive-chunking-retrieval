# Retrieval Weight Evaluation Guide

## Question

How do I evaluate different BM25 and vector retrieval weights?

## Answer

Use a controlled A/B test.

### 1. Record the baseline

Run the existing **Run RAG Performance Test** with:

- The same uploaded documents
- The same evaluation dataset
- The same `top_k`
- The same question count

Record:

```text
Recall@5
Recall@10
Recall@25
Context loss
Latency
```

### 2. Choose one alternative

For example:

```text
Baseline: BM25 0.80, Vector 0.20
Candidate: BM25 0.70, Vector 0.30
```

Change the CDK environment values:

```python
"PRA_BM25_FUSION_WEIGHT": "0.70",
"PRA_VECTOR_FUSION_WEIGHT": "0.30",
"PRA_RETRIEVAL_BM25_WEIGHT": "0.70",
"PRA_RETRIEVAL_VECTOR_WEIGHT": "0.30",
```

Keep the weights totaling `1.0`.

### 3. Deploy the candidate

Deploy the branch containing only the weight change, then run the exact same
performance test again.

### 4. Compare results

Keep the candidate only if it improves evidence coverage and retrieval recall
without an unacceptable latency increase.

### 5. Run RAGAS last

After retrieval testing, run a small RAGAS test to check:

```text
Faithfulness
Context precision
Context recall
Answer relevancy
Answer correctness
```

Use the performance test first because it does not call the LLM and costs less.
Change only one retrieval setting at a time so the result identifies which
change caused the improvement or regression.

## Post-Rebuild Optimization Workflow

After rebuilding `rag_eval_dataset_v2.json`, use this guide to compare a small
set of BM25/vector weight candidates against the same documents, questions,
chunking settings, and `top_k` value. Keep BM25-first retrieval, hybrid
fallback, disabled-by-default reranking, adaptive context sizing, and all
public response behavior unchanged during the comparison.

Select a candidate using the combined results for Recall@5, Recall@10,
Recall@25, context loss, final-context coverage, and latency. Do not select a
weight set from one metric alone. Run a small answer-quality or RAGAS check
only after retrieval-only testing identifies the best candidate, so model cost
does not obscure retrieval results.

The optimization process is intentionally a controlled comparison rather than
a broad parameter sweep. Record the baseline, record each candidate and its
metrics, document the selected values, and retain the baseline if no candidate
improves evidence coverage without an unacceptable latency or accuracy cost.

## Latest Controlled Result

The rebuilt `data/rag_eval_dataset_v2.json` contains 22 questions using:

- `docs/acmecorp-benefits-policy.md`
- `docs/test-corpus/scenario-northstar-meridian.md`
- 600-token chunks
- 100-token overlap

The retrieval-only comparison used the same 25-question `top_k` setting,
local deterministic embeddings, BM25 preservation, and reranking disabled for
every run. It did not call the LLM.

| Candidate | Recall@5 | Recall@10 | Recall@25 | Context loss | Context coverage | Avg latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25-heavy `0.80/0.20` | 0.8644 | 0.9508 | 0.9773 | 3 | 0.9474 | 7.16 ms |
| Balanced `0.70/0.30` | 0.8644 | 0.9508 | 0.9773 | 3 | 0.9474 | 7.51 ms |
| Moderate semantic `0.60/0.40` | 0.8644 | 0.9508 | 0.9773 | 3 | 0.9474 | 11.20 ms |

The selected legacy hybrid weights are:

```text
PRA_BM25_FUSION_WEIGHT=0.80
PRA_VECTOR_FUSION_WEIGHT=0.20
```

Because the quality metrics tied, the BM25-heavy candidate was selected on
latency and consistency with BM25-first retrieval. The adaptive retrieval
weights are kept consistent with the deployment baseline:

```text
PRA_RETRIEVAL_BM25_WEIGHT=0.60
PRA_RETRIEVAL_VECTOR_WEIGHT=0.40
PRA_RETRIEVAL_RELATIONSHIP_WEIGHT=0.00
```

These three adaptive weights are normalized by configuration loading. The
selection preserves hybrid fallback, adaptive context sizing, and reranking
disabled by default. A small answer-quality or RAGAS run should follow this
retrieval comparison; it was not used to select the retrieval weights.
