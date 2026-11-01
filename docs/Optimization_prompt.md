Optimize the existing AWS RAG pipeline using the current async evaluation framework and dataset.

Current findings:
- BM25 outperforms hybrid.
- Vector retrieval is weak.
- Reranking adds little value.
- Dataset/index sync is nearly complete but 1 question is unlabeled and 2 IDs are missing.
- Latency is already good.

Tasks:

1. Fix evaluation integrity
- restore all 25 labeled questions
- resolve all missing IDs
- fail clearly on stale labels

2. Run controlled sweeps by changing existing env/config variables only where possible:
- PRA_BM25_TOP_K: 25, 50, 75, 100
- PRA_VECTOR_TOP_K: 25, 50, 75, 100
- PRA_RRF_K: 20, 30, 60, 90
- PRA_ENABLE_RERANKING: true, false
- PRA_RERANK_TOP_K: 25, 50, 75
- PRA_LLM_CONTEXT_TOP_K: 5, 8, 10, 15
- PRA_CHUNK_SIZE / PRA_CHUNK_OVERLAP:
  - 300/50
  - 450/75
  - 600/100

3. Compare:
- BM25 only
- vector only
- current RRF hybrid
- BM25-weighted hybrid
- normalized weighted fusion, including 0.8/0.2, 0.7/0.3, 0.6/0.4
- preserve BM25 candidates when hybrid would rank them lower

4. Report:
- Recall, Precision, MRR, NDCG at 5/10/25
- stage-level losses
- final-context coverage
- BM25-only, vector-only, and shared contribution
- latency
- best configuration by quality, then latency/cost

5. Apply the best configuration to production defaults without breaking current behavior. Prefer simple deterministic logic over new dependencies.

6. Add or update tests for sweeps, fusion, config handling, label integrity, and regression thresholds.

Documentation:
- do not update existing docs except `rag-master.md`
- create/update `rag-master.md` with findings, chosen configuration, commands, and rationale
- add one link to `rag-master.md` from README

Do not call the LLM during retrieval evaluation. Reuse existing code and avoid duplicating production logic.