from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps" / "api" / "src"))
sys.path.insert(0, str(ROOT / "platform" / "core" / "src"))

from pra_core.embeddings.local import LocalHashEmbeddingProvider
from pra_core.evaluation.rag_retrieval import load_rag_evaluation_dataset
from pra_core.evaluation.retrieval_diagnostics import (
    ChunkDiagnosticConfig,
    RetrievalDiagnosticConfig,
    format_retrieval_diagnostics_summary,
    run_retrieval_diagnostics,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run retrieval-only RAG diagnostics without calling an LLM."
    )
    parser.add_argument(
        "--dataset",
        default="data/rag_eval_dataset_v2.json",
        help="Path to the RAG evaluation dataset JSON file.",
    )
    parser.add_argument(
        "--docs-root",
        default="docs",
        help="Directory containing source Markdown documents referenced by the dataset.",
    )
    parser.add_argument(
        "--output",
        default=".pra_diagnostics/rag_retrieval_diagnostics.json",
        help="Where to write the full JSON diagnostic report.",
    )
    parser.add_argument("--embedding-dimensions", type=int, default=1024)
    parser.add_argument("--bm25-top-k", type=int, default=50)
    parser.add_argument("--vector-top-k", type=int, default=50)
    parser.add_argument("--rrf-k", type=int, default=30)
    parser.add_argument("--fusion-strategy", default="weighted_rrf")
    parser.add_argument("--bm25-fusion-weight", type=float, default=0.8)
    parser.add_argument("--vector-fusion-weight", type=float, default=0.2)
    parser.add_argument(
        "--preserve-bm25-candidates",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--rerank-top-k", type=int, default=50)
    parser.add_argument("--llm-context-top-k", type=int, default=10)
    parser.add_argument(
        "--enable-reranking",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--chunk-size", type=int, default=600)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    args = parser.parse_args()

    dataset = load_rag_evaluation_dataset(args.dataset)
    report = run_retrieval_diagnostics(
        dataset=dataset,
        embedding_provider=LocalHashEmbeddingProvider(
            dimensions=args.embedding_dimensions
        ),
        docs_root=Path(args.docs_root),
        base_config=RetrievalDiagnosticConfig(
            bm25_top_k=args.bm25_top_k,
            vector_top_k=args.vector_top_k,
            rrf_k=args.rrf_k,
            fusion_strategy=args.fusion_strategy,
            bm25_fusion_weight=args.bm25_fusion_weight,
            vector_fusion_weight=args.vector_fusion_weight,
            preserve_bm25_candidates=args.preserve_bm25_candidates,
            enable_reranking=args.enable_reranking,
            rerank_top_k=args.rerank_top_k,
            llm_context_top_k=args.llm_context_top_k,
            eval_k=dataset.eval_k,
        ),
        base_chunk_config=ChunkDiagnosticConfig(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        ),
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report.model_dump(), indent=2),
        encoding="utf-8",
    )
    print(format_retrieval_diagnostics_summary(report))
    print(f"\nWrote JSON report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
