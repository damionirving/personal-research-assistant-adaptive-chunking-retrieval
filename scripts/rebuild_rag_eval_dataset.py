from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps" / "api" / "src"))
sys.path.insert(0, str(ROOT / "platform" / "core" / "src"))

from pra_core.documents.chunking import TextChunker
from pra_core.evaluation.dataset_relabel import (
    rebuild_rag_evaluation_dataset,
    write_rag_evaluation_dataset,
)
from pra_core.evaluation.rag_retrieval import load_rag_evaluation_dataset


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild RAG evaluation labels against the current deterministic chunker."
    )
    parser.add_argument(
        "--input",
        default="data/rag_eval_dataset_v1.json",
        help="Input RAG evaluation dataset JSON file.",
    )
    parser.add_argument(
        "--output",
        default="data/rag_eval_dataset_v2.json",
        help="Output dataset path for rebuilt labels.",
    )
    parser.add_argument(
        "--docs-root",
        default="docs",
        help="Directory containing the evaluation Markdown documents.",
    )
    parser.add_argument("--chunk-size", type=int, default=600)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    args = parser.parse_args()

    dataset = load_rag_evaluation_dataset(args.input)
    rebuilt = rebuild_rag_evaluation_dataset(
        dataset=dataset,
        docs_root=Path(args.docs_root),
        chunker=TextChunker(
            chunk_size=args.chunk_size,
            overlap=args.chunk_overlap,
        ),
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_rag_evaluation_dataset(rebuilt, output_path)
    print(f"Wrote rebuilt dataset: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
