from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps" / "api" / "src"))
sys.path.insert(0, str(ROOT / "platform" / "core" / "src"))

from pra_core.evaluation.answer_quality import (
    AnswerQualityAnswer,
    AnswerQualitySummary,
    evaluate_answer_quality,
)
from pra_core.evaluation.rag_retrieval import load_rag_evaluation_dataset


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate captured RAG answers without calling an LLM."
    )
    parser.add_argument(
        "--dataset",
        default="data/rag_eval_dataset_v2.json",
        help="Path to the existing RAG evaluation dataset JSON file.",
    )
    parser.add_argument(
        "--answers",
        required=True,
        help="JSON file containing {\"answers\": [{\"id\", \"answer\", ...}] }.",
    )
    parser.add_argument(
        "--output",
        default=".pra_diagnostics/rag_answer_quality.json",
        help="Where to write the full JSON answer-quality report.",
    )
    args = parser.parse_args()

    dataset = load_rag_evaluation_dataset(args.dataset)
    answers = _load_answers(Path(args.answers))
    report = evaluate_answer_quality(dataset=dataset, answers=answers)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    _print_summary(report)
    print(f"\nWrote JSON report: {output_path}")
    return 0


def _load_answers(path: Path) -> list[AnswerQualityAnswer]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        values = payload
    elif isinstance(payload, dict) and isinstance(payload.get("answers"), list):
        values = payload["answers"]
    else:
        raise ValueError("Answer input must be a JSON list or an object with an 'answers' list.")
    return [AnswerQualityAnswer.model_validate(value) for value in values]


def _print_summary(report: AnswerQualitySummary) -> None:
    print(f"Dataset: {report.dataset}")
    print(f"Questions: {report.total_questions}; answered: {report.answered_questions}")
    for name, value in report.metrics.items():
        rendered = "n/a" if value is None else f"{value:.4f}"
        print(f"{name}: {rendered}")


if __name__ == "__main__":
    raise SystemExit(main())
