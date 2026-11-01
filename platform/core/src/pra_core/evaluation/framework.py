from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationCase:
    name: str
    question: str
    answer: str
    expected_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvaluationResult:
    name: str
    passed: bool
    score: float
    details: str


class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, case: EvaluationCase) -> EvaluationResult:
        raise NotImplementedError


class ExpectedTermsEvaluator(Evaluator):
    """Simple deterministic evaluator for local regression checks."""

    def evaluate(self, case: EvaluationCase) -> EvaluationResult:
        if not case.expected_terms:
            return EvaluationResult(
                name=case.name,
                passed=True,
                score=1.0,
                details="No expected terms configured.",
            )

        answer = case.answer.lower()
        matched = [
            term for term in case.expected_terms if term.lower() in answer
        ]
        score = len(matched) / len(case.expected_terms)
        return EvaluationResult(
            name=case.name,
            passed=score == 1.0,
            score=score,
            details=f"Matched {len(matched)} of {len(case.expected_terms)} expected terms.",
        )


class EvaluationRunner:
    def __init__(self, evaluators: list[Evaluator]) -> None:
        self._evaluators = evaluators

    def run(self, cases: list[EvaluationCase]) -> list[EvaluationResult]:
        results: list[EvaluationResult] = []
        for case in cases:
            for evaluator in self._evaluators:
                results.append(evaluator.evaluate(case))
        return results

