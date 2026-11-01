from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class GuardrailResult:
    allowed: bool
    reason: str | None = None


class Guardrail(ABC):
    @abstractmethod
    def validate(self, text: str) -> GuardrailResult:
        raise NotImplementedError

