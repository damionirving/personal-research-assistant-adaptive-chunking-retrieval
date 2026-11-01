from pra_core.guardrails.base import Guardrail, GuardrailResult


class LocalPromptGuardrail(Guardrail):
    """Small deterministic prompt guardrail for local safety checks."""

    def __init__(self, blocked_terms: tuple[str, ...] = ()) -> None:
        self._blocked_terms = tuple(term.lower() for term in blocked_terms)

    def validate(self, text: str) -> GuardrailResult:
        lowered = text.lower()
        for term in self._blocked_terms:
            if term in lowered:
                return GuardrailResult(
                    allowed=False,
                    reason=f"Prompt contains blocked term: {term}",
                )
        return GuardrailResult(allowed=True)


class GuardrailChain(Guardrail):
    def __init__(self, guardrails: list[Guardrail]) -> None:
        self._guardrails = guardrails

    def validate(self, text: str) -> GuardrailResult:
        for guardrail in self._guardrails:
            result = guardrail.validate(text)
            if not result.allowed:
                return result
        return GuardrailResult(allowed=True)
