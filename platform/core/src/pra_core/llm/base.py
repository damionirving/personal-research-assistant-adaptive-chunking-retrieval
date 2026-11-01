from abc import ABC, abstractmethod

from pra_core.documents.models import RetrievedDocument


class LLMProviderError(Exception):
    """Raised when the configured LLM provider cannot generate a response."""


class RetryableLLMProviderError(LLMProviderError):
    """Raised when generation failed after retries and may be safely retried."""


class LLMProvider(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        context: list[RetrievedDocument],
        history: list[str],
    ) -> str:
        raise NotImplementedError

    def resolved_model_id(self) -> str | None:
        return None

    def generate_bounded(
        self,
        prompt: str,
        context: list[RetrievedDocument],
        history: list[str],
        max_output_tokens: int,
    ) -> str:
        """Generate with a per-request budget when the provider supports it."""
        return self.generate(prompt=prompt, context=context, history=history)
