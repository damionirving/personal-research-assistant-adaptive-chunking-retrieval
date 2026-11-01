from pra_core.documents.models import RetrievedDocument
from pra_core.llm.base import LLMProvider, LLMProviderError, RetryableLLMProviderError
from pra_core.llm.bedrock import _is_retryable_error
from pra_core.logging import get_logger


logger = get_logger(__name__)


class RetryableFallbackLLMProvider(LLMProvider):
    """Use a fallback provider only after a retryable primary failure."""

    def __init__(self, primary: LLMProvider, fallback: LLMProvider) -> None:
        self._primary = primary
        self._fallback = fallback
        self._resolved_model_id = primary.resolved_model_id()

    def generate(self, prompt: str, context: list[RetrievedDocument], history: list[str]) -> str:
        return self._run(prompt, context, history, None)

    def generate_bounded(
        self,
        prompt: str,
        context: list[RetrievedDocument],
        history: list[str],
        max_output_tokens: int,
    ) -> str:
        return self._run(prompt, context, history, max_output_tokens)

    def resolved_model_id(self) -> str | None:
        return self._resolved_model_id

    def _run(self, prompt, context, history, max_output_tokens):
        try:
            if max_output_tokens is None:
                answer = self._primary.generate(prompt, context, history)
            else:
                answer = self._primary.generate_bounded(
                    prompt, context, history, max_output_tokens
                )
            self._resolved_model_id = self._primary.resolved_model_id()
            return answer
        except (RetryableLLMProviderError, LLMProviderError) as exc:
            cause = exc.__cause__
            if not isinstance(exc, RetryableLLMProviderError) and not (
                cause is not None and _is_retryable_error(cause)
            ):
                raise
            logger.warning(
                "llm_retryable_fallback_invoked",
                extra={
                    "primary_model_id": self._primary.resolved_model_id(),
                    "fallback_model_id": self._fallback.resolved_model_id(),
                },
            )
            if max_output_tokens is None:
                answer = self._fallback.generate(prompt, context, history)
            else:
                answer = self._fallback.generate_bounded(
                    prompt, context, history, max_output_tokens
                )
            self._resolved_model_id = self._fallback.resolved_model_id()
            return answer
