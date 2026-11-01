from collections.abc import Callable
import time

from pra_core.documents.models import RetrievedDocument
from pra_core.llm.base import LLMProvider, LLMProviderError, RetryableLLMProviderError
from pra_core.llm.bedrock import (
    _SYSTEM_PROMPT,
    _build_user_prompt,
    _error_code,
    _is_retryable_error,
    _retry_delay_seconds,
)
from pra_core.logging import get_logger


logger = get_logger(__name__)


class BedrockMantleLLMProvider(LLMProvider):
    """OpenAI-compatible Bedrock Mantle provider using short-lived IAM tokens."""

    def __init__(
        self,
        model_id: str,
        region_name: str,
        base_url: str | None = None,
        max_tokens: int = 700,
        temperature: float = 0.2,
        retry_attempts: int = 2,
        retry_base_delay_seconds: float = 0.5,
        token_provider: Callable[[], str] | None = None,
        client_factory: Callable[..., object] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._model_id = model_id
        self._region_name = region_name
        self._base_url = base_url or f"https://bedrock-mantle.{region_name}.api.aws/openai/v1"
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._retry_attempts = retry_attempts
        self._retry_base_delay_seconds = retry_base_delay_seconds
        self._token_provider = token_provider or self._default_token_provider
        self._client_factory = client_factory
        self._sleep = sleep

    def generate(self, prompt, context, history) -> str:
        return self._generate(prompt, context, history, self._max_tokens)

    def generate_bounded(self, prompt, context, history, max_output_tokens) -> str:
        return self._generate(prompt, context, history, max_output_tokens)

    def resolved_model_id(self) -> str | None:
        return self._model_id

    def _generate(self, prompt, context, history, max_tokens) -> str:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(prompt, context, history)},
        ]
        for attempt in range(self._retry_attempts + 1):
            try:
                token = self._token_provider()
                client = self._client(token)
                response = client.chat.completions.create(
                    model=self._model_id,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=self._temperature,
                )
                answer = self._extract_text(response)
                if not answer:
                    raise LLMProviderError("Bedrock Mantle returned an empty response.")
                return answer
            except Exception as exc:
                if attempt < self._retry_attempts and _is_retryable_error(exc):
                    delay = _retry_delay_seconds(attempt, self._retry_base_delay_seconds)
                    logger.warning(
                        "bedrock_mantle_generation_retry",
                        extra={
                            "bedrock_mantle_model_id": self._model_id,
                            "bedrock_mantle_region": self._region_name,
                            "attempt": attempt + 1,
                            "max_attempts": self._retry_attempts + 1,
                            "delay_seconds": round(delay, 3),
                            "error_type": exc.__class__.__name__,
                            "error_code": _error_code(exc),
                        },
                    )
                    self._sleep(delay)
                    continue
                logger.exception(
                    "bedrock_mantle_generation_failed",
                    extra={
                        "bedrock_mantle_model_id": self._model_id,
                        "bedrock_mantle_region": self._region_name,
                        "error_type": exc.__class__.__name__,
                    },
                )
                error_type = RetryableLLMProviderError if _is_retryable_error(exc) else LLMProviderError
                raise error_type(
                    "LLM provider is unavailable. Check Bedrock Mantle access and configuration."
                ) from exc
        raise RetryableLLMProviderError("Bedrock Mantle did not complete generation.")

    def _client(self, token: str):
        if self._client_factory is not None:
            return self._client_factory(api_key=token, base_url=self._base_url)
        from openai import OpenAI

        return OpenAI(api_key=token, base_url=self._base_url)

    def _default_token_provider(self) -> str:
        from aws_bedrock_token_generator import provide_token

        return provide_token(region=self._region_name)

    @staticmethod
    def _extract_text(response: object) -> str:
        choices = getattr(response, "choices", None) or []
        if not choices:
            return ""
        content = getattr(getattr(choices[0], "message", None), "content", None)
        return content.strip() if isinstance(content, str) else ""
