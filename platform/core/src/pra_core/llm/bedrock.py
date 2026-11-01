import random
import time

from pra_core.documents.models import RetrievedDocument
from pra_core.llm.base import LLMProvider, LLMProviderError
from pra_core.logging import get_logger


logger = get_logger(__name__)


class BedrockLLMProvider(LLMProvider):
    """Amazon Bedrock Runtime provider using the Converse API."""

    def __init__(
        self,
        model_id: str,
        region_name: str,
        max_tokens: int = 700,
        temperature: float = 0.2,
        fallback_model_id: str | None = None,
        client=None,
        retry_attempts: int = 2,
        retry_base_delay_seconds: float = 0.5,
        sleep=time.sleep,
    ) -> None:
        self._model_id = model_id
        self._region_name = region_name
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._fallback_model_id = fallback_model_id
        self._client = client
        self._retry_attempts = retry_attempts
        self._retry_base_delay_seconds = retry_base_delay_seconds
        self._sleep = sleep
        self._resolved_model_id = model_id

    def generate(
        self,
        prompt: str,
        context: list[RetrievedDocument],
        history: list[str],
    ) -> str:
        try:
            self._resolved_model_id = self._model_id
            request = {
                "system": [{"text": _SYSTEM_PROMPT}],
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"text": _build_user_prompt(prompt, context, history)}
                        ],
                    }
                ],
                "inferenceConfig": {
                    "maxTokens": self._max_tokens,
                    "temperature": self._temperature,
                },
            }
            response = self._generate_with_fallback(request)
        except Exception as exc:
            logger.exception(
                "bedrock_generation_failed",
                extra={
                    "bedrock_model_id": self._resolved_model_id,
                    "bedrock_region": self._region_name,
                    "error_type": exc.__class__.__name__,
                },
            )
            raise LLMProviderError(
                "LLM provider is unavailable. Check Bedrock model access and configuration."
            ) from exc

        answer = _extract_text(response)
        if not answer:
            raise LLMProviderError("LLM provider returned an empty response.")
        return answer

    def generate_bounded(
        self,
        prompt: str,
        context: list[RetrievedDocument],
        history: list[str],
        max_output_tokens: int,
    ) -> str:
        return self._generate(prompt, context, history, max_output_tokens)

    def _generate(
        self,
        prompt: str,
        context: list[RetrievedDocument],
        history: list[str],
        max_output_tokens: int | None = None,
    ) -> str:
        try:
            self._resolved_model_id = self._model_id
            request = {
                "system": [{"text": _SYSTEM_PROMPT}],
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"text": _build_user_prompt(prompt, context, history)}
                        ],
                    }
                ],
                "inferenceConfig": {
                    "maxTokens": max_output_tokens or self._max_tokens,
                    "temperature": self._temperature,
                },
            }
            response = self._generate_with_fallback(request)
        except Exception as exc:
            logger.exception(
                "bedrock_generation_failed",
                extra={
                    "bedrock_model_id": self._resolved_model_id,
                    "bedrock_region": self._region_name,
                    "error_type": exc.__class__.__name__,
                },
            )
            raise LLMProviderError(
                "LLM provider is unavailable. Check Bedrock model access and configuration."
            ) from exc

        answer = _extract_text(response)
        if not answer:
            raise LLMProviderError("Bedrock provider returned an empty response.")
        return answer

    def resolved_model_id(self) -> str | None:
        return self._resolved_model_id

    def _generate_with_fallback(self, request: dict) -> dict:
        try:
            return self._converse_with_retries(
                {
                    **request,
                    "modelId": self._model_id,
                }
            )
        except Exception as primary_exc:
            if not self._should_fallback(primary_exc):
                raise

            self._resolved_model_id = self._fallback_model_id or self._model_id
            logger.warning(
                "bedrock_reasoning_fallback_invoked",
                extra={
                    "bedrock_primary_model_id": self._model_id,
                    "bedrock_fallback_model_id": self._fallback_model_id,
                    "bedrock_region": self._region_name,
                    "error_type": primary_exc.__class__.__name__,
                    "error_code": _error_code(primary_exc),
                },
            )
            response = self._converse_with_retries(
                {
                    **request,
                    "modelId": self._fallback_model_id,
                }
            )
            logger.info(
                "bedrock_reasoning_fallback_succeeded",
                extra={
                    "bedrock_primary_model_id": self._model_id,
                    "bedrock_fallback_model_id": self._fallback_model_id,
                    "bedrock_region": self._region_name,
                },
            )
            return response

    def _converse_with_retries(self, request: dict) -> dict:
        model_id = request["modelId"]
        for attempt in range(self._retry_attempts + 1):
            try:
                return self._bedrock().converse(**request)
            except Exception as exc:
                if attempt >= self._retry_attempts or not _is_retryable_error(exc):
                    raise

                delay_seconds = _retry_delay_seconds(
                    attempt,
                    self._retry_base_delay_seconds,
                )
                logger.warning(
                    "bedrock_generation_retry",
                    extra={
                        "bedrock_model_id": model_id,
                        "bedrock_region": self._region_name,
                        "attempt": attempt + 1,
                        "max_attempts": self._retry_attempts + 1,
                        "delay_seconds": round(delay_seconds, 3),
                        "error_type": exc.__class__.__name__,
                        "error_code": _error_code(exc),
                    },
                )
                self._sleep(delay_seconds)

        raise LLMProviderError("LLM provider did not complete generation.")

    def _should_fallback(self, exc: Exception) -> bool:
        return bool(self._fallback_model_id) and _is_retryable_error(exc)

    def _bedrock(self):
        if self._client is None:
            import boto3
            from botocore.config import Config

            self._client = boto3.client(
                "bedrock-runtime",
                region_name=self._region_name,
                config=Config(read_timeout=3600),
            )
        return self._client


_SYSTEM_PROMPT = (
    "You are a personal research assistant. Answer directly and concisely. "
    "Default to 1-3 short sentences. Do not include introductions, summaries, "
    "or restatements of the question. Use bullets only when the user asks for "
    "a list or the answer requires multiple distinct points. "
    "For multi-condition incident, credit, approval, exception, or comparison "
    "questions, completeness takes priority over brevity: enumerate each "
    "applicable category, amount, timing condition, and approval requirement. "
    "Do not omit numeric values or conditions when summarizing retrieved evidence. "
    "For greetings or small talk, reply with one short sentence and do not "
    "summarize uploaded documents unless asked. Use retrieved documents only "
    "when they are relevant to the user's question. If the context does not "
    "contain enough information, say so clearly in a natural conversational "
    "way. Do not mechanically repeat the user's topic or say that the "
    "available documents are not relevant unless that detail is genuinely "
    "helpful."
)


def _build_user_prompt(
    prompt: str,
    context: list[RetrievedDocument],
    history: list[str],
) -> str:
    context_text = "\n\n".join(
        (
            f"Source: {document.filename}\n"
            f"Document ID: {document.document_id}\n"
            f"Content: {document.text}"
        )
        for document in context
    )
    history_text = "\n".join(history[-8:])
    return (
        f"Conversation history:\n{history_text or 'No prior messages.'}\n\n"
        f"Retrieved document context:\n{context_text or 'No matching documents.'}\n\n"
        f"User question:\n{prompt}"
    )


def _extract_text(response: dict) -> str:
    content = response.get("output", {}).get("message", {}).get("content", [])
    texts = [
        item["text"]
        for item in content
        if isinstance(item, dict) and isinstance(item.get("text"), str)
    ]
    return "\n".join(texts).strip()


_RETRYABLE_ERROR_CODES = {
    "APIConnectionError",
    "APITimeoutError",
    "ConnectionClosedError",
    "EndpointConnectionError",
    "InternalFailure",
    "InternalServerException",
    "ModelNotReadyException",
    "ModelTimeoutException",
    "ReadTimeoutError",
    "RequestTimeout",
    "RequestTimeoutException",
    "ServiceUnavailable",
    "ServiceUnavailableException",
    "ThrottlingException",
    "TooManyRequestsException",
    "InternalServerError",
    "RateLimitError",
}

_NON_RETRYABLE_ERROR_CODES = {
    "AccessDeniedException",
    "AuthorizationError",
    "InvalidSignatureException",
    "ResourceNotFoundException",
    "UnrecognizedClientException",
    "UnauthorizedException",
    "ValidationException",
}


def _is_retryable_error(exc: Exception) -> bool:
    code = _error_code(exc)
    if code in _NON_RETRYABLE_ERROR_CODES:
        return False
    if code in _RETRYABLE_ERROR_CODES:
        return True

    status_code = _status_code(exc)
    return status_code in {408, 429, 500, 502, 503, 504}


def _error_code(exc: Exception) -> str:
    class_name = exc.__class__.__name__
    if class_name in _RETRYABLE_ERROR_CODES:
        return class_name
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        error = response.get("Error", {})
        code = error.get("Code")
        if isinstance(code, str):
            return code
    return exc.__class__.__name__


def _status_code(exc: Exception) -> int | None:
    direct_status = getattr(exc, "status", None)
    if isinstance(direct_status, int):
        return direct_status
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        metadata = response.get("ResponseMetadata", {})
        status_code = metadata.get("HTTPStatusCode")
        if isinstance(status_code, int):
            return status_code
    return None


def _retry_delay_seconds(attempt: int, base_delay_seconds: float) -> float:
    exponential_delay = base_delay_seconds * (2**attempt)
    jitter = random.uniform(0, base_delay_seconds)
    return exponential_delay + jitter
