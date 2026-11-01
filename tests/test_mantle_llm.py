from types import SimpleNamespace

from pra_core.documents.models import RetrievedDocument
from pra_core.llm.base import LLMProvider, LLMProviderError, RetryableLLMProviderError
from pra_core.llm.fallback import RetryableFallbackLLMProvider
from pra_core.llm.mantle import BedrockMantleLLMProvider


class FakeCompletions:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, completions):
        self.chat = SimpleNamespace(completions=completions)


def _response(text="Mantle answer"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


def test_mantle_provider_uses_short_lived_token_and_openai_compatible_request():
    completions = FakeCompletions(_response())
    clients = []

    def factory(**kwargs):
        clients.append(kwargs)
        return FakeClient(completions)

    provider = BedrockMantleLLMProvider(
        model_id="google.gemma-4-31b",
        region_name="us-east-1",
        token_provider=lambda: "short-lived-test-token",
        client_factory=factory,
    )

    answer = provider.generate(
        "What is the policy?",
        [RetrievedDocument("doc-1", "policy.md", "Use within 90 days.", 0.9)],
        [],
    )

    assert answer == "Mantle answer"
    assert clients == [
        {
            "api_key": "short-lived-test-token",
            "base_url": "https://bedrock-mantle.us-east-1.api.aws/openai/v1",
        }
    ]
    assert completions.calls[0]["model"] == "google.gemma-4-31b"
    assert completions.calls[0]["messages"][1]["content"].endswith(
        "User question:\nWhat is the policy?"
    )


def test_nonretryable_primary_error_does_not_use_fallback():
    class NonRetryable(Exception):
        status = 400

    primary = BedrockMantleLLMProvider(
        model_id="google.gemma-4-31b",
        region_name="us-east-1",
        token_provider=lambda: "token",
        client_factory=lambda **kwargs: FakeClient(FakeCompletions(error=NonRetryable())),
        retry_attempts=0,
    )
    fallback = BedrockMantleLLMProvider(
        model_id="mistral.ministral-3-14b-instruct",
        region_name="us-east-1",
        token_provider=lambda: "token",
        client_factory=lambda **kwargs: FakeClient(FakeCompletions(_response("fallback"))),
    )

    try:
        RetryableFallbackLLMProvider(primary, fallback).generate("hello", [], [])
    except LLMProviderError:
        pass
    else:
        raise AssertionError("Expected nonretryable primary failure")


def test_retryable_primary_error_uses_fallback():
    class Primary(LLMProvider):
        def generate(self, prompt, context, history):
            raise RetryableLLMProviderError("temporary")

        def resolved_model_id(self):
            return "google.gemma-4-31b"

    class Fallback(LLMProvider):
        def generate(self, prompt, context, history):
            return "fallback answer"

        def resolved_model_id(self):
            return "mistral.ministral-3-14b-instruct"

    provider = RetryableFallbackLLMProvider(Primary(), Fallback())

    assert provider.generate("hello", [], []) == "fallback answer"
    assert provider.resolved_model_id() == "mistral.ministral-3-14b-instruct"
