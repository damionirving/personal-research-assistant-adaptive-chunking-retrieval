from pra_api.main import create_app
from pra_core.config import Settings
from pra_core.documents.models import RetrievedDocument
from pra_core.llm.bedrock import BedrockLLMProvider
from pra_core.llm.base import LLMProviderError
from pra_core.llm.fallback import RetryableFallbackLLMProvider
from pra_core.llm.mock import MockLLMProvider


class FakeBedrockClient:
    def __init__(self) -> None:
        self.request = None

    def converse(self, **kwargs):
        self.request = kwargs
        return {
            "output": {
                "message": {
                    "content": [
                        {"text": "Bedrock answer from fake client."},
                    ]
                }
            }
        }


class FailingBedrockClient:
    def converse(self, **kwargs):
        raise RuntimeError("model access denied")


class BedrockError(Exception):
    def __init__(self, code: str, status_code: int) -> None:
        super().__init__(code)
        self.response = {
            "Error": {"Code": code},
            "ResponseMetadata": {"HTTPStatusCode": status_code},
        }


class FlakyBedrockClient:
    def __init__(self) -> None:
        self.calls = 0

    def converse(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise BedrockError("ThrottlingException", 429)
        return {
            "output": {
                "message": {
                    "content": [
                        {"text": "Recovered Bedrock answer."},
                    ]
                }
            }
        }


class PrimaryThenFallbackClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def converse(self, **kwargs):
        model_id = kwargs["modelId"]
        self.calls.append(model_id)
        if model_id == "mistral.mistral-large-3-675b-instruct":
            raise BedrockError("ThrottlingException", 429)
        return {
            "output": {
                "message": {
                    "content": [
                        {"text": "Fallback Haiku answer."},
                    ]
                }
            }
        }


class SimplePrimaryThenFallbackClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def converse(self, **kwargs):
        model_id = kwargs["modelId"]
        self.calls.append(model_id)
        if model_id == "google.gemma-3-12b-it":
            raise BedrockError("ThrottlingException", 429)
        return {
            "output": {
                "message": {
                    "content": [{"text": "Nova Lite fallback answer."}],
                }
            }
        }


class AccessDeniedBedrockClient:
    def __init__(self) -> None:
        self.calls = 0

    def converse(self, **kwargs):
        self.calls += 1
        raise BedrockError("AccessDeniedException", 403)


def test_bedrock_provider_calls_converse_with_context() -> None:
    client = FakeBedrockClient()
    provider = BedrockLLMProvider(
        model_id="us.anthropic.claude-opus-4-1-20250805-v1:0",
        region_name="us-east-1",
        max_tokens=500,
        temperature=0.1,
        client=client,
    )

    answer = provider.generate(
        prompt="What is the policy?",
        context=[
            RetrievedDocument(
                document_id="doc-1",
                filename="policy.md",
                text="Remote stipend must be used within 90 days.",
                score=0.9,
            )
        ],
        history=["user: hello"],
    )

    assert answer == "Bedrock answer from fake client."
    assert client.request["modelId"] == "us.anthropic.claude-opus-4-1-20250805-v1:0"
    assert client.request["inferenceConfig"] == {
        "maxTokens": 500,
        "temperature": 0.1,
    }
    system_prompt = client.request["system"][0]["text"]
    assert "Answer directly and concisely" in system_prompt
    assert "Default to 1-3 short sentences" in system_prompt
    assert "Do not include introductions" in system_prompt
    assert "enumerate each applicable category, amount, timing condition, and approval requirement" in system_prompt
    assert "Do not omit numeric values or conditions" in system_prompt
    assert "For greetings or small talk" in system_prompt
    assert "natural conversational way" in system_prompt
    assert "Do not mechanically repeat the user's topic" in system_prompt
    user_prompt = client.request["messages"][0]["content"][0]["text"]
    assert "policy.md" in user_prompt
    assert "Remote stipend" in user_prompt
    assert "What is the policy?" in user_prompt


def test_bedrock_provider_raises_provider_error_for_runtime_failure() -> None:
    provider = BedrockLLMProvider(
        model_id="us.anthropic.claude-opus-4-1-20250805-v1:0",
        region_name="us-east-1",
        client=FailingBedrockClient(),
    )

    try:
        provider.generate(prompt="hello", context=[], history=[])
    except LLMProviderError as exc:
        assert "Check Bedrock model access" in str(exc)
    else:
        raise AssertionError("Expected LLMProviderError")


def test_bedrock_provider_retries_transient_runtime_failures() -> None:
    client = FlakyBedrockClient()
    provider = BedrockLLMProvider(
        model_id="us.anthropic.claude-opus-4-1-20250805-v1:0",
        region_name="us-east-1",
        client=client,
        retry_attempts=2,
        retry_base_delay_seconds=0,
        sleep=lambda seconds: None,
    )

    answer = provider.generate(prompt="hello", context=[], history=[])

    assert answer == "Recovered Bedrock answer."
    assert client.calls == 2


def test_bedrock_provider_does_not_retry_access_errors() -> None:
    client = AccessDeniedBedrockClient()
    provider = BedrockLLMProvider(
        model_id="us.anthropic.claude-opus-4-1-20250805-v1:0",
        region_name="us-east-1",
        client=client,
        retry_attempts=2,
        retry_base_delay_seconds=0,
        sleep=lambda seconds: None,
    )

    try:
        provider.generate(prompt="hello", context=[], history=[])
    except LLMProviderError as exc:
        assert "Check Bedrock model access" in str(exc)
    else:
        raise AssertionError("Expected LLMProviderError")
    assert client.calls == 1


def test_bedrock_reasoning_provider_uses_fallback_for_retryable_primary_failure() -> None:
    client = PrimaryThenFallbackClient()
    provider = BedrockLLMProvider(
        model_id="mistral.mistral-large-3-675b-instruct",
        fallback_model_id="arn:aws:bedrock:us-east-1::inference-profile/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        region_name="us-east-1",
        client=client,
        retry_attempts=0,
        retry_base_delay_seconds=0,
        sleep=lambda seconds: None,
    )

    answer = provider.generate(prompt="hello", context=[], history=[])

    assert answer == "Fallback Haiku answer."
    assert client.calls == [
        "mistral.mistral-large-3-675b-instruct",
        "arn:aws:bedrock:us-east-1::inference-profile/us.anthropic.claude-haiku-4-5-20251001-v1:0",
    ]
    assert provider.resolved_model_id() == (
        "arn:aws:bedrock:us-east-1::inference-profile/"
        "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    )


def test_bedrock_reasoning_provider_tries_primary_again_on_next_request() -> None:
    client = FakeBedrockClient()
    provider = BedrockLLMProvider(
        model_id="mistral.mistral-large-3-675b-instruct",
        fallback_model_id="arn:aws:bedrock:us-east-1::inference-profile/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        region_name="us-east-1",
        client=client,
    )

    answer = provider.generate(prompt="hello", context=[], history=[])

    assert answer == "Bedrock answer from fake client."
    assert client.request["modelId"] == "mistral.mistral-large-3-675b-instruct"
    assert provider.resolved_model_id() == "mistral.mistral-large-3-675b-instruct"


def test_bedrock_simple_provider_uses_nova_lite_after_retryable_failure() -> None:
    client = SimplePrimaryThenFallbackClient()
    provider = BedrockLLMProvider(
        model_id="google.gemma-3-12b-it",
        fallback_model_id="amazon.nova-lite-v1:0",
        region_name="us-east-1",
        client=client,
        retry_attempts=0,
        retry_base_delay_seconds=0,
        sleep=lambda seconds: None,
    )

    answer = provider.generate(prompt="hello", context=[], history=[])

    assert answer == "Nova Lite fallback answer."
    assert client.calls == [
        "google.gemma-3-12b-it",
        "amazon.nova-lite-v1:0",
    ]
    assert provider.resolved_model_id() == "amazon.nova-lite-v1:0"


def test_app_defaults_to_mock_llm_provider() -> None:
    app = create_app(Settings(environment="test"))

    assert isinstance(app.state.workflow._llm_provider, MockLLMProvider)


def test_app_can_use_bedrock_llm_provider() -> None:
    app = create_app(
        Settings(
            environment="test",
            llm_provider="bedrock",
            bedrock_model_id="us.anthropic.claude-opus-4-1-20250805-v1:0",
        )
    )

    assert isinstance(app.state.workflow._llm_provider, RetryableFallbackLLMProvider)
