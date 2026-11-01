from pra_core.documents.models import RetrievedDocument
from pra_core.llm.base import LLMProvider


class MockLLMProvider(LLMProvider):
    """Deterministic local LLM substitute for Phase 1."""

    def generate(
        self,
        prompt: str,
        context: list[RetrievedDocument],
        history: list[str],
    ) -> str:
        if not context:
            return (
                "I can help once I have material on that topic. If you upload "
                "something relevant, I can work from it."
            )

        filenames = ", ".join(document.filename for document in context)
        snippets = " ".join(document.text[:160] for document in context)
        return f"Here's what I found in the local documents ({filenames}): {snippets}"
