from pra_core.agentcore.base import AgentMemoryGateway, AgentRequest, AgentResponse, AgentRuntime
from pra_core.memory.store import MemoryStore


class LocalAgentRuntime(AgentRuntime):
    """Local placeholder runtime preserving the AgentCore boundary."""

    def invoke(self, request: AgentRequest) -> AgentResponse:
        return AgentResponse(
            answer=f"Local AgentCore placeholder received: {request.message}",
            metadata={
                "session_id": request.session_id,
                "agentcore_production": False,
            },
        )


class LocalAgentMemoryGateway(AgentMemoryGateway):
    """Local adapter over the existing memory store."""

    def __init__(self, memory_store: MemoryStore) -> None:
        self._memory_store = memory_store

    def load(self, session_id: str) -> list[str]:
        return self._memory_store.get_messages(session_id)

    def save(self, session_id: str, role: str, content: str) -> None:
        self._memory_store.add_message(session_id=session_id, role=role, content=content)

