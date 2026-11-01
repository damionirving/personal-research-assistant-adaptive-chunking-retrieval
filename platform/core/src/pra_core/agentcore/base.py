from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AgentRequest:
    message: str
    session_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentResponse:
    answer: str
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentRuntime(ABC):
    @abstractmethod
    def invoke(self, request: AgentRequest) -> AgentResponse:
        raise NotImplementedError


class AgentMemoryGateway(ABC):
    @abstractmethod
    def load(self, session_id: str) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def save(self, session_id: str, role: str, content: str) -> None:
        raise NotImplementedError

