from abc import ABC, abstractmethod
from threading import Lock


class MemoryStore(ABC):
    @abstractmethod
    def add_message(self, session_id: str, role: str, content: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_messages(self, session_id: str) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def clear_session(self, session_id: str) -> None:
        raise NotImplementedError


class LocalMemoryStore(MemoryStore):
    """In-memory conversation history for local Phase 1 development."""

    def __init__(self) -> None:
        self._messages: dict[str, list[str]] = {}
        self._lock = Lock()

    def add_message(self, session_id: str, role: str, content: str) -> None:
        with self._lock:
            self._messages.setdefault(session_id, []).append(f"{role}: {content}")

    def get_messages(self, session_id: str) -> list[str]:
        with self._lock:
            return list(self._messages.get(session_id, []))

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            self._messages.pop(session_id, None)
