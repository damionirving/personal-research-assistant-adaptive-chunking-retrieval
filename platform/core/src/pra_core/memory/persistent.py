from threading import Lock

from pra_core.memory.store import MemoryStore
from pra_core.persistence.object_store import JsonObjectStore


class PersistentMemoryStore(MemoryStore):
    """Conversation memory persisted through a JSON object store."""

    def __init__(self, object_store: JsonObjectStore, key: str = "memory.json") -> None:
        self._object_store = object_store
        self._key = key
        self._lock = Lock()
        self._messages = self._object_store.read_json(self._key, {})

    def add_message(self, session_id: str, role: str, content: str) -> None:
        with self._lock:
            self._messages.setdefault(session_id, []).append(f"{role}: {content}")
            self._object_store.write_json(self._key, self._messages)

    def get_messages(self, session_id: str) -> list[str]:
        with self._lock:
            return list(self._messages.get(session_id, []))

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            if session_id in self._messages:
                del self._messages[session_id]
                self._object_store.write_json(self._key, self._messages)
