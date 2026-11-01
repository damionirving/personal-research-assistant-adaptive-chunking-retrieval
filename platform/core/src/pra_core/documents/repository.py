from abc import ABC, abstractmethod
from threading import Lock
from uuid import uuid4

from pra_core.documents.filenames import normalize_filename
from pra_core.documents.hashing import compute_content_hash
from pra_core.documents.metadata import infer_document_metadata
from pra_core.documents.models import Document, ParsedDocument


class DocumentRepository(ABC):
    @abstractmethod
    def add(self, document: ParsedDocument) -> Document:
        raise NotImplementedError

    @abstractmethod
    def list(self) -> list[Document]:
        raise NotImplementedError

    @abstractmethod
    def get(self, document_id: str) -> Document | None:
        raise NotImplementedError

    @abstractmethod
    def find_by_content_hash(self, content_hash: str) -> Document | None:
        raise NotImplementedError


class LocalDocumentRepository(DocumentRepository):
    """In-memory document repository for local Phase 1 development."""

    def __init__(self) -> None:
        self._documents: dict[str, Document] = {}
        self._lock = Lock()

    def add(self, document: ParsedDocument) -> Document:
        stored = Document(
            id=str(uuid4()),
            filename=normalize_filename(document.filename),
            content_type=document.content_type,
            text=document.text,
            content_hash=compute_content_hash(document.text),
            metadata=infer_document_metadata(document.filename, document.content_type, document.text),
        )
        with self._lock:
            self._documents[stored.id] = stored
        return stored

    def list(self) -> list[Document]:
        with self._lock:
            return list(self._documents.values())

    def get(self, document_id: str) -> Document | None:
        with self._lock:
            return self._documents.get(document_id)

    def find_by_content_hash(self, content_hash: str) -> Document | None:
        with self._lock:
            for document in self._documents.values():
                if document.content_hash == content_hash:
                    return document
        return None
