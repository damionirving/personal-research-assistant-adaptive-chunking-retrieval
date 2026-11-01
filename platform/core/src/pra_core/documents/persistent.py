from datetime import datetime
from threading import Lock
from uuid import uuid4

from pra_core.documents.filenames import normalize_filename
from pra_core.documents.hashing import compute_content_hash
from pra_core.documents.models import Document, ParsedDocument
from pra_core.documents.metadata import infer_document_metadata
from pra_core.documents.repository import DocumentRepository
from pra_core.persistence.object_store import JsonObjectStore


class PersistentDocumentRepository(DocumentRepository):
    """Document repository persisted through a JSON object store."""

    def __init__(self, object_store: JsonObjectStore, key: str = "documents.json") -> None:
        self._object_store = object_store
        self._key = key
        self._lock = Lock()
        self._documents = self._load()

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
            self._save()
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

    def _load(self) -> dict[str, Document]:
        rows = self._object_store.read_json(self._key, [])
        documents: dict[str, Document] = {}
        for row in rows:
            document = Document(
                id=row["id"],
                filename=normalize_filename(row["filename"]),
                content_type=row["content_type"],
                text=row["text"],
                content_hash=row.get("content_hash", compute_content_hash(row["text"])),
                metadata=dict(row.get("metadata", {})),
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            documents[document.id] = document
        return documents

    def _save(self) -> None:
        rows = [
            {
                "id": document.id,
                "filename": document.filename,
                "content_type": document.content_type,
                "text": document.text,
                "content_hash": document.content_hash,
                "metadata": document.metadata,
                "created_at": document.created_at.isoformat(),
            }
            for document in self._documents.values()
        ]
        self._object_store.write_json(self._key, rows)
