import json
import re

from pra_core.documents.models import DocumentChunk
from pra_core.vectorstores.base import VectorSearchResult, VectorStore


class PgVectorStore(VectorStore):
    """Postgres pgvector store for production deployments that choose Postgres."""

    def __init__(
        self,
        dsn: str,
        table_name: str,
        dimensions: int,
        connection_factory=None,
    ) -> None:
        self._dsn = dsn
        if not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", table_name):
            raise ValueError("pgvector table_name must be a simple SQL identifier.")
        self._table_name = table_name
        self._dimensions = dimensions
        self._connection_factory = connection_factory
        self._schema_ready = False

    def add(self, chunk: DocumentChunk, embedding: list[float]) -> None:
        self._ensure_schema()
        with self._connect() as connection:
            connection.execute(
                (
                    f"insert into {self._table_name} "
                    "(chunk_id, document_id, filename, text, chunk_index, chunk_metadata, embedding) "
                    "values (%s, %s, %s, %s, %s, %s::jsonb, %s::vector) "
                    "on conflict (chunk_id) do update set "
                    "document_id = excluded.document_id, "
                    "filename = excluded.filename, "
                    "text = excluded.text, "
                    "chunk_index = excluded.chunk_index, "
                    "chunk_metadata = excluded.chunk_metadata, "
                    "embedding = excluded.embedding"
                ),
                (
                    chunk.id,
                    chunk.document_id,
                    chunk.filename,
                    chunk.text,
                    chunk.index,
                    json.dumps(chunk.metadata),
                    _vector_literal(embedding),
                ),
            )
            connection.commit()

    def search(self, embedding: list[float], k: int = 3) -> list[VectorSearchResult]:
        self._ensure_schema()
        with self._connect() as connection:
            rows = connection.execute(
                (
                    f"select chunk_id, document_id, filename, text, chunk_index, chunk_metadata, "
                    f"1 - (embedding <=> %s::vector) as score "
                    f"from {self._table_name} "
                    "order by embedding <=> %s::vector "
                    "limit %s"
                ),
                (_vector_literal(embedding), _vector_literal(embedding), k),
            ).fetchall()
        return [
            VectorSearchResult(
                chunk=DocumentChunk(
                    id=row[0],
                    document_id=row[1],
                    filename=row[2],
                    text=row[3],
                    index=int(row[4]),
                    metadata=dict(row[5] or {}),
                ),
                score=float(row[6]),
            )
            for row in rows
        ]

    def clear(self) -> None:
        self._ensure_schema()
        with self._connect() as connection:
            connection.execute(f"truncate table {self._table_name}")
            connection.commit()

    def count(self) -> int:
        self._ensure_schema()
        with self._connect() as connection:
            row = connection.execute(f"select count(*) from {self._table_name}").fetchone()
        return int(row[0])

    def contains_document(self, document_id: str) -> bool:
        self._ensure_schema()
        with self._connect() as connection:
            row = connection.execute(
                f"select count(*) from {self._table_name} where document_id = %s",
                (document_id,),
            ).fetchone()
        return int(row[0]) > 0

    def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        with self._connect() as connection:
            connection.execute("create extension if not exists vector")
            connection.execute(
                (
                    f"create table if not exists {self._table_name} ("
                    "chunk_id text primary key, "
                    "document_id text not null, "
                    "filename text not null, "
                    "text text not null, "
                    "chunk_index integer not null, "
                    "chunk_metadata jsonb not null default '{}'::jsonb, "
                    f"embedding vector({self._dimensions}) not null)"
                )
            )
            connection.commit()
        self._schema_ready = True

    def _connect(self):
        if self._connection_factory is not None:
            return self._connection_factory()

        import psycopg

        return psycopg.connect(self._dsn)


def _vector_literal(embedding: list[float]) -> str:
    return json.dumps([float(value) for value in embedding])
