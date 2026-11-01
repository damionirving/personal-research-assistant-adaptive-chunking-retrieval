from fastapi import APIRouter, Depends, File, Header, HTTPException, Request, UploadFile
from pydantic import BaseModel

from pra_api.dependencies import (
    get_auth_provider,
    get_document_indexer,
    get_document_repository,
    get_knowledge_graph,
    get_rate_limiter,
    get_vector_index_ensure,
    get_vector_store,
)
from pra_api.rate_limits import enforce_upload_rate_limit
from pra_core.auth.base import AuthError, AuthProvider
from pra_core.documents.hashing import compute_content_hash
from pra_core.documents.parser import DocumentParser, UnsupportedDocumentTypeError
from pra_core.documents.repository import DocumentRepository
from pra_core.indexing import DocumentIndexer
from pra_core.knowledge_graph import KnowledgeGraphService
from pra_core.logging import get_logger
from pra_core.rate_limit import FixedWindowRateLimiter
from pra_core.vectorstores.base import VectorStore

router = APIRouter(prefix="/documents", tags=["documents"])
logger = get_logger(__name__)


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    content_type: str
    character_count: int
    chunk_count: int
    indexed: bool
    already_uploaded: bool
    message: str
    principal: str


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    authorization: str | None = Header(default=None),
    repository: DocumentRepository = Depends(get_document_repository),
    indexer: DocumentIndexer = Depends(get_document_indexer),
    auth_provider: AuthProvider = Depends(get_auth_provider),
    rate_limiter: FixedWindowRateLimiter = Depends(get_rate_limiter),
    vector_store: VectorStore = Depends(get_vector_store),
    ensure_vector_index=Depends(get_vector_index_ensure),
    knowledge_graph: KnowledgeGraphService = Depends(get_knowledge_graph),
) -> UploadResponse:
    try:
        principal = auth_provider.authenticate(authorization)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    if not request.app.state.settings.document_upload_enabled:
        raise HTTPException(status_code=403, detail="Document upload is disabled.")

    enforce_upload_rate_limit(
        request=request,
        principal=principal,
        limiter=rate_limiter,
    )

    content = await file.read()
    max_upload_bytes = request.app.state.settings.max_upload_bytes
    if len(content) > max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                "Uploaded file is too large. "
                f"Maximum allowed size is {max_upload_bytes} bytes."
            ),
        )

    parser = DocumentParser()
    try:
        parsed = parser.parse(
            content=content,
            filename=file.filename or "uploaded-document.txt",
            content_type=file.content_type or "application/octet-stream",
        )
    except (UnicodeDecodeError, UnsupportedDocumentTypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    existing_document = repository.find_by_content_hash(
        compute_content_hash(parsed.text)
    )
    if existing_document is not None:
        if not vector_store.contains_document(existing_document.id):
            try:
                ensure_vector_index()
                chunk_count = indexer.index(existing_document)
                _update_knowledge_graph(knowledge_graph, existing_document, indexer)
                return UploadResponse(
                    document_id=existing_document.id,
                    filename=existing_document.filename,
                    content_type=existing_document.content_type,
                    character_count=len(existing_document.text),
                    chunk_count=chunk_count,
                    indexed=True,
                    already_uploaded=True,
                    message=(
                        f"{existing_document.filename} was already uploaded. "
                        "Existing document reused and indexing retried."
                    ),
                    principal=principal.subject,
                )
            except Exception as exc:
                logger.exception(
                    "document_reindexing_failed_for_existing_upload",
                    extra={
                        "document_id": existing_document.id,
                        "document_filename": existing_document.filename,
                        "document_content_type": existing_document.content_type,
                        "error_type": exc.__class__.__name__,
                    },
                )
        _update_knowledge_graph(knowledge_graph, existing_document, indexer)
        return UploadResponse(
            document_id=existing_document.id,
            filename=existing_document.filename,
            content_type=existing_document.content_type,
            character_count=len(existing_document.text),
            chunk_count=0,
            indexed=False,
            already_uploaded=True,
            message=(
                f"{existing_document.filename} was already uploaded. "
                "Existing document reused; indexing skipped."
            ),
            principal=principal.subject,
        )

    document = repository.add(parsed)
    indexed = True
    response_message = "Document indexed."
    try:
        ensure_vector_index()
        chunk_count = indexer.index(document)
        _update_knowledge_graph(knowledge_graph, document, indexer)
    except Exception as exc:
        indexed = False
        chunk_count = 0
        response_message = (
            "Document uploaded, but indexing is temporarily unavailable. "
            "Please try again later."
        )
        logger.exception(
            "document_indexing_failed",
            extra={
                "document_id": document.id,
                "document_filename": document.filename,
                "document_content_type": document.content_type,
                "error_type": exc.__class__.__name__,
            },
        )

    return UploadResponse(
        document_id=document.id,
        filename=document.filename,
        content_type=document.content_type,
        character_count=len(document.text),
        chunk_count=chunk_count,
        indexed=indexed,
        already_uploaded=False,
        message=response_message,
        principal=principal.subject,
    )


def _update_knowledge_graph(
    knowledge_graph: KnowledgeGraphService,
    document,
    indexer: DocumentIndexer,
) -> None:
    if not knowledge_graph.enabled:
        return
    try:
        knowledge_graph.ingest(document, indexer.chunks(document))
    except Exception as exc:
        logger.exception(
            "knowledge_graph_ingestion_failed",
            extra={"document_id": document.id, "error_type": exc.__class__.__name__},
        )
