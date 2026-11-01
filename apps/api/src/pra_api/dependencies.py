from fastapi import Request

from pra_core.agent.workflow import ResearchAssistantWorkflow
from pra_core.auth.base import AuthProvider
from pra_core.documents.repository import DocumentRepository
from pra_core.guardrails.base import Guardrail
from pra_core.indexing import DocumentIndexer
from pra_core.knowledge_graph import KnowledgeGraphService
from pra_core.memory.store import MemoryStore
from pra_core.rate_limit import FixedWindowRateLimiter
from pra_core.retrieval.base import Retriever
from pra_core.vectorstores.base import VectorStore


def get_document_repository(request: Request) -> DocumentRepository:
    return request.app.state.document_repository


def get_document_indexer(request: Request) -> DocumentIndexer:
    return request.app.state.document_indexer


def get_knowledge_graph(request: Request) -> KnowledgeGraphService:
    return request.app.state.knowledge_graph


def get_auth_provider(request: Request) -> AuthProvider:
    return request.app.state.auth_provider


def get_guardrail(request: Request) -> Guardrail:
    return request.app.state.guardrail


def get_memory_store(request: Request) -> MemoryStore:
    return request.app.state.memory_store


def get_workflow(request: Request) -> ResearchAssistantWorkflow:
    return request.app.state.workflow


def get_retriever(request: Request) -> Retriever:
    return request.app.state.retriever


def get_vector_store(request: Request) -> VectorStore:
    return request.app.state.vector_store


def get_vector_index_ensure(request: Request):
    return request.app.state.ensure_vector_index


def get_rate_limiter(request: Request) -> FixedWindowRateLimiter:
    return request.app.state.rate_limiter
