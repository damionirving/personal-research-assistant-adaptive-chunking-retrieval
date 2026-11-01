from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

from pra_api.error_handlers import register_error_handlers
from pra_api.middleware import ObservabilityMiddleware, RootPathPrefixMiddleware
from fastapi.middleware.cors import CORSMiddleware
from pra_api.routes.auth import router as auth_router
from pra_api.routes.chat import router as chat_router
from pra_api.routes.documents import router as documents_router
from pra_api.routes.health import router as health_router
from pra_api.routes.metrics import router as metrics_router
from pra_api.routes.rag import router as rag_router
from pra_api.routes.ui import router as ui_router
from pra_core.agent.workflow import ResearchAssistantWorkflow
from pra_core.adaptive_rag import AdaptiveDocumentIndexer, AdaptiveRagArtifacts, AdaptiveRetriever
from pra_core.auth.cognito import CognitoAuthProvider
from pra_core.config import Settings, get_settings
from pra_core.documents.chunking import TextChunker
from pra_core.documents.persistent import PersistentDocumentRepository
from pra_core.documents.repository import LocalDocumentRepository
from pra_core.embeddings.bedrock import BedrockEmbeddingProvider
from pra_core.embeddings.local import LocalHashEmbeddingProvider
from pra_core.evaluation.framework import EvaluationRunner, ExpectedTermsEvaluator
from pra_core.evaluation.jobs import (
    MemoryRagEvaluationJobStore,
    ObjectStoreRagEvaluationJobStore,
    RagEvaluationJobCoordinator,
)
from pra_core.evaluation.ragas_jobs import (
    MemoryRagasEvaluationJobStore,
    ObjectStoreRagasEvaluationJobStore,
    RagasEvaluationJobCoordinator,
)
from pra_core.graphstores.local import LocalGraphStore
from pra_core.guardrails.local import GuardrailChain, LocalPromptGuardrail
from pra_core.indexing import DocumentIndexer
from pra_core.knowledge_graph import KnowledgeGraphService
from pra_core.llm.bedrock import BedrockLLMProvider
from pra_core.llm.fallback import RetryableFallbackLLMProvider
from pra_core.llm.mantle import BedrockMantleLLMProvider
from pra_core.llm.mock import MockLLMProvider
from pra_core.logging import configure_logging, get_logger
from pra_core.memory.persistent import PersistentMemoryStore
from pra_core.metrics import MetricsRegistry
from pra_core.memory.store import LocalMemoryStore
from pra_core.persistence.object_store import (
    FileJsonObjectStore,
    JsonObjectStore,
    S3JsonObjectStore,
)
from pra_core.retrieval.hybrid import HybridRetriever
from pra_core.rate_limit import FixedWindowRateLimiter
from pra_core.routing.model import DeterministicModelRouter
from pra_core.tools.registry import ToolDefinition, ToolRegistry
from pra_core.tracing.langsmith import NoOpLangSmithTracer, build_langsmith_tracer
from pra_core.vectorstores.local import LocalVectorStore
from pra_core.vectorstores.opensearch import OpenSearchServerlessVectorStore
from pra_core.vectorstores.faiss import FaissVectorStore
from pra_core.vectorstores.persistent import PersistentVectorStore
from pra_core.vectorstores.pgvector import PgVectorStore

_PACKAGE_ROOT = Path(__file__).resolve().parent
def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the local-first FastAPI application."""
    active_settings = settings or get_settings()
    configure_logging(active_settings)

    app = FastAPI(
        title=active_settings.service_name,
        version=active_settings.service_version,
        debug=active_settings.debug,
        root_path=active_settings.api_root_path,
    )
    app.state.settings = active_settings
    app.state.metrics = MetricsRegistry()

    document_repository, memory_store, vector_store, object_store = _build_persistent_services(
        active_settings
    )
    embedding_provider = _build_embedding_provider(active_settings)
    evaluation_embedding_provider = LocalHashEmbeddingProvider(
        dimensions=active_settings.embedding_dimensions
    )
    chunker = TextChunker(
        chunk_size=active_settings.chunk_size,
        overlap=active_settings.chunk_overlap,
    )
    legacy_document_indexer = DocumentIndexer(
        chunker=chunker,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )
    adaptive_artifacts = AdaptiveRagArtifacts(object_store)
    document_indexer = AdaptiveDocumentIndexer(
        legacy=legacy_document_indexer,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        artifacts=adaptive_artifacts,
        settings=active_settings,
    )
    vector_index_lock = Lock()
    blocked_terms = tuple(
        term.strip()
        for term in active_settings.guardrail_blocked_terms.split(",")
        if term.strip()
    )
    guardrail = GuardrailChain(
        [LocalPromptGuardrail(blocked_terms=blocked_terms)]
    )
    graph_store = LocalGraphStore()
    knowledge_graph = KnowledgeGraphService(
        object_store=object_store,
        graph_store=graph_store,
        enabled=active_settings.knowledge_graph_enabled,
        benefit_threshold=active_settings.kg_benefit_threshold,
        reuse_threshold=active_settings.ontology_reuse_threshold,
        extension_threshold=active_settings.ontology_extension_threshold,
        max_hops=active_settings.graph_max_hops,
        max_results=active_settings.graph_max_results,
        llm_enabled=active_settings.graph_extraction_llm_enabled,
    )
    tool_registry = ToolRegistry()
    tool_registry.register(
        ToolDefinition(
            name="document_count",
            description="Return the number of locally uploaded documents.",
            handler=lambda _payload: {"count": len(document_repository.list())},
        )
    )
    evaluation_runner = EvaluationRunner([ExpectedTermsEvaluator()])
    legacy_retriever = HybridRetriever(
        repository=document_repository,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        bm25_top_k=active_settings.bm25_top_k,
        vector_top_k=active_settings.vector_top_k,
        rrf_k=active_settings.rrf_k,
        fusion_strategy=active_settings.fusion_strategy,
        bm25_fusion_weight=active_settings.bm25_fusion_weight,
        vector_fusion_weight=active_settings.vector_fusion_weight,
        preserve_bm25_candidates=active_settings.preserve_bm25_candidates,
        enable_bm25_lexical=active_settings.enable_bm25_lexical,
        enable_vector_search=active_settings.enable_vector_search,
        enable_reciprocal_rank_fusion=active_settings.enable_reciprocal_rank_fusion,
        enable_reranking=active_settings.enable_reranking,
        rerank_top_k=active_settings.rerank_top_k,
        graph_retriever=(
            knowledge_graph.search if active_settings.graph_retrieval_enabled else None
        ),
    )
    retriever = AdaptiveRetriever(
        legacy=legacy_retriever,
        documents=document_repository,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        artifacts=adaptive_artifacts,
        settings=active_settings,
        knowledge_graph=knowledge_graph,
    )
    llm_provider, llm_providers, model_router = _build_llm_routing(active_settings)
    tracer = build_langsmith_tracer(active_settings)
    ragas_runtime_settings = active_settings.model_copy(
        update={"ragas_evaluation_worker": True}
    )
    ragas_llm_provider, ragas_llm_providers, ragas_model_router = _build_llm_routing(
        ragas_runtime_settings
    )
    evaluation_docs_root = Path(__file__).resolve().parents[4] / "docs"

    def build_ragas_workflow(evaluation_retriever):
        return ResearchAssistantWorkflow(
            retriever=evaluation_retriever,
            llm_provider=ragas_llm_provider,
            llm_providers=ragas_llm_providers,
            model_router=ragas_model_router,
            memory_store=LocalMemoryStore(),
            metrics_registry=None,
            llm_context_top_k=active_settings.llm_context_top_k,
            knowledge_graph_enabled=False,
            simple_answer_max_tokens=active_settings.simple_answer_max_tokens,
            reasoning_answer_max_tokens=active_settings.reasoning_answer_max_tokens,
            tracer=NoOpLangSmithTracer(active_settings.langsmith_project),
        )
    rag_evaluation_job_store = _build_rag_evaluation_job_store(active_settings, object_store)
    rag_evaluation_executor = _build_rag_evaluation_executor(active_settings)
    rag_evaluation_coordinator = RagEvaluationJobCoordinator(
        job_store=rag_evaluation_job_store,
        retriever=retriever,
        chunker=chunker,
        embedding_provider=embedding_provider,
        evaluation_embedding_provider=evaluation_embedding_provider,
        llm_context_top_k=active_settings.llm_context_top_k,
        ensure_vector_index=lambda: None,
        executor=rag_evaluation_executor,
        queue_url=active_settings.rag_evaluation_queue_url,
    )

    app.state.document_repository = document_repository
    app.state.document_indexer = document_indexer
    app.state.embedding_provider = embedding_provider
    app.state.chunker = chunker
    app.state.vector_store = vector_store
    app.state.retriever = retriever
    app.state.object_store = object_store
    app.state.adaptive_artifacts = adaptive_artifacts
    app.state.auth_provider = CognitoAuthProvider(active_settings)
    app.state.guardrail = guardrail
    app.state.graph_store = graph_store
    app.state.knowledge_graph = knowledge_graph
    app.state.tool_registry = tool_registry
    app.state.evaluation_runner = evaluation_runner
    app.state.memory_store = memory_store
    app.state.rate_limiter = FixedWindowRateLimiter()
    app.state.ensure_vector_index = _build_vector_index_ensure_function(
        object_store=object_store,
        embedding_provider_id=embedding_provider.provider_id,
        document_repository=document_repository,
        document_indexer=document_indexer,
        vector_store=vector_store,
        lock=vector_index_lock,
    )
    rag_evaluation_coordinator._ensure_vector_index = app.state.ensure_vector_index
    app.state.rag_evaluation_job_store = rag_evaluation_job_store
    app.state.rag_evaluation_coordinator = rag_evaluation_coordinator
    app.state.workflow = ResearchAssistantWorkflow(
        retriever=retriever,
        llm_provider=llm_provider,
        llm_providers=llm_providers,
        model_router=model_router,
        memory_store=memory_store,
        metrics_registry=app.state.metrics,
        llm_context_top_k=active_settings.llm_context_top_k,
        knowledge_graph_enabled=active_settings.graph_retrieval_enabled,
        simple_answer_max_tokens=active_settings.simple_answer_max_tokens,
        reasoning_answer_max_tokens=active_settings.reasoning_answer_max_tokens,
        tracer=tracer,
    )
    ragas_job_store = _build_ragas_job_store(active_settings, object_store)
    ragas_executor = _build_ragas_executor(active_settings)
    ragas_coordinator = RagasEvaluationJobCoordinator(
        job_store=ragas_job_store,
        workflow=app.state.workflow,
        chunker=chunker,
        evaluation_embedding_provider=evaluation_embedding_provider,
        evaluation_docs_root=evaluation_docs_root,
        evaluation_workflow_factory=build_ragas_workflow,
        retrieval_config_source=retriever,
        executor=ragas_executor,
        queue_url=active_settings.ragas_evaluation_queue_url,
        evaluator_model_id=active_settings.ragas_evaluator_model_id,
        embedding_model_id=active_settings.ragas_embedding_model_id,
        ragas_enabled=active_settings.ragas_enabled,
        allow_inline_continuation=not active_settings.ragas_evaluation_worker,
        ensure_vector_index=app.state.ensure_vector_index,
        batch_size=active_settings.ragas_batch_size,
        evaluator_max_tokens=active_settings.ragas_evaluator_max_tokens,
        metric_timeout_seconds=active_settings.ragas_metric_timeout_seconds,
    )
    app.state.ragas_job_store = ragas_job_store
    app.state.ragas_coordinator = ragas_coordinator

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(active_settings.cors_allowed_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(ObservabilityMiddleware)
    app.add_middleware(
        RootPathPrefixMiddleware,
        root_path=active_settings.api_root_path,
    )
    register_error_handlers(app)

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> FileResponse:
        return FileResponse(
            _PACKAGE_ROOT / "static" / "favicon.ico",
            media_type="image/x-icon",
        )

    app.mount(
        "/static",
        StaticFiles(directory=_PACKAGE_ROOT / "static"),
        name="static",
    )

    app.include_router(health_router)
    app.include_router(auth_router)
    app.include_router(chat_router)
    app.include_router(documents_router)
    app.include_router(metrics_router)
    app.include_router(rag_router)
    app.include_router(ui_router)

    logger = get_logger(__name__)
    logger.info(
        "application_created",
        extra={
            "service_name": active_settings.service_name,
            "environment": active_settings.environment,
        },
    )
    return app


def _build_persistent_services(settings: Settings):
    object_store: JsonObjectStore | None = None
    if settings.persistence_backend == "file":
        object_store = FileJsonObjectStore(settings.persistence_data_dir)
        return (
            PersistentDocumentRepository(object_store),
            PersistentMemoryStore(object_store),
            _build_vector_store(settings, object_store),
            object_store,
        )

    if settings.persistence_backend == "s3":
        if not settings.document_bucket_name:
            raise ValueError("PRA_DOCUMENT_BUCKET_NAME is required for S3 persistence.")
        object_store = S3JsonObjectStore(
            bucket_name=settings.document_bucket_name,
            prefix=settings.persistence_s3_prefix,
        )
        return (
            PersistentDocumentRepository(object_store),
            PersistentMemoryStore(object_store),
            _build_vector_store(settings, object_store),
            object_store,
        )

    return (
        LocalDocumentRepository(),
        LocalMemoryStore(),
        _build_vector_store(settings, object_store),
        object_store,
    )


def _build_embedding_provider(settings: Settings):
    if settings.embedding_provider == "bedrock":
        return BedrockEmbeddingProvider(
            model_id=settings.bedrock_embedding_model_id,
            region_name=settings.aws_region,
            dimensions=settings.bedrock_embedding_dimensions,
        )
    return LocalHashEmbeddingProvider(dimensions=settings.embedding_dimensions)


def _build_vector_store(settings: Settings, object_store: JsonObjectStore | None):
    if settings.vector_store_backend == "local" and object_store is not None:
        return PersistentVectorStore(object_store)

    if settings.vector_store_backend == "persistent":
        if object_store is None:
            raise ValueError("persistent vector store requires persistent object storage.")
        return PersistentVectorStore(object_store)

    if settings.vector_store_backend == "faiss":
        if object_store is None:
            raise ValueError("faiss vector store requires persistent object storage.")
        return FaissVectorStore(object_store)

    if settings.vector_store_backend == "pgvector":
        if not settings.pgvector_dsn:
            raise ValueError("PRA_PGVECTOR_DSN is required for pgvector storage.")
        return PgVectorStore(
            dsn=settings.pgvector_dsn,
            table_name=settings.pgvector_table_name,
            dimensions=_active_embedding_dimensions(settings),
        )

    if settings.vector_store_backend == "opensearch":
        if not settings.opensearch_endpoint:
            raise ValueError("PRA_OPENSEARCH_ENDPOINT is required for OpenSearch vectors.")
        return OpenSearchServerlessVectorStore(
            endpoint=settings.opensearch_endpoint,
            index_name=settings.opensearch_index_name,
            dimensions=_active_embedding_dimensions(settings),
            region_name=settings.aws_region,
        )

    return LocalVectorStore()


def _active_embedding_dimensions(settings: Settings) -> int:
    if settings.embedding_provider == "bedrock":
        return settings.bedrock_embedding_dimensions
    return settings.embedding_dimensions


def _ensure_vector_index_matches_embedding_provider(
    object_store: JsonObjectStore,
    embedding_provider_id: str,
    document_repository,
    document_indexer: DocumentIndexer,
    vector_store,
) -> None:
    key = "vector_index_metadata.json"
    metadata = object_store.read_json(key, {})
    if metadata.get("embedding_provider_id") == embedding_provider_id:
        return

    vector_store.clear()
    documents = document_repository.list()
    for document in documents:
        document_indexer.index(document)

    object_store.write_json(
        key,
        {
            "embedding_provider_id": embedding_provider_id,
            "indexed_document_count": len(documents),
            "indexed_chunk_count": vector_store.count(),
        },
    )


def _build_vector_index_ensure_function(
    object_store: JsonObjectStore | None,
    embedding_provider_id: str,
    document_repository,
    document_indexer: DocumentIndexer,
    vector_store,
    lock: Lock,
):
    if object_store is None:
        return lambda: None

    def ensure_vector_index() -> None:
        with lock:
            _ensure_vector_index_matches_embedding_provider(
                object_store=object_store,
                embedding_provider_id=embedding_provider_id,
                document_repository=document_repository,
                document_indexer=document_indexer,
                vector_store=vector_store,
            )

    return ensure_vector_index


def _build_llm_routing(settings: Settings):
    if settings.llm_provider == "bedrock":
        if settings.ragas_evaluation_worker and not settings.ragas_use_normal_model_routing:
            model_id = settings.ragas_model_only_id
            provider = BedrockLLMProvider(
                model_id=model_id,
                region_name=settings.aws_region,
                max_tokens=settings.bedrock_max_tokens,
                temperature=settings.bedrock_temperature,
            )
            return (
                provider,
                {model_id: provider},
                DeterministicModelRouter(
                    simple_model_id=model_id,
                    reasoning_model_id=model_id,
                ),
            )
        simple_provider = BedrockLLMProvider(
            model_id=settings.bedrock_simple_model_id,
            region_name=settings.aws_region,
            max_tokens=settings.bedrock_max_tokens,
            temperature=settings.bedrock_temperature,
            fallback_model_id=settings.bedrock_simple_fallback_model_id,
        )
        mantle_provider = BedrockMantleLLMProvider(
            model_id=settings.bedrock_reasoning_model_id,
            region_name=settings.aws_region,
            base_url=settings.bedrock_mantle_base_url,
            max_tokens=settings.bedrock_max_tokens,
            temperature=settings.bedrock_temperature,
        )
        fallback_provider = BedrockLLMProvider(
            model_id=settings.bedrock_reasoning_fallback_model_id,
            region_name=settings.aws_region,
            max_tokens=settings.bedrock_max_tokens,
            temperature=settings.bedrock_temperature,
        )
        reasoning_provider = RetryableFallbackLLMProvider(
            primary=mantle_provider,
            fallback=fallback_provider,
        )
        providers = {
            settings.bedrock_simple_model_id: simple_provider,
            settings.bedrock_simple_fallback_model_id: simple_provider,
            settings.bedrock_reasoning_model_id: reasoning_provider,
            settings.bedrock_reasoning_fallback_model_id: reasoning_provider,
        }
        return (
            reasoning_provider,
            providers,
            DeterministicModelRouter(
                simple_model_id=settings.bedrock_simple_model_id,
                reasoning_model_id=settings.bedrock_reasoning_model_id,
            ),
        )

    mock_provider = MockLLMProvider()
    return (
        mock_provider,
        {"mock": mock_provider},
        DeterministicModelRouter(
            simple_model_id="mock",
            reasoning_model_id="mock",
        ),
    )


def _build_rag_evaluation_job_store(
    settings: Settings, object_store: JsonObjectStore | None
):
    if object_store is not None:
        return ObjectStoreRagEvaluationJobStore(object_store)
    return MemoryRagEvaluationJobStore()


def _build_rag_evaluation_executor(settings: Settings):
    if (
        settings.environment == "test"
        or settings.rag_evaluation_queue_url
        or settings.rag_evaluation_worker
    ):
        return None
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag-eval")


def _build_ragas_job_store(settings: Settings, object_store: JsonObjectStore | None):
    if object_store is not None:
        return ObjectStoreRagasEvaluationJobStore(object_store)
    return MemoryRagasEvaluationJobStore()


def _build_ragas_executor(settings: Settings):
    if (
        settings.environment == "test"
        or settings.ragas_evaluation_queue_url
        or settings.ragas_evaluation_worker
    ):
        return None
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix="ragas-eval")


app = create_app()
