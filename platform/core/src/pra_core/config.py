from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env files."""

    service_name: str = Field(default="personal-research-assistant")
    service_version: str = Field(default="0.1.0")
    environment: Literal["local", "test", "dev", "staging", "prod"] = "local"
    debug: bool = False
    retrieval_diagnostics_enabled: bool = True
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: Literal["json", "console"] = "json"
    api_root_path: str = ""
    allowed_cors_origins: str = (
        "http://localhost:5000,"
        "http://127.0.0.1:5000"
    )
    auth_provider: Literal["cognito"] = "cognito"
    aws_region: str = "us-east-1"
    cognito_user_pool_id: str | None = None
    cognito_client_id: str | None = None
    cognito_issuer: str | None = None
    cognito_hosted_ui_base_url: str | None = None
    document_bucket_name: str | None = None
    persistence_backend: Literal["memory", "file", "s3"] = "memory"
    persistence_data_dir: str = ".pra_data"
    persistence_s3_prefix: str = "pra-prod"
    vector_store_backend: Literal["local", "persistent", "faiss", "opensearch", "pgvector"] = (
        "local"
    )
    opensearch_endpoint: str | None = None
    opensearch_index_name: str = "pra-chunks"
    pgvector_dsn: str | None = None
    pgvector_table_name: str = "pra_chunks"
    cloudwatch_log_group: str | None = None
    cloudwatch_log_stream: str | None = None
    langsmith_project: str = Field(
        default="personal-research-assistant",
        validation_alias="LANGSMITH_PROJECT",
    )
    max_upload_bytes: int = 9_500_000
    document_upload_enabled: bool = True
    llm_provider: Literal["mock", "bedrock"] = "mock"
    bedrock_model_id: str = "us.anthropic.claude-opus-4-1-20250805-v1:0"
    bedrock_simple_model_id: str = "google.gemma-3-12b-it"
    bedrock_simple_fallback_model_id: str = "amazon.nova-lite-v1:0"
    bedrock_reasoning_model_id: str = "google.gemma-4-31b"
    bedrock_reasoning_fallback_model_id: str = "mistral.ministral-3-14b-instruct"
    bedrock_mantle_base_url: str | None = None
# ----------------------------------------------------------------------------
    bedrock_max_tokens: int = 700
    simple_answer_max_tokens: int = 150
    reasoning_answer_max_tokens: int = 400
# ----------------------------------------------------------------------------
    bedrock_temperature: float = 0.2
    embedding_provider: Literal["local", "bedrock"] = "local"
    bedrock_embedding_model_id: str = "amazon.titan-embed-text-v2:0"
# ----------------------------------------------------------------------------
    bedrock_embedding_dimensions: int = 1024
# ----------------------------------------------------------------------------
# Retreival values used locally and sent to AWS
# ----------------------------------------------------------------------------
    bm25_top_k: int = 50
    vector_top_k: int = 50
    rrf_k: int = 30
    fusion_strategy: Literal["rrf", "weighted_rrf", "normalized_weighted"] = "weighted_rrf"
    bm25_fusion_weight: float = 0.8
    vector_fusion_weight: float = 0.2
    preserve_bm25_candidates: bool = True
    enable_bm25_lexical: bool = True
    enable_vector_search: bool = True
    enable_reciprocal_rank_fusion: bool = True
    enable_reranking: bool = False
    rerank_top_k: int = 50
    llm_context_top_k: int = 15
    bm25_first_enabled: bool = True
    chunk_size: int = 600
    chunk_overlap: int = 100
    # Adaptive RAG is opt-in so deployments retain the MVP behavior until enabled.
    adaptive_chunking_enabled: bool = False
    adaptive_retrieval_enabled: bool = False
    table_processing_enabled: bool = True
    domain_detection_enabled: bool = True
    document_type_detection_enabled: bool = True
    relationship_extraction_enabled: bool = True
    hierarchical_chunking_enabled: bool = True
    relationship_retrieval_enabled: bool = True
    parent_child_expansion_enabled: bool = True
    query_expansion_enabled: bool = True
    adaptive_chunking_profile: Literal["auto", "structure", "semantic", "fixed"] = "auto"
    adaptive_retrieval_profile: Literal["auto", "lexical", "semantic", "relationship"] = "auto"
    chunk_min_tokens: int = 80
    chunk_target_tokens: int = 300
    chunk_max_tokens: int = 600
    chunk_overlap_tokens: int = 60
    retrieval_vector_top_k: int = 30
    retrieval_bm25_top_k: int = 30
    retrieval_fused_top_k: int = 30
    retrieval_final_top_k: int = 10
    retrieval_vector_weight: float = 0.40
    retrieval_bm25_weight: float = 0.60
    retrieval_relationship_weight: float = 0.00
    relationship_max_depth: int = 1
    relationship_max_neighbors: int = 4
    relationship_min_confidence: float = 0.6
    knowledge_graph_enabled: bool = True
    kg_benefit_threshold: float = 60.0
    ontology_reuse_threshold: float = 0.90
    ontology_extension_threshold: float = 0.70
    graph_retrieval_enabled: bool = True
    graph_max_hops: int = 2
    graph_max_results: int = 20
    graph_extraction_llm_enabled: bool = True
    adaptive_context_token_budget: int = 2400
    adaptive_simple_context_top_k: int = 5
    adaptive_policy_context_top_k: int = 8
    adaptive_complex_context_top_k: int = 15
    adaptive_low_confidence_context_top_k: int = 15
    adaptive_low_confidence_score_gap: float = 0.15
    adaptive_min_strong_matches: int = 2
    reranker_mode: Literal["deterministic", "bedrock"] = "deterministic"
    reranker_model_id: str | None = None
    chunk_analysis_model_id: str | None = None
    query_planner_model_id: str | None = None
# ----------------------------------------------------------------------------
    embedding_dimensions: int = 64 #for local testing not bedrock
# ----------------------------------------------------------------------------
    guardrail_blocked_terms: str = "production-agentcore"
    rate_limit_chat_requests: int = 30
    rate_limit_chat_window_seconds: int = 60
    rate_limit_upload_requests: int = 5
    rate_limit_upload_window_seconds: int = 3600
    rate_limit_public_requests: int = 60
    rate_limit_public_window_seconds: int = 60
    rate_limit_ragas_requests: int = 2
    rate_limit_ragas_window_seconds: int = 3600
    rag_evaluation_queue_url: str | None = None
    rag_evaluation_worker: bool = False
    rag_evaluation_enabled: bool = True
    rag_eval_dataset_path: str = "data/rag_eval_dataset_v2.json"
    ragas_enabled: bool = True
    ragas_evaluation_queue_url: str | None = None
    ragas_evaluation_worker: bool = False
    ragas_eval_dataset_path: str = "data/rag_eval_dataset_v2.json"
    ragas_evaluator_model_id: str | None = None
    ragas_embedding_model_id: str | None = None
    ragas_max_questions: int = 50
    ragas_standard_max_questions: int = 12
    ragas_standard_top_k: int = 15
    ragas_batch_size: int = 1
    ragas_model_only_id: str = "google.gemma-3-12b-it"
    ragas_use_normal_model_routing: bool = False
    ragas_evaluator_max_tokens: int = 768
    ragas_metric_timeout_seconds: int = 120

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="PRA_",
        extra="ignore",
    )

    @model_validator(mode="after")
    def enforce_retrieval_switch_dependencies(self) -> "Settings":
        self.allowed_cors_origins = ",".join(
            dict.fromkeys(
                origin.strip()
                for origin in self.allowed_cors_origins.split(",")
                if origin.strip()
            )
        )
        self.enable_reciprocal_rank_fusion = (
            self.enable_bm25_lexical and self.enable_vector_search
        )
        total_weight = self.bm25_fusion_weight + self.vector_fusion_weight
        if total_weight > 0:
            self.bm25_fusion_weight = self.bm25_fusion_weight / total_weight
            self.vector_fusion_weight = self.vector_fusion_weight / total_weight
        adaptive_total_weight = (
            self.retrieval_bm25_weight
            + self.retrieval_vector_weight
            + self.retrieval_relationship_weight
        )
        if adaptive_total_weight > 0:
            self.retrieval_bm25_weight /= adaptive_total_weight
            self.retrieval_vector_weight /= adaptive_total_weight
            self.retrieval_relationship_weight /= adaptive_total_weight
        if not (
            self.chunk_min_tokens <= self.chunk_target_tokens <= self.chunk_max_tokens
        ):
            raise ValueError(
                "chunk_min_tokens must be less than or equal to chunk_target_tokens, "
                "which must be less than or equal to chunk_max_tokens."
            )
        if self.chunk_overlap_tokens >= self.chunk_target_tokens:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_target_tokens.")
        if self.retrieval_final_top_k > self.retrieval_fused_top_k:
            raise ValueError("retrieval_final_top_k cannot exceed retrieval_fused_top_k.")
        if self.relationship_max_depth < 0 or self.relationship_max_neighbors < 0:
            raise ValueError("relationship traversal limits must be non-negative.")
        if self.graph_max_hops < 0 or self.graph_max_results < 0:
            raise ValueError("graph traversal limits must be non-negative.")
        if not 0 <= self.ontology_extension_threshold <= self.ontology_reuse_threshold <= 1:
            raise ValueError(
                "ontology thresholds must satisfy 0 <= extension <= reuse <= 1."
            )
        return self

    @property
    def cors_allowed_origins(self) -> tuple[str, ...]:
        return tuple(
            origin.strip()
            for origin in self.allowed_cors_origins.split(",")
            if origin.strip()
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
