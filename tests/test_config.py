import pytest
from pydantic import ValidationError

from pra_core.config import Settings


def test_settings_accept_phase_zero_defaults() -> None:
    settings = Settings()

    assert settings.service_name == "personal-research-assistant"
    assert settings.environment == "local"
    assert settings.log_format == "json"
    assert settings.api_root_path == ""
    assert settings.allowed_cors_origins == (
        "http://localhost:5000,"
        "http://127.0.0.1:5000,"
        "https://<your-domain.com>,"
        "https://www.<your-domain.com>,"
        "https://tractlyticswebsite.web.app,"
        "https://tractlyticswebsite.firebaseapp.com"
    )
    assert settings.cors_allowed_origins == (
        "http://localhost:5000",
        "http://127.0.0.1:5000",
        "https://<your-domain.com>",
        "https://www.<your-domain.com>",
        "https://tractlyticswebsite.web.app",
        "https://tractlyticswebsite.firebaseapp.com",
    )
    assert settings.auth_provider == "cognito"
    assert settings.aws_region == "us-east-1"
    assert settings.cognito_user_pool_id is None
    assert settings.cognito_client_id is None
    assert settings.cognito_issuer is None
    assert settings.cognito_hosted_ui_base_url is None
    assert settings.document_bucket_name is None
    assert settings.persistence_backend == "memory"
    assert settings.vector_store_backend == "local"
    assert settings.opensearch_endpoint is None
    assert settings.opensearch_index_name == "pra-chunks"
    assert settings.pgvector_dsn is None
    assert settings.pgvector_table_name == "pra_chunks"
    assert settings.rag_eval_dataset_path == "data/rag_eval_dataset_v2.json"
    assert settings.langsmith_project == "personal-research-assistant"
    assert settings.llm_provider == "mock"
    assert settings.bedrock_model_id == "us.anthropic.claude-opus-4-1-20250805-v1:0"
    assert settings.bedrock_simple_model_id == "google.gemma-3-12b-it"
    assert settings.bedrock_simple_fallback_model_id == "amazon.nova-lite-v1:0"
    assert settings.bedrock_reasoning_model_id == "google.gemma-4-31b"
    assert settings.bedrock_reasoning_fallback_model_id == "mistral.ministral-3-14b-instruct"
    assert settings.bedrock_max_tokens == 700
    assert settings.simple_answer_max_tokens == 150
    assert settings.reasoning_answer_max_tokens == 400
    assert settings.bedrock_temperature == 0.2
    assert settings.embedding_provider == "local"
    assert settings.bedrock_embedding_model_id == "amazon.titan-embed-text-v2:0"
    assert settings.bedrock_embedding_dimensions == 1024
    assert settings.bm25_top_k == 50
    assert settings.vector_top_k == 50
    assert settings.rrf_k == 30
    assert settings.fusion_strategy == "weighted_rrf"
    assert settings.bm25_fusion_weight == 0.8
    assert settings.vector_fusion_weight == 0.2
    assert settings.preserve_bm25_candidates is True
    assert settings.enable_bm25_lexical is True
    assert settings.enable_vector_search is True
    assert settings.enable_reciprocal_rank_fusion is True
    assert settings.enable_reranking is False
    assert settings.retrieval_bm25_weight == 0.6
    assert settings.retrieval_vector_weight == 0.4
    assert settings.retrieval_relationship_weight == 0.0
    assert settings.document_upload_enabled is True
    assert settings.retrieval_diagnostics_enabled is True
    assert settings.rag_evaluation_enabled is True
    assert settings.ragas_enabled is True
    assert settings.ragas_batch_size == 1
    assert settings.ragas_standard_max_questions == 12
    assert settings.ragas_standard_top_k == 15
    assert settings.ragas_evaluator_max_tokens == 768
    assert settings.rate_limit_ragas_requests == 2
    assert settings.ragas_model_only_id == "google.gemma-3-12b-it"
    assert settings.ragas_use_normal_model_routing is False
    assert settings.bm25_first_enabled is True
    assert settings.chunk_size == 600
    assert settings.chunk_overlap == 100
    assert settings.llm_context_top_k == 15
    assert settings.embedding_dimensions == 64
    assert settings.guardrail_blocked_terms == "production-agentcore"


def test_settings_reject_unknown_environment() -> None:
    with pytest.raises(ValidationError):
        Settings(environment="sandbox")


def test_settings_disable_rrf_when_a_retrieval_leg_is_disabled() -> None:
    settings = Settings(enable_vector_search=False)

    assert settings.enable_bm25_lexical is True
    assert settings.enable_vector_search is False
    assert settings.enable_reciprocal_rank_fusion is False


def test_settings_normalize_fusion_weights() -> None:
    settings = Settings(bm25_fusion_weight=4.0, vector_fusion_weight=1.0)

    assert settings.bm25_fusion_weight == 0.8
    assert settings.vector_fusion_weight == 0.2


def test_settings_normalize_adaptive_retrieval_weights() -> None:
    settings = Settings(
        retrieval_bm25_weight=6.0,
        retrieval_vector_weight=3.0,
        retrieval_relationship_weight=1.0,
    )

    assert settings.retrieval_bm25_weight == 0.6
    assert settings.retrieval_vector_weight == 0.3
    assert settings.retrieval_relationship_weight == 0.1


def test_settings_normalize_cors_origin_list() -> None:
    settings = Settings(
        allowed_cors_origins=(
            " https://<your-domain.com>,https://<your-domain.com>,"
            "http://localhost:5000 "
        )
    )

    assert settings.cors_allowed_origins == (
        "https://<your-domain.com>",
        "http://localhost:5000",
    )
