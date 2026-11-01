from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_phase_two_cdk_project_files_exist() -> None:
    expected_files = [
        "infra/app.py",
        "infra/cdk.json",
        "infra/requirements.txt",
        "infra/pra_infra/stack.py",
        "requirements-lambda.txt",
        "docs/github-oidc.md",
    ]

    for filename in expected_files:
        assert (ROOT / filename).is_file()


def test_ragas_dependencies_are_isolated_to_container_worker() -> None:
    lambda_requirements = (ROOT / "requirements-lambda.txt").read_text(encoding="utf-8")
    ragas_requirements = (ROOT / "requirements-ragas-worker.txt").read_text(encoding="utf-8")
    dockerfile = (ROOT / "Dockerfile.ragas-worker").read_text(encoding="utf-8")

    assert "ragas" not in lambda_requirements.lower()
    assert "-r requirements-lambda.txt" in ragas_requirements
    assert "ragas" in ragas_requirements.lower()
    assert "langchain-community==0.4.1" in ragas_requirements
    assert "requirements-ragas-worker.txt" in dockerfile
    assert "import datasets, ragas" in dockerfile


def test_cdk_stack_declares_required_phase_two_resources() -> None:
    stack_source = (ROOT / "infra/pra_infra/stack.py").read_text(encoding="utf-8")

    assert "s3.Bucket" in stack_source
    assert "BucketDeployment" in stack_source
    assert 'destination_key_prefix="pra-prod/testing/rag-eval"' in stack_source
    assert 'exclude=["rag_eval_dataset_v1.json", "research-assistant/**"]' in stack_source
    assert "lambda_.Function" in stack_source
    assert "apigateway.LambdaRestApi" in stack_source
    assert "cognito.UserPool" in stack_source
    assert "cognito.UserPool.from_user_pool_id" in stack_source
    assert '"cognito_user_pool_id"' in stack_source
    assert '"us-east-1_IxEu5C7oJ"' in stack_source
    assert '"cognito_domain_prefix"' in stack_source
    assert "PRA_AUTH_PROVIDER" in stack_source
    assert "PRA_COGNITO_CLIENT_ID" in stack_source
    assert "hosted_ui_callback_url" in stack_source
    assert "CfnUserPoolUICustomizationAttachment" in stack_source
    assert "iam.Role" in stack_source
    assert "PRA_DOCUMENT_BUCKET_NAME" in stack_source
    assert "PRA_LLM_PROVIDER" in stack_source
    assert "PRA_BEDROCK_MODEL_ID" in stack_source
    assert "PRA_BEDROCK_SIMPLE_MODEL_ID" in stack_source
    assert "PRA_BEDROCK_SIMPLE_FALLBACK_MODEL_ID" in stack_source
    assert "PRA_BEDROCK_REASONING_MODEL_ID" in stack_source
    assert '"PRA_SIMPLE_ANSWER_MAX_TOKENS": "150"' in stack_source
    assert '"PRA_REASONING_ANSWER_MAX_TOKENS": "400"' in stack_source
    assert "PRA_BEDROCK_REASONING_FALLBACK_MODEL_ID" in stack_source
    assert "bedrock-mantle:CreateInference" in stack_source
    assert "bedrock-mantle:CallWithBearerToken" in stack_source
    assert "PRA_BEDROCK_MANTLE_BASE_URL" in stack_source
    assert "PRA_EMBEDDING_PROVIDER" in stack_source
    assert "secretsmanager.Secret" in stack_source
    assert "LANGSMITH_PROJECT" in stack_source
    assert "LangSmithApiKeySecret" in stack_source
    assert 'secret_name="LANGSMITH_API_KEY"' in stack_source
    assert "PRA_VECTOR_STORE_BACKEND" in stack_source
    assert '"PRA_VECTOR_STORE_BACKEND": "faiss"' in stack_source
    assert "RagasEvaluationQueue" in stack_source
    assert "ragas-evaluation-worker" in stack_source
    assert "RagasEvaluationStarterFunction" in stack_source
    assert "ragas-evaluation-starter" in stack_source
    assert "RagasEvaluationStateMachine" in stack_source
    assert "InvokeRagasWorkerStep" in stack_source
    assert "RagasStatusCheckpointFunction" in stack_source
    assert "CheckpointRagasOrchestration" in stack_source
    assert "CheckpointRagasCapacityWait" in stack_source
    assert "WaitForRagasWorkerCapacity" in stack_source
    assert "RagasWorkerCapacityRetryLimit" in stack_source
    assert "RagasEvaluationNeedsAnotherStep" in stack_source
    assert "RagasEvaluationStarterRole" in stack_source
    assert "role=ragas_evaluation_starter_role" in stack_source
    assert "ragas_evaluation_queue.grant_consume_messages(" in stack_source
    assert "ragas_evaluation_state_machine.grant_start_execution(" in stack_source
    assert "ragas_evaluation_state_machine.grant_start_execution(execution_role)" not in stack_source
    assert "ragas_evaluation_starter.add_event_source" in stack_source
    assert "ragas_evaluation_worker.add_event_source" not in stack_source
    assert "cp -r apps/api/src/pra_api " in stack_source
    assert "platform/core/src/pra_core /asset-output" in stack_source
    assert "DockerImageFunction" in stack_source
    assert "DockerImageCode.from_image_asset" in stack_source
    assert (ROOT / "Dockerfile.ragas-worker").is_file()
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    assert "infra/cdk.out" in dockerignore
    assert "PRA_RAGAS_ENABLED" in stack_source
    assert stack_source.count('"PRA_RAGAS_ENABLED": ragas_enabled') == 3
    assert stack_source.count('"PRA_RAG_EVALUATION_ENABLED": rag_evaluation_enabled') == 2
    assert '_context_bool(self, "rag_evaluation_enabled", True)' in stack_source
    assert 'return normalized if normalized in {"true", "false"} else str(default).lower()' in stack_source
    assert "document_upload_enabled" in stack_source
    assert '"PRA_DOCUMENT_UPLOAD_ENABLED": document_upload_enabled' in stack_source
    assert "retrieval_diagnostics_enabled" in stack_source
    assert '"PRA_RETRIEVAL_DIAGNOSTICS_ENABLED": retrieval_diagnostics_enabled' in stack_source
    assert "PRA_RAGAS_EVALUATOR_MODEL_ID" in stack_source
    assert "PRA_RAGAS_BATCH_SIZE" in stack_source
    assert "PRA_RAGAS_MODEL_ONLY_ID" in stack_source
    assert "RagasEvaluationDeadLetterQueue" in stack_source
    assert "reserved_concurrent_executions=1" in stack_source
    assert 'errors=["Lambda.TooManyRequestsException"]' in stack_source
    assert "capacity_wait_count" in stack_source
    assert "Duration.minutes(2)" in stack_source
    assert 'memory_size=1024' in stack_source
    assert 'timeout=Duration.minutes(14)' in stack_source
    assert '"PRA_RAGAS_BATCH_SIZE": "1"' in stack_source
    assert '"PRA_RAGAS_EVALUATION_QUEUE_URL": ragas_evaluation_queue.queue_url' in stack_source
    assert 'visibility_timeout=Duration.minutes(30)' in stack_source
    assert 'max_receive_count=5' in stack_source
    assert "RagasDeadLetterQueueAlarm" in stack_source
    assert '"PRA_RAGAS_STANDARD_MAX_QUESTIONS": "12"' in stack_source
    assert '"PRA_RAGAS_STANDARD_TOP_K": "15"' in stack_source
    assert '"PRA_RAGAS_EVALUATOR_MAX_TOKENS": "768"' in stack_source
    assert '"PRA_BEDROCK_REASONING_MODEL_ID": bedrock_reasoning_model_id' in stack_source
    assert '"PRA_RAGAS_USE_NORMAL_MODEL_ROUTING": ragas_use_normal_model_routing' in stack_source
    assert '"PRA_BEDROCK_REASONING_FALLBACK_MODEL_ID": bedrock_reasoning_fallback_model_id' in stack_source
    assert "CostAlertTopic" in stack_source
    assert "CostDashboard" in stack_source
    assert "owner_email" in stack_source
    assert "aws_owner_email" in stack_source
    assert "ApiLambdaErrorAlarm" in stack_source
    assert "ApiLambdaDurationAlarm" in stack_source
    assert "ApiGateway5xxAlarm" in stack_source
    assert "CfnCollection" not in stack_source
    assert "VECTORSEARCH" not in stack_source
    assert "bedrock:InvokeModel" in stack_source
    assert "aoss:APIAccessAll" not in stack_source
    assert "authorization_code_grant=True" in stack_source
    assert "access_token_validity=Duration.hours(1)" in stack_source
    assert "refresh_token_validity=Duration.days(30)" in stack_source
    assert "http://localhost:5000/research-assistant/" in stack_source
    assert "http://127.0.0.1:5000/research-assistant/" in stack_source
    assert "https://<your-domain.com>/research-assistant/" in stack_source
    assert "https://www.<your-domain.com>/research-assistant/" in stack_source
    assert "https://tractlyticswebsite.web.app/research-assistant/" in stack_source
    assert "https://tractlyticswebsite.firebaseapp.com/research-assistant/" in stack_source
    assert "DeploymentAutomation" in stack_source


def test_cdk_stack_does_not_include_excluded_services() -> None:
    stack_source = (ROOT / "infra/pra_infra/stack.py").read_text(encoding="utf-8")

    excluded_terms = ["agentcore"]
    assert all(term not in stack_source.lower() for term in excluded_terms)


def test_lambda_handler_strips_api_gateway_base_path() -> None:
    handler_source = (
        ROOT / "apps/api/src/pra_api/lambda_handler.py"
    ).read_text(encoding="utf-8")

    assert "api_gateway_base_path=settings.api_root_path" in handler_source
