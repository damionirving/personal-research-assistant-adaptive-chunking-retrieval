from pathlib import Path

from aws_cdk import (
    BundlingOptions,
    CfnOutput,
    Duration,
    RemovalPolicy,
    Stack,
)
from aws_cdk import aws_apigateway as apigateway
from aws_cdk import aws_cloudwatch as cloudwatch
from aws_cdk import aws_cloudwatch_actions as cloudwatch_actions
from aws_cdk import aws_cognito as cognito
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_lambda_event_sources as lambda_event_sources
from aws_cdk import aws_logs as logs
from aws_cdk import aws_sns as sns
from aws_cdk import aws_sns_subscriptions as sns_subscriptions
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_s3_deployment as s3_deployment
from aws_cdk import aws_secretsmanager as secretsmanager
from aws_cdk import aws_sqs as sqs
from aws_cdk import aws_stepfunctions as sfn
from aws_cdk import aws_stepfunctions_tasks as sfn_tasks
from constructs import Construct


def _context_bool(scope: Construct, key: str, default: bool) -> str:
    """Return a Lambda-safe boolean string for a CDK context value."""
    value = scope.node.try_get_context(key)
    if isinstance(value, bool):
        return str(value).lower()
    normalized = str(value).strip().lower() if value is not None else ""
    return normalized if normalized in {"true", "false"} else str(default).lower()


class PersonalResearchAssistantStack(Stack):
    """Phase 2 AWS stack for the local-first research assistant."""

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        project_root = Path(__file__).resolve().parents[2]

        document_bucket = s3.Bucket(
            self,
            "DocumentBucket",
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            versioned=True,
            removal_policy=RemovalPolicy.RETAIN,
        )
        s3_deployment.BucketDeployment(
            self,
            "RagEvaluationDatasetDeployment",
            sources=[
                s3_deployment.Source.asset(
                    str(project_root / "data"),
                    exclude=["rag_eval_dataset_v1.json", "research-assistant/**"],
                )
            ],
            destination_bucket=document_bucket,
            destination_key_prefix="pra-prod/testing/rag-eval",
            prune=False,
        )
        cognito_user_pool_id = (
            self.node.try_get_context("cognito_user_pool_id")
            or "us-east-1_IxEu5C7oJ"
        )
        if cognito_user_pool_id:
            user_pool = cognito.UserPool.from_user_pool_id(
                self,
                "UserPool",
                cognito_user_pool_id,
            )
        else:
            user_pool = cognito.UserPool(
                self,
                "UserPool",
                self_sign_up_enabled=True,
                sign_in_aliases=cognito.SignInAliases(email=True),
                standard_attributes=cognito.StandardAttributes(
                    email=cognito.StandardAttribute(required=True, mutable=True),
                ),
                account_recovery=cognito.AccountRecovery.EMAIL_ONLY,
                removal_policy=RemovalPolicy.RETAIN,
            )
        cognito_domain_prefix = (
            self.node.try_get_context("cognito_domain_prefix")
            or f"pra-{Stack.of(self).account}-{Stack.of(self).region}-retained"
        )
        user_pool_domain = user_pool.add_domain(
            "UserPoolDomain",
            cognito_domain=cognito.CognitoDomainOptions(
                domain_prefix=cognito_domain_prefix,
            ),
        )
        hosted_ui_callback_url = (
            self.node.try_get_context("hosted_ui_callback_url")
            or "https://svr1n7ghpk.execute-api.us-east-1.amazonaws.com/prod/"
        )
        hosted_ui_callback_urls = list(
            dict.fromkeys(
                [
                    hosted_ui_callback_url,
                    "http://localhost:8000/",
                    "http://localhost:5000/research-assistant/",
                    "http://127.0.0.1:5000/research-assistant/",
                    "https://<your-domain.com>/research-assistant/",
                    "https://www.<your-domain.com>/research-assistant/",
                    "https://tractlyticswebsite.web.app/research-assistant/",
                    "https://tractlyticswebsite.firebaseapp.com/research-assistant/",
                ]
            )
        )
        bedrock_model_id = (
            self.node.try_get_context("bedrock_model_id")
            or "us.anthropic.claude-opus-4-1-20250805-v1:0"
        )
        bedrock_simple_model_id = (
            self.node.try_get_context("bedrock_simple_model_id")
            or "google.gemma-3-12b-it"
        )
        bedrock_simple_fallback_model_id = (
            self.node.try_get_context("bedrock_simple_fallback_model_id")
            or "amazon.nova-lite-v1:0"
        )
        bedrock_reasoning_model_id = (
            self.node.try_get_context("bedrock_reasoning_model_id")
            or "google.gemma-4-31b"
        )
        bedrock_reasoning_fallback_model_id = (
            self.node.try_get_context("bedrock_reasoning_fallback_model_id")
            or "mistral.ministral-3-14b-instruct"
        )
        owner_email = (
            self.node.try_get_context("owner_email")
            or self.node.try_get_context("aws_owner_email")
            or "[REDACTED_EMAIL]"
        )
        ragas_enabled = _context_bool(self, "ragas_enabled", True)
        rag_evaluation_enabled = _context_bool(self, "rag_evaluation_enabled", True)
        ragas_use_normal_model_routing = _context_bool(
            self, "ragas_use_normal_model_routing", False
        )
        document_upload_enabled = _context_bool(self, "document_upload_enabled", True)
        retrieval_diagnostics_enabled = _context_bool(
            self, "retrieval_diagnostics_enabled", True
        )
        bedrock_embedding_model_id = (
            self.node.try_get_context("bedrock_embedding_model_id")
            or "amazon.titan-embed-text-v2:0"
        )
        bedrock_embedding_dimensions = (
            self.node.try_get_context("bedrock_embedding_dimensions") or "1024"
        )
        langsmith_project = (
            self.node.try_get_context("langsmith_project")
            or "personal-research-assistant"
        )
        execution_role = iam.Role(
            self,
            "ApiLambdaExecutionRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            description="Execution role for the Personal Research Assistant API Lambda.",
        )
        execution_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name(
                "service-role/AWSLambdaBasicExecutionRole"
            )
        )
        document_bucket.grant_read_write(execution_role)
        execution_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                ],
                resources=[
                    f"arn:aws:bedrock:{Stack.of(self).region}:"
                    f"{Stack.of(self).account}:inference-profile/*",
                    "arn:aws:bedrock:*::foundation-model/*",
                ],
            )
        )
        execution_role.add_to_policy(
            iam.PolicyStatement(
                actions=["bedrock-mantle:CreateInference"],
                resources=[
                    f"arn:aws:bedrock-mantle:{Stack.of(self).region}:"
                    f"{Stack.of(self).account}:project/*"
                ],
            )
        )
        execution_role.add_to_policy(
            iam.PolicyStatement(
                actions=["bedrock-mantle:CallWithBearerToken"],
                resources=["*"],
            )
        )
        rag_evaluation_queue = sqs.Queue(
            self,
            "RagEvaluationQueue",
            visibility_timeout=Duration.minutes(15),
            retention_period=Duration.days(4),
        )
        ragas_evaluation_dead_letter_queue = sqs.Queue(
            self,
            "RagasEvaluationDeadLetterQueue",
            encryption=sqs.QueueEncryption.SQS_MANAGED,
            enforce_ssl=True,
            retention_period=Duration.days(14),
        )
        ragas_evaluation_queue = sqs.Queue(
            self,
            "RagasEvaluationQueue",
            encryption=sqs.QueueEncryption.SQS_MANAGED,
            enforce_ssl=True,
            visibility_timeout=Duration.minutes(30),
            retention_period=Duration.days(1),
            dead_letter_queue=sqs.DeadLetterQueue(
                queue=ragas_evaluation_dead_letter_queue,
                # The queue starts a Step Functions execution. Keep retries
                # bounded, but tolerate transient starter or service failures.
                max_receive_count=5,
            ),
        )
        ragas_evaluation_starter_role = iam.Role(
            self,
            "RagasEvaluationStarterRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            description="Execution role for the RAGAS evaluation starter Lambda.",
        )
        ragas_evaluation_starter_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name(
                "service-role/AWSLambdaBasicExecutionRole"
            )
        )
        ragas_evaluation_queue.grant_consume_messages(
            ragas_evaluation_starter_role
        )
        execution_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "sqs:SendMessage",
                    "sqs:ReceiveMessage",
                    "sqs:DeleteMessage",
                    "sqs:GetQueueAttributes",
                    "sqs:ChangeMessageVisibility",
                ],
                resources=[rag_evaluation_queue.queue_arn, ragas_evaluation_queue.queue_arn],
            )
        )
        langsmith_secret = secretsmanager.Secret.from_secret_name_v2(
            self,
            "LangSmithApiKeySecret",
            secret_name="LANGSMITH_API_KEY",
        )
        langsmith_secret.grant_read(execution_role)
        api_function_name = f"{Stack.of(self).stack_name}-api"
        api_function = lambda_.Function(
            self,
            "ApiFunction",
            function_name=api_function_name,
            runtime=lambda_.Runtime.PYTHON_3_12,
            architecture=lambda_.Architecture.X86_64,
            handler="pra_api.lambda_handler.handler",
            role=execution_role,
            timeout=Duration.seconds(30),
            memory_size=512,
            code=lambda_.Code.from_asset(
                str(project_root),
                bundling=BundlingOptions(
                    image=lambda_.Runtime.PYTHON_3_12.bundling_image,
                    command=[
                        "bash",
                        "-c",
                        "python -m pip install --quiet --disable-pip-version-check "
                        "-r requirements-lambda.txt -t /asset-output "
                        "&& cp -r apps platform docs data /asset-output",
                    ],
                ),
            ),
            environment={
                "PYTHONPATH": "/var/task/apps/api/src:/var/task/platform/core/src",
                "PRA_ENVIRONMENT": "prod",
                "PRA_LOG_FORMAT": "json",
                "PRA_LOG_LEVEL": "INFO",
                "PRA_API_ROOT_PATH": "/prod",
                "PRA_AUTH_PROVIDER": "cognito",
                "PRA_AWS_REGION": Stack.of(self).region,
                "PRA_COGNITO_USER_POOL_ID": user_pool.user_pool_id,
                "PRA_DOCUMENT_BUCKET_NAME": document_bucket.bucket_name,
                "PRA_PERSISTENCE_BACKEND": "s3",
                "PRA_PERSISTENCE_S3_PREFIX": "pra-prod",
                "PRA_VECTOR_STORE_BACKEND": "faiss",
                "PRA_RAG_EVAL_DATASET_PATH": (
                    f"s3://{document_bucket.bucket_name}/"
                    "pra-prod/testing/rag-eval/rag_eval_dataset_v2.json"
                ),
                "PRA_LLM_PROVIDER": "bedrock",
                "PRA_BEDROCK_MODEL_ID": bedrock_model_id,
                "PRA_BEDROCK_SIMPLE_MODEL_ID": bedrock_simple_model_id,
                "PRA_BEDROCK_SIMPLE_FALLBACK_MODEL_ID": bedrock_simple_fallback_model_id,
                "PRA_BEDROCK_REASONING_MODEL_ID": bedrock_reasoning_model_id,
                "PRA_BEDROCK_REASONING_FALLBACK_MODEL_ID": bedrock_reasoning_fallback_model_id,
                "PRA_BEDROCK_MANTLE_BASE_URL": (
                    f"https://bedrock-mantle.{Stack.of(self).region}.api.aws/openai/v1"
                ),
                "PRA_BEDROCK_MAX_TOKENS": "700",
                "PRA_SIMPLE_ANSWER_MAX_TOKENS": "150",
                "PRA_REASONING_ANSWER_MAX_TOKENS": "400",
                "PRA_BEDROCK_TEMPERATURE": "0.2",
                "PRA_EMBEDDING_PROVIDER": "bedrock",
                "PRA_BEDROCK_EMBEDDING_MODEL_ID": bedrock_embedding_model_id,
                "PRA_BEDROCK_EMBEDDING_DIMENSIONS": str(
                    bedrock_embedding_dimensions
                ),
                "PRA_BM25_TOP_K": "50",
                "PRA_VECTOR_TOP_K": "50",
                "PRA_RRF_K": "30",
                "PRA_FUSION_STRATEGY": "weighted_rrf",
                "PRA_BM25_FUSION_WEIGHT": "0.8",
                "PRA_VECTOR_FUSION_WEIGHT": "0.2",
                "PRA_PRESERVE_BM25_CANDIDATES": "true",
                "PRA_ENABLE_BM25_LEXICAL": "true",
                "PRA_ENABLE_VECTOR_SEARCH": "true",
                "PRA_ENABLE_RECIPROCAL_RANK_FUSION": "true",
                "PRA_ENABLE_RERANKING": "false",
                "PRA_RERANK_TOP_K": "50",
                "PRA_LLM_CONTEXT_TOP_K": "15",
                "PRA_CHUNK_SIZE": "600",
                "PRA_CHUNK_OVERLAP": "100",
                "PRA_DOCUMENT_UPLOAD_ENABLED": document_upload_enabled,
                "PRA_RETRIEVAL_DIAGNOSTICS_ENABLED": retrieval_diagnostics_enabled,
                "PRA_ADAPTIVE_CHUNKING_ENABLED": "true",
                "PRA_ADAPTIVE_RETRIEVAL_ENABLED": "true",
                "PRA_TABLE_PROCESSING_ENABLED": "true",
                "PRA_RETRIEVAL_BM25_WEIGHT": "0.60",
                "PRA_RETRIEVAL_VECTOR_WEIGHT": "0.40",
                "PRA_RETRIEVAL_RELATIONSHIP_WEIGHT": "0.00",
                "PRA_KNOWLEDGE_GRAPH_ENABLED": "true",
                "PRA_GRAPH_RETRIEVAL_ENABLED": "true",
                "PRA_GRAPH_EXTRACTION_LLM_ENABLED": "true",
                "PRA_CLOUDWATCH_LOG_GROUP": f"/aws/lambda/{api_function_name}",
                "LANGSMITH_PROJECT": langsmith_project,
                "PRA_RAG_EVALUATION_QUEUE_URL": rag_evaluation_queue.queue_url,
                "PRA_RAG_EVALUATION_ENABLED": rag_evaluation_enabled,
                "PRA_RAGAS_ENABLED": ragas_enabled,
                "PRA_RAGAS_BATCH_SIZE": "1",
                "PRA_RAGAS_STANDARD_MAX_QUESTIONS": "12",
                "PRA_RAGAS_STANDARD_TOP_K": "15",
                "PRA_RAGAS_EVALUATOR_MAX_TOKENS": "768",
                "PRA_RAGAS_METRIC_TIMEOUT_SECONDS": "120",
                "PRA_RATE_LIMIT_RAGAS_REQUESTS": "2",
                "PRA_RATE_LIMIT_RAGAS_WINDOW_SECONDS": "3600",
                "PRA_RAGAS_EVALUATION_QUEUE_URL": ragas_evaluation_queue.queue_url,
                "PRA_RAGAS_EVAL_DATASET_PATH": (
                    f"s3://{document_bucket.bucket_name}/"
                    "pra-prod/testing/rag-eval/rag_eval_dataset_v2.json"
                ),
                "PRA_RAGAS_EVALUATOR_MODEL_ID": (
                    "bedrock/" + bedrock_simple_model_id
                ),
                "PRA_RAGAS_EMBEDDING_MODEL_ID": (
                    "bedrock/" + bedrock_embedding_model_id
                ),
            },
            log_retention=logs.RetentionDays.ONE_MONTH,
        )

        rag_evaluation_worker = lambda_.Function(
            self,
            "RagEvaluationWorkerFunction",
            function_name=f"{Stack.of(self).stack_name}-rag-evaluation-worker",
            runtime=lambda_.Runtime.PYTHON_3_12,
            architecture=lambda_.Architecture.X86_64,
            handler="pra_api.rag_worker.handler",
            role=execution_role,
            timeout=Duration.minutes(15),
            memory_size=1024,
            code=lambda_.Code.from_asset(
                str(project_root),
                bundling=BundlingOptions(
                    image=lambda_.Runtime.PYTHON_3_12.bundling_image,
                    command=[
                        "bash",
                        "-c",
                        "python -m pip install --quiet --disable-pip-version-check "
                        "-r requirements-lambda.txt -t /asset-output "
                        "&& cp -r apps platform docs data /asset-output",
                    ],
                ),
            ),
            environment={
                "PYTHONPATH": "/var/task/apps/api/src:/var/task/platform/core/src",
                "PRA_ENVIRONMENT": "prod",
                "PRA_LOG_FORMAT": "json",
                "PRA_LOG_LEVEL": "INFO",
                "PRA_API_ROOT_PATH": "/prod",
                "PRA_AUTH_PROVIDER": "cognito",
                "PRA_AWS_REGION": Stack.of(self).region,
                "PRA_COGNITO_USER_POOL_ID": user_pool.user_pool_id,
                "PRA_DOCUMENT_BUCKET_NAME": document_bucket.bucket_name,
                "PRA_PERSISTENCE_BACKEND": "s3",
                "PRA_PERSISTENCE_S3_PREFIX": "pra-prod",
                "PRA_VECTOR_STORE_BACKEND": "faiss",
                "PRA_RAG_EVAL_DATASET_PATH": (
                    f"s3://{document_bucket.bucket_name}/"
                    "pra-prod/testing/rag-eval/rag_eval_dataset_v2.json"
                ),
                "PRA_LLM_PROVIDER": "bedrock",
                "PRA_BEDROCK_MODEL_ID": bedrock_model_id,
                "PRA_BEDROCK_SIMPLE_MODEL_ID": bedrock_simple_model_id,
                "PRA_BEDROCK_SIMPLE_FALLBACK_MODEL_ID": bedrock_simple_fallback_model_id,
                "PRA_BEDROCK_REASONING_MODEL_ID": bedrock_reasoning_model_id,
                "PRA_BEDROCK_REASONING_FALLBACK_MODEL_ID": bedrock_reasoning_fallback_model_id,
                "PRA_BEDROCK_MAX_TOKENS": "700",
                "PRA_SIMPLE_ANSWER_MAX_TOKENS": "150",
                "PRA_REASONING_ANSWER_MAX_TOKENS": "400",
                "PRA_BEDROCK_TEMPERATURE": "0.2",
                "PRA_EMBEDDING_PROVIDER": "bedrock",
                "PRA_BEDROCK_EMBEDDING_MODEL_ID": bedrock_embedding_model_id,
                "PRA_BEDROCK_EMBEDDING_DIMENSIONS": str(
                    bedrock_embedding_dimensions
                ),
                "PRA_BM25_TOP_K": "50",
                "PRA_VECTOR_TOP_K": "50",
                "PRA_RRF_K": "30",
                "PRA_FUSION_STRATEGY": "weighted_rrf",
                "PRA_BM25_FUSION_WEIGHT": "0.8",
                "PRA_VECTOR_FUSION_WEIGHT": "0.2",
                "PRA_PRESERVE_BM25_CANDIDATES": "true",
                "PRA_ENABLE_BM25_LEXICAL": "true",
                "PRA_ENABLE_VECTOR_SEARCH": "true",
                "PRA_ENABLE_RECIPROCAL_RANK_FUSION": "true",
                "PRA_ENABLE_RERANKING": "false",
                "PRA_RERANK_TOP_K": "50",
                "PRA_LLM_CONTEXT_TOP_K": "15",
                "PRA_CHUNK_SIZE": "600",
                "PRA_CHUNK_OVERLAP": "100",
                "PRA_DOCUMENT_UPLOAD_ENABLED": document_upload_enabled,
                "PRA_RETRIEVAL_DIAGNOSTICS_ENABLED": retrieval_diagnostics_enabled,
                "PRA_ADAPTIVE_CHUNKING_ENABLED": "true",
                "PRA_ADAPTIVE_RETRIEVAL_ENABLED": "true",
                "PRA_TABLE_PROCESSING_ENABLED": "true",
                "PRA_RETRIEVAL_BM25_WEIGHT": "0.60",
                "PRA_RETRIEVAL_VECTOR_WEIGHT": "0.40",
                "PRA_RETRIEVAL_RELATIONSHIP_WEIGHT": "0.00",
                "PRA_KNOWLEDGE_GRAPH_ENABLED": "true",
                "PRA_GRAPH_RETRIEVAL_ENABLED": "true",
                "PRA_GRAPH_EXTRACTION_LLM_ENABLED": "true",
                "PRA_CLOUDWATCH_LOG_GROUP": f"/aws/lambda/{api_function_name}",
                "LANGSMITH_PROJECT": langsmith_project,
                "PRA_RAG_EVALUATION_WORKER": "true",
                "PRA_RAG_EVALUATION_ENABLED": rag_evaluation_enabled,
                "PRA_RAGAS_ENABLED": ragas_enabled,
                "PRA_RAGAS_BATCH_SIZE": "1",
                "PRA_RAGAS_EVALUATION_QUEUE_URL": ragas_evaluation_queue.queue_url,
                "PRA_RAGAS_EVAL_DATASET_PATH": (
                    f"s3://{document_bucket.bucket_name}/"
                    "pra-prod/testing/rag-eval/rag_eval_dataset_v2.json"
                ),
                "PRA_RAGAS_EVALUATOR_MODEL_ID": (
                    "bedrock/" + bedrock_simple_model_id
                ),
                "PRA_RAGAS_EMBEDDING_MODEL_ID": (
                    "bedrock/" + bedrock_embedding_model_id
                ),
            },
            log_retention=logs.RetentionDays.ONE_MONTH,
        )
        rag_evaluation_worker.add_event_source(
            lambda_event_sources.SqsEventSource(
                rag_evaluation_queue,
                batch_size=1,
                report_batch_item_failures=False,
            )
        )

        ragas_evaluation_worker = lambda_.DockerImageFunction(
            self,
            "RagasEvaluationWorkerFunction",
            function_name=f"{Stack.of(self).stack_name}-ragas-evaluation-worker",
            architecture=lambda_.Architecture.X86_64,
            role=execution_role,
            timeout=Duration.minutes(14),
            memory_size=1024,
            reserved_concurrent_executions=1,
            code=lambda_.DockerImageCode.from_image_asset(
                str(project_root),
                file="Dockerfile.ragas-worker",
            ),
            environment={
                "PYTHONPATH": "/var/task/apps/api/src:/var/task/platform/core/src",
                "PRA_ENVIRONMENT": "prod",
                "PRA_LOG_FORMAT": "json",
                "PRA_LOG_LEVEL": "INFO",
                "PRA_AUTH_PROVIDER": "cognito",
                "PRA_AWS_REGION": Stack.of(self).region,
                "PRA_COGNITO_USER_POOL_ID": user_pool.user_pool_id,
                "PRA_DOCUMENT_BUCKET_NAME": document_bucket.bucket_name,
                "PRA_PERSISTENCE_BACKEND": "s3",
                "PRA_PERSISTENCE_S3_PREFIX": "pra-prod",
                "PRA_VECTOR_STORE_BACKEND": "faiss",
                "PRA_RAG_EVAL_DATASET_PATH": (
                    f"s3://{document_bucket.bucket_name}/"
                    "pra-prod/testing/rag-eval/rag_eval_dataset_v2.json"
                ),
                "PRA_LLM_PROVIDER": "bedrock",
                "PRA_BEDROCK_MODEL_ID": bedrock_model_id,
                "PRA_BEDROCK_SIMPLE_MODEL_ID": bedrock_simple_model_id,
                "PRA_BEDROCK_SIMPLE_FALLBACK_MODEL_ID": bedrock_simple_fallback_model_id,
                "PRA_BEDROCK_REASONING_MODEL_ID": bedrock_reasoning_model_id,
                "PRA_BEDROCK_REASONING_FALLBACK_MODEL_ID": bedrock_reasoning_fallback_model_id,
                "PRA_BEDROCK_MANTLE_BASE_URL": (
                    f"https://bedrock-mantle.{Stack.of(self).region}.api.aws/openai/v1"
                ),
                "PRA_BEDROCK_MAX_TOKENS": "400",
                "PRA_SIMPLE_ANSWER_MAX_TOKENS": "150",
                "PRA_REASONING_ANSWER_MAX_TOKENS": "400",
                "PRA_BEDROCK_TEMPERATURE": "0.2",
                "PRA_EMBEDDING_PROVIDER": "bedrock",
                "PRA_BEDROCK_EMBEDDING_MODEL_ID": bedrock_embedding_model_id,
                "PRA_BEDROCK_EMBEDDING_DIMENSIONS": str(bedrock_embedding_dimensions),
                "PRA_BM25_TOP_K": "50",
                "PRA_VECTOR_TOP_K": "50",
                "PRA_RRF_K": "30",
                "PRA_FUSION_STRATEGY": "weighted_rrf",
                "PRA_BM25_FUSION_WEIGHT": "0.8",
                "PRA_VECTOR_FUSION_WEIGHT": "0.2",
                "PRA_PRESERVE_BM25_CANDIDATES": "true",
                "PRA_ENABLE_BM25_LEXICAL": "true",
                "PRA_ENABLE_VECTOR_SEARCH": "true",
                "PRA_ENABLE_RECIPROCAL_RANK_FUSION": "true",
                "PRA_ENABLE_RERANKING": "false",
                "PRA_RERANK_TOP_K": "50",
                "PRA_LLM_CONTEXT_TOP_K": "15",
                "PRA_CHUNK_SIZE": "600",
                "PRA_CHUNK_OVERLAP": "100",
                "PRA_DOCUMENT_UPLOAD_ENABLED": document_upload_enabled,
                "PRA_RETRIEVAL_DIAGNOSTICS_ENABLED": retrieval_diagnostics_enabled,
                "PRA_ADAPTIVE_CHUNKING_ENABLED": "true",
                "PRA_ADAPTIVE_RETRIEVAL_ENABLED": "true",
                "PRA_TABLE_PROCESSING_ENABLED": "true",
                "PRA_RETRIEVAL_BM25_WEIGHT": "0.60",
                "PRA_RETRIEVAL_VECTOR_WEIGHT": "0.40",
                "PRA_RETRIEVAL_RELATIONSHIP_WEIGHT": "0.00",
                "PRA_KNOWLEDGE_GRAPH_ENABLED": "true",
                "PRA_GRAPH_RETRIEVAL_ENABLED": "true",
                "PRA_GRAPH_EXTRACTION_LLM_ENABLED": "true",
                "PRA_RAGAS_ENABLED": ragas_enabled,
                "PRA_RAGAS_EVALUATION_WORKER": "true",
                "PRA_RAGAS_STEP_FUNCTIONS_ORCHESTRATION": "true",
                "PRA_RAGAS_BATCH_SIZE": "1",
                "PRA_RAGAS_STANDARD_MAX_QUESTIONS": "12",
                "PRA_RAGAS_STANDARD_TOP_K": "15",
                "PRA_RAGAS_EVALUATOR_MAX_TOKENS": "768",
                "PRA_RAGAS_METRIC_TIMEOUT_SECONDS": "120",
                "PRA_RAGAS_MODEL_ONLY_ID": "google.gemma-3-12b-it",
                "PRA_RAGAS_USE_NORMAL_MODEL_ROUTING": ragas_use_normal_model_routing,
                "PRA_RAGAS_EVAL_DATASET_PATH": (
                    f"s3://{document_bucket.bucket_name}/"
                    "pra-prod/testing/rag-eval/rag_eval_dataset_v2.json"
                ),
                "PRA_RAGAS_EVALUATOR_MODEL_ID": "bedrock/google.gemma-3-12b-it",
                "PRA_RAGAS_EMBEDDING_MODEL_ID": "bedrock/" + bedrock_embedding_model_id,
                "PRA_CLOUDWATCH_LOG_GROUP": (
                    f"/aws/lambda/{Stack.of(self).stack_name}-ragas-evaluation-worker"
                ),
                "LANGSMITH_PROJECT": langsmith_project,
            },
            log_retention=logs.RetentionDays.ONE_MONTH,
        )

        ragas_status_checkpoint = lambda_.Function(
            self,
            "RagasStatusCheckpointFunction",
            function_name=f"{Stack.of(self).stack_name}-ragas-status-checkpoint",
            runtime=lambda_.Runtime.PYTHON_3_12,
            architecture=lambda_.Architecture.X86_64,
            handler="pra_api.ragas_status_checkpoint.handler",
            timeout=Duration.seconds(30),
            memory_size=256,
            code=lambda_.Code.from_asset(
                str(project_root),
                bundling=BundlingOptions(
                    image=lambda_.Runtime.PYTHON_3_12.bundling_image,
                    command=[
                        "bash",
                        "-c",
                        "python -m pip install --quiet --disable-pip-version-check "
                        "-r requirements-lambda.txt -t /asset-output "
                        "&& cp -r apps/api/src/pra_api /asset-output",
                    ],
                ),
            ),
            environment={
                "PRA_DOCUMENT_BUCKET_NAME": document_bucket.bucket_name,
                "PRA_PERSISTENCE_S3_PREFIX": "pra-prod",
                "PRA_ENVIRONMENT": "prod",
                "PRA_LOG_FORMAT": "json",
                "PRA_LOG_LEVEL": "INFO",
            },
            log_retention=logs.RetentionDays.ONE_MONTH,
        )
        document_bucket.grant_read_write(ragas_status_checkpoint)

        checkpoint_ragas_orchestration = sfn_tasks.LambdaInvoke(
            self,
            "CheckpointRagasOrchestration",
            lambda_function=ragas_status_checkpoint,
            payload=sfn.TaskInput.from_object(
                {
                    "job_id": sfn.JsonPath.string_at("$.job_id"),
                    "checkpoint": "orchestration_started",
                }
            ),
            result_path="$.orchestration_checkpoint",
        )

        invoke_ragas_worker_step = sfn_tasks.LambdaInvoke(
            self,
            "InvokeRagasWorkerStep",
            lambda_function=ragas_evaluation_worker,
            payload=sfn.TaskInput.from_object(
                {"job_id": sfn.JsonPath.string_at("$.job_id")}
            ),
            result_path="$.step",
            task_timeout=sfn.Timeout.duration(Duration.minutes(14)),
        )
        checkpoint_ragas_capacity_wait = sfn_tasks.LambdaInvoke(
            self,
            "CheckpointRagasCapacityWait",
            lambda_function=ragas_status_checkpoint,
            payload=sfn.TaskInput.from_object(
                {
                    "job_id": sfn.JsonPath.string_at("$.job_id"),
                    "checkpoint": "waiting_for_capacity",
                }
            ),
            result_path="$.capacity_checkpoint",
        )
        wait_for_ragas_worker_capacity = sfn.Wait(
            self,
            "WaitForRagasWorkerCapacity",
            time=sfn.WaitTime.duration(Duration.minutes(2)),
        )
        checkpoint_ragas_capacity_failure = sfn_tasks.LambdaInvoke(
            self,
            "CheckpointRagasCapacityFailure",
            lambda_function=ragas_status_checkpoint,
            payload=sfn.TaskInput.from_object(
                {
                    "job_id": sfn.JsonPath.string_at("$.job_id"),
                    "checkpoint": "capacity_exhausted",
                }
            ),
            result_path="$.capacity_failure",
        )
        ragas_execution_succeeded = sfn.Succeed(
            self,
            "RagasEvaluationExecutionSucceeded",
        )
        ragas_execution_cancelled = sfn.Succeed(
            self,
            "RagasEvaluationExecutionCancelled",
        )
        ragas_execution_failed = sfn.Fail(
            self,
            "RagasEvaluationExecutionFailed",
            error="RagasEvaluationFailed",
            cause="The RAGAS job record was marked failed by the bounded worker step.",
        )
        capacity_retry_limit = sfn.Choice(
            self,
            "RagasWorkerCapacityRetryLimit",
        )
        capacity_retry_limit.when(
            sfn.Condition.number_greater_than_equals(
                "$.capacity_checkpoint.Payload.capacity_wait_count", 60
            ),
            checkpoint_ragas_capacity_failure.next(ragas_execution_failed),
        )
        capacity_retry_limit.otherwise(wait_for_ragas_worker_capacity)
        wait_for_ragas_worker_capacity.next(invoke_ragas_worker_step)
        invoke_ragas_worker_step.add_catch(
            checkpoint_ragas_capacity_wait.next(capacity_retry_limit),
            errors=["Lambda.TooManyRequestsException"],
            result_path="$.worker_capacity_error",
        )
        checkpoint_ragas_orchestration.next(invoke_ragas_worker_step)
        continue_ragas_evaluation = sfn.Choice(
            self,
            "RagasEvaluationNeedsAnotherStep",
        )
        continue_ragas_evaluation.when(
            sfn.Condition.string_equals("$.step.Payload.status", "running"),
            invoke_ragas_worker_step,
        )
        continue_ragas_evaluation.when(
            sfn.Condition.string_equals("$.step.Payload.status", "completed"),
            ragas_execution_succeeded,
        )
        continue_ragas_evaluation.when(
            sfn.Condition.string_equals("$.step.Payload.status", "cancelled"),
            ragas_execution_cancelled,
        )
        continue_ragas_evaluation.otherwise(ragas_execution_failed)
        invoke_ragas_worker_step.next(continue_ragas_evaluation)

        ragas_evaluation_state_machine = sfn.StateMachine(
            self,
            "RagasEvaluationStateMachine",
            state_machine_name=f"{Stack.of(self).stack_name}-ragas-evaluation",
            definition_body=sfn.DefinitionBody.from_chainable(checkpoint_ragas_orchestration),
            timeout=Duration.hours(6),
            tracing_enabled=True,
        )

        ragas_evaluation_starter = lambda_.Function(
            self,
            "RagasEvaluationStarterFunction",
            function_name=f"{Stack.of(self).stack_name}-ragas-evaluation-starter",
            runtime=lambda_.Runtime.PYTHON_3_12,
            architecture=lambda_.Architecture.X86_64,
            handler="pra_api.ragas_starter.handler",
            role=ragas_evaluation_starter_role,
            timeout=Duration.seconds(30),
            memory_size=256,
            code=lambda_.Code.from_asset(
                str(project_root),
                bundling=BundlingOptions(
                    image=lambda_.Runtime.PYTHON_3_12.bundling_image,
                    command=[
                        "bash",
                        "-c",
                        "python -m pip install --quiet --disable-pip-version-check "
                        "-r requirements-lambda.txt -t /asset-output "
                        "&& cp -r apps/api/src/pra_api "
                        "platform/core/src/pra_core /asset-output",
                    ],
                ),
            ),
            environment={
                "PRA_ENVIRONMENT": "prod",
                "PRA_LOG_FORMAT": "json",
                "PRA_LOG_LEVEL": "INFO",
                "PRA_RAGAS_STATE_MACHINE_ARN": ragas_evaluation_state_machine.state_machine_arn,
            },
            log_retention=logs.RetentionDays.ONE_MONTH,
        )
        ragas_evaluation_state_machine.grant_start_execution(
            ragas_evaluation_starter_role
        )
        ragas_evaluation_starter.add_event_source(
            lambda_event_sources.SqsEventSource(
                ragas_evaluation_queue,
                batch_size=1,
                report_batch_item_failures=False,
            )
        )

        api = apigateway.LambdaRestApi(
            self,
            "RestApi",
            handler=api_function,
            proxy=True,
            deploy_options=apigateway.StageOptions(
                stage_name="prod",
                metrics_enabled=True,
                logging_level=apigateway.MethodLoggingLevel.INFO,
            ),
        )

        cost_alert_topic = sns.Topic(
            self,
            "CostAlertTopic",
            display_name="Personal Research Assistant cost alerts",
        )
        cost_alert_topic.add_subscription(
            sns_subscriptions.EmailSubscription(owner_email)
        )

        lambda_error_alarm = cloudwatch.Alarm(
            self,
            "ApiLambdaErrorAlarm",
            alarm_description="Triggers when the API Lambda records errors.",
            metric=api_function.metric_errors(
                period=Duration.minutes(5),
                statistic="sum",
            ),
            threshold=1,
            evaluation_periods=1,
            datapoints_to_alarm=1,
        )
        lambda_error_alarm.add_alarm_action(
            cloudwatch_actions.SnsAction(cost_alert_topic)
        )

        lambda_duration_alarm = cloudwatch.Alarm(
            self,
            "ApiLambdaDurationAlarm",
            alarm_description="Triggers when API Lambda duration stays elevated.",
            metric=api_function.metric_duration(
                period=Duration.minutes(5),
                statistic="p95",
            ),
            threshold=5000,
            evaluation_periods=3,
            datapoints_to_alarm=3,
        )
        lambda_duration_alarm.add_alarm_action(
            cloudwatch_actions.SnsAction(cost_alert_topic)
        )

        api_5xx_alarm = cloudwatch.Alarm(
            self,
            "ApiGateway5xxAlarm",
            alarm_description="Triggers when API Gateway returns server errors.",
            metric=api.metric_server_error(
                period=Duration.minutes(5),
                statistic="sum",
            ),
            threshold=1,
            evaluation_periods=1,
            datapoints_to_alarm=1,
        )
        api_5xx_alarm.add_alarm_action(
            cloudwatch_actions.SnsAction(cost_alert_topic)
        )

        ragas_invocation_alarm = cloudwatch.Alarm(
            self,
            "RagasInvocationVolumeAlarm",
            alarm_description="RAGAS worker invocation volume exceeded the expected admin workload.",
            metric=ragas_evaluation_worker.metric_invocations(
                period=Duration.hours(1),
                statistic="sum",
            ),
            threshold=8,
            evaluation_periods=1,
            datapoints_to_alarm=1,
        )
        ragas_invocation_alarm.add_alarm_action(
            cloudwatch_actions.SnsAction(cost_alert_topic)
        )

        ragas_dlq_alarm = cloudwatch.Alarm(
            self,
            "RagasDeadLetterQueueAlarm",
            alarm_description="A RAGAS batch exhausted its bounded retry allowance.",
            metric=ragas_evaluation_dead_letter_queue.metric_approximate_number_of_messages_visible(
                period=Duration.minutes(5),
                statistic="max",
            ),
            threshold=1,
            evaluation_periods=1,
            datapoints_to_alarm=1,
        )
        ragas_dlq_alarm.add_alarm_action(
            cloudwatch_actions.SnsAction(cost_alert_topic)
        )

        cost_dashboard = cloudwatch.Dashboard(
            self,
            "CostDashboard",
            dashboard_name=f"{Stack.of(self).stack_name}-cost-dashboard",
        )
        cost_dashboard.add_widgets(
            cloudwatch.GraphWidget(
                title="Request Volume",
                left=[
                    api.metric_count(period=Duration.minutes(5)),
                    api_function.metric_invocations(period=Duration.minutes(5)),
                ],
            ),
            cloudwatch.GraphWidget(
                title="Errors and Retries",
                left=[
                    api_function.metric_errors(period=Duration.minutes(5)),
                    api.metric_server_error(period=Duration.minutes(5)),
                    api.metric_client_error(period=Duration.minutes(5)),
                ],
            ),
            cloudwatch.GraphWidget(
                title="Latency",
                left=[
                    api.metric_latency(
                        period=Duration.minutes(5),
                        statistic="p95",
                    ),
                    api_function.metric_duration(
                        period=Duration.minutes(5),
                        statistic="p95",
                    ),
                ],
            ),
            cloudwatch.GraphWidget(
                title="Operational Cost Signals",
                left=[
                    api_function.metric_throttles(period=Duration.minutes(5)),
                    ragas_evaluation_worker.metric_invocations(period=Duration.minutes(5)),
                    ragas_evaluation_worker.metric_errors(period=Duration.minutes(5)),
                ],
            ),
            cloudwatch.GraphWidget(
                title="Integration Latency",
                left=[
                    api.metric_integration_latency(period=Duration.minutes(5)),
                ],
            ),
        )

        user_pool_client = user_pool.add_client(
            "UserPoolClient",
            generate_secret=False,
            o_auth=cognito.OAuthSettings(
                flows=cognito.OAuthFlows(authorization_code_grant=True),
                scopes=[
                    cognito.OAuthScope.OPENID,
                    cognito.OAuthScope.EMAIL,
                    cognito.OAuthScope.PROFILE,
                ],
                callback_urls=hosted_ui_callback_urls,
                logout_urls=hosted_ui_callback_urls,
            ),
            access_token_validity=Duration.hours(1),
            id_token_validity=Duration.hours(1),
            refresh_token_validity=Duration.days(30),
            auth_flows=cognito.AuthFlow(
                user_password=True,
                user_srp=True,
            ),
        )
        cognito_issuer = (
            f"https://cognito-idp.{Stack.of(self).region}.amazonaws.com/"
            f"{user_pool.user_pool_id}"
        )
        hosted_ui_base_url = (
            f"https://{user_pool_domain.domain_name}.auth."
            f"{Stack.of(self).region}.amazoncognito.com"
        )
        cognito_hosted_ui_css = """
.background-customizable {
  background:
    linear-gradient(135deg, rgba(3, 21, 41, 0.98), rgba(5, 50, 82, 0.96)),
    #031529;
}

.banner-customizable {
  background-color: #071f3a;
  border-bottom: 1px solid #1c6a93;
  padding: 1.5rem 0;
}

.label-customizable {
  color: #95c6df;
}

.textDescription-customizable {
  color: #95c6df;
}

.idpDescription-customizable {
  color: #95c6df;
}

.legalText-customizable {
  color: #95c6df;
}

.inputField-customizable {
  background-color: #02101f;
  border: 1px solid #1c6a93;
  border-radius: 6px;
  color: #e9f7ff;
}

.inputField-customizable:focus {
  border-color: #19c9ff;
  box-shadow: 0 0 0 2px rgba(25, 201, 255, 0.36);
}

.submitButton-customizable {
  background-color: #58b322;
  border: 0;
  border-radius: 6px;
  color: #061327;
  font-weight: 700;
}

.submitButton-customizable:hover {
  background-color: #3f8a15;
}

.redirect-customizable {
  color: #ffb21c;
}

.errorMessage-customizable {
  color: #ff8f7f;
}
"""
        cognito.CfnUserPoolUICustomizationAttachment(
            self,
            "UserPoolHostedUiBranding",
            client_id=user_pool_client.user_pool_client_id,
            user_pool_id=user_pool.user_pool_id,
            css=cognito_hosted_ui_css,
        )
        api_function.add_environment(
            "PRA_COGNITO_CLIENT_ID",
            user_pool_client.user_pool_client_id,
        )
        api_function.add_environment("PRA_COGNITO_ISSUER", cognito_issuer)
        api_function.add_environment(
            "PRA_COGNITO_HOSTED_UI_BASE_URL",
            hosted_ui_base_url,
        )

        CfnOutput(
            self,
            "ApiUrl",
            value=api.url,
            description="API Gateway invoke URL.",
        )
        CfnOutput(
            self,
            "DocumentBucketName",
            value=document_bucket.bucket_name,
            description="S3 bucket used for document and metadata persistence.",
        )
        CfnOutput(
            self,
            "CognitoUserPoolId",
            value=user_pool.user_pool_id,
            description="Cognito user pool used for application login.",
        )
        CfnOutput(
            self,
            "CognitoUserPoolClientId",
            value=user_pool_client.user_pool_client_id,
            description="Cognito app client ID used by the browser UI.",
        )
        CfnOutput(
            self,
            "CognitoHostedUiBaseUrl",
            value=hosted_ui_base_url,
            description="Cognito Hosted UI base URL.",
        )
        CfnOutput(
            self,
            "DeploymentAutomation",
            value="GitHub OIDC manual deployment workflow enabled.",
            description="Documents that this stack can be deployed through GitHub Actions OIDC.",
        )
        CfnOutput(
            self,
            "CostDashboardUrl",
            value=f"https://{Stack.of(self).region}.console.aws.amazon.com/cloudwatch/home"
            f"?region={Stack.of(self).region}#dashboards:name={Stack.of(self).stack_name}-cost-dashboard",
            description="CloudWatch dashboard URL for the cost and operations view.",
        )
        CfnOutput(
            self,
            "CostAlertEmail",
            value=owner_email,
            description="Email address subscribed to dashboard and alarm notifications.",
        )
