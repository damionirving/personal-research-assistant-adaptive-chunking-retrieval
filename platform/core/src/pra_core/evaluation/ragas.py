from __future__ import annotations

import asyncio
from dataclasses import dataclass
import inspect
import json
import math
from statistics import mean
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field
from pra_core.logging import get_logger

logger = get_logger(__name__)


class RagasEvaluationSample(BaseModel):
    id: str
    user_input: str
    response: str
    retrieved_contexts: list[str] = Field(default_factory=list)
    reference: str = ""
    citations: list[str] = Field(default_factory=list)
    retrieval_details: list[dict[str, Any]] = Field(default_factory=list)
    context_compression: dict[str, Any] = Field(default_factory=dict)
    expected_behavior: Literal["answer", "insufficient_evidence"] = "answer"


class RagasQuestionResult(BaseModel):
    id: str
    question: str
    answer: str
    metrics: dict[str, float | None] = Field(default_factory=dict)
    citations: list[str] = Field(default_factory=list)
    context_count: int = 0
    context_compression: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    error_category: str | None = None


class RagasEvaluationSummary(BaseModel):
    evaluator: str = "ragas"
    total_questions: int
    completed_questions: int
    failed_questions: int
    evaluation_success_rate: float = 0.0
    first_error: str | None = None
    metrics: dict[str, float | None] = Field(default_factory=dict)
    failed_metric_counts: dict[str, int] = Field(default_factory=dict)
    results: list[RagasQuestionResult] = Field(default_factory=list)


@dataclass(frozen=True)
class RagasRunConfig:
    evaluator_model_id: str | None = None
    embedding_model_id: str | None = None
    metric_names: tuple[str, ...] = (
        "faithfulness",
        "context_precision",
        "answer_correctness",
    )
    evaluator_max_tokens: int = 768
    max_retries: int = 1
    metric_timeout_seconds: int = 120


STANDARD_RAGAS_METRICS = (
    "faithfulness",
    "context_precision",
    "answer_correctness",
)
FULL_RAGAS_METRICS = (
    "faithfulness",
    "context_precision",
    "context_recall",
    "answer_relevancy",
    "answer_correctness",
)


RagasMetricRunner = Callable[[list[RagasEvaluationSample]], list[dict[str, float | None]]]

RAGAS_STRUCTURED_OUTPUT_SYSTEM_PROMPT = (
    "Return only valid JSON matching the requested schema. Do not use Markdown "
    "code fences, explanations, or introductory text."
)


class _LegacyEmbeddingAdapter:
    """Expose RAGAS modern embeddings through the legacy metric interface."""

    def __init__(self, embeddings: Any) -> None:
        self.embeddings = embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.embeddings.embed_text(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embeddings.embed_texts(texts)

    async def aembed_query(self, text: str) -> list[float]:
        if hasattr(self.embeddings, "aembed_text"):
            return await self.embeddings.aembed_text(text)
        return await asyncio.to_thread(self.embed_query, text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        if hasattr(self.embeddings, "aembed_texts"):
            return await self.embeddings.aembed_texts(texts)
        return await asyncio.to_thread(self.embed_documents, texts)


def sanitize_json_response(text: str) -> str:
    """Remove only an outer Markdown JSON fence from model output."""
    value = text.strip()
    if value.lower().startswith("```json"):
        value = value[len("```json") :].strip()
    elif value.startswith("```"):
        value = value[3:].strip()
    if value.endswith("```"):
        value = value[:-3].strip()
    return value


def _assistant_content(response: Any) -> str | None:
    choices = response.get("choices") if isinstance(response, dict) else getattr(response, "choices", None)
    if not choices:
        return None
    choice = choices[0]
    message = choice.get("message") if isinstance(choice, dict) else getattr(choice, "message", None)
    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
    return content if isinstance(content, str) else None


def _replace_assistant_content(response: Any, content: str) -> Any:
    choices = response.get("choices") if isinstance(response, dict) else getattr(response, "choices", None)
    if not choices:
        return response
    choice = choices[0]
    message = choice.get("message") if isinstance(choice, dict) else getattr(choice, "message", None)
    if isinstance(message, dict):
        message["content"] = content
    else:
        setattr(message, "content", content)
    return response


def _normalize_ragas_response(response: Any, kwargs: dict[str, Any]) -> Any:
    original = _assistant_content(response)
    if original is None:
        return response
    cleaned = sanitize_json_response(original)
    logger.info(
        "ragas_structured_output_sanitized",
        extra={
            "evaluator_model_id": kwargs.get("model"),
            "fence_detected": cleaned != original,
            "original_length": len(original),
            "sanitized_length": len(cleaned),
        },
    )
    try:
        json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.warning(
            "ragas_structured_output_invalid",
            extra={
                "error_type": type(exc).__name__,
                "evaluator_model_id": kwargs.get("model"),
                "retry_attempt": kwargs.get("retry_attempt"),
            },
        )
    if cleaned != original:
        _replace_assistant_content(response, cleaned)
    return response


def ragas_completion(*args, **kwargs):
    """Call LiteLLM and normalize fenced JSON before RAGAS validates it."""
    import litellm

    response = litellm.completion(*args, **kwargs)
    return _normalize_ragas_response(response, kwargs)


async def ragas_acompletion(*args, **kwargs):
    """Async counterpart for RAGAS clients that use LiteLLM acompletion."""
    import litellm

    response = await litellm.acompletion(*args, **kwargs)
    return _normalize_ragas_response(response, kwargs)


def evaluate_ragas(
    samples: list[RagasEvaluationSample],
    *,
    config: RagasRunConfig | None = None,
    metric_runner: RagasMetricRunner | None = None,
) -> RagasEvaluationSummary:
    """Evaluate generated RAG samples with RAGAS.

    The runner is injectable so API tests never need to invoke a judge model.
    Production uses the lazy RAGAS adapter below and therefore keeps the
    dependency and model calls out of retrieval-only evaluation.
    """
    run_config = config or RagasRunConfig()
    if not samples:
        return RagasEvaluationSummary(
            total_questions=0,
            completed_questions=0,
            failed_questions=0,
            evaluation_success_rate=0.0,
        )

    try:
        scores = (metric_runner or _run_ragas_library)(
            samples,
            config=run_config,
        )
    except Exception as exc:
        error = str(exc)
        logger.exception(
            "ragas_judge_batch_failed",
            extra={
                "error_type": type(exc).__name__,
                "error": error,
                "evaluator_model_id": run_config.evaluator_model_id,
                "embedding_model_id": run_config.embedding_model_id,
                "metric_names": list(run_config.metric_names),
                "sample_count": len(samples),
            },
        )
        return RagasEvaluationSummary(
            total_questions=len(samples),
            completed_questions=0,
            failed_questions=len(samples),
            evaluation_success_rate=0.0,
            first_error=error,
            results=[
                RagasQuestionResult(
                    id=sample.id,
                    question=sample.user_input,
                    answer=sample.response,
                    citations=sample.citations,
                    context_count=len(sample.retrieved_contexts),
                    error=error,
                    error_category=classify_ragas_error(exc),
                )
                for sample in samples
            ],
        )

    results: list[RagasQuestionResult] = []
    score_rows = list(scores)
    for index, sample in enumerate(samples):
        score = score_rows[index] if index < len(score_rows) else {}
        combined_score = {
            name: _as_float(value)
            for name, value in score.items()
        }
        citation_correctness = _citation_correctness(sample)
        if citation_correctness is not None:
            combined_score["citation_correctness"] = citation_correctness
        results.append(
            RagasQuestionResult(
                id=sample.id,
                question=sample.user_input,
                answer=sample.response,
                metrics=combined_score,
                citations=sample.citations,
                context_count=len(sample.retrieved_contexts),
                context_compression=sample.context_compression,
            )
        )

    # Recover only questions with missing judge metrics. A valid low score is
    # evidence, not an infrastructure failure, and must not trigger recovery.
    incomplete_indexes = [
        index
        for index, result in enumerate(results)
        if _missing_requested_metrics(result, run_config.metric_names)
    ]
    if incomplete_indexes and run_config.max_retries > 0:
        retry_samples = [samples[index] for index in incomplete_indexes]
        try:
            retry_scores = (metric_runner or _run_ragas_library)(
                retry_samples,
                config=run_config,
            )
            for offset, index in enumerate(incomplete_indexes):
                retry_score = retry_scores[offset] if offset < len(retry_scores) else {}
                for name, value in retry_score.items():
                    normalized = _as_float(value)
                    if normalized is not None:
                        results[index].metrics[name] = normalized
        except Exception as exc:
            logger.warning(
                "ragas_missing_metric_recovery_failed",
                extra={
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "affected_question_count": len(incomplete_indexes),
                },
            )

    aggregate: dict[str, float | None] = {}
    metric_names = sorted({name for result in results for name in result.metrics})
    for name in metric_names:
        values = [
            _as_float(result.metrics.get(name))
            for result in results
            if result.metrics.get(name) is not None
        ]
        values = [value for value in values if value is not None]
        aggregate[name] = round(mean(values), 4) if values else None

    failed_metric_counts: dict[str, int] = {}
    completed_questions = 0
    for result in results:
        missing = _missing_requested_metrics(result, run_config.metric_names)
        if result.error is None and not missing:
            completed_questions += 1
        elif result.error is None:
            result.error = (
                "Missing RAGAS metric scores after one retry: "
                + ", ".join(missing)
            )
            result.error_category = "missing_metric_scores"
        for name in missing:
            failed_metric_counts[name] = failed_metric_counts.get(name, 0) + 1
    failed_questions = len(results) - completed_questions
    return RagasEvaluationSummary(
        total_questions=len(samples),
        completed_questions=completed_questions,
        failed_questions=failed_questions,
        evaluation_success_rate=round(completed_questions / len(samples), 4),
        first_error=next((result.error for result in results if result.error), None),
        metrics=aggregate,
        failed_metric_counts=failed_metric_counts,
        results=results,
    )


def _run_ragas_library(
    samples: list[RagasEvaluationSample],
    *,
    config: RagasRunConfig,
) -> list[dict[str, float | None]]:
    """Run the supported RAGAS library lazily with optional Bedrock models."""
    if not config.evaluator_model_id or not config.embedding_model_id:
        raise RuntimeError(
            "RAGAS evaluator and embedding model IDs must be configured explicitly. "
            "Set PRA_RAGAS_EVALUATOR_MODEL_ID and PRA_RAGAS_EMBEDDING_MODEL_ID."
        )
    try:
        from ragas import evaluate
        from ragas.dataset_schema import EvaluationDataset
        from ragas.run_config import RunConfig
    except ImportError as exc:
        raise RuntimeError(
            "RAGAS worker imports failed. Missing or incompatible dependency: "
            f"{exc}. Rebuild the deployed RAGAS worker image from "
            "Dockerfile.ragas-worker."
        ) from exc

    rows = [
        {
            "user_input": sample.user_input,
            "response": sample.response,
            "retrieved_contexts": sample.retrieved_contexts,
            "reference": sample.reference,
        }
        for sample in samples
    ]
    # Keep evaluation data in RAGAS's native Pydantic-backed dataset. Passing a
    # Hugging Face Dataset here routes through PyArrow and can fail while
    # serializing extension types such as MonthDayNano in the worker runtime.
    dataset = EvaluationDataset.from_list(rows)
    llm, embeddings = _build_models(config)
    metrics = _build_metrics(
        llm=llm,
        embeddings=embeddings,
        metric_names=config.metric_names,
    )
    kwargs: dict[str, Any] = {"dataset": dataset, "metrics": metrics}
    kwargs["run_config"] = RunConfig(
        max_retries=max(0, min(config.max_retries, 1)),
        timeout=max(1, config.metric_timeout_seconds),
    )
    if llm is not None:
        kwargs["llm"] = llm
    if embeddings is not None:
        kwargs["embeddings"] = embeddings

    evaluation = evaluate(**kwargs)
    return _result_rows(evaluation, len(samples), metric_names=config.metric_names)


def _build_metrics(
    *,
    llm: Any | None = None,
    embeddings: Any | None = None,
    metric_names: tuple[str, ...] = STANDARD_RAGAS_METRICS,
) -> list[Any]:
    try:
        # The evaluate() API validates against the legacy Metric interface.
        # These exports are deprecated in RAGAS 0.4, but remain the compatible
        # metric objects for this evaluation entry point.
        from ragas.metrics import (  # type: ignore[attr-defined]
            answer_correctness,
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        metrics_by_name = {
            "faithfulness": faithfulness,
            "context_precision": context_precision,
            "context_recall": context_recall,
            "answer_relevancy": answer_relevancy,
            "answer_correctness": answer_correctness,
        }
        return [metrics_by_name[name] for name in metric_names]
    except ImportError:
        from ragas.metrics.collections import (  # type: ignore[attr-defined]
            AnswerCorrectness,
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )

        metric_types = {
            "faithfulness": Faithfulness,
            "context_precision": ContextPrecision,
            "context_recall": ContextRecall,
            "answer_relevancy": AnswerRelevancy,
            "answer_correctness": AnswerCorrectness,
        }
        return [
            _instantiate_metric(metric_types[name], llm=llm, embeddings=embeddings)
            for name in metric_names
        ]


def _instantiate_metric(metric_type: Any, *, llm: Any | None, embeddings: Any | None) -> Any:
    """Initialize a Collections metric across compatible RAGAS 0.4 releases."""
    parameters = inspect.signature(metric_type).parameters
    kwargs = {
        name: value
        for name, value in {"llm": llm, "embeddings": embeddings}.items()
        if value is not None and name in parameters
    }
    return metric_type(**kwargs)


def _build_models(config: RagasRunConfig) -> tuple[Any | None, Any | None]:
    if not config.evaluator_model_id and not config.embedding_model_id:
        return None, None
    try:
        from ragas.embeddings.base import embedding_factory
        from ragas.llms import llm_factory
    except ImportError as exc:
        raise RuntimeError(
            "RAGAS Bedrock evaluation requires the RAGAS model factories and LiteLLM."
        ) from exc

    def bounded_completion(*args, **kwargs):
        # RAGAS's RunConfig bounds metric tasks, but the provider callback also
        # needs a network timeout so one Bedrock request cannot consume the
        # entire Lambda invocation.
        kwargs.setdefault("timeout", max(1, config.metric_timeout_seconds))
        return ragas_completion(*args, **kwargs)

    llm = (
        llm_factory(
            config.evaluator_model_id,
            provider="litellm",
            client=bounded_completion,
            system_prompt=RAGAS_STRUCTURED_OUTPUT_SYSTEM_PROMPT,
            max_tokens=config.evaluator_max_tokens,
            max_retries=max(0, min(config.max_retries, 1)),
        )
        if config.evaluator_model_id
        else None
    )
    embeddings = None
    if config.embedding_model_id:
        raw_embeddings = embedding_factory("litellm", model=config.embedding_model_id)
        if hasattr(raw_embeddings, "embed_query") and hasattr(raw_embeddings, "embed_documents"):
            embeddings = raw_embeddings
        else:
            from ragas.embeddings.base import LangchainEmbeddingsWrapper

            embeddings = LangchainEmbeddingsWrapper(
                _LegacyEmbeddingAdapter(raw_embeddings)
            )
    return llm, embeddings


def _result_rows(
    evaluation: Any,
    count: int,
    *,
    metric_names: tuple[str, ...] = STANDARD_RAGAS_METRICS,
) -> list[dict[str, float | None]]:
    columns: list[str] = []
    if hasattr(evaluation, "to_pandas"):
        dataframe = evaluation.to_pandas()
        columns = [str(column) for column in dataframe.columns]
        rows = dataframe.to_dict(orient="records")
    elif isinstance(evaluation, dict):
        rows = [evaluation]
        columns = [str(column) for column in evaluation]
    else:
        rows = []
    logger.info(
        "ragas_result_schema",
        extra={
            "columns": columns,
            "row_count": len(rows),
            "metric_names": list(metric_names),
        },
    )
    if rows and not set(columns).intersection(metric_names):
        raise ValueError(
            "RAGAS evaluation returned rows but none of the requested metric columns were found. "
            f"Requested metrics: {', '.join(metric_names)}. "
            f"Returned columns: {', '.join(columns) or '<none>'}."
        )
    selected_metric_names = set(metric_names)
    normalized: list[dict[str, float | None]] = []
    for row in rows[:count]:
        normalized.append(
            {
                name: _as_float(row.get(name))
                for name in sorted(selected_metric_names)
                if name in row
            }
        )
    while len(normalized) < count:
        normalized.append({})
    return normalized


def _citation_correctness(sample: RagasEvaluationSample) -> float | None:
    if not sample.citations:
        return 0.0
    source_ids = {
        str(detail.get("document_id", ""))
        for detail in sample.retrieval_details
        if str(detail.get("document_id", "")).strip()
    }
    if not source_ids:
        return None
    aligned = sum(1 for citation in sample.citations if citation in source_ids)
    return round(aligned / len(sample.citations), 4)


def classify_ragas_error(exc: BaseException) -> str:
    text = str(exc).lower()
    if "no context" in text or ("retrieval" in text and "context" in text):
        return "no_context"
    if "credential" in text or "accessdenied" in text or "access denied" in text:
        return "bedrock_auth"
    if "throttl" in text or "rate exceeded" in text:
        return "bedrock_throttle"
    if "import" in text or "dependency" in text or "incompatible" in text:
        return "dependency_incompatibility"
    if "embed" in text:
        return "embedding_initialization"
    if "json" in text or "structured" in text or "parse" in text:
        return "structured_output"
    if "metric" in text or "evaluate" in text:
        return "metric_runtime"
    if "model" in text or "llm" in text:
        return "evaluator_initialization"
    return "unknown"


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
        return round(parsed, 4) if math.isfinite(parsed) else None
    except (TypeError, ValueError):
        return None


def _has_valid_requested_metric(
    result: RagasQuestionResult,
    metric_names: tuple[str, ...],
) -> bool:
    return all(
        result.metrics.get(name) is not None
        and isinstance(result.metrics.get(name), (int, float))
        and math.isfinite(float(result.metrics[name]))
        for name in metric_names
    )


def _missing_requested_metrics(
    result: RagasQuestionResult,
    metric_names: tuple[str, ...],
) -> list[str]:
    return [
        name
        for name in metric_names
        if not (
            result.metrics.get(name) is not None
            and isinstance(result.metrics.get(name), (int, float))
            and math.isfinite(float(result.metrics[name]))
        )
    ]
