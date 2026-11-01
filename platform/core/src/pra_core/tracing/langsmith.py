import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter
from typing import Any
from uuid import UUID, uuid4

from pra_core.config import Settings
from pra_core.logging import get_logger

try:
    from langsmith import Client as LangSmithClient
except ImportError:  # pragma: no cover - dependency is installed in normal app flows
    LangSmithClient = None


@dataclass(frozen=True)
class TraceRun:
    trace_id: str
    name: str
    started_at: float
    run_id: str | None = None
    langsmith_trace_id: str | None = None
    parent_run_id: str | None = None
    dotted_order: str | None = None
    run_type: str = "chain"


class LangSmithTracer(ABC):
    @abstractmethod
    def start_run(
        self,
        name: str,
        trace_id: str,
        inputs: dict[str, Any],
        run_type: str = "chain",
        parent: TraceRun | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> TraceRun:
        raise NotImplementedError

    @abstractmethod
    def end_run(
        self,
        run: TraceRun,
        outputs: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        raise NotImplementedError


class NoOpLangSmithTracer(LangSmithTracer):
    """Local tracer that preserves LangSmith boundaries without network calls."""

    def __init__(self, project_name: str) -> None:
        self._project_name = project_name
        self._logger = get_logger(__name__)

    def start_run(
        self,
        name: str,
        trace_id: str,
        inputs: dict[str, Any],
        run_type: str = "chain",
        parent: TraceRun | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> TraceRun:
        self._logger.info(
            "trace_started",
            extra={
                "trace_id": trace_id,
                "trace_name": name,
                "trace_run_type": run_type,
                "trace_parent_run_id": parent.run_id if parent else None,
                "langsmith_project": self._project_name,
                "langsmith_tags": tags or [],
                "input_keys": sorted(inputs.keys()),
                "metadata_keys": sorted((metadata or {}).keys()),
            },
        )
        return TraceRun(
            trace_id=trace_id,
            name=name,
            started_at=perf_counter(),
            parent_run_id=parent.run_id if parent else None,
            run_type=run_type,
        )

    def end_run(
        self,
        run: TraceRun,
        outputs: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        duration_ms = (perf_counter() - run.started_at) * 1000
        self._logger.info(
            "trace_finished",
            extra={
                "trace_id": run.trace_id,
                "trace_name": run.name,
                "trace_run_type": run.run_type,
                "trace_parent_run_id": run.parent_run_id,
                "langsmith_project": self._project_name,
                "duration_ms": round(duration_ms, 2),
                "output_keys": sorted((outputs or {}).keys()),
                "error": error,
            },
        )


class HostedLangSmithTracer(LangSmithTracer):
    """Real LangSmith tracer that keeps app execution safe if tracing fails."""

    def __init__(
        self,
        project_name: str,
        service_name: str,
        environment: str,
        client: Any,
    ) -> None:
        self._project_name = project_name
        self._service_name = service_name
        self._environment = environment
        self._client = client
        self._logger = get_logger(__name__)
        self._base_tags = [
            self._service_name,
            f"environment:{self._environment}",
        ]

    def start_run(
        self,
        name: str,
        trace_id: str,
        inputs: dict[str, Any],
        run_type: str = "chain",
        parent: TraceRun | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> TraceRun:
        run_id = _generate_langsmith_run_id(trace_id if parent is None else None)
        start_time = datetime.now(UTC)
        langsmith_trace_id = (
            parent.langsmith_trace_id if parent and parent.langsmith_trace_id else run_id
        )
        current_dotted_order = _create_current_dotted_order(
            start_time=start_time,
            run_id=run_id,
        )
        dotted_order = (
            f"{parent.dotted_order}.{current_dotted_order}"
            if parent and parent.dotted_order
            else current_dotted_order
        )
        started_at = perf_counter()
        run = TraceRun(
            trace_id=trace_id,
            name=name,
            started_at=started_at,
            run_id=run_id,
            langsmith_trace_id=langsmith_trace_id,
            parent_run_id=parent.run_id if parent else None,
            dotted_order=dotted_order,
            run_type=run_type,
        )
        payload_metadata = {
            "app_trace_id": trace_id,
            "service_name": self._service_name,
            "environment": self._environment,
            **(metadata or {}),
        }
        payload_tags = [*self._base_tags, *(tags or [])]

        try:
            self._client.create_run(
                id=run_id,
                trace_id=langsmith_trace_id,
                parent_run_id=run.parent_run_id,
                dotted_order=dotted_order,
                name=name,
                inputs=inputs,
                run_type=run_type,
                project_name=self._project_name,
                start_time=start_time,
                extra={"metadata": payload_metadata},
                tags=payload_tags,
            )
            self._logger.info(
                "langsmith_run_created",
                extra={
                    "trace_id": trace_id,
                    "trace_name": name,
                    "trace_run_type": run_type,
                    "langsmith_run_id": run_id,
                    "langsmith_trace_id": langsmith_trace_id,
                    "trace_parent_run_id": run.parent_run_id,
                    "langsmith_dotted_order": dotted_order,
                    "langsmith_project": self._project_name,
                },
            )
        except Exception as exc:  # pragma: no cover - network failure path
            self._logger.warning(
                "langsmith_run_create_failed",
                extra={
                    "trace_id": trace_id,
                    "trace_name": name,
                    "trace_run_type": run_type,
                    "langsmith_project": self._project_name,
                    "langsmith_dotted_order": dotted_order,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
        return run

    def end_run(
        self,
        run: TraceRun,
        outputs: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        duration_ms = (perf_counter() - run.started_at) * 1000
        try:
            if run.run_id is not None:
                self._client.update_run(
                    run.run_id,
                    end_time=datetime.now(UTC),
                    outputs=outputs or {},
                    error=error,
                    trace_id=run.langsmith_trace_id,
                    dotted_order=run.dotted_order,
                    extra={
                        "metadata": {
                            "duration_ms": round(duration_ms, 2),
                            "app_trace_id": run.trace_id,
                            "service_name": self._service_name,
                            "environment": self._environment,
                        }
                    },
                )
            self._logger.info(
                "langsmith_run_finished",
                extra={
                    "trace_id": run.trace_id,
                    "trace_name": run.name,
                    "trace_run_type": run.run_type,
                    "langsmith_project": self._project_name,
                    "langsmith_run_id": run.run_id,
                    "langsmith_trace_id": run.langsmith_trace_id,
                    "langsmith_dotted_order": run.dotted_order,
                    "duration_ms": round(duration_ms, 2),
                    "error": error,
                },
            )
        except Exception as exc:  # pragma: no cover - network failure path
            self._logger.warning(
                "langsmith_run_update_failed",
                extra={
                    "trace_id": run.trace_id,
                    "trace_name": run.name,
                    "trace_run_type": run.run_type,
                    "langsmith_project": self._project_name,
                    "langsmith_run_id": run.run_id,
                    "langsmith_dotted_order": run.dotted_order,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )


def build_langsmith_tracer(
    settings: Settings,
    client: Any | None = None,
    secrets_client: Any | None = None,
) -> LangSmithTracer:
    project_name = _resolve_langsmith_project_name(settings)

    logger = get_logger(__name__)
    if settings.ragas_evaluation_worker:
        logger.info(
            "langsmith_disabled_for_ragas_evaluation_worker",
            extra={"langsmith_project": project_name},
        )
        return NoOpLangSmithTracer(project_name=project_name)

    # Tests must remain offline and deterministic. An explicitly injected
    # client or secrets client is still supported for tracer unit tests.
    if settings.environment == "test" and client is None and secrets_client is None:
        logger.info(
            "langsmith_disabled_for_test_environment",
            extra={"langsmith_project": project_name},
        )
        return NoOpLangSmithTracer(project_name=project_name)

    api_key, api_url = _resolve_langsmith_credentials(
        settings=settings,
        logger=logger,
        secrets_client=secrets_client,
    )

    if client is None and LangSmithClient is None:
        logger.warning(
            "langsmith_unavailable",
            extra={
                "langsmith_project": project_name,
                "reason": "langsmith_package_missing",
            },
        )
        return NoOpLangSmithTracer(project_name=project_name)

    if client is None and not api_key:
        logger.warning(
            "langsmith_unavailable",
            extra={
                "langsmith_project": project_name,
                "reason": "missing_langsmith_api_key",
            },
        )
        return NoOpLangSmithTracer(project_name=project_name)

    active_client = client
    if active_client is None:
        #active_client = LangSmithClient(api_key=api_key, api_url=api_url)

        active_client = LangSmithClient(
            api_key=api_key,
            api_url=api_url,
            auto_batch_tracing=False,
        )

    return HostedLangSmithTracer(
        project_name=project_name,
        service_name=settings.service_name,
        environment=settings.environment,
        client=active_client,
    )


def _resolve_langsmith_credentials(
    settings: Settings,
    logger,
    secrets_client: Any | None = None,
) -> tuple[str | None, str | None]:
    direct_api_key = os.getenv("LANGSMITH_API_KEY")
    direct_endpoint = os.getenv("LANGSMITH_ENDPOINT")

    if (
        settings.environment == "local"
        and direct_api_key
        and not _looks_like_placeholder_secret(direct_api_key)
    ):
        return direct_api_key, direct_endpoint

    try:
        secret_payload = _load_langsmith_secret_payload(
            secret_name="LANGSMITH_API_KEY",
            region_name=settings.aws_region,
            secrets_client=secrets_client,
        )
    except Exception as exc:
        logger.warning(
            "langsmith_unavailable",
            extra={
                "langsmith_project": settings.langsmith_project,
                "reason": "secret_lookup_failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        return None, direct_endpoint if settings.environment == "local" else None

    api_key = _extract_langsmith_api_key(secret_payload)
    endpoint = _extract_langsmith_endpoint(secret_payload)
    if settings.environment == "local" and direct_endpoint and direct_endpoint.strip():
        endpoint = direct_endpoint.strip()

    if not api_key or _looks_like_placeholder_secret(api_key):
        logger.warning(
            "langsmith_unavailable",
            extra={
                "langsmith_project": settings.langsmith_project,
                "reason": "secret_missing_api_key",
            },
        )
        return None, endpoint

    return api_key, endpoint


def _load_langsmith_secret_payload(
    secret_name: str,
    region_name: str,
    secrets_client: Any | None = None,
) -> str:
    active_client = secrets_client
    if active_client is None:
        import boto3

        active_client = boto3.client("secretsmanager", region_name=region_name)

    response = active_client.get_secret_value(SecretId=secret_name)
    return response.get("SecretString", "")


def _extract_langsmith_api_key(secret_payload: str) -> str | None:
    payload = secret_payload.strip()
    if not payload:
        return None

    if payload.startswith("{"):
        import json

        secret_data = json.loads(payload)
        for key_name in ("LANGSMITH_API_KEY", "langsmith_api_key", "api_key"):
            value = secret_data.get(key_name)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    return payload


def _extract_langsmith_endpoint(secret_payload: str) -> str | None:
    payload = secret_payload.strip()
    if not payload or not payload.startswith("{"):
        return None

    import json

    secret_data = json.loads(payload)
    for key_name in ("LANGSMITH_ENDPOINT", "langsmith_endpoint", "endpoint"):
        value = secret_data.get(key_name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _looks_like_placeholder_secret(value: str) -> bool:
    normalized = value.strip().lower()
    return normalized in {"replace-me", "changeme", "change-me", "placeholder"}


def _resolve_langsmith_project_name(settings: Settings) -> str:
    return settings.langsmith_project


def _create_current_dotted_order(start_time: datetime, run_id: str) -> str:
    return start_time.strftime("%Y%m%dT%H%M%S%fZ") + run_id


def _generate_langsmith_run_id(candidate: str | None = None) -> str:
    if candidate:
        try:
            return str(UUID(candidate))
        except ValueError:
            pass
    return str(uuid4())
