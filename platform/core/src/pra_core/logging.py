import json
import logging
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

from pra_core.config import Settings

trace_id_context: ContextVar[str | None] = ContextVar("trace_id", default=None)
_cloudwatch_context: dict[str, str | None] = {}


class JsonFormatter(logging.Formatter):
    """Structured formatter for local logs and CloudWatch ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "trace_id": getattr(record, "trace_id", None) or trace_id_context.get(),
            "service": _cloudwatch_context.get("service"),
            "environment": _cloudwatch_context.get("environment"),
            "cloudwatch_log_group": _cloudwatch_context.get("log_group"),
            "cloudwatch_log_stream": _cloudwatch_context.get("log_stream"),
        }

        for key, value in record.__dict__.items():
            if key.startswith("_") or key in _RESERVED_LOG_RECORD_KEYS:
                continue
            payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


_RESERVED_LOG_RECORD_KEYS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
}


def configure_logging(settings: Settings) -> None:
    level = getattr(logging, settings.log_level)
    handler = logging.StreamHandler()
    _cloudwatch_context.update(
        {
            "service": settings.service_name,
            "environment": settings.environment,
            "log_group": settings.cloudwatch_log_group,
            "log_stream": settings.cloudwatch_log_stream,
        }
    )

    if settings.log_format == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s [%(name)s] %(message)s",
            )
        )

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def set_trace_id(trace_id: str | None):
    return trace_id_context.set(trace_id)


def reset_trace_id(token) -> None:
    trace_id_context.reset(token)
