from time import perf_counter
from uuid import uuid4

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from pra_core.logging import get_logger, reset_trace_id, set_trace_id


class RootPathPrefixMiddleware(BaseHTTPMiddleware):
    """Normalizes API Gateway stage-prefixed paths before route matching."""

    def __init__(self, app, root_path: str = "") -> None:
        super().__init__(app)
        self.root_path = root_path.rstrip("/")

    async def dispatch(self, request: Request, call_next) -> Response:
        path = request.scope.get("path", "")
        if self.root_path and path.startswith("/static/"):
            request.scope["root_path"] = ""
        elif self.root_path:
            if path == self.root_path:
                request.scope["path"] = "/"
                request.scope["raw_path"] = b"/"
                request.scope["root_path"] = ""
            elif path.startswith(f"{self.root_path}/"):
                normalized_path = path[len(self.root_path) :]
                request.scope["path"] = normalized_path
                request.scope["raw_path"] = normalized_path.encode("utf-8")
                request.scope["root_path"] = ""

        return await call_next(request)


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """Adds trace IDs, request logs, and in-memory metrics."""

    async def dispatch(self, request: Request, call_next) -> Response:
        trace_id = request.headers.get("x-trace-id") or str(uuid4())
        request.state.trace_id = trace_id
        token = set_trace_id(trace_id)
        started_at = perf_counter()
        status_code = 500
        logger = get_logger(__name__)

        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception:
            logger.exception(
                "request_failed",
                extra={
                    "trace_id": trace_id,
                    "http_method": request.method,
                    "http_path": request.url.path,
                },
            )
            raise
        finally:
            duration_ms = (perf_counter() - started_at) * 1000
            request.app.state.metrics.record_request(
                latency_ms=duration_ms,
                is_error=status_code >= 500,
            )
            logger.info(
                "request_completed",
                extra={
                    "trace_id": trace_id,
                    "http_method": request.method,
                    "http_path": request.url.path,
                    "http_status_code": status_code,
                    "duration_ms": round(duration_ms, 2),
                    "metric_name": "http.server.duration",
                },
            )
            reset_trace_id(token)
            if "response" in locals():
                response.headers["x-trace-id"] = trace_id
