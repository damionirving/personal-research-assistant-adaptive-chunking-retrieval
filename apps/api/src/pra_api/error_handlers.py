from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse

from pra_core.logging import get_logger, trace_id_context


def _trace_id(request: Request) -> str | None:
    return trace_id_context.get() or getattr(request.state, "trace_id", None)


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request,
        exc: StarletteHTTPException,
    ) -> JSONResponse:
        trace_id = _trace_id(request)
        get_logger(__name__).warning(
            "http_error",
            extra={
                "trace_id": trace_id,
                "http_method": request.method,
                "http_path": request.url.path,
                "http_status_code": exc.status_code,
            },
        )
        return JSONResponse(
            status_code=exc.status_code,
            headers=exc.headers,
            content={"error": exc.detail, "trace_id": trace_id},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        trace_id = _trace_id(request)
        get_logger(__name__).warning(
            "validation_error",
            extra={
                "trace_id": trace_id,
                "http_method": request.method,
                "http_path": request.url.path,
                "http_status_code": 422,
            },
        )
        return JSONResponse(
            status_code=422,
            content={"error": "Validation failed.", "details": exc.errors(), "trace_id": trace_id},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        trace_id = _trace_id(request)
        get_logger(__name__).exception(
            "unhandled_error",
            extra={
                "trace_id": trace_id,
                "http_method": request.method,
                "http_path": request.url.path,
                "http_status_code": 500,
            },
        )
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error.", "trace_id": trace_id},
        )
