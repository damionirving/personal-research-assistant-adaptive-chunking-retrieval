from fastapi import HTTPException, Request

from pra_core.auth.base import Principal
from pra_core.config import Settings
from pra_core.rate_limit import FixedWindowRateLimiter, RateLimitExceeded, RateLimitPolicy


def enforce_chat_rate_limit(
    request: Request,
    principal: Principal,
    limiter: FixedWindowRateLimiter,
) -> None:
    settings: Settings = request.app.state.settings
    _enforce(
        request=request,
        limiter=limiter,
        subject=principal.subject,
        route="/chat",
        policy=RateLimitPolicy(
            name="chat",
            max_requests=settings.rate_limit_chat_requests,
            window_seconds=settings.rate_limit_chat_window_seconds,
        ),
    )


def enforce_upload_rate_limit(
    request: Request,
    principal: Principal,
    limiter: FixedWindowRateLimiter,
) -> None:
    settings: Settings = request.app.state.settings
    _enforce(
        request=request,
        limiter=limiter,
        subject=principal.subject,
        route="/documents/upload",
        policy=RateLimitPolicy(
            name="upload",
            max_requests=settings.rate_limit_upload_requests,
            window_seconds=settings.rate_limit_upload_window_seconds,
        ),
    )


def enforce_ragas_rate_limit(
    request: Request,
    principal: Principal,
    limiter: FixedWindowRateLimiter,
) -> None:
    settings: Settings = request.app.state.settings
    _enforce(
        request=request,
        limiter=limiter,
        subject=principal.subject,
        route="/rag/evaluate/ragas",
        policy=RateLimitPolicy(
            name="ragas",
            max_requests=settings.rate_limit_ragas_requests,
            window_seconds=settings.rate_limit_ragas_window_seconds,
        ),
    )


def enforce_public_rate_limit(
    request: Request,
    limiter: FixedWindowRateLimiter,
    route: str,
) -> None:
    settings: Settings = request.app.state.settings
    _enforce(
        request=request,
        limiter=limiter,
        subject=_client_identifier(request),
        route=route,
        policy=RateLimitPolicy(
            name="public",
            max_requests=settings.rate_limit_public_requests,
            window_seconds=settings.rate_limit_public_window_seconds,
        ),
    )


def _enforce(
    request: Request,
    limiter: FixedWindowRateLimiter,
    subject: str,
    route: str,
    policy: RateLimitPolicy,
) -> None:
    try:
        limiter.enforce(policy=policy, subject=subject, route=route)
    except RateLimitExceeded as exc:
        raise HTTPException(
            status_code=429,
            detail=str(exc),
            headers={"Retry-After": str(exc.retry_after_seconds)},
        ) from exc


def _client_identifier(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        first_ip = forwarded_for.split(",")[0].strip()
        if first_ip:
            return first_ip
    client = request.client
    if client and client.host:
        return client.host
    return "unknown-client"
