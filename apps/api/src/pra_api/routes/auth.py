from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from pra_api.dependencies import get_rate_limiter
from pra_api.rate_limits import enforce_public_rate_limit
from pra_core.rate_limit import FixedWindowRateLimiter

router = APIRouter(prefix="/auth", tags=["auth"])


class AuthConfigResponse(BaseModel):
    provider: str
    cognito_client_id: str | None
    cognito_hosted_ui_base_url: str | None
    redirect_uri: str
    document_upload_enabled: bool
    retrieval_diagnostics_enabled: bool
    rag_evaluation_enabled: bool
    ragas_enabled: bool


@router.get("/config", response_model=AuthConfigResponse)
def auth_config(
    request: Request,
    rate_limiter: FixedWindowRateLimiter = Depends(get_rate_limiter),
) -> AuthConfigResponse:
    enforce_public_rate_limit(request=request, limiter=rate_limiter, route="/auth/config")
    settings = request.app.state.settings
    root_path = settings.api_root_path.rstrip("/")
    redirect_uri = str(request.url_for("index"))
    if root_path and redirect_uri.endswith(f"{root_path}/"):
        redirect_uri = redirect_uri

    return AuthConfigResponse(
        provider=settings.auth_provider,
        cognito_client_id=settings.cognito_client_id,
        cognito_hosted_ui_base_url=settings.cognito_hosted_ui_base_url,
        redirect_uri=redirect_uri,
        document_upload_enabled=settings.document_upload_enabled,
        retrieval_diagnostics_enabled=settings.retrieval_diagnostics_enabled,
        rag_evaluation_enabled=settings.rag_evaluation_enabled,
        ragas_enabled=settings.ragas_enabled,
    )
