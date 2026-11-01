from collections.abc import Mapping
from datetime import datetime
from typing import Protocol

from pra_core.auth.base import AuthError, AuthProvider, Principal
from pra_core.config import Settings


class TokenVerifier(Protocol):
    def verify(self, token: str) -> Mapping[str, object]:
        raise NotImplementedError


class PyJwtCognitoTokenVerifier:
    """Verifies Cognito JWTs against the user pool JWKS endpoint."""

    def __init__(self, issuer: str, client_id: str) -> None:
        self._issuer = issuer.rstrip("/")
        self._client_id = client_id
        self._jwks_url = f"{self._issuer}/.well-known/jwks.json"
        self._jwks_client = None

    def verify(self, token: str) -> Mapping[str, object]:
        try:
            import jwt
            from jwt import PyJWKClient
        except ImportError as exc:
            raise AuthError("JWT verification dependency is not installed.") from exc

        if self._jwks_client is None:
            self._jwks_client = PyJWKClient(self._jwks_url)

        try:
            signing_key = self._jwks_client.get_signing_key_from_jwt(token)
            return jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                audience=self._client_id,
                issuer=self._issuer,
                options={"require": ["exp", "iat", "iss"]},
            )
        except Exception as exc:
            raise AuthError(sign_in_again_message()) from exc


class CognitoAuthProvider(AuthProvider):
    """Authenticates requests using Cognito-issued JWT bearer tokens."""

    def __init__(
        self,
        settings: Settings,
        token_verifier: TokenVerifier | None = None,
    ) -> None:
        issuer = settings.cognito_issuer
        client_id = settings.cognito_client_id
        self._configuration_error = (
            "Cognito auth requires issuer and client ID settings."
            if token_verifier is None and (not issuer or not client_id)
            else None
        )

        self._token_verifier = token_verifier or (
            PyJwtCognitoTokenVerifier(
                issuer=issuer or "",
                client_id=client_id or "",
            )
        )

    def authenticate(self, authorization_header: str | None) -> Principal:
        if self._configuration_error:
            raise AuthError(self._configuration_error)

        if not authorization_header:
            raise AuthError("Missing authorization token.")

        scheme, _, token = authorization_header.partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            raise AuthError("Authorization header must use a bearer token.")

        claims = self._token_verifier.verify(token.strip())
        subject = _string_claim(claims, "sub")
        if not subject:
            raise AuthError("Cognito token is missing a subject.")

        roles = _roles_from_claims(claims)
        return Principal(subject=subject, roles=roles)


def _string_claim(claims: Mapping[str, object], name: str) -> str | None:
    value = claims.get(name)
    return value if isinstance(value, str) and value else None


def _roles_from_claims(claims: Mapping[str, object]) -> tuple[str, ...]:
    groups = claims.get("cognito:groups")
    if isinstance(groups, list):
        return tuple(group for group in groups if isinstance(group, str))
    return ("cognito",)


def sign_in_again_message() -> str:
    return f"Please Sign in Again. {datetime.now().strftime('%m/%d/%Y %I:%M %p')}"
