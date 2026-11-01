from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Principal:
    subject: str
    roles: tuple[str, ...] = ()


class AuthError(ValueError):
    """Raised when request credentials are missing or invalid."""


class AuthProvider(ABC):
    @abstractmethod
    def authenticate(self, authorization_header: str | None) -> Principal:
        raise NotImplementedError
