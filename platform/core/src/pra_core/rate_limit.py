from dataclasses import dataclass
import math
from threading import Lock
import time

from pra_core.logging import get_logger


logger = get_logger(__name__)


@dataclass(frozen=True)
class RateLimitPolicy:
    name: str
    max_requests: int
    window_seconds: int


class RateLimitExceeded(Exception):
    def __init__(self, message: str, retry_after_seconds: int) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class FixedWindowRateLimiter:
    """Process-local fixed-window limiter for the MVP."""

    def __init__(self, clock=time.time) -> None:
        self._clock = clock
        self._lock = Lock()
        self._windows: dict[str, tuple[float, int]] = {}

    def enforce(self, policy: RateLimitPolicy, subject: str, route: str) -> None:
        now = self._clock()
        key = f"{policy.name}:{subject}"
        with self._lock:
            window_started_at, request_count = self._windows.get(key, (now, 0))
            if now - window_started_at >= policy.window_seconds:
                window_started_at = now
                request_count = 0

            if request_count >= policy.max_requests:
                retry_after = max(
                    1,
                    math.ceil(policy.window_seconds - (now - window_started_at)),
                )
                logger.warning(
                    "rate_limit_exceeded",
                    extra={
                        "rate_limit_policy": policy.name,
                        "rate_limit_subject": subject,
                        "rate_limit_route": route,
                        "retry_after_seconds": retry_after,
                    },
                )
                raise RateLimitExceeded(
                    (
                        f"Rate limit exceeded for {route}. "
                        "Please wait and try again."
                    ),
                    retry_after_seconds=retry_after,
                )

            self._windows[key] = (window_started_at, request_count + 1)
