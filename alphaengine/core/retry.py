"""Retry with exponential backoff for calls to external services (data vendors, brokers)."""

from __future__ import annotations

import functools
import random
import time
from collections.abc import Callable
from typing import ParamSpec, TypeVar

from alphaengine.core.logging import get_logger

P = ParamSpec("P")
R = TypeVar("R")

log = get_logger(__name__)


def retry(
    *,
    attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 10.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    sleep: Callable[[float], None] = time.sleep,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Retry ``fn`` on ``exceptions`` with jittered exponential backoff; re-raise the last error."""
    if attempts < 1:
        raise ValueError("attempts must be >= 1")

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            for attempt in range(1, attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:
                    if attempt == attempts:
                        raise
                    delay = min(max_delay, base_delay * 2 ** (attempt - 1))
                    delay *= 0.5 + random.random() / 2
                    log.warning(
                        "retrying",
                        fn=fn.__qualname__,
                        attempt=attempt,
                        delay=round(delay, 3),
                        error=repr(exc),
                    )
                    sleep(delay)
            raise AssertionError("unreachable")  # pragma: no cover

        return wrapper

    return decorator
