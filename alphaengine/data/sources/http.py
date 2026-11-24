"""Polite, retrying HTTP fetcher shared by vendor adapters."""

from __future__ import annotations

import time
from collections.abc import Callable

import httpx

from alphaengine.core.errors import DataSourceError
from alphaengine.core.logging import get_logger
from alphaengine.core.retry import retry

log = get_logger(__name__)

# Generic browser-like headers; NSE rejects default library user agents.
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
}


class _RetryableHTTPError(Exception):
    pass


class HttpFetcher:
    """GET with retries on transport errors / 429 / 5xx. ``404`` returns ``None`` (no data)."""

    def __init__(
        self,
        client: httpx.Client | None = None,
        *,
        warmup_url: str | None = None,
        min_interval: float = 0.35,
        attempts: int = 4,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._client = client or httpx.Client(
            headers=DEFAULT_HEADERS, timeout=30.0, follow_redirects=True
        )
        self._warmup_url = warmup_url
        self._warmed = False
        self._min_interval = min_interval
        self._sleep = sleep
        self._monotonic = monotonic
        self._last: float | None = None
        self._get_with_retry = retry(
            attempts=attempts,
            base_delay=1.0,
            exceptions=(httpx.TransportError, _RetryableHTTPError),
            sleep=sleep,
        )(self._get_once)

    def close(self) -> None:
        self._client.close()

    def _throttle(self) -> None:
        now = self._monotonic()
        if self._last is not None:
            wait = self._min_interval - (now - self._last)
            if wait > 0:
                self._sleep(wait)
        self._last = self._monotonic()

    def _warmup(self) -> None:
        if self._warmup_url and not self._warmed:
            self._warmed = True
            try:
                self._client.get(self._warmup_url)  # collects session cookies
            except httpx.HTTPError as exc:
                log.warning("warmup failed", url=self._warmup_url, error=repr(exc))

    def _get_once(self, url: str) -> bytes | None:
        self._throttle()
        resp = self._client.get(url)
        if resp.status_code == 404:
            return None
        if resp.status_code == 429 or resp.status_code >= 500:
            raise _RetryableHTTPError(f"HTTP {resp.status_code} for {url}")
        if resp.status_code != 200:
            raise DataSourceError(f"HTTP {resp.status_code} for {url}")
        return resp.content

    def get(self, url: str) -> bytes | None:
        self._warmup()
        try:
            return self._get_with_retry(url)
        except (httpx.TransportError, _RetryableHTTPError) as exc:
            raise DataSourceError(f"GET {url} failed after retries: {exc!r}") from exc
