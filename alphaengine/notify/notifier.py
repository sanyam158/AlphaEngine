"""Notifications: Telegram when configured, otherwise a local log file."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Protocol

import httpx

from alphaengine.config.settings import Settings
from alphaengine.core.clock import IST
from alphaengine.core.errors import AlphaEngineError
from alphaengine.core.logging import get_logger
from alphaengine.core.retry import retry

log = get_logger(__name__)
TELEGRAM_LIMIT = 4000


class NotifyError(AlphaEngineError):
    pass


class Notifier(Protocol):
    def send(self, text: str) -> None: ...


class LogNotifier:
    """Appends messages to a file (default when Telegram isn't configured)."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def send(self, text: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(tz=IST).strftime("%Y-%m-%d %H:%M")
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(f"--- {stamp}\n{text}\n")
        log.info("notification", chars=len(text), path=str(self.path))


class TelegramNotifier:
    def __init__(self, token: str, chat_id: str, client: httpx.Client | None = None) -> None:
        self._url = f"https://api.telegram.org/bot{token}/sendMessage"
        self._chat = chat_id
        self._client = client or httpx.Client(timeout=20)
        self._post = retry(attempts=3, base_delay=2.0, exceptions=(httpx.TransportError,))(
            self._client.post
        )

    def send(self, text: str) -> None:
        for chunk in _chunks(text, TELEGRAM_LIMIT):
            resp = self._post(
                self._url,
                json={"chat_id": self._chat, "text": chunk, "disable_web_page_preview": True},
            )
            if resp.status_code != 200:
                raise NotifyError(f"telegram HTTP {resp.status_code}: {resp.text[:200]}")


class MultiNotifier:
    def __init__(self, notifiers: Sequence[Notifier]) -> None:
        self.notifiers = list(notifiers)

    def send(self, text: str) -> None:
        errors = []
        for n in self.notifiers:
            try:
                n.send(text)
            except Exception as exc:  # one failing channel must not silence the others
                errors.append(repr(exc))
        if errors and len(errors) == len(self.notifiers):
            raise NotifyError("; ".join(errors))


def _chunks(text: str, size: int) -> list[str]:
    out, cur = [], ""
    for line in text.splitlines(keepends=True):
        if len(cur) + len(line) > size and cur:
            out.append(cur)
            cur = ""
        rest = line
        while len(rest) > size:
            out.append(rest[:size])
            rest = rest[size:]
        cur += rest
    if cur:
        out.append(cur)
    return out or [""]


def build_notifier(settings: Settings) -> Notifier:
    log_n = LogNotifier(settings.data_dir / "notifications.log")
    if settings.telegram_bot_token and settings.telegram_chat_id:
        tg = TelegramNotifier(
            settings.telegram_bot_token.get_secret_value(), settings.telegram_chat_id
        )
        return MultiNotifier([tg, log_n])
    return log_n
