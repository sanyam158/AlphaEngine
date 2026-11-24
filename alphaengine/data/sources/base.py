"""Source protocols. Every adapter returns frames in the canonical schema (``data.schema``)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Protocol

import pandas as pd

from alphaengine.core.types import Symbol


class DailyBarSource(Protocol):
    """Anything that can produce raw daily bars for a date range."""

    name: str

    def fetch_bars(
        self, start: date, end: date, symbols: Sequence[Symbol] | None = None
    ) -> pd.DataFrame:
        """Canonical bars frame (possibly empty), sorted by (date, symbol)."""
        ...


@dataclass(frozen=True)
class RawPayload:
    """A raw vendor file for one trading day, as downloaded."""

    source: str
    day: date
    name: str  # file name used in the raw store
    fmt: str  # parser selector, e.g. "legacy" / "udiff"
    content: bytes


class DailyFileSource(Protocol):
    """A source that publishes one whole-market file per trading day (e.g. NSE bhavcopy)."""

    name: str

    def fetch_raw(self, day: date) -> RawPayload | None:
        """Download the file for ``day``; ``None`` when the exchange published nothing."""
        ...

    def parse(self, payload: RawPayload) -> pd.DataFrame:
        """Parse a payload into a canonical (unvalidated) bars frame."""
        ...


CORPORATE_ACTION_COLUMNS = ["symbol", "ex_date", "kind", "ratio", "amount", "source"]
