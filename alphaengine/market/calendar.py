"""NSE trading calendar.

Two sources of truth, in priority order:
1. **Observed trading days**: dates for which the exchange published a bhavcopy. Inside the
   observed range this is authoritative (it captures holidays, Muhurat and special sessions).
2. **Rules** for dates outside the observed range (e.g. the future): weekdays, minus the
   holidays and plus the special sessions listed in a CSV you maintain from NSE's annual
   holiday circular (``date,kind,description`` with kind ``holiday`` or ``special_session``).
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from datetime import date, timedelta
from pathlib import Path


class TradingCalendar:
    def __init__(
        self,
        holidays: Iterable[date] = (),
        special_sessions: Iterable[date] = (),
        observed_trading_days: Iterable[date] = (),
    ) -> None:
        self._holidays = frozenset(holidays)
        self._special = frozenset(special_sessions)
        self._observed = frozenset(observed_trading_days)
        self._obs_min = min(self._observed) if self._observed else None
        self._obs_max = max(self._observed) if self._observed else None

    @classmethod
    def from_csv(cls, path: Path, observed_trading_days: Iterable[date] = ()) -> TradingCalendar:
        holidays: list[date] = []
        special: list[date] = []
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                d = date.fromisoformat(row["date"].strip())
                kind = row.get("kind", "holiday").strip().lower()
                if kind == "holiday":
                    holidays.append(d)
                elif kind == "special_session":
                    special.append(d)
                else:
                    raise ValueError(f"unknown calendar kind {kind!r} for {d}")
        return cls(holidays, special, observed_trading_days)

    def with_observed(self, days: Iterable[date]) -> TradingCalendar:
        return TradingCalendar(self._holidays, self._special, self._observed | frozenset(days))

    def _in_observed_range(self, d: date) -> bool:
        return self._obs_min is not None and self._obs_min <= d <= self._obs_max  # type: ignore[operator]

    def is_trading_day(self, d: date) -> bool:
        if self._in_observed_range(d):
            return d in self._observed
        if d in self._special:
            return True
        return d.weekday() < 5 and d not in self._holidays

    def trading_days(self, start: date, end: date) -> list[date]:
        """All trading days in ``[start, end]``, ascending."""
        if end < start:
            return []
        n = (end - start).days + 1
        return [d for d in (start + timedelta(days=i) for i in range(n)) if self.is_trading_day(d)]

    def next_trading_day(self, d: date) -> date:
        cur = d + timedelta(days=1)
        for _ in range(30):
            if self.is_trading_day(cur):
                return cur
            cur += timedelta(days=1)
        raise ValueError(f"no trading day within 30 days after {d}")

    def previous_trading_day(self, d: date) -> date:
        cur = d - timedelta(days=1)
        for _ in range(30):
            if self.is_trading_day(cur):
                return cur
            cur -= timedelta(days=1)
        raise ValueError(f"no trading day within 30 days before {d}")


def candidate_days(start: date, end: date, *, include_weekends: bool = False) -> list[date]:
    """Days worth asking a data source about: weekdays (and weekends if requested)."""
    n = max((end - start).days + 1, 0)
    days = (start + timedelta(days=i) for i in range(n))
    return [d for d in days if include_weekends or d.weekday() < 5]
