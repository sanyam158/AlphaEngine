"""Time handling. All internal timestamps are timezone-aware; the market timezone is IST.

Code should depend on the :class:`Clock` protocol rather than calling ``datetime.now``
directly, so backtests and tests can control time.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Protocol
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
UTC = ZoneInfo("UTC")


def to_ist(ts: datetime) -> datetime:
    """Convert an aware datetime to IST. Naive datetimes are rejected to avoid silent bugs."""
    if ts.tzinfo is None or ts.utcoffset() is None:
        raise ValueError("naive datetime; attach a timezone before converting")
    return ts.astimezone(IST)


class Clock(Protocol):
    def now(self) -> datetime:
        """Current aware datetime in IST."""
        ...

    def today(self) -> date:
        """Current calendar date in IST."""
        ...


class SystemClock:
    """Wall-clock time in IST."""

    def now(self) -> datetime:
        return datetime.now(tz=IST)

    def today(self) -> date:
        return self.now().date()


class FixedClock:
    """A manually controlled clock for tests and simulations."""

    def __init__(self, start: datetime) -> None:
        self._now = to_ist(start)

    def now(self) -> datetime:
        return self._now

    def today(self) -> date:
        return self._now.date()

    def advance(self, delta: timedelta) -> None:
        if delta < timedelta(0):
            raise ValueError("clock cannot move backwards")
        self._now = self._now + delta

    def set(self, ts: datetime) -> None:
        new = to_ist(ts)
        if new < self._now:
            raise ValueError("clock cannot move backwards")
        self._now = new
