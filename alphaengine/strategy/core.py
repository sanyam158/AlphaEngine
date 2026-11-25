"""Strategy primitives: the target portfolio a strategy emits, the context it sees, schedules."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol

import pandas as pd

from alphaengine.core.errors import StrategyError
from alphaengine.features.data import MarketData

WEIGHT_TOL = 1e-6


@dataclass(frozen=True)
class TargetPortfolio:
    """Long-only target weights at the close of ``as_of``. Unallocated weight is cash."""

    as_of: pd.Timestamp
    weights: Mapping[str, float]
    rationale: Mapping[str, list[str]] = field(default_factory=dict)  # symbol -> reasons
    notes: tuple[str, ...] = ()  # portfolio-level notes (e.g. regime switches)
    strategy: str = ""

    def __post_init__(self) -> None:
        bad = {s: w for s, w in self.weights.items() if w < -WEIGHT_TOL}
        if bad:
            raise StrategyError(f"negative weights are not allowed (long-only): {bad}")
        total = sum(self.weights.values())
        if total > 1 + WEIGHT_TOL:
            raise StrategyError(f"weights sum to {total:.6f} > 1 (no leverage)")

    @property
    def invested(self) -> float:
        return float(sum(self.weights.values()))

    @property
    def cash(self) -> float:
        return max(0.0, 1.0 - self.invested)

    def describe(self) -> str:
        lines = [
            f"{self.strategy} @ {self.as_of.date()}: {len(self.weights)} positions, "
            f"cash {self.cash:.1%}"
        ]
        lines += [f"  note: {n}" for n in self.notes]
        for sym, w in sorted(self.weights.items(), key=lambda kv: -kv[1]):
            why = "; ".join(self.rationale.get(sym, []))
            lines.append(f"  {sym:<14} {w:6.2%}  {why}")
        return "\n".join(lines)


@dataclass
class StrategyContext:
    as_of: pd.Timestamp
    data: MarketData
    current_weights: Mapping[str, float] = field(default_factory=dict)
    rationale: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    notes: list[str] = field(default_factory=list)

    def explain(self, symbol: str, reason: str) -> None:
        self.rationale[symbol].append(reason)

    def note(self, text: str) -> None:
        self.notes.append(text)


class Strategy(Protocol):
    name: str

    def target(self, ctx: StrategyContext) -> TargetPortfolio: ...

    def rebalance_dates(self, dates: pd.DatetimeIndex) -> list[pd.Timestamp]: ...


Frequency = Literal["daily", "weekly", "monthly", "quarterly", "yearly"]


@dataclass(frozen=True)
class Schedule:
    """Rebalance on the last trading day of each period (signals use that day's close)."""

    freq: Frequency = "monthly"

    def dates(self, trading_days: pd.DatetimeIndex) -> list[pd.Timestamp]:
        if len(trading_days) == 0:
            return []
        if self.freq == "daily":
            return list(trading_days)
        idx = pd.Series(trading_days, index=trading_days)
        key = {
            "weekly": lambda d: (d.isocalendar().year, d.isocalendar().week),
            "monthly": lambda d: (d.year, d.month),
            "quarterly": lambda d: (d.year, (d.month - 1) // 3),
            "yearly": lambda d: (d.year,),
        }[self.freq]
        groups = idx.groupby(idx.map(key)).max()
        # Note: the final period may still be in progress, so its "last day" is just the last
        # day in the data. Backtests should treat it accordingly.
        return sorted(pd.Timestamp(x) for x in groups)
