"""Exit rules for rule-driven (swing) strategies.

In *rules mode* (a strategy with ``exits``), rebalancing only opens new positions into free
slots (``max_positions``, equal size). Existing positions are held until an exit rule fires.
Rules are checked on every trading day's close, and the sell happens at the next open.

All prices here are split-adjusted, so stops survive splits and bonuses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from alphaengine.core.errors import StrategyError
from alphaengine.features.base import FeatureSpec
from alphaengine.features.data import MarketData
from alphaengine.strategy.stages import component


@dataclass(frozen=True)
class PositionState:
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float  # adjusted close on the entry day
    peak: float  # highest adjusted close since entry
    close: float  # today's adjusted close
    days_held: int  # trading days since entry


def _atr(md: MarketData, symbol: str, as_of: pd.Timestamp, window: int) -> float:
    if md.high is None or md.low is None:
        raise StrategyError("ATR-based exits need OHLC data (loaded automatically in backtests)")
    v = md.feature_row(FeatureSpec.of("atr", window=window), as_of).get(symbol, np.nan)
    return float(v)


@component("exit", "stop_loss")
@dataclass(frozen=True)
class StopLoss:
    """Exit when the close falls ``pct`` below entry, or ``atr_mult`` ATRs below entry."""

    pct: float | None = None
    atr_mult: float | None = None
    atr_window: int = 14

    def __post_init__(self) -> None:
        if (self.pct is None) == (self.atr_mult is None):
            raise StrategyError("stop_loss needs exactly one of pct or atr_mult")

    def check(self, p: PositionState, md: MarketData, as_of: pd.Timestamp) -> str | None:
        if self.pct is not None:
            level = p.entry_price * (1 - self.pct)
        else:
            assert self.atr_mult is not None
            atr = _atr(md, p.symbol, as_of, self.atr_window)
            if not np.isfinite(atr):
                return None
            level = p.entry_price - self.atr_mult * atr
        return f"stop loss ({p.close:.2f} < {level:.2f})" if p.close < level else None


@component("exit", "trailing_stop")
@dataclass(frozen=True)
class TrailingStop:
    """Exit when the close falls ``pct`` (or ``atr_mult`` ATRs) below the highest close since
    entry: lets winners run, cuts reversals."""

    pct: float | None = None
    atr_mult: float | None = None
    atr_window: int = 14

    def __post_init__(self) -> None:
        if (self.pct is None) == (self.atr_mult is None):
            raise StrategyError("trailing_stop needs exactly one of pct or atr_mult")

    def check(self, p: PositionState, md: MarketData, as_of: pd.Timestamp) -> str | None:
        if self.pct is not None:
            level = p.peak * (1 - self.pct)
        else:
            assert self.atr_mult is not None
            atr = _atr(md, p.symbol, as_of, self.atr_window)
            if not np.isfinite(atr):
                return None
            level = p.peak - self.atr_mult * atr
        return f"trailing stop ({p.close:.2f} < {level:.2f})" if p.close < level else None


@component("exit", "take_profit")
@dataclass(frozen=True)
class TakeProfit:
    pct: float

    def check(self, p: PositionState, md: MarketData, as_of: pd.Timestamp) -> str | None:
        target = p.entry_price * (1 + self.pct)
        return f"take profit ({p.close:.2f} >= {target:.2f})" if p.close >= target else None


@component("exit", "time_stop")
@dataclass(frozen=True)
class TimeStop:
    """Exit after ``days`` trading days (keeps capital moving; caps exposure to stale trades)."""

    days: int

    def check(self, p: PositionState, md: MarketData, as_of: pd.Timestamp) -> str | None:
        return f"time stop ({p.days_held} days)" if p.days_held >= self.days else None


_OPS = {">": np.greater, ">=": np.greater_equal, "<": np.less, "<=": np.less_equal}


@component("exit", "feature")
@dataclass(frozen=True)
class FeatureExit:
    """Exit when a feature crosses a level, e.g. ``rsi(window=2) > 70`` or ``price_to_sma < 0``."""

    feature: str
    op: Literal[">", ">=", "<", "<="]
    value: float
    params: dict[str, Any] = field(default_factory=dict)

    def check(self, p: PositionState, md: MarketData, as_of: pd.Timestamp) -> str | None:
        v = float(
            md.feature_row(FeatureSpec.of(self.feature, **self.params), as_of).get(p.symbol, np.nan)
        )
        if np.isfinite(v) and bool(_OPS[self.op](v, self.value)):
            return f"{self.feature} {v:.2f} {self.op} {self.value}"
        return None
