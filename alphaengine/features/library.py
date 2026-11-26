"""Built-in causal features. All operate on full panels with rolling/shift operations, so row
*t* never depends on rows after *t* (verified by tests/unit/test_features.py).

Short gaps (a stock not trading for a few days) are forward-filled up to ``_FILL_LIMIT`` rows
before computing, so one missing print doesn't blank out a 12-month momentum value.
"""

from __future__ import annotations

from typing import Literal, cast

import numpy as np
import pandas as pd

from alphaengine.core.errors import StrategyError
from alphaengine.features.base import feature
from alphaengine.features.data import MarketData

_FILL_LIMIT = 5
TRADING_DAYS = 252
_MIN_VOL = 0.01
Price = Literal["close", "tr"]


def _prices(data: MarketData, price: Price) -> pd.DataFrame:
    p = data.tr_close if price == "tr" else data.close
    return p.ffill(limit=_FILL_LIMIT)


def _daily_returns(data: MarketData) -> pd.DataFrame:
    return _prices(data, "tr").pct_change(fill_method=None)


@feature("returns", history=lambda p: p["window"], window=1, price="tr")
def returns(data: MarketData, window: int, price: Price) -> pd.DataFrame:
    """Simple return over ``window`` rows."""
    return _prices(data, price).pct_change(window, fill_method=None)


@feature("momentum", history=lambda p: p["lookback"], lookback=252, skip=21, price="tr")
def momentum(data: MarketData, lookback: int, skip: int, price: Price) -> pd.DataFrame:
    """Return from ``lookback`` rows ago to ``skip`` rows ago (12-1 momentum by default)."""
    if not 0 <= skip < lookback:
        raise StrategyError("momentum needs 0 <= skip < lookback")
    p = _prices(data, price)
    return p.shift(skip) / p.shift(lookback) - 1


@feature("volatility", history=lambda p: p["window"], window=63)
def volatility(data: MarketData, window: int) -> pd.DataFrame:
    """Annualised standard deviation of daily total returns."""
    r = _daily_returns(data)
    vol = r.rolling(window, min_periods=int(window * 0.8)).std()
    return cast(pd.DataFrame, vol * np.sqrt(TRADING_DAYS))


@feature(
    "risk_adj_momentum",
    history=lambda p: max(p["lookback"], p["vol_window"]),
    lookback=252,
    skip=21,
    vol_window=126,
)
def risk_adj_momentum(data: MarketData, lookback: int, skip: int, vol_window: int) -> pd.DataFrame:
    """Momentum divided by annualised volatility (a Sharpe-like trend strength). Undefined (NaN)
    below 1% annual volatility, where the ratio explodes (e.g. liquid-fund ETFs)."""
    mom = momentum(data, lookback, skip, "tr")
    vol = volatility(data, vol_window)
    return mom / vol.where(vol >= _MIN_VOL)


@feature("max_abs_return", history=lambda p: p["window"], window=252)
def max_abs_return(data: MarketData, window: int) -> pd.DataFrame:
    """Largest absolute one-day return over ``window`` rows, on split-adjusted prices. A value
    far above NSE's 20% circuit usually means a missing corporate action (bad data)."""
    r = _prices(data, "close").pct_change(fill_method=None).abs()
    return r.rolling(window, min_periods=1).max()


@feature("sma", history=lambda p: p["window"], window=200)
def sma(data: MarketData, window: int) -> pd.DataFrame:
    """Simple moving average of the adjusted close."""
    return _prices(data, "close").rolling(window, min_periods=window).mean()


@feature("price_to_sma", history=lambda p: p["window"], window=200)
def price_to_sma(data: MarketData, window: int) -> pd.DataFrame:
    """Adjusted close / SMA − 1 (positive = above the moving average)."""
    p = _prices(data, "close")
    return p / p.rolling(window, min_periods=window).mean() - 1


@feature("high_proximity", history=lambda p: p["window"], window=252)
def high_proximity(data: MarketData, window: int) -> pd.DataFrame:
    """Adjusted close / highest close of the last ``window`` rows (1.0 = at the high)."""
    p = _prices(data, "close")
    return p / p.rolling(window, min_periods=window).max()


@feature("adv_value", history=lambda p: p["window"], window=63)
def adv_value(data: MarketData, window: int) -> pd.DataFrame:
    """Median daily traded value (₹) over ``window`` rows; days without trades count as 0."""
    listed = data.close.notna().cumsum() > 0
    v = data.value.where(data.value.notna(), 0.0).where(listed)
    return v.rolling(window, min_periods=window).median()


@feature("history_days", history=lambda p: 0)
def history_days(data: MarketData) -> pd.DataFrame:
    """Number of trading days with a price so far."""
    return data.close.notna().cumsum().astype(np.float32)


@feature("rsi", history=lambda p: p["window"] * 3, window=14)
def rsi(data: MarketData, window: int) -> pd.DataFrame:
    """Wilder's relative strength index (0–100)."""
    delta = _prices(data, "close").diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = gain / loss.where(loss > 0)
    out = 100 - 100 / (1 + rs)
    return out.where(loss > 0, 100.0).where(gain.notna())


@feature("atr", history=lambda p: p["window"] * 3, window=14)
def atr(data: MarketData, window: int) -> pd.DataFrame:
    """Average true range (Wilder), in adjusted price units. Needs OHLC data."""
    if data.high is None or data.low is None:
        raise StrategyError("atr needs OHLC panels: load MarketData with ohlc=True")
    prev = data.close.shift(1)
    tr = (
        pd.concat([data.high - data.low, (data.high - prev).abs(), (data.low - prev).abs()])
        .groupby(level=0)
        .max()
    )
    return tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


@feature("zscore", history=lambda p: p["window"], window=20)
def zscore(data: MarketData, window: int) -> pd.DataFrame:
    """(close − SMA) / rolling std: the Bollinger-band position."""
    p = _prices(data, "close")
    roll = p.rolling(window, min_periods=window)
    std = roll.std()
    return (p - roll.mean()) / std.where(std > 0)
