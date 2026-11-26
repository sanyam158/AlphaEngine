"""How simulated orders get filled: slippage, liquidity caps, circuit limits, settlement.

These are deliberately simple and conservative; their job is to stop backtests from assuming
fills that couldn't have happened.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import Protocol

from alphaengine.core.types import Side

TICK = Decimal("0.05")  # NSE equity tick size (₹0.05 for most stocks)
T1_FROM = date(2023, 1, 27)  # T+1 settlement fully phased in for all NSE equities


class SlippageModel(Protocol):
    def slippage_bps(
        self, order_value: float, adv_value: float | None, daily_vol: float | None
    ) -> float:
        """Adverse price move in basis points for an order of ``order_value`` rupees."""
        ...


@dataclass(frozen=True)
class FixedBpsSlippage:
    bps: float = 10.0

    def slippage_bps(
        self, order_value: float, adv_value: float | None, daily_vol: float | None
    ) -> float:
        return self.bps


@dataclass(frozen=True)
class SqrtImpactSlippage:
    """Half-spread plus square-root market impact: ``base + k · σ_daily · sqrt(Q / ADV)``.

    ``k`` around 0.5–1 is a common empirical range. Without ADV/volatility data it falls back to
    ``fallback_bps``.
    """

    base_bps: float = 5.0
    k: float = 0.7
    fallback_bps: float = 25.0

    def slippage_bps(
        self, order_value: float, adv_value: float | None, daily_vol: float | None
    ) -> float:
        if not adv_value or adv_value <= 0 or daily_vol is None or daily_vol <= 0:
            return self.fallback_bps
        participation = max(order_value, 0.0) / adv_value
        return self.base_bps + self.k * daily_vol * math.sqrt(participation) * 10_000


def apply_slippage(price: Decimal, side: Side, bps: float) -> Decimal:
    """Move ``price`` against the trader by ``bps`` and round to the tick (away from trader)."""
    factor = Decimal(1) + (Decimal(str(bps)) / Decimal(10_000)) * (1 if side is Side.BUY else -1)
    raw = price * factor
    ticks = raw / TICK
    rounding = math.ceil if side is Side.BUY else math.floor
    return max(TICK, Decimal(rounding(ticks)) * TICK)


def max_fill_quantity(adv_shares: float | None, max_participation: float = 0.1) -> int | None:
    """Cap on shares fillable in one day (``None`` = unknown ADV, no cap)."""
    if adv_shares is None:
        return None
    if not 0 < max_participation <= 1:
        raise ValueError("max_participation must be in (0, 1]")
    return int(adv_shares * max_participation)


def circuit_locked(
    side: Side,
    *,
    open_: float,
    high: float,
    low: float,
    close: float,
    prev_close: float,
    min_move: float = 0.0195,
) -> bool:
    """Heuristic: a bar that traded at a single price after moving at least ``min_move`` (the
    smallest NSE band is 2%) was locked at a circuit, so there were no sellers (upper) or no
    buyers (lower). Buys are blocked at an upper circuit, sells at a lower one."""
    if prev_close <= 0 or not (high == low == close == open_):
        return False
    move = close / prev_close - 1
    if side is Side.BUY:
        return move >= min_move
    return move <= -min_move


def settlement_date(trade_date: date, trading_days_after: int | None = None) -> date:
    """Business-day settlement estimate (T+1 from 2023-01-27, T+2 before). Holidays are
    ignored; use the trading calendar when exactness matters."""
    n = (
        trading_days_after
        if trading_days_after is not None
        else (1 if trade_date >= T1_FROM else 2)
    )
    d = trade_date
    while n > 0:
        d += timedelta(days=1)
        if d.weekday() < 5:
            n -= 1
    return d


def shares_for_value(value: Decimal, price: Decimal) -> int:
    """Whole shares purchasable with ``value`` at ``price`` (no fractional shares on NSE)."""
    if price <= 0:
        raise ValueError("price must be positive")
    return int((value / price).to_integral_value(rounding="ROUND_FLOOR"))


def round_to_tick(price: Decimal) -> Decimal:
    return (price / TICK).quantize(Decimal(1), rounding=ROUND_HALF_UP) * TICK
