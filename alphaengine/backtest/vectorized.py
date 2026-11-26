"""Fast weight-based simulation for research (parameter sweeps, walk-forward).

Same timing as the full engine (signal at the close, trade at the next open), but on weights
instead of shares: total-return prices (dividends reinvested), fractional positions, costs charged
as ``turnover × cost_per_side``, and **no taxes**. A parity test keeps it in line with the full
engine. Use it to compare many variants quickly, then confirm finalists with
:func:`alphaengine.backtest.engine.simulate`, which models taxes, integer shares and fills.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

import numpy as np
import pandas as pd

from alphaengine.core.types import Side
from alphaengine.features.data import MarketData
from alphaengine.market.costs import CostModel
from alphaengine.strategy.core import Strategy, StrategyContext, TargetPortfolio

TRADING_DAYS = 252


def default_cost_per_side(model: CostModel | None = None, on: date | None = None) -> float:
    """Average buy/sell charges as a fraction of a ₹1 lakh order (Zerodha delivery ≈ 0.12%)."""
    m = model or CostModel()
    d = on or date.today()  # noqa: DTZ011 - rate lookup only
    qty, px = 100, Decimal("1000")
    buy = m.charges(Side.BUY, qty, px, d).total
    sell = m.charges(Side.SELL, qty, px, d).total
    return float((buy + sell) / 2 / (qty * px))


@dataclass
class VectorResult:
    strategy: str
    equity: pd.DataFrame  # value, flow, contributions (initial capital 1.0 by default)
    turnover: pd.Series  # sum |Δw| per rebalance day
    targets: list[TargetPortfolio] = field(default_factory=list)

    @property
    def returns(self) -> pd.Series:
        v = self.equity["value"]
        return v.pct_change().iloc[1:].fillna(0.0)

    @property
    def total_costs(self) -> float:
        return float(self.equity.attrs.get("costs", 0.0))


def run_vectorized(  # noqa: PLR0915 - one linear simulation loop
    strategy: Strategy,
    md: MarketData,
    start: date,
    end: date | None = None,
    *,
    cost_per_side: float | None = None,
    slippage_bps: float = 10.0,
    cash_rate: float = 0.06,
    capital: float = 1.0,
) -> VectorResult:
    if md.raw_open is None:
        raise ValueError("MarketData must be loaded with execution=True (needs open prices)")
    if getattr(strategy, "exits", None):
        raise ValueError("strategies with exit rules need the full engine (backtest run)")
    cost = (
        default_cost_per_side() if cost_per_side is None else cost_per_side
    ) + slippage_bps / 1e4
    dates = md.dates
    mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end or dates[-1]))
    days = dates[mask]
    if len(days) < 2:
        raise ValueError("need at least two trading days")
    cols = {s: i for i, s in enumerate(md.close.columns)}
    tr = md.tr_close.ffill().to_numpy(dtype=float)
    factor = (md.tr_close / md.raw_close).ffill().to_numpy(dtype=float)
    opens = md.raw_open.to_numpy(dtype=float) * factor
    row = {d: i for i, d in enumerate(dates)}
    rebalance = {days[0], *strategy.rebalance_dates(days)}
    rf = cash_rate / TRADING_DAYS

    value = capital
    w: dict[str, float] = {}
    pending: dict[str, float] | None = None
    values, turnover, targets = [], {}, []
    total_cost = 0.0

    def grow(weights: dict[str, float], ratio: dict[str, float], cash_growth: float) -> float:
        cash = 1.0 - sum(weights.values())
        g = cash * cash_growth + sum(x * ratio[s] for s, x in weights.items())
        for s, x in list(weights.items()):
            weights[s] = x * ratio[s] / g if g > 0 else 0.0
        return g

    for k, d in enumerate(days):
        i = row[d]
        if k > 0:
            p = i - 1
            if pending is not None:
                ratio_on = {s: _ratio(opens[i, cols[s]], tr[p, cols[s]]) for s in w}
                value *= grow(w, ratio_on, 1.0)
                trade = sum(abs(pending.get(s, 0.0) - w.get(s, 0.0)) for s in set(w) | set(pending))
                charged = value * trade * cost
                value -= charged
                total_cost += charged
                turnover[d] = trade
                w = {s: x for s, x in pending.items() if x > 0}
                pending = None
                ratio_id = {s: _ratio(tr[i, cols[s]], opens[i, cols[s]]) for s in w}
                value *= grow(w, ratio_id, 1.0 + rf)
            else:
                ratio = {s: _ratio(tr[i, cols[s]], tr[p, cols[s]]) for s in w}
                value *= grow(w, ratio, 1.0 + rf)
        values.append(value)
        if d in rebalance and k < len(days) - 1:
            t = strategy.target(StrategyContext(d, md, dict(w)))
            targets.append(t)
            pending = {s: x for s, x in t.weights.items() if s in cols}

    equity = pd.DataFrame(
        {"value": values, "flow": 0.0, "contributions": capital}, index=pd.DatetimeIndex(days)
    )
    equity.attrs["costs"] = total_cost
    return VectorResult(
        strategy=getattr(strategy, "name", "strategy"),
        equity=equity,
        turnover=pd.Series(turnover, dtype=float),
        targets=targets,
    )


def _ratio(now: float, before: float) -> float:
    if not (np.isfinite(now) and np.isfinite(before)) or before <= 0:
        return 1.0  # no print: assume unchanged
    return now / before
