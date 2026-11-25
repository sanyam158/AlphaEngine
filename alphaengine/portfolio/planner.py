"""Rebalance planner: current holdings + target weights → the smallest sensible trade list.

- **No-trade band:** skip a position whose weight is within ``band`` of target.
- **Minimum trade:** skip trades smaller than ``min_trade_value``.
- **Tax-aware:** for each sell, preview the FIFO lots it would realise. If short-term lots
  turn long-term within ``ltcg_wait_days``, the trade gets a note with the estimated tax saved
  by waiting, and is deferred when ``defer_near_ltcg`` is on (exits included, trims only if
  ``defer_exits`` is off).
- Cash is budgeted: buys are scaled down so they never exceed cash (after sells) minus a buffer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from alphaengine.core.types import Side
from alphaengine.market.costs import CostModel
from alphaengine.market.tax import regime_on
from alphaengine.portfolio.lots import LotBook, add_months, is_long_term


@dataclass(frozen=True)
class PlannerSettings:
    band: float = 0.0
    min_trade_value: float = 2_000
    cash_buffer: float = 0.005
    ltcg_wait_days: int = 30
    defer_near_ltcg: bool = True
    defer_exits: bool = False  # also defer full exits (stop following the strategy for a while)


@dataclass(frozen=True)
class PlannedTrade:
    symbol: str
    side: Side
    quantity: int
    est_price: float
    reason: str
    note: str = ""

    @property
    def value(self) -> float:
        return self.quantity * self.est_price


@dataclass(frozen=True)
class Plan:
    trades: list[PlannedTrade]
    deferred: list[PlannedTrade]
    total_value: float
    est_charges: float

    def describe(self) -> str:
        lines = [
            f"{len(self.trades)} trades (est. charges ₹{self.est_charges:,.0f}), "
            f"{len(self.deferred)} deferred"
        ]
        for t in self.trades:
            lines.append(
                f"  {t.side.value:<4} {t.quantity:>6} {t.symbol:<14} @ ~₹{t.est_price:,.2f} "
                f"(₹{t.value:,.0f}) {t.reason}{' - ' + t.note if t.note else ''}"
            )
        for t in self.deferred:
            lines.append(f"  WAIT {t.quantity:>6} {t.symbol:<14} {t.note}")
        return "\n".join(lines)


def _ltcg_note(
    lots: LotBook, symbol: str, qty: int, price: float, on: date, wait_days: int
) -> tuple[str, float]:
    """(note, estimated tax saved by waiting) for selling ``qty`` FIFO shares on ``on``."""
    remaining, saving, soonest = qty, 0.0, None
    reg = regime_on(on)
    lt_rate = float(reg.ltcg) if reg.ltcg is not None else 0.0
    for lot in lots.lots(symbol):
        if remaining <= 0:
            break
        take = min(remaining, lot.quantity)
        remaining -= take
        if is_long_term(lot.acquired, on):
            continue
        becomes = add_months(lot.acquired, 12)
        gain = (price - float(lot.cost_per_share)) * take
        if gain > 0 and (becomes - on).days <= wait_days:
            saving += gain * (float(reg.stcg) - lt_rate)
            soonest = becomes if soonest is None else max(soonest, becomes)
    if saving <= 0 or soonest is None:
        return "", 0.0
    return f"short-term until {soonest}; waiting saves ~₹{saving:,.0f} tax", saving


def plan_rebalance(
    lots: LotBook,
    cash: float,
    prices: dict[str, float],
    target: dict[str, float],
    on: date,
    settings: PlannerSettings | None = None,
    cost_model: CostModel | None = None,
) -> Plan:
    s = settings or PlannerSettings()
    costs = cost_model or CostModel()
    held = {sym: lots.quantity(sym) for sym in lots.symbols()}
    missing = [sym for sym in set(held) | set(target) if not prices.get(sym)]
    if missing:
        raise ValueError(f"no price for {sorted(missing)}")
    total = cash + sum(q * prices[sym] for sym, q in held.items())
    sells, buys, deferred = [], [], []
    for sym in sorted(set(held) | set(target)):
        px, q = prices[sym], held.get(sym, 0)
        cw, tw = q * px / total if total else 0.0, target.get(sym, 0.0)
        if tw > 0 and cw > 0 and abs(tw - cw) < s.band:
            continue
        diff = tw * total - cw * total
        if tw == 0 and q > 0:
            qty, reason = q, "exit"
        elif abs(diff) < s.min_trade_value:
            continue
        elif diff < 0:
            qty, reason = min(q, math.ceil(-diff / px)), "trim"
        else:
            buys.append(
                PlannedTrade(sym, Side.BUY, int(diff // px), px, "buy" if q == 0 else "add")
            )
            continue
        note, saving = _ltcg_note(lots, sym, qty, px, on, s.ltcg_wait_days)
        trade = PlannedTrade(sym, Side.SELL, qty, px, reason, note)
        if saving > 0 and s.defer_near_ltcg and (reason == "trim" or s.defer_exits):
            deferred.append(trade)
        else:
            sells.append(trade)

    proceeds = sum(t.value for t in sells)
    sell_charges = sum(
        float(costs.charges(Side.SELL, t.quantity, Decimal(str(t.est_price)), on).total)
        for t in sells
        if t.quantity > 0
    )
    budget = (cash + proceeds - sell_charges) * (1 - s.cash_buffer)
    wanted = sum(t.value for t in buys)
    scale = min(1.0, budget / wanted) if wanted > 0 else 1.0
    scaled = [
        PlannedTrade(t.symbol, t.side, int(t.quantity * scale), t.est_price, t.reason, t.note)
        for t in buys
    ]
    scaled = [t for t in scaled if t.quantity > 0 and t.value >= s.min_trade_value]
    trades = [t for t in sells if t.quantity > 0] + scaled
    est = sum(
        float(costs.charges(t.side, t.quantity, Decimal(str(t.est_price)), on).total)
        for t in trades
    )
    return Plan(trades, deferred, total, est)
