"""Brokers.

- :class:`PaperBroker` fills open orders at the trading day's **open** (with slippage, circuit
  checks and real charges), like the backtester. Unfillable orders retry, then expire.
- :class:`ManualBroker`: *you* place the orders at your real broker and record each fill
  (quantity, price, optional actual charges from the contract note).

Both update the portfolio's tax lots and cash the same way (:func:`apply_fill`).
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path

from alphaengine.core.types import OrderStatus, Side, can_transition, make_symbol
from alphaengine.data.models import PortfolioOrder
from alphaengine.market.costs import ChargeBreakdown, CostModel
from alphaengine.market.execution_model import apply_slippage, circuit_locked
from alphaengine.portfolio.store import PortfolioError, PortfolioState


@dataclass(frozen=True)
class DayBar:
    open: float
    high: float
    low: float
    close: float
    prev_close: float | None


BarLookup = Callable[[str], DayBar | None]


def _transition(order: PortfolioOrder, status: OrderStatus) -> None:
    current = OrderStatus(order.status)
    if current != status and not can_transition(current, status):
        raise PortfolioError(f"order {order.id}: illegal transition {current} -> {status}")
    order.status = status.value


def apply_fill(
    state: PortfolioState,
    side: Side,
    symbol: str,
    qty: int,
    price: Decimal,
    day: date,
    charges: ChargeBreakdown,
) -> None:
    if side is Side.SELL:
        held = state.lots.quantity(symbol)
        if qty > held:
            raise PortfolioError(f"cannot sell {qty} {symbol}: only {held} held")
        state.new_disposals.extend(state.lots.sell(symbol, qty, price, day, charges))
        state.cash += price * qty - charges.total
    else:
        state.lots.buy(symbol, qty, price, day, charges)
        state.cash -= price * qty + charges.total


def _record(order: PortfolioOrder, qty: int, price: Decimal, charges: Decimal, day: date) -> None:
    prev_filled = order.filled_quantity
    total_qty = prev_filled + qty
    prev_px = Decimal(str(order.fill_price or 0))
    order.fill_price = float((prev_px * prev_filled + price * qty) / total_qty)
    order.filled_quantity = total_qty
    order.charges = float(Decimal(str(order.charges or 0)) + charges)
    order.filled_at = day
    if order.status == OrderStatus.NEW.value:
        _transition(order, OrderStatus.SUBMITTED)
    _transition(order, OrderStatus.FILLED if total_qty >= order.quantity else OrderStatus.PARTIAL)


class PaperBroker:
    def __init__(
        self, costs: CostModel | None = None, slippage_bps: float = 10.0, max_attempts: int = 5
    ) -> None:
        self.costs = costs or CostModel()
        self.slippage_bps = slippage_bps
        self.max_attempts = max_attempts

    def submit(self, orders: list[PortfolioOrder]) -> None:
        for o in orders:
            if o.status == OrderStatus.NEW.value:
                _transition(o, OrderStatus.SUBMITTED)

    def fill(
        self, state: PortfolioState, orders: list[PortfolioOrder], day: date, bars: BarLookup
    ) -> list[str]:
        """Fill ``orders`` at ``day``'s open. Returns human-readable notes."""
        notes: list[str] = []
        sold_today: set[str] = set()
        for o in sorted(orders, key=lambda x: (x.side != Side.SELL.value, x.id)):
            if o.signal_date >= day:
                continue  # decided at today's close or later: fill on a later day
            side = Side(o.side)
            bar = bars(o.symbol)
            remaining = o.quantity - o.filled_quantity
            blocked = (
                bar is None
                or bar.open <= 0
                or (
                    bar.prev_close is not None
                    and circuit_locked(
                        side,
                        open_=bar.open,
                        high=bar.high,
                        low=bar.low,
                        close=bar.close,
                        prev_close=bar.prev_close,
                    )
                )
            )
            if blocked:
                o.attempts += 1
                if o.attempts >= self.max_attempts:
                    _transition(o, OrderStatus.CANCELLED)
                    o.note = "expired: not fillable (no trade or circuit)"
                    notes.append(f"expired {o.side} {o.symbol}")
                continue
            assert bar is not None
            price = apply_slippage(Decimal(str(round(bar.open, 2))), side, self.slippage_bps)
            if side is Side.SELL:
                qty = min(remaining, state.lots.quantity(o.symbol))
            else:
                affordable = int(state.cash * Decimal("0.995") / price)
                qty = min(remaining, max(affordable, 0))
            if qty <= 0:
                o.attempts += 1
                o.note = "no shares to sell" if side is Side.SELL else "insufficient cash"
                if o.attempts >= self.max_attempts:
                    _transition(o, OrderStatus.CANCELLED)
                continue
            ch = self.costs.charges(side, qty, price, day, dp=o.symbol not in sold_today)
            if side is Side.SELL:
                sold_today.add(o.symbol)
            apply_fill(state, side, o.symbol, qty, price, day, ch)
            _record(o, qty, price, ch.total, day)
            notes.append(f"filled {o.side} {qty} {o.symbol} @ ₹{price}")
        return notes


class ManualBroker:
    """Orders are executed by you; record fills from your broker's trade book."""

    def __init__(self, costs: CostModel | None = None) -> None:
        self.costs = costs or CostModel()

    def submit(self, orders: list[PortfolioOrder]) -> None:
        for o in orders:
            if o.status == OrderStatus.NEW.value:
                _transition(o, OrderStatus.SUBMITTED)

    def record_fill(
        self,
        state: PortfolioState,
        order: PortfolioOrder,
        qty: int,
        price: Decimal,
        day: date,
        charges: Decimal | None = None,
    ) -> None:
        if order.status not in ("NEW", "SUBMITTED", "PARTIAL"):
            raise PortfolioError(f"order {order.id} is {order.status}")
        if qty <= 0 or qty > order.quantity - order.filled_quantity:
            raise PortfolioError(
                f"fill quantity must be 1..{order.quantity - order.filled_quantity}"
            )
        side = Side(order.side)
        ch = self.costs.charges(side, qty, price, day)
        if charges is not None:  # actual contract-note total: keep STT split, adjust the rest
            ch = ChargeBreakdown(
                ch.turnover,
                ch.brokerage,
                ch.stt,
                ch.exchange_txn,
                ch.sebi,
                ch.stamp,
                ch.gst,
                ch.dp + (charges - ch.total),
            )
        apply_fill(state, side, order.symbol, qty, price, day, ch)
        _record(order, qty, price, ch.total, day)


def cancel(order: PortfolioOrder, note: str = "cancelled by user") -> None:
    _transition(order, OrderStatus.CANCELLED)
    order.note = note


# ---------------------------------------------------------------- reconciliation
def read_holdings_csv(path: Path) -> dict[str, int]:
    """Broker holdings export. Accepts ``symbol,quantity`` or Zerodha Console's
    ``Instrument,Qty.`` (other columns are ignored)."""
    with path.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return {}
    keys = {k.strip().lower(): k for k in rows[0]}
    sym_k = keys.get("symbol") or keys.get("instrument") or keys.get("tradingsymbol")
    qty_k = keys.get("quantity") or keys.get("qty.") or keys.get("qty")
    if sym_k is None or qty_k is None:
        raise PortfolioError(f"{path.name}: need symbol/instrument and quantity/qty columns")
    out: dict[str, int] = {}
    for r in rows:
        if not (r.get(sym_k) or "").strip():
            continue
        out[make_symbol(r[sym_k])] = out.get(make_symbol(r[sym_k]), 0) + int(float(r[qty_k]))
    return out


def reconcile(book: dict[str, int], broker: dict[str, int]) -> list[tuple[str, int, int]]:
    """(symbol, book quantity, broker quantity) for every mismatch."""
    return [
        (s, book.get(s, 0), broker.get(s, 0))
        for s in sorted(set(book) | set(broker))
        if book.get(s, 0) != broker.get(s, 0)
    ]
