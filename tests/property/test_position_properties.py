"""Accounting invariants that must hold for any sequence of fills."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from hypothesis import given
from hypothesis import strategies as st

from alphaengine.core.clock import IST
from alphaengine.core.types import Fill, Position, Side, Symbol

SYM = Symbol("INFY")
TS = datetime(2026, 1, 5, tzinfo=IST)

prices = st.decimals(min_value="1", max_value="10000", places=2)
qtys = st.integers(min_value=1, max_value=1000)
charges = st.decimals(min_value="0", max_value="50", places=2)


@given(st.lists(st.tuples(st.booleans(), qtys, prices, charges), min_size=1, max_size=40))
def test_cash_plus_value_conserved(events: list[tuple[bool, int, Decimal, Decimal]]) -> None:
    """cash + realized-at-cost identity: cash spent == cost basis held - realized pnl + charges.

    Equivalent statement checked here: for every step,
    initial_cash + Σcash_flows + qty*avg_cost == initial_cash + realized_pnl.
    """
    pos = Position(SYM)
    cash = Decimal("0")
    for is_buy, want, price, fee in events:
        qty = want if is_buy else min(want, pos.quantity)
        if qty == 0:
            continue
        fill = Fill("o", SYM, Side.BUY if is_buy else Side.SELL, qty, price, TS, fee)
        pos.apply(fill)
        cash += fill.cash_flow
        assert pos.quantity >= 0
        held_cost = pos.avg_cost * pos.quantity
        assert abs((cash + held_cost) - pos.realized_pnl) < Decimal("1e-6")


@given(qtys, prices, prices)
def test_round_trip_pnl_matches_price_change(qty: int, buy: Decimal, sell: Decimal) -> None:
    pos = Position(SYM)
    pos.apply(Fill("b", SYM, Side.BUY, qty, buy, TS))
    pos.apply(Fill("s", SYM, Side.SELL, qty, sell, TS))
    assert pos.quantity == 0
    assert pos.realized_pnl == (sell - buy) * qty
