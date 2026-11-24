from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

import pytest

from alphaengine.core.clock import IST
from alphaengine.core.errors import OrderError
from alphaengine.core.types import (
    Bar,
    Fill,
    Order,
    OrderStatus,
    OrderType,
    Position,
    Side,
    Symbol,
    can_transition,
    make_symbol,
)

TS = datetime(2026, 1, 5, 9, 15, tzinfo=IST)
REL = Symbol("RELIANCE")


class TestMakeSymbol:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("reliance", "RELIANCE"),
            ("RELIANCE.NS", "RELIANCE"),
            ("tcs.bo", "TCS"),
            (" M&M ", "M&M"),
            ("BAJAJ-AUTO", "BAJAJ-AUTO"),
        ],
    )
    def test_normalizes(self, raw: str, expected: str) -> None:
        assert make_symbol(raw) == expected

    @pytest.mark.parametrize("raw", ["", "  ", "REL IANCE", "-ABC", "A" * 21, "ABC$"])
    def test_rejects_invalid(self, raw: str) -> None:
        with pytest.raises(ValueError, match="invalid symbol"):
            make_symbol(raw)


class TestBar:
    def test_valid(self) -> None:
        bar = Bar(REL, date(2026, 1, 5), 100, 105, 99, 104, 1000)
        assert bar.close == 104

    @pytest.mark.parametrize(
        ("ohlcv", "msg"),
        [
            ((0, 105, 99, 104, 10), "positive"),
            ((100, 101, 99, 104, 10), "high below"),
            ((100, 105, 101, 104, 10), "low above"),
            ((100, 105, 99, 104, -1), "negative volume"),
        ],
    )
    def test_invalid(self, ohlcv: tuple[float, float, float, float, int], msg: str) -> None:
        o, h, lo, c, v = ohlcv
        with pytest.raises(ValueError, match=msg):
            Bar(REL, date(2026, 1, 5), o, h, lo, c, int(v))


class TestOrder:
    def _order(self, **kw: object) -> Order:
        base: dict[str, object] = {"id": "ord_1", "symbol": REL, "side": Side.BUY, "quantity": 10}
        base.update(kw)
        return Order(**base)  # type: ignore[arg-type]

    def test_validation(self) -> None:
        with pytest.raises(OrderError, match="positive"):
            self._order(quantity=0)
        with pytest.raises(OrderError, match="limit_price"):
            self._order(order_type=OrderType.LIMIT)
        with pytest.raises(OrderError, match="market orders"):
            self._order(limit_price=Decimal("10"))
        with pytest.raises(OrderError, match="out of range"):
            self._order(filled_quantity=11)
        assert self._order(order_type=OrderType.LIMIT, limit_price=Decimal("10")).limit_price

    def test_happy_path_lifecycle(self) -> None:
        o = self._order()
        o = o.with_status(OrderStatus.SUBMITTED)
        o = o.with_status(OrderStatus.PARTIAL, filled_quantity=4)
        assert o.remaining_quantity == 6
        assert not o.is_terminal
        o = o.with_status(OrderStatus.FILLED, filled_quantity=10)
        assert o.is_terminal
        assert o.remaining_quantity == 0

    def test_illegal_transitions(self) -> None:
        o = self._order()
        with pytest.raises(OrderError, match="illegal"):
            o.with_status(OrderStatus.FILLED, filled_quantity=10)
        sub = o.with_status(OrderStatus.SUBMITTED)
        with pytest.raises(OrderError, match="FILLED requires"):
            sub.with_status(OrderStatus.FILLED, filled_quantity=5)
        with pytest.raises(OrderError, match="PARTIAL requires"):
            sub.with_status(OrderStatus.PARTIAL, filled_quantity=10)
        part = sub.with_status(OrderStatus.PARTIAL, filled_quantity=5)
        with pytest.raises(OrderError, match="cannot decrease"):
            part.with_status(OrderStatus.PARTIAL, filled_quantity=3)

    @pytest.mark.parametrize(
        "terminal", [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]
    )
    def test_terminal_states_are_final(self, terminal: OrderStatus) -> None:
        assert not any(can_transition(terminal, s) for s in OrderStatus)


class TestFillAndPosition:
    def test_fill_cash_flow(self) -> None:
        buy = Fill("o1", REL, Side.BUY, 10, Decimal("100"), TS, Decimal("2"))
        sell = Fill("o2", REL, Side.SELL, 10, Decimal("110"), TS, Decimal("3"))
        assert buy.cash_flow == Decimal("-1002")
        assert sell.cash_flow == Decimal("1097")

    def test_fill_validation(self) -> None:
        with pytest.raises(OrderError):
            Fill("o", REL, Side.BUY, 0, Decimal("1"), TS)
        with pytest.raises(OrderError):
            Fill("o", REL, Side.BUY, 1, Decimal("0"), TS)
        with pytest.raises(OrderError):
            Fill("o", REL, Side.BUY, 1, Decimal("1"), TS, Decimal("-1"))

    def test_position_round_trip(self) -> None:
        p = Position(REL)
        p.apply(Fill("o1", REL, Side.BUY, 10, Decimal("100"), TS, Decimal("10")))
        assert p.avg_cost == Decimal("101")  # charges capitalised into cost basis
        p.apply(Fill("o2", REL, Side.BUY, 10, Decimal("121"), TS))
        assert p.avg_cost == Decimal("111")
        assert p.market_value(Decimal("120")) == Decimal("2400")
        assert p.unrealized_pnl(Decimal("120")) == Decimal("180")
        p.apply(Fill("o3", REL, Side.SELL, 20, Decimal("120"), TS, Decimal("5")))
        assert p.quantity == 0
        assert p.avg_cost == 0
        assert p.realized_pnl == Decimal("175")

    def test_position_rejects_oversell_and_wrong_symbol(self) -> None:
        p = Position(REL)
        with pytest.raises(OrderError, match="cannot sell"):
            p.apply(Fill("o", REL, Side.SELL, 1, Decimal("1"), TS))
        with pytest.raises(OrderError, match="applied to"):
            p.apply(Fill("o", Symbol("TCS"), Side.BUY, 1, Decimal("1"), TS))
