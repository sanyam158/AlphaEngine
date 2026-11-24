"""Charges are checked against hand calculations from the published rate card (not against
actual contract notes, which this project doesn't have)."""

from __future__ import annotations

from datetime import date
from decimal import Decimal as D

import pytest
from hypothesis import given
from hypothesis import strategies as st

from alphaengine.core.types import Side
from alphaengine.market.costs import FLAT_20, NO_COSTS, ZERO_CHARGES, CostModel, rates_on
from alphaengine.market.execution_model import (
    FixedBpsSlippage,
    SqrtImpactSlippage,
    apply_slippage,
    circuit_locked,
    max_fill_quantity,
    round_to_tick,
    settlement_date,
    shares_for_value,
)

NOW = date(2026, 1, 15)


class TestCharges:
    def test_zerodha_buy_known_answer(self) -> None:
        c = CostModel().charges(Side.BUY, 100, D("1000"), NOW)  # ₹1,00,000 turnover
        assert (c.brokerage, c.stt, c.exchange_txn, c.sebi, c.stamp, c.gst, c.dp) == (
            D("0.00"),
            D("100"),
            D("3.07"),
            D("0.10"),
            D("15.00"),
            D("0.57"),
            D("0.00"),
        )
        assert c.total == D("118.74")
        assert c.tax_deductible == D("18.74")

    def test_zerodha_sell_known_answer(self) -> None:
        c = CostModel().charges(Side.SELL, 100, D("1100"), NOW)  # ₹1,10,000
        # GST = 18% x (txn 3.38 + SEBI 0.11 + DP 13.00) = 2.9682
        assert (c.stt, c.exchange_txn, c.sebi, c.stamp, c.dp, c.gst) == (
            D("110"),
            D("3.38"),
            D("0.11"),
            D("0.00"),
            D("13.00"),
            D("2.97"),
        )
        assert c.total == D("129.46")
        no_dp = CostModel().charges(Side.SELL, 100, D("1100"), NOW, dp=False)
        assert no_dp.dp == 0
        assert no_dp.gst == D("0.63")

    def test_historical_rates(self) -> None:
        c = CostModel().charges(Side.BUY, 100, D("1000"), date(2012, 1, 2))
        assert c.stt == D("125")  # 0.125% before June 2013
        assert c.exchange_txn == D("3.22")
        assert c.gst == D("0.50")  # 15% service tax on 3.32
        assert rates_on(date(2017, 7, 1)).gst == D("0.18")
        with pytest.raises(ValueError, match="no statutory rates"):
            rates_on(date(1999, 1, 1))

    def test_flat_broker_and_cap(self) -> None:
        c = CostModel(FLAT_20).charges(Side.BUY, 100, D("1000"), NOW)
        assert c.brokerage == D("20.00")  # 0.1% = ₹100, capped at ₹20
        assert c.gst == D("4.17")  # 18% x (20 + 3.07 + 0.10)
        small = CostModel(FLAT_20).charges(Side.BUY, 1, D("500"), NOW)
        assert small.brokerage == D("0.50")

    def test_stt_rounding_option_and_costless(self) -> None:
        c = CostModel(round_stt_to_rupee=False).charges(Side.BUY, 3, D("333.33"), NOW)
        assert c.stt == D("1.00")  # 0.999990 -> paisa
        c2 = CostModel().charges(Side.BUY, 3, D("333.33"), NOW)
        assert c2.stt == D("1")
        free = CostModel(NO_COSTS).without_statutory().charges(Side.SELL, 10, D("100"), NOW)
        assert free.total == 0

    def test_sum_and_round_trip(self) -> None:
        m = CostModel()
        both = m.charges(Side.BUY, 100, D("1000"), NOW) + m.charges(Side.SELL, 100, D("1100"), NOW)
        assert both.total == D("248.20")
        assert (both + ZERO_CHARGES).total == both.total
        assert D("0.0023") < m.round_trip_cost_pct(D("100000"), NOW) < D("0.0026")

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            CostModel().charges(Side.BUY, 0, D("10"), NOW)

    @given(
        st.integers(min_value=1, max_value=100_000),
        st.decimals(min_value="0.05", max_value="50000", places=2),
        st.sampled_from([Side.BUY, Side.SELL]),
    )
    def test_non_negative_and_monotonic(self, qty: int, price: D, side: Side) -> None:
        m = CostModel()
        a = m.charges(side, qty, price, NOW)
        b = m.charges(side, qty * 2, price, NOW)
        assert all(getattr(a, f) >= 0 for f in ("stt", "exchange_txn", "sebi", "stamp", "gst"))
        assert b.total >= a.total
        assert a.tax_deductible == a.total - a.stt


class TestExecutionModel:
    def test_slippage_models(self) -> None:
        assert FixedBpsSlippage(12).slippage_bps(1e6, None, None) == 12
        m = SqrtImpactSlippage(base_bps=5, k=1.0)
        assert m.slippage_bps(1e6, None, 0.02) == m.fallback_bps
        # 1% of ADV, 2% daily vol: 5 + 1 * 0.02 * 0.1 * 1e4 = 25 bps
        assert m.slippage_bps(1e6, 1e8, 0.02) == pytest.approx(25.0)

    def test_apply_slippage_rounds_against_trader(self) -> None:
        assert apply_slippage(D("100.00"), Side.BUY, 10) == D("100.10")
        assert apply_slippage(D("100.00"), Side.SELL, 10) == D("99.90")
        assert apply_slippage(D("100.02"), Side.BUY, 0) == D("100.05")
        assert apply_slippage(D("100.02"), Side.SELL, 0) == D("100.00")
        assert apply_slippage(D("0.05"), Side.SELL, 5000) == D("0.05")
        assert round_to_tick(D("100.03")) == D("100.05")

    def test_circuit_detection(self) -> None:
        def bar(o: float, h: float, lo: float, c: float, pc: float) -> dict[str, float]:
            return {"open_": o, "high": h, "low": lo, "close": c, "prev_close": pc}

        up = bar(110, 110, 110, 110, 100)
        assert circuit_locked(Side.BUY, **up)
        assert not circuit_locked(Side.SELL, **up)
        assert circuit_locked(Side.SELL, **bar(95, 95, 95, 95, 100))
        assert not circuit_locked(Side.BUY, **bar(100, 105, 99, 104, 100))
        assert not circuit_locked(Side.BUY, **bar(110, 110, 110, 110, 0))

    def test_settlement(self) -> None:
        assert settlement_date(date(2026, 1, 16)) == date(2026, 1, 19)  # Fri -> Mon, T+1
        assert settlement_date(date(2020, 1, 2)) == date(2020, 1, 6)  # Thu, T+2 -> Mon
        assert settlement_date(date(2026, 1, 16), 3) == date(2026, 1, 21)

    def test_sizing_helpers(self) -> None:
        assert shares_for_value(D("10000"), D("333")) == 30
        with pytest.raises(ValueError):  # noqa: PT011
            shares_for_value(D("1"), D("0"))
        assert max_fill_quantity(None) is None
        assert max_fill_quantity(50_000, 0.1) == 5000
        with pytest.raises(ValueError, match="participation"):
            max_fill_quantity(1000, 0)
