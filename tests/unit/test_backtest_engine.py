"""Engine known-answer tests on synthetic data (all exact or within rupee rounding)."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from alphaengine.backtest.config import BacktestConfig
from alphaengine.backtest.engine import BacktestError, simulate
from alphaengine.core.types import Side
from alphaengine.features.data import MarketData
from alphaengine.market.costs import CostModel
from alphaengine.market.tax import compute_fy_tax
from alphaengine.strategy.core import Schedule
from alphaengine.strategy.pipeline import PipelineStrategy
from alphaengine.strategy.stages import FixedWeights, StaticUniverse
from tests.helpers import make_bars

FRICTIONLESS = {
    "broker": "none",
    "statutory_costs": False,
    "slippage": "fixed",
    "slippage_bps": 0,
    "taxes": False,
    "cash_rate": 0.0,
    "cash_buffer": 0.0,
    "min_trade_value": 0.0,
    "max_participation": 1.0,
    "benchmark": "A",
}


def cfg(start: str = "2024-01-01", **kw: object) -> BacktestConfig:
    params = {"initial_capital": 10_000, **FRICTIONLESS, **kw}
    return BacktestConfig(start=date.fromisoformat(start), **params)  # type: ignore[arg-type]


def hold(weights: dict[str, float], freq: str = "yearly") -> PipelineStrategy:
    return PipelineStrategy(
        "hold",
        StaticUniverse(list(weights) or ["A"]),
        FixedWeights(weights),
        schedule=Schedule(freq),
    )  # type: ignore[arg-type]


def md_from(
    spec: dict[str, list[float]],
    start: str = "2024-01-01",
    actions=None,  # type: ignore[no-untyped-def]
    volume: int = 1_000_000,
) -> MarketData:
    return MarketData.from_bars(make_bars(spec, start, volume=volume), actions, execution=True)


def acts(rows: list[tuple[str, str, str, float | None, float | None]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["symbol", "ex_date", "kind", "ratio", "amount"]).assign(
        source="t"
    )


class TestKnownAnswers:
    def test_buy_and_hold_frictionless(self) -> None:
        prices = [100.0, 101.0, 103.0, 99.0, 110.0, 120.0]
        res = simulate(hold({"A": 1.0}), md_from({"A": prices}), cfg())
        shares = int(10_000 // 101.0)  # decided at day-0 close, filled at day-1 open (=101)
        leftover = 10_000 - shares * 101.0
        assert res.final_value == pytest.approx(shares * 120.0 + leftover)
        assert res.equity["value"].iloc[0] == 10_000  # nothing happens before the first fill
        assert len(res.trades) == 1
        assert res.trades.iloc[0]["date"] == pd.Timestamp("2024-01-02")

    def test_buy_charges_match_cost_model(self) -> None:
        c = cfg(broker="zerodha", statutory_costs=True, cash_buffer=0.003)
        res = simulate(hold({"A": 1.0}), md_from({"A": [100.0] * 5}), c)
        t = res.trades.iloc[0]
        expected = CostModel().charges(
            Side.BUY, int(t["quantity"]), Decimal("100.00"), date(2024, 1, 2)
        )
        assert t["charges"] == pytest.approx(float(expected.total))
        assert res.equity["cash"].iloc[-1] >= 0

    def test_cash_earns_interest_net_of_tax(self) -> None:
        md = md_from({"A": [100.0] * 11})
        res = simulate(hold({}), md, cfg(cash_rate=0.0504))
        assert res.final_value == pytest.approx(10_000 * (1 + 0.0504 / 252) ** 10)
        taxed = simulate(hold({}), md, cfg(cash_rate=0.0504, taxes=True, income_tax_rate=0.3))
        assert taxed.final_value == pytest.approx(10_000 * (1 + 0.7 * 0.0504 / 252) ** 10)

    def test_sip_flows(self) -> None:
        md = md_from({"A": [100.0] * 70}, start="2024-01-01")  # ~3 calendar months
        res = simulate(hold({}), md, cfg(sip_amount=1_000))
        assert res.equity["flow"].sum() == 3_000  # Feb, Mar, Apr first trading days
        assert res.equity["contributions"].iloc[-1] == 13_000
        assert res.final_value == pytest.approx(13_000)

    def test_split_keeps_value_continuous(self) -> None:
        raw = [100.0, 100.0, 50.0, 50.0, 55.0]
        md = md_from({"A": raw}, actions=acts([("A", "2024-01-03", "SPLIT", 2.0, None)]))
        res = simulate(hold({"A": 1.0}), md, cfg())
        v = res.equity["value"]
        assert v.iloc[2] == pytest.approx(v.iloc[1])
        assert res.final_value == pytest.approx(10_000 * 1.1)

    def test_ambiguous_action_as_bonus_creates_zero_cost_lot(self) -> None:
        raw = [100.0, 100.0, 50.0, 60.0]
        md = md_from({"A": raw}, actions=acts([("A", "2024-01-03", "SPLIT_OR_BONUS", 2.0, None)]))
        sell_after = PipelineStrategy("x", StaticUniverse(["A"]), FixedWeights({"A": 1.0}))
        res = simulate(sell_after, md, cfg())
        assert res.final_value == pytest.approx(10_000 * 1.2)
        assert res.liquidation["value"] == pytest.approx(res.final_value)

    def test_ambiguous_action_on_etf_is_a_split(self) -> None:
        raw = [100.0, 100.0, 50.0, 60.0, 60.0, 60.0]
        actions = acts([("A", "2024-01-03", "SPLIT_OR_BONUS", 2.0, None)])
        bars = make_bars({"A": raw}, volume=1_000_000)

        class ExitAfterAction(PipelineStrategy):
            def target(self, ctx):  # type: ignore[no-untyped-def]
                t = super().target(ctx)
                if ctx.as_of >= pd.Timestamp("2024-01-04"):
                    return type(t)(ctx.as_of, {}, {}, (), self.name)
                return t

        s = ExitAfterAction(
            "x", StaticUniverse(["A"]), FixedWeights({"A": 1.0}), schedule=Schedule("daily")
        )
        etf = MarketData.from_bars(bars, actions, execution=True, asset_classes={"A": "ETF"})
        stock = MarketData.from_bars(bars, actions, execution=True)
        etf_lots = {d.acquired for d in simulate(s, etf, cfg()).disposals}
        stock_lots = {(d.acquired, d.lot_source) for d in simulate(s, stock, cfg()).disposals}
        assert etf_lots == {date(2024, 1, 2)}  # one split-adjusted lot, original date
        assert (date(2024, 1, 3), "BONUS") in stock_lots  # stock: conservative bonus lot

    def test_dividends_credited_net_of_tax(self) -> None:
        md = md_from(
            {"A": [100.0, 100.0, 100.0, 98.0]},
            actions=acts([("A", "2024-01-04", "DIVIDEND", None, 2.0)]),
        )
        res = simulate(hold({"A": 1.0}), md, cfg(taxes=True, income_tax_rate=0.3))
        assert res.final_value == pytest.approx(10_000 * 0.98 + 100 * 2.0 * 0.7)
        assert res.liquidation["total_dividends"] == pytest.approx(200.0)


class TestTaxesAndLiquidation:
    def test_fy_end_tax_and_liquidation(self) -> None:
        # Buy in Feb 2025 (FY24-25), sell in March, FY ends 31 Mar: STCG @20% paid from cash.
        days = pd.bdate_range("2025-02-03", "2025-04-30")
        prices = np.linspace(100, 130, len(days)).tolist()
        md = md_from({"A": prices}, start="2025-02-03")

        class SellInMarch(PipelineStrategy):
            def target(self, ctx):  # type: ignore[no-untyped-def]
                t = super().target(ctx)
                if ctx.as_of >= pd.Timestamp("2025-03-14"):
                    return type(t)(ctx.as_of, {}, {}, (), self.name)
                return t

        s = SellInMarch(
            "s", StaticUniverse(["A"]), FixedWeights({"A": 1.0}), schedule=Schedule("weekly")
        )
        res = simulate(s, md, cfg(start="2025-02-03", taxes=True))
        fy = res.taxes[2024]
        assert fy.taxable_stcg > 0
        assert fy.total_tax == compute_fy_tax(res.disposals, 2024).total_tax
        assert res.liquidation["total_capital_gains_tax"] == pytest.approx(float(fy.total_tax))
        assert res.liquidation["tax_on_liquidation"] == 0  # nothing left to sell

    def test_liquidation_estimate_for_open_position(self) -> None:
        md = md_from({"A": [100.0, 100.0, 150.0]})
        res = simulate(hold({"A": 1.0}), md, cfg(taxes=True))
        liq = res.liquidation
        assert liq["tax_on_liquidation"] > 0  # short-term gain if sold now
        assert liq["after_tax_value"] == pytest.approx(
            liq["value"] - liq["sell_charges"] - liq["tax_on_liquidation"]
        )


class TestTaxCashAndGrandfathering:
    def test_tax_due_with_no_cash_triggers_sales(self) -> None:
        # Rotate from a big winner (A) into B in March; the FY-end tax bill finds no cash.
        days = pd.bdate_range("2025-01-01", "2025-05-30")
        a = np.linspace(100, 200, len(days)).tolist()
        b = [100.0] * len(days)
        md = md_from({"A": a, "B": b}, start="2025-01-01")

        class Rotate(PipelineStrategy):
            def target(self, ctx):  # type: ignore[no-untyped-def]
                t = super().target(ctx)
                w = {"B": 1.0} if ctx.as_of >= pd.Timestamp("2025-03-10") else {"A": 1.0}
                return type(t)(ctx.as_of, w, {}, (), self.name)

        s = Rotate("r", StaticUniverse(["A", "B"]), FixedWeights({}), schedule=Schedule("weekly"))
        res = simulate(s, md, cfg(start="2025-01-01", taxes=True, min_trade_value=1))
        assert res.taxes[2024].total_tax > 0
        assert res.trades["reason"].str.startswith("raise cash").any()
        after = res.equity.loc["2025-04-07":, "cash"]
        assert (after >= -1e-6).all()

    def test_grandfathering_uses_31_jan_2018_high_and_later_splits(self) -> None:
        days = pd.bdate_range("2017-06-01", "2019-06-28")
        prices = np.linspace(100, 300, len(days))
        split_day = pd.Timestamp("2018-06-01")
        raw = [p / 2 if d >= split_day else p for d, p in zip(days, prices, strict=True)]
        md = md_from(
            {"A": raw}, start="2017-06-01", actions=acts([("A", "2018-06-01", "SPLIT", 2.0, None)])
        )
        res = simulate(
            hold({"A": 1.0}), md, cfg(start="2017-06-01", taxes=True, initial_capital=10_000_000)
        )
        buy = res.trades.iloc[0]
        q0, cost = int(buy["quantity"]), float(buy["value"])
        fmv_share = round(float(md.raw_high.loc["2018-01-31", "A"]), 2) / 2  # paisa, then split
        fmv_total = fmv_share * 2 * q0
        gross = 2 * q0 * round(float(md.raw_close["A"].iloc[-1]), 2)
        gain = gross - max(cost, min(fmv_total, gross))
        tax = round(0.10 * (gain - 100_000))  # FY2019-20: 10% LTCG above ₹1 lakh
        assert res.liquidation["tax_on_liquidation"] == pytest.approx(
            tax + round(tax * 0.04), abs=2
        )
        assert gain < gross - cost  # grandfathering really reduced the taxable gain


class TestExecutionRealism:
    def test_circuit_lock_delays_buy(self) -> None:
        bars = make_bars({"A": [100.0, 100.0, 105.0, 106.0]}, volume=1_000_000)
        locked = bars["date"] == pd.Timestamp("2024-01-02")
        bars.loc[locked, ["open", "high", "low", "close"]] = 110.0  # locked at upper circuit
        bars.loc[bars["date"] == pd.Timestamp("2024-01-03"), ["open"]] = 105.0
        md = MarketData.from_bars(bars, execution=True)
        res = simulate(hold({"A": 1.0}), md, cfg())
        assert res.trades.iloc[0]["date"] == pd.Timestamp("2024-01-03")

    def test_participation_cap_spreads_fills(self) -> None:
        # 15 warm-up days give a known ADV of ₹10,000 -> 10% cap = 10 shares/day.
        md = md_from({"A": [100.0] * 40}, volume=100)
        start = str(md.dates[15].date())
        res = simulate(
            hold({"A": 1.0}), md, cfg(start=start, max_participation=0.1, max_fill_attempts=10)
        )
        assert (res.trades["quantity"] <= 10).all()
        assert len(res.trades) == 10  # ₹10,000 at ₹100 = 100 shares over 10 days
        dropped = simulate(
            hold({"A": 1.0}), md, cfg(start=start, max_participation=0.1, max_fill_attempts=3)
        )
        assert len(dropped.trades) == 3
        assert any("dropped" in n for n in dropped.notes)
        no_history = simulate(hold({"A": 1.0}), md_from({"A": [100.0] * 5}, volume=1), cfg())
        assert len(no_history.trades) == 1  # unknown ADV: not capped

    def test_rebalance_band_reduces_trades(self) -> None:
        rng = np.random.default_rng(1)
        a = (100 * np.cumprod(1 + rng.normal(0, 0.01, 120))).tolist()
        b = (100 * np.cumprod(1 + rng.normal(0, 0.01, 120))).tolist()
        md = md_from({"A": a, "B": b})
        s = hold({"A": 0.5, "B": 0.5}, "weekly")
        tight = simulate(s, md, cfg(min_trade_value=1))
        banded = simulate(s, md, cfg(min_trade_value=1, rebalance_band=0.05))
        assert len(banded.trades) < len(tight.trades)

    def test_no_look_ahead(self) -> None:
        """Equity up to day t is identical whether or not later data exists."""
        rng = np.random.default_rng(3)
        spec = {s: (100 * np.cumprod(1 + rng.normal(0, 0.02, 80))).tolist() for s in "AB"}
        md = md_from(spec)
        s = hold({"A": 0.6, "B": 0.4}, "weekly")
        full = simulate(s, md, cfg())
        cut_day = md.dates[50]
        part = simulate(s, md.truncate(cut_day), cfg())
        pd.testing.assert_series_equal(
            full.equity["value"].loc[: md.dates[49]], part.equity["value"].loc[: md.dates[49]]
        )

    def test_errors(self) -> None:
        md = MarketData.from_bars(make_bars({"A": [1.0, 2.0]}))
        with pytest.raises(BacktestError, match="execution"):
            simulate(hold({"A": 1.0}), md, cfg())
        with pytest.raises(BacktestError, match="two trading days"):
            simulate(hold({"A": 1.0}), md_from({"A": [1.0, 2.0]}), cfg(start="2024-01-02"))
        with pytest.raises(ValueError, match="end must be after start"):
            BacktestConfig(start=date(2024, 2, 1), end=date(2024, 1, 1))
        assert cfg().config_hash() == cfg().config_hash()
        assert cfg().config_hash() != cfg(cash_rate=0.01).config_hash()
