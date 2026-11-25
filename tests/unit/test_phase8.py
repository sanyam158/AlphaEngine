"""Exit rules / rules mode, harvesting ideas and the goal planner."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from alphaengine.backtest.config import BacktestConfig
from alphaengine.backtest.engine import simulate
from alphaengine.backtest.vectorized import run_vectorized
from alphaengine.core.errors import StrategyError
from alphaengine.features.data import MarketData
from alphaengine.market.costs import NO_COSTS, CostModel
from alphaengine.planning.goal import Goal, monthly_returns, plan_goal
from alphaengine.portfolio.harvest import harvest_ideas
from alphaengine.portfolio.lots import Disposal, LotBook
from alphaengine.strategy.config import build_strategy, load_strategy, parse_config
from alphaengine.strategy.core import Schedule
from alphaengine.strategy.exits import (
    FeatureExit,
    PositionState,
    StopLoss,
    TakeProfit,
    TimeStop,
    TrailingStop,
)
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


def pos(close: float, entry: float = 100.0, peak: float = 100.0, days: int = 1) -> PositionState:
    return PositionState("A", pd.Timestamp("2024-01-02"), entry, peak, close, days)


class TestExitRules:
    def test_price_rules(self) -> None:
        md = MarketData.from_bars(make_bars({"A": [100.0] * 30}), ohlc=True)
        t = md.dates[-1]
        assert StopLoss(pct=0.1).check(pos(89), md, t)
        assert StopLoss(pct=0.1).check(pos(91), md, t) is None
        assert TrailingStop(pct=0.1).check(pos(107, peak=120), md, t)
        assert TrailingStop(pct=0.1).check(pos(110, peak=120), md, t) is None
        assert TakeProfit(0.2).check(pos(121), md, t)
        assert TimeStop(5).check(pos(100, days=5), md, t)
        assert TimeStop(5).check(pos(100, days=4), md, t) is None
        # ATR of a flat ±1% bar series is about 2 points: 3 ATRs below 100 is ~94
        assert StopLoss(atr_mult=3).check(pos(90), md, t)
        assert StopLoss(atr_mult=3).check(pos(97), md, t) is None
        assert TrailingStop(atr_mult=3).check(pos(90, peak=100), md, t)
        with pytest.raises(StrategyError, match="exactly one"):
            StopLoss()
        with pytest.raises(StrategyError, match="exactly one"):
            TrailingStop(pct=0.1, atr_mult=2)

    def test_feature_exit(self) -> None:
        md = MarketData.from_bars(make_bars({"A": [100.0 + i for i in range(30)]}))
        t = md.dates[-1]
        assert FeatureExit("price_to_sma", ">", 0.0, {"window": 5}).check(pos(129), md, t)
        assert FeatureExit("price_to_sma", "<", 0.0, {"window": 5}).check(pos(129), md, t) is None
        nan_md = MarketData.from_bars(make_bars({"A": [100.0] * 3}))
        assert (
            FeatureExit("price_to_sma", ">", 0.0, {"window": 20}).check(
                pos(1), nan_md, nan_md.dates[-1]
            )
            is None
        )
        flat = MarketData.from_bars(make_bars({"A": [100.0] * 3}))
        with pytest.raises(StrategyError, match="OHLC"):
            StopLoss(atr_mult=2).check(pos(1), flat, flat.dates[-1])

    def test_config_builds_rules_mode(self) -> None:
        for name in ("swing_breakout", "swing_pullback"):
            s = load_strategy(name)
            assert isinstance(s, PipelineStrategy)
            assert s.rules_mode
            assert s.max_positions == 10
        plain = build_strategy(
            parse_config(
                {
                    "name": "x_rules",
                    "universe": {"type": "static", "symbols": ["A"]},
                    "exits": [{"type": "time_stop", "days": 3}],
                }
            )
        )
        assert plain.max_positions == 10  # type: ignore[union-attr]
        with pytest.raises(StrategyError, match="exits"):
            build_strategy(
                parse_config(
                    {
                        "name": "bad_rules",
                        "universe": {"type": "static", "symbols": ["A"]},
                        "exits": [{"type": "stop_loss"}],
                    }
                )
            )


class TestRulesModeEngine:
    def _md(self) -> MarketData:
        # A rises then crashes; B is flat. Strategy always wants both.
        a = [100.0, 100.0, 105.0, 110.0, 120.0, 100.0, 90.0, 85.0, 85.0, 85.0]
        b = [50.0] * 10
        return MarketData.from_bars(make_bars({"A": a, "B": b}), execution=True)

    def test_slots_and_stop_exit(self) -> None:
        md = self._md()
        s = PipelineStrategy(
            "swing",
            StaticUniverse(["A", "B"]),
            FixedWeights({"A": 0.5, "B": 0.5}),
            schedule=Schedule("daily"),
            exits=[TrailingStop(pct=0.15)],
            max_positions=1,
        )
        res = simulate(
            s, md, BacktestConfig(start=date(2024, 1, 1), initial_capital=10_000, **FRICTIONLESS)
        )
        t = res.trades
        assert t.iloc[0]["symbol"] == "A"  # only one slot: first pick
        assert (t["side"] == "BUY").sum() >= 1
        exits = t[t["reason"].str.startswith("exit:")]
        assert len(exits) == 1
        assert "trailing stop" in exits.iloc[0]["reason"]
        # after the exit the free slot is refilled (A is out, so B gets in)
        assert "B" in set(t.loc[t.index > exits.index[0], "symbol"])
        # existing positions are never trimmed by rebalancing
        assert not t["reason"].isin(["trim", "rebalance", "exit"]).any()

    def test_time_stop_and_vectorized_guard(self) -> None:
        md = self._md()
        s = PipelineStrategy(
            "swing",
            StaticUniverse(["B"]),
            FixedWeights({"B": 1.0}),
            schedule=Schedule("weekly"),
            exits=[TimeStop(3)],
            max_positions=2,
        )
        res = simulate(
            s, md, BacktestConfig(start=date(2024, 1, 1), initial_capital=10_000, **FRICTIONLESS)
        )
        assert any("time stop" in r for r in res.trades["reason"])
        with pytest.raises(ValueError, match="full engine"):
            run_vectorized(s, md, date(2024, 1, 1))


class TestHarvest:
    FREE = CostModel(NO_COSTS, statutory=False)

    def test_loss_harvest_offsets_realised_gains(self) -> None:
        on = date(2026, 3, 2)  # FY 2025-26
        book = LotBook()
        book.buy("LOSER", 100, Decimal("500"), date(2025, 9, 1))  # short-term, now 300 -> -20k
        realised = [
            Disposal(
                "WIN",
                10,
                date(2025, 6, 1),
                date(2025, 12, 1),
                Decimal("10000"),
                Decimal("40000"),
                Decimal("0"),
                Decimal("0"),
            )
        ]  # STCG 30k
        ideas = harvest_ideas(book, {"LOSER": 300.0}, on, realised, costs=self.FREE)
        loss = [i for i in ideas if i.kind == "loss"]
        assert len(loss) == 1
        assert loss[0].gain == pytest.approx(-20_000)
        assert loss[0].tax_change == pytest.approx(-20_000 * 0.20 * 1.04, abs=2)
        assert loss[0].net_benefit > 0
        # no gains realised: the loss only carries forward
        later = harvest_ideas(book, {"LOSER": 300.0}, on, [], costs=self.FREE)
        assert "carries forward" in later[0].note

    def test_ltcg_gain_harvest_respects_exemption(self) -> None:
        on = date(2026, 3, 2)
        book = LotBook()
        book.buy("OLD", 1000, Decimal("100"), date(2023, 1, 2))  # long-term, +200/share = 2L gain
        ideas = harvest_ideas(book, {"OLD": 300.0}, on, [], costs=self.FREE)
        gain = [i for i in ideas if i.kind == "ltcg_gain"]
        assert len(gain) == 1
        assert gain[0].gain <= 125_000 + 1
        assert gain[0].quantity == 625  # 1.25L / ₹200 per share
        used = [
            Disposal(
                "X",
                1,
                date(2020, 1, 1),
                date(2025, 5, 1),
                Decimal("0"),
                Decimal("125000"),
                Decimal("0"),
                Decimal("0"),
            )
        ]
        assert not [
            i
            for i in harvest_ideas(book, {"OLD": 300.0}, on, used, costs=self.FREE)
            if i.kind == "ltcg_gain"
        ]  # exemption already used this FY
        assert harvest_ideas(book, {}, on, []) == []  # no price, no idea


class TestGoal:
    hist = np.full(120, 0.01)  # exactly 1% a month

    def test_deterministic_growth_and_required_sip(self) -> None:
        g = Goal(
            target=1_000_000,
            years=10,
            current=100_000,
            monthly_sip=5_000,
            inflation=0.0,
            after_tax=False,
        )
        r = plan_goal(g, self.hist, n_paths=50)
        months = 120
        expect = 100_000 * 1.01**months + sum(5_000 * 1.01 ** (months - m) for m in range(months))
        assert r.final_percentiles["p50"] == pytest.approx(expect, rel=1e-9)
        assert r.probability == 1.0
        assert r.required_sip == 0.0 or r.required_sip <= 5_000
        assert len(r.yearly_bands) == 10
        hard = plan_goal(
            Goal(target=5e6, years=10, inflation=0.0, after_tax=False), self.hist, n_paths=50
        )
        assert hard.probability == 0.0
        assert hard.required_sip is not None and hard.required_sip > 0
        assert (
            plan_goal(
                Goal(
                    target=5e6,
                    years=10,
                    inflation=0.0,
                    after_tax=False,
                    monthly_sip=hard.required_sip,
                ),
                self.hist,
                n_paths=50,
            ).probability
            == 1.0
        )

    def test_inflation_tax_and_inputs(self) -> None:
        r = plan_goal(
            Goal(target=1e6, years=5, inflation=0.06, monthly_sip=10_000), self.hist, n_paths=20
        )
        assert r.target_nominal == pytest.approx(1e6 * 1.06**5)
        pre = plan_goal(
            Goal(target=1e6, years=5, monthly_sip=10_000, after_tax=False), self.hist, n_paths=20
        )
        assert r.final_percentiles["p50"] < pre.final_percentiles["p50"]
        with pytest.raises(ValueError):  # noqa: PT011
            Goal(target=0, years=5)
        with pytest.raises(ValueError):  # noqa: PT011
            Goal(target=1, years=0)
        prices = pd.Series(
            np.cumprod(np.full(800, 1.0005)), index=pd.bdate_range("2020-01-01", periods=800)
        )
        assert len(monthly_returns(prices)) >= 35
        with pytest.raises(ValueError, match="3 years"):
            monthly_returns(prices.iloc[:300])
