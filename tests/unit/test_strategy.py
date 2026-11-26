from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaengine.core.errors import StrategyError
from alphaengine.features.base import FeatureSpec
from alphaengine.features.data import MarketData
from alphaengine.strategy.core import Schedule, StrategyContext, TargetPortfolio
from alphaengine.strategy.pipeline import Blend, PipelineStrategy
from alphaengine.strategy.stages import (
    CompositeScore,
    EqualWeight,
    Factor,
    FeatureFilter,
    FeatureScore,
    FixedWeights,
    IndexUniverse,
    InverseVolatility,
    LiquidityUniverse,
    MaxWeight,
    MinAdv,
    MinHistory,
    MinPrice,
    RegimeFilter,
    SectorCap,
    SelectAll,
    StaticUniverse,
    TopN,
    VolatilityTarget,
    component,
)
from tests.helpers import make_bars, market, random_walk_bars


@pytest.fixture(scope="module")
def md() -> MarketData:
    bench = list(np.linspace(100, 130, 300))
    return market(
        random_walk_bars(n_symbols=10, gap_prob=0.0),
        benchmark=bench,
        industries={"S00": "Banks", "S01": "Banks", "S02": "IT"},
        asset_classes={"S09": "ETF"},
    )


def ctx(md: MarketData, i: int = -1, current: dict[str, float] | None = None) -> StrategyContext:
    return StrategyContext(md.dates[i], md, current or {})


def momentum_strategy(**kw: object) -> PipelineStrategy:
    return PipelineStrategy(
        name="mom",
        universe=LiquidityUniverse(top_n=8, window=20),
        filters=[MinHistory(60)],
        scorer=FeatureScore("momentum", {"lookback": 60, "skip": 5}),
        selector=TopN(3, buffer=kw.pop("buffer", None)),  # type: ignore[arg-type]
        weighter=EqualWeight(),
        overlays=[MaxWeight(0.5)],
    )


class TestSchedule:
    def test_period_ends(self) -> None:
        days = pd.bdate_range("2024-01-01", "2024-06-30")
        m = Schedule("monthly").dates(days)
        assert [d.strftime("%m-%d") for d in m] == [
            "01-31",
            "02-29",
            "03-29",
            "04-30",
            "05-31",
            "06-28",
        ]
        assert [d.strftime("%m-%d") for d in Schedule("quarterly").dates(days)] == [
            "03-29",
            "06-28",
        ]
        assert len(Schedule("weekly").dates(days)) == 26
        assert Schedule("yearly").dates(days) == [pd.Timestamp("2024-06-28")]
        assert len(Schedule("daily").dates(days)) == len(days)
        assert Schedule().dates(pd.DatetimeIndex([])) == []


class TestTargetPortfolio:
    def test_validation_and_describe(self) -> None:
        t = TargetPortfolio(
            pd.Timestamp("2024-01-31"), {"A": 0.6, "B": 0.3}, {"A": ["rank 1"]}, ("regime on",), "x"
        )
        assert t.cash == pytest.approx(0.1)
        text = t.describe()
        assert "cash 10.0%" in text
        assert "rank 1" in text
        assert "regime on" in text
        with pytest.raises(StrategyError, match="negative"):
            TargetPortfolio(pd.Timestamp("2024-01-31"), {"A": -0.1})
        with pytest.raises(StrategyError, match="leverage"):
            TargetPortfolio(pd.Timestamp("2024-01-31"), {"A": 0.7, "B": 0.4})


class TestUniversesAndFilters:
    def test_static(self, md: MarketData) -> None:
        c = ctx(md)
        assert StaticUniverse(["S01", "NOPE"]).select(c) == ["S01"]
        assert "NOPE" in c.notes[0]

    def test_liquidity_ranks_and_excludes_etfs(self, md: MarketData) -> None:
        c = ctx(md)
        u = LiquidityUniverse(top_n=3, window=20).select(c)
        adv = md.feature_row(FeatureSpec.of("adv_value", window=20), c.as_of).drop("S09")
        assert u == list(adv.sort_values(ascending=False).index[:3])
        assert "S09" not in LiquidityUniverse(top_n=10, window=20).select(c)
        assert LiquidityUniverse(top_n=1, window=20, asset_class="ETF").select(c) == ["S09"]
        assert LiquidityUniverse(top_n=1, window=20, asset_class="ALL").select(c) == ["S09"]
        assert u[0] not in LiquidityUniverse(top_n=2, window=20, exclude=[u[0]]).select(c)

    def test_index_universe(self, md: MarketData) -> None:
        c = ctx(md)
        assert IndexUniverse("NIFTY50").select(c) == []
        assert "no membership" in c.notes[0]

    def test_filters(self, md: MarketData) -> None:
        c = ctx(md)
        syms = md.symbols
        px = md.raw_close.iloc[-1]
        assert MinPrice(float(px.median())).apply(c, syms) == [
            s for s in syms if px[s] >= px.median()
        ]
        assert MinHistory(10_000).apply(c, syms) == []
        assert MinHistory(10).apply(c, syms) == syms
        assert MinAdv(1e15).apply(c, syms) == []
        up = FeatureFilter("price_to_sma", ">", 0.0, {"window": 20}).apply(c, syms)
        down = FeatureFilter("price_to_sma", "<=", 0.0, {"window": 20}).apply(c, syms)
        assert sorted(up + down) == sorted(syms)


class TestScoringAndSelection:
    def test_composite_zscores(self, md: MarketData) -> None:
        c = ctx(md)
        cs = CompositeScore(
            [
                Factor("momentum", {"lookback": 60, "skip": 5}),
                {"feature": "volatility", "params": {"window": 20}, "higher_is_better": False},  # type: ignore[list-item]
            ]
        )
        s = cs.score(c, md.symbols)
        assert len(s) == len(md.symbols)
        assert abs(float(s.mean())) < 1e-9  # z-scores sum to zero per factor
        with pytest.raises(StrategyError):
            CompositeScore([])
        low = FeatureScore("volatility", {"window": 20}, higher_is_better=False).score(
            c, md.symbols
        )
        assert (low <= 0).all()

    def test_top_n_hysteresis(self, md: MarketData) -> None:
        scores = pd.Series({"A": 5.0, "B": 4.0, "C": 3.0, "D": 2.0, "E": 1.0})
        c = ctx(md, current={"D": 0.5})
        assert TopN(2).select(c, scores) == ["A", "B"]
        c2 = ctx(md, current={"D": 0.5})
        assert TopN(2, buffer=4).select(c2, scores) == ["D", "A"]  # D kept: rank 4 <= buffer
        assert "within buffer" in c2.rationale["D"][0]
        c3 = ctx(md, current={"E": 0.5})
        assert TopN(2, buffer=4).select(c3, scores) == ["A", "B"]  # E rank 5 > buffer
        assert TopN(5, min_score=2.5).select(ctx(md), scores) == ["A", "B", "C"]
        with pytest.raises(StrategyError):
            TopN(3, buffer=2)
        assert SelectAll(min_score=3.5).select(ctx(md), scores) == ["A", "B"]
        assert SelectAll().select(ctx(md), scores) == list("ABCDE")


class TestWeightsAndOverlays:
    def test_weighters(self, md: MarketData) -> None:
        c = ctx(md)
        assert EqualWeight().weights(c, []).empty
        assert EqualWeight().weights(c, ["S00", "S01"]).tolist() == [0.5, 0.5]
        fw = FixedWeights({"S00": 0.6, "S01": 0.3}).weights(c, ["S00", "S01", "S02"])
        assert fw.tolist() == [0.6, 0.3, 0.0]
        with pytest.raises(StrategyError):
            FixedWeights({"S00": 0.8, "S01": 0.3})
        iv = InverseVolatility(20).weights(c, ["S00", "S01", "S02"])
        assert iv.sum() == pytest.approx(1.0)
        assert InverseVolatility(20).weights(c, []).empty

    def test_max_weight_redistributes(self, md: MarketData) -> None:
        w = pd.Series({"A": 0.7, "B": 0.2, "C": 0.1})
        out = MaxWeight(0.4).apply(ctx(md), w)
        assert out["A"] == pytest.approx(0.4)
        assert out.sum() == pytest.approx(1.0)
        assert (out <= 0.4 + 1e-9).all()
        capped = MaxWeight(0.3).apply(ctx(md), pd.Series({"A": 0.5, "B": 0.5}))
        assert capped.sum() == pytest.approx(0.6)  # nowhere to put the excess -> cash
        with pytest.raises(StrategyError):
            MaxWeight(0).apply(ctx(md), w)

    def test_regime(self) -> None:
        bars = make_bars({"A": [10.0] * 30})
        w = pd.Series({"A": 1.0})
        up = market(bars, benchmark=[100.0 + i for i in range(30)])
        c = StrategyContext(up.dates[-1], up)
        assert RegimeFilter(sma=10).apply(c, w)["A"] == 1.0
        assert "risk-on" in c.notes[0]
        down = market(bars, benchmark=[100.0 - i for i in range(30)])
        c = StrategyContext(down.dates[-1], down)
        assert RegimeFilter(sma=10, risk_off_exposure=0.25).apply(c, w)["A"] == 0.25
        assert "risk-off" in c.notes[0]
        c = StrategyContext(down.dates[3], down)
        assert RegimeFilter(sma=10).apply(c, w)["A"] == 1.0  # not enough history
        with pytest.raises(StrategyError, match="benchmark"):
            RegimeFilter(benchmark="NOPE").apply(c, w)

    def test_vol_target_and_sector_cap(self, md: MarketData) -> None:
        c = ctx(md)
        w = pd.Series({"S00": 0.5, "S01": 0.5})
        scaled = VolatilityTarget(target=0.05, window=20).apply(c, w)
        assert scaled.sum() < 1.0
        assert "vol target" in c.notes[-1]
        assert VolatilityTarget(target=5.0, window=20).apply(c, w).sum() == 1.0
        assert VolatilityTarget().apply(c, pd.Series(dtype=float)).empty
        early = StrategyContext(md.dates[5], md)
        assert VolatilityTarget(window=63).apply(early, w).sum() == 1.0
        sc = SectorCap(0.3).apply(c, pd.Series({"S00": 0.4, "S01": 0.4, "S02": 0.2}))
        assert sc[["S00", "S01"]].sum() == pytest.approx(0.3)
        assert sc["S02"] == pytest.approx(0.2)


class TestPipeline:
    def test_end_to_end(self, md: MarketData) -> None:
        s = momentum_strategy()
        t = s.target(ctx(md))
        assert len(t.weights) == 3
        assert t.invested == pytest.approx(1.0)
        assert t.notes[0].startswith("universe 8 -> after filters 8 -> selected 3")
        assert all(t.rationale[sym] for sym in t.weights)
        assert s.rebalance_dates(md.dates)[-1] == md.dates[-1]
        with pytest.raises(StrategyError, match="not a trading day"):
            s.target(StrategyContext(pd.Timestamp("2030-01-01"), md))

    def test_no_look_ahead_end_to_end(self, md: MarketData) -> None:
        """Targets computed on the full history equal those computed on truncated history."""
        s = momentum_strategy(buffer=5)
        for i in (120, 200, 260):
            full = s.target(ctx(md, i))
            cut = s.target(StrategyContext(md.dates[i], md.truncate(md.dates[i])))
            assert full.weights == cut.weights

    def test_hysteresis_reduces_turnover(self, md: MarketData) -> None:
        def turnover(strategy: PipelineStrategy) -> float:
            held: dict[str, float] = {}
            total = 0.0
            for d in md.dates[100::5]:
                t = strategy.target(StrategyContext(d, md, held))
                syms = set(held) | set(t.weights)
                total += sum(abs(t.weights.get(s, 0) - held.get(s, 0)) for s in syms) / 2
                held = dict(t.weights)
            return total

        assert turnover(momentum_strategy(buffer=6)) < turnover(momentum_strategy())

    def test_blend(self, md: MarketData) -> None:
        core = PipelineStrategy("core", StaticUniverse(["S00"]), FixedWeights({"S00": 1.0}))
        sat = momentum_strategy()
        b = Blend("cs", [(core, 0.7), (sat, 0.3)])
        t = b.target(ctx(md, current={"S00": 0.7}))
        assert t.weights["S00"] >= 0.7 - 1e-9
        assert t.invested == pytest.approx(1.0)
        assert any(r.startswith("[mom 30%]") for rs in t.rationale.values() for r in rs)
        assert b.rebalance_dates(md.dates)
        with pytest.raises(StrategyError):
            Blend("bad", [(core, 0.8), (sat, 0.5)])


def test_component_registry_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="kind"):
        component("nope", "x")
