from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import pytest

from alphaengine.core.errors import StrategyError
from alphaengine.features.base import FeatureSpec, compute, feature, registered_features
from alphaengine.features.data import MarketData
from tests.helpers import make_bars, market, random_walk_bars

# Small windows so every feature has values within 300 rows of synthetic data.
SMALL: dict[str, dict[str, Any]] = {
    "returns": {"window": 5},
    "momentum": {"lookback": 40, "skip": 5},
    "volatility": {"window": 20},
    "risk_adj_momentum": {"lookback": 40, "skip": 5, "vol_window": 20},
    "sma": {"window": 20},
    "price_to_sma": {"window": 20},
    "high_proximity": {"window": 30},
    "adv_value": {"window": 20},
    "rsi": {"window": 14},
    "atr": {"window": 14},
    "zscore": {"window": 20},
    "max_abs_return": {"window": 30},
}


@pytest.fixture(scope="module")
def md() -> MarketData:
    return market(random_walk_bars(), ohlc=True)


def test_every_feature_has_a_causality_case() -> None:
    assert set(registered_features()) - {"history_days"} == set(SMALL)


@pytest.mark.parametrize("name", sorted(registered_features()))
def test_features_are_causal(md: MarketData, name: str) -> None:
    """Row t computed on the full history must equal row t computed on data truncated at t."""
    spec = FeatureSpec.of(name, **SMALL.get(name, {}))
    full = md.feature(spec)
    assert full.notna().to_numpy().any(), f"{spec} produced no values"
    for t in md.dates[[60, 150, 299]]:
        trunc = compute(spec, md.truncate(t))
        pd.testing.assert_series_equal(
            full.loc[t], trunc.loc[t], check_names=False, check_dtype=False, rtol=1e-5
        )


def test_future_changes_do_not_leak(md: MarketData) -> None:
    """Shuffling everything after t must leave row t unchanged."""
    t = md.dates[200]
    spec = FeatureSpec.of("momentum", lookback=40, skip=5)
    before = compute(spec, md).loc[t]
    shuffled = md.truncate(md.dates[-1])
    for panel in (shuffled.close, shuffled.tr_close):
        future = panel.index > t
        panel.loc[future] = panel.loc[future].to_numpy()[::-1]
    after = compute(spec, shuffled).loc[t]
    pd.testing.assert_series_equal(before, after)


class TestKnownAnswers:
    def setup_method(self) -> None:
        # 1% growth per day for A; B flat then jumps
        a = [100 * 1.01**i for i in range(30)]
        b = [50.0] * 29 + [60.0]
        self.md = market(make_bars({"A": a, "B": b}), ohlc=True)
        self.last = self.md.dates[-1]

    def row(self, name: str, **params: Any) -> pd.Series:
        return self.md.feature_row(FeatureSpec.of(name, **params), self.last)

    def test_returns_and_momentum(self) -> None:
        assert self.row("returns", window=1)["B"] == pytest.approx(0.2)
        assert self.row("momentum", lookback=10, skip=2)["A"] == pytest.approx(
            1.01**8 - 1, rel=1e-5
        )
        with pytest.raises(StrategyError, match="skip"):
            self.row("momentum", lookback=5, skip=5)

    def test_trend_features(self) -> None:
        sma = self.row("sma", window=5)["B"]
        assert sma == pytest.approx(52.0)
        assert self.row("price_to_sma", window=5)["B"] == pytest.approx(60 / 52 - 1, rel=1e-5)
        assert self.row("high_proximity", window=10)["A"] == pytest.approx(1.0)
        assert self.row("zscore", window=5)["B"] > 1

    def test_risk_features(self) -> None:
        assert self.row("volatility", window=10)["A"] == pytest.approx(0.0, abs=1e-5)
        assert self.row("max_abs_return", window=5)["B"] == pytest.approx(0.2)
        assert self.row("rsi", window=5)["A"] == pytest.approx(100.0)
        assert self.row("atr", window=5)["B"] > 0
        assert np.isnan(self.row("risk_adj_momentum", lookback=10, skip=0, vol_window=10)["A"])

    def test_liquidity_and_history(self) -> None:
        assert self.row("history_days")["A"] == 30
        adv = self.row("adv_value", window=5)
        assert adv["A"] == pytest.approx(1000 * 100 * 1.01**27, rel=1e-4)

    def test_atr_needs_ohlc(self) -> None:
        md = market(make_bars({"A": [1.0, 2.0, 3.0]}))
        with pytest.raises(StrategyError, match="OHLC"):
            md.feature(FeatureSpec.of("atr", window=2))


class TestFeatureSpec:
    def test_spec_basics(self) -> None:
        a = FeatureSpec.of("momentum")
        b = FeatureSpec.of("momentum", lookback=252)
        assert a == b
        assert hash(a) == hash(b)
        assert str(a) == "momentum(lookback=252,price=tr,skip=21)"
        assert a.history == 252
        with pytest.raises(StrategyError, match="no parameter"):
            FeatureSpec.of("momentum", window=3)
        with pytest.raises(StrategyError, match="unknown feature"):
            FeatureSpec.of("nope")

    def test_registry_guards(self) -> None:
        with pytest.raises(ValueError, match="already registered"):
            feature("momentum", history=lambda p: 0)(lambda data: pd.DataFrame())

    def test_bad_feature_output(self, md: MarketData) -> None:
        feature("bad_output_for_test", history=lambda p: 0)(lambda data: pd.Series(dtype=float))
        with pytest.raises(StrategyError, match="did not return"):
            compute(FeatureSpec.of("bad_output_for_test"), md)

    def test_cache(self, md: MarketData) -> None:
        spec = FeatureSpec.of("sma", window=20)
        assert md.feature(spec) is md.feature(spec)


class TestMarketData:
    def test_panels_and_adjustment(self) -> None:
        bars = make_bars({"A": [100, 102, 51, 52], "B": [10, 10, 10, 10]})
        acts = pd.DataFrame(
            [
                {
                    "symbol": "A",
                    "ex_date": "2024-01-03",
                    "kind": "SPLIT",
                    "ratio": 2.0,
                    "amount": None,
                },
                {
                    "symbol": "B",
                    "ex_date": "2024-01-04",
                    "kind": "DIVIDEND",
                    "ratio": None,
                    "amount": 1.0,
                },
            ]
        )
        md = MarketData.from_bars(bars, acts, asset_classes={"B": "ETF"})
        assert list(md.close["A"]) == [50, 51, 51, 52]
        assert list(md.raw_close["A"]) == [100, 102, 51, 52]
        assert md.tr_close["B"].iloc[0] == pytest.approx(9.0)
        assert md.symbols == ["A", "B"]
        assert md.asset_class("B") == "ETF"
        assert md.asset_class("A") == "EQUITY"
        assert md.close.dtypes.unique().tolist() == [np.float32]

    def test_truncate_and_dates(self, md: MarketData) -> None:
        t = md.dates[10]
        cut = md.truncate(t)
        assert cut.dates[-1] == t
        assert cut.high is not None
        assert md.last_date_on_or_before(date(2023, 1, 1)) is None
        assert md.last_date_on_or_before(pd.Timestamp("2023-01-08")) == pd.Timestamp("2023-01-06")
        with pytest.raises(KeyError, match="not a trading day"):
            md.feature_row(FeatureSpec.of("sma", window=5), pd.Timestamp("2023-01-07"))

    def test_benchmarks_and_memberships(self) -> None:
        bars = make_bars({"A": [1.0] * 5})
        mem = pd.DataFrame(
            [
                {
                    "index_name": "NIFTY50",
                    "symbol": "A",
                    "start_date": date(2024, 1, 2),
                    "end_date": None,
                },
                {
                    "index_name": "NIFTY50",
                    "symbol": "B",
                    "start_date": date(2023, 1, 1),
                    "end_date": date(2024, 1, 3),
                },
            ]
        )
        md = market(bars, benchmark=[10.0, 11.0, 12.0], memberships=mem)
        assert md.benchmarks["NIFTY50"].tolist() == [10, 11, 12, 12, 12]  # forward-filled
        assert md.index_members("nifty50", date(2024, 1, 2)) == ["A", "B"]
        assert md.index_members("NIFTY50", date(2024, 1, 3)) == ["A"]
        assert market(bars).index_members("NIFTY50", date(2024, 1, 3)) == []
