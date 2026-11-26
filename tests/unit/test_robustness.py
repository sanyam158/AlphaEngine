from __future__ import annotations

import json
import math
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from alphaengine.backtest.config import BacktestConfig
from alphaengine.backtest.engine import simulate
from alphaengine.backtest.vectorized import default_cost_per_side, run_vectorized
from alphaengine.core.errors import StrategyError
from alphaengine.features.data import MarketData
from alphaengine.robustness import stats
from alphaengine.robustness.analysis import analyze, warnings_for
from alphaengine.robustness.report import write
from alphaengine.robustness.sweep import (
    expand_grid,
    get_path,
    run_sweep,
    set_path,
    stability,
    variant,
)
from alphaengine.strategy.config import parse_config
from tests.helpers import random_walk_bars
from tests.unit.test_strategy import momentum_strategy

BASE = {
    "name": "syn",
    "universe": {"type": "liquidity", "top_n": 8, "window": 20},
    "filters": [{"type": "min_history", "days": 60}],
    "scorer": {"type": "feature", "feature": "momentum", "params": {"lookback": 60, "skip": 5}},
    "selector": {"type": "top_n", "n": 3, "buffer": 5},
}


@pytest.fixture(scope="module")
def md() -> MarketData:
    return MarketData.from_bars(
        random_walk_bars(n_symbols=10, n_days=900, seed=9, gap_prob=0.0), execution=True
    )


class TestVectorized:
    def test_parity_with_full_engine(self, md: MarketData) -> None:
        s = momentum_strategy(buffer=5)
        start = md.dates[80].date()
        cfg = BacktestConfig(
            start=start,
            initial_capital=1e6,
            broker="none",
            statutory_costs=False,
            slippage="fixed",
            slippage_bps=0,
            taxes=False,
            cash_rate=0.0,
            cash_buffer=0.0,
            min_trade_value=0,
            max_participation=1.0,
            benchmark="S00",
        )
        full = simulate(s, md, cfg)
        fast = run_vectorized(
            s, md, start, cost_per_side=0, slippage_bps=0, cash_rate=0, capital=1e6
        )
        a, b = full.equity["value"], fast.equity["value"]
        assert float((a / b - 1).abs().max()) < 0.01  # integer shares / sizing at the close only
        assert [t.weights.keys() for t in full.targets] == [t.weights.keys() for t in fast.targets]

    def test_costs_and_cash(self, md: MarketData) -> None:
        s = momentum_strategy()
        start = md.dates[80].date()
        free = run_vectorized(s, md, start, cost_per_side=0, slippage_bps=0, cash_rate=0)
        costly = run_vectorized(s, md, start, cost_per_side=0.002, slippage_bps=10, cash_rate=0)
        assert costly.equity["value"].iloc[-1] < free.equity["value"].iloc[-1]
        assert costly.total_costs > 0
        assert free.turnover.iloc[0] == pytest.approx(1.0)  # cash -> fully invested
        assert len(free.returns) == len(free.equity) - 1
        assert 0.001 < default_cost_per_side(on=date(2026, 1, 15)) < 0.0015

    def test_cash_only_and_errors(self, md: MarketData) -> None:
        from alphaengine.strategy.pipeline import PipelineStrategy
        from alphaengine.strategy.stages import FixedWeights, StaticUniverse

        cash = PipelineStrategy("cash", StaticUniverse(["S00"]), FixedWeights({}))
        res = run_vectorized(cash, md, md.dates[0].date(), md.dates[10].date(), cash_rate=0.0504)
        assert res.equity["value"].iloc[-1] == pytest.approx((1 + 0.0504 / 252) ** 10)
        with pytest.raises(ValueError, match="execution"):
            run_vectorized(
                cash, MarketData.from_bars(random_walk_bars(n_days=10)), date(2023, 1, 2)
            )
        with pytest.raises(ValueError, match="two trading days"):
            run_vectorized(cash, md, md.dates[-1].date())


class TestSweep:
    def test_paths(self) -> None:
        cfg = {"a": {"b": [1, {"c": 2}]}, "x": 1}
        assert set_path(cfg, "a.b.1.c", 5)["a"]["b"][1]["c"] == 5
        assert cfg["a"]["b"][1]["c"] == 2  # deep copy
        assert set_path(cfg, "a.new", 1)["a"]["new"] == 1
        assert get_path(cfg, "a.b.0") == 1
        with pytest.raises(StrategyError, match="out of range"):
            set_path(cfg, "a.b.5", 1)
        with pytest.raises(StrategyError, match="no key"):
            set_path(cfg, "a.zz.c", 1)
        with pytest.raises(StrategyError, match="not a mapping"):
            set_path(cfg, "x.y", 1)
        assert expand_grid({"a": [1, 2], "b": ["x"]}) == [{"a": 1, "b": "x"}, {"a": 2, "b": "x"}]
        assert expand_grid({}) == [{}]

    def test_variant_and_sweep(self, md: MarketData) -> None:
        base = parse_config(BASE)
        v = variant(base, {"selector.n": 4, "overlays": []})
        assert v.name.startswith("syn__n4")
        assert v.selector is not None and v.selector.params()["n"] == 4  # type: ignore[union-attr]
        res = run_sweep(base, {"selector.n": [2, 4]}, md, md.dates[100].date())
        assert list(res.metrics["selector.n"]) == [2, 4]
        assert res.returns_matrix().shape[1] == 2
        st = stability(res.metrics, ["selector.n"])
        assert st["best"] >= st["median"] >= st["worst"]
        assert stability(res.metrics.iloc[:0], []) == {}


class TestStats:
    def setup_method(self) -> None:
        rng = np.random.default_rng(0)
        idx = pd.bdate_range("2012-01-02", periods=2500)
        self.noise = pd.DataFrame(rng.normal(0, 0.01, (2500, 10)), index=idx)
        good = self.noise.copy()
        good[0] = rng.normal(0.0015, 0.01, 2500)  # one genuinely better variant
        self.good = good

    def test_sharpe_significance(self) -> None:
        strong = self.good[0]
        weak = self.noise[self.noise.mean().idxmin()]  # the unluckiest noise column
        assert stats.probabilistic_sharpe(strong) > 0.99
        assert stats.probabilistic_sharpe(weak) < 0.5
        d = stats.deflated_sharpe(strong, self.good)
        assert d["deflated_sharpe"] <= d["psr_vs_zero"]
        assert d["trials"] == 10
        assert d["deflated_sharpe"] > 0.95
        assert stats.deflated_sharpe(weak, self.noise)["deflated_sharpe"] < 0.5
        assert math.isnan(stats.probabilistic_sharpe(pd.Series([0.0, 0.0, 0.0])))
        assert stats.expected_max_sharpe(np.array([0.1])) == 0.0

    def test_pbo(self) -> None:
        assert stats.pbo(self.good)["pbo"] < 0.1
        # For pure noise PBO should hover around 50% (any single sample can be lucky).
        noise_pbo = [
            stats.pbo(pd.DataFrame(np.random.default_rng(s).normal(0, 0.01, (1500, 8))))["pbo"]
            for s in range(6)
        ]
        assert 0.3 < float(np.mean(noise_pbo)) < 0.7
        assert math.isnan(stats.pbo(self.noise[[0]])["pbo"])
        assert stats.pbo(self.noise, metric="mean")["splits"] == 252

    def test_bootstrap_and_walk_forward(self) -> None:
        b = stats.stationary_bootstrap(self.good[0], n_paths=200)
        assert b["cagr"]["p5"] < b["cagr"]["p50"] < b["cagr"]["p95"]
        assert b["max_drawdown"]["p50"] < 0
        assert stats.stationary_bootstrap(self.good[0].iloc[:10]) == {}
        wf = stats.walk_forward(self.good, n_folds=4)
        assert all(f["chosen"] == 0 for f in wf["folds"])  # always picks the real edge
        assert wf["oos_sharpe"] > 1
        noise_wf = stats.walk_forward(self.noise, n_folds=4)
        assert noise_wf["mean_in_sample_sharpe"] > noise_wf["oos_sharpe"]
        assert stats.walk_forward(self.noise.iloc[:550]) == {}

    def test_regimes(self) -> None:
        bench = pd.Series(
            np.r_[np.linspace(100, 200, 1250), np.linspace(200, 100, 1250)], index=self.good.index
        )
        out = stats.regime_split(self.good[0], bench)
        assert set(out) == {"benchmark_uptrend", "benchmark_downtrend"}
        assert (
            out["benchmark_uptrend"]["benchmark_annual"]
            > 0
            > out["benchmark_downtrend"]["benchmark_annual"]
        )
        assert math.isnan(stats.annualise(pd.Series(dtype=float)))


def test_analyze_report_and_warnings(md: MarketData, tmp_path: Path) -> None:
    base = parse_config(BASE)
    r = analyze(base, {"selector.n": [2, 4]}, md, md.dates[150].date(), benchmark="S00", folds=3)
    assert r.sweep.params[r.base_index] == {"selector.n": 3}  # base value added to the grid
    assert len(r.sweep.params) == 3
    assert r.confirmation is not None
    assert set(r.costs) == {"0x", "1x", "2x"}
    assert r.costs["0x"] > r.costs["1x"] > r.costs["2x"]
    assert isinstance(r.warnings, list)
    h, j = write(r, tmp_path)
    assert "Parameter stability" in h.read_text(encoding="utf-8")
    assert json.loads(j.read_text(encoding="utf-8"))["strategy"] == "syn"
    two = analyze(
        base,
        {"selector.n": [2], "scorer.params.lookback": [40]},
        md,
        md.dates[150].date(),
        benchmark="S00",
        folds=3,
        confirm=False,
    )
    assert two.confirmation is None
    assert "rows selector.n" in write(two, tmp_path / "b")[0].read_text(encoding="utf-8")
    # warnings fire on clearly bad inputs
    two.overfitting["pbo"]["pbo"] = 0.9
    two.costs["2x"] = -0.5
    msgs = " ".join(warnings_for(two))
    assert "PBO 90%" in msgs
    assert "double costs" in msgs


def test_robust_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from alphaengine.cli.main import app
    from alphaengine.cli.robust_commands import parse_grid
    from alphaengine.config.settings import get_settings
    from alphaengine.data.store import PriceStore

    assert parse_grid(["selector.n=10, 20", "x.y=true,0.5,abc"]) == {
        "selector.n": [10, 20],
        "x.y": [True, 0.5, "abc"],
    }
    monkeypatch.setenv("ALPHAENGINE_DATABASE_URL", f"sqlite:///{(tmp_path / 'ae.db').as_posix()}")
    monkeypatch.setenv("ALPHAENGINE_DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()
    bars = random_walk_bars(n_symbols=4, n_days=700, seed=3, gap_prob=0.0)
    names = {"S00": "NIFTYBEES", "S01": "JUNIORBEES", "S02": "GOLDBEES", "S03": "BANKBEES"}
    bars["symbol"] = bars["symbol"].map(lambda s: names[str(s)]).astype("string")
    PriceStore(tmp_path / "data" / "curated", "equities").write(bars)
    runner = CliRunner()
    assert runner.invoke(app, ["db", "upgrade"]).exit_code == 0
    r = runner.invoke(
        app,
        [
            "robust",
            "run",
            "dual_momentum",
            "--start",
            "2023-09-01",
            "--grid",
            "scorer.params.lookback=63,126",
            "--folds",
            "2",
        ],
    )
    assert r.exit_code == 0, r.output
    assert "deflated Sharpe" in r.output
    out = re.search(r"report: (.+robustness\.html)", r.output)
    assert out is not None
    assert Path(out.group(1).strip()).exists()
    bad = runner.invoke(app, ["robust", "run", "dual_momentum", "--grid", "nonsense"])
    assert bad.exit_code != 0
