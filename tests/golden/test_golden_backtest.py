"""Golden regression: a fixed synthetic market + strategy must keep producing the same metrics.

If an intentional engine change moves these numbers, regenerate with
``UPDATE_GOLDEN=1 pytest tests/golden`` and review the diff of the JSON in the PR.
"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

import pytest

from alphaengine.backtest.config import BacktestConfig
from alphaengine.backtest.engine import simulate
from alphaengine.backtest.metrics import compute_metrics
from alphaengine.features.data import MarketData
from alphaengine.strategy.core import Schedule
from alphaengine.strategy.pipeline import PipelineStrategy
from alphaengine.strategy.stages import (
    EqualWeight,
    FeatureScore,
    LiquidityUniverse,
    MaxWeight,
    MinHistory,
    TopN,
)
from tests.helpers import random_walk_bars

pytestmark = pytest.mark.golden
GOLDEN = Path(__file__).with_name("momentum_synthetic.json")
KEYS = ["final_value", "cagr", "volatility", "max_drawdown", "turnover", "trades", "costs", "taxes"]


def run() -> dict[str, float]:
    md = MarketData.from_bars(
        random_walk_bars(n_symbols=10, n_days=420, seed=11, gap_prob=0.01), execution=True
    )
    strategy = PipelineStrategy(
        name="golden_momentum",
        universe=LiquidityUniverse(top_n=8, window=20),
        filters=[MinHistory(60)],
        scorer=FeatureScore("momentum", {"lookback": 60, "skip": 5}),
        selector=TopN(3, buffer=5),
        weighter=EqualWeight(),
        overlays=[MaxWeight(0.5)],
        schedule=Schedule("monthly"),
    )
    cfg = BacktestConfig(start=md.dates[80].date(), initial_capital=1_000_000, benchmark="S00")
    res = simulate(strategy, md, cfg)
    m = compute_metrics(
        res.equity,
        trades=res.trades,
        disposals=res.disposals,
        rf=cfg.cash_rate,
        totals={k.removeprefix("total_"): v for k, v in res.liquidation.items()},
    )
    return {k: round(float(m[k]), 6) for k in KEYS}


def test_golden_momentum_backtest() -> None:
    got = run()
    if os.environ.get("UPDATE_GOLDEN") == "1" or not GOLDEN.exists():
        GOLDEN.write_text(json.dumps(got, indent=2) + "\n", encoding="utf-8")
        if os.environ.get("CI"):
            pytest.fail("golden file was missing in CI")
    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))
    for k in KEYS:
        assert got[k] == pytest.approx(expected[k], rel=1e-6, abs=1e-6), k
    assert date(2023, 1, 2)  # keeps the date import meaningful for readers of this file
