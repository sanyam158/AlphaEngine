"""Runner + persistence + CLI on a small synthetic store (no network)."""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from alphaengine.app import build_context
from alphaengine.backtest.config import BacktestConfig
from alphaengine.backtest.engine import BacktestError
from alphaengine.backtest.runner import (
    benchmark_series,
    get_run,
    list_runs,
    required_symbols,
    run_backtest,
)
from alphaengine.data.db import Base
from alphaengine.data.store import PriceStore
from alphaengine.features.data import MarketData
from alphaengine.strategy.config import build_strategy, load_config, parse_config
from tests.helpers import random_walk_bars


def _store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPHAENGINE_DATABASE_URL", f"sqlite:///{(tmp_path / 'ae.db').as_posix()}")
    monkeypatch.setenv("ALPHAENGINE_DATA_DIR", str(tmp_path / "data"))
    from alphaengine.config.settings import get_settings

    get_settings.cache_clear()
    bars = random_walk_bars(n_symbols=6, n_days=320, seed=2, gap_prob=0.0)
    names = {"S00": "NIFTYBEES", "S01": "JUNIORBEES", "S02": "GOLDBEES"}
    bars["symbol"] = bars["symbol"].map(lambda s: names.get(str(s), str(s))).astype("string")
    PriceStore(tmp_path / "data" / "curated", "equities").write(bars)


@pytest.fixture
def cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> CliRunner:
    _store(tmp_path, monkeypatch)
    from alphaengine.cli.main import app

    runner = CliRunner()
    assert runner.invoke(app, ["db", "upgrade"]).exit_code == 0
    return runner


def test_run_persist_and_cli(cli: CliRunner, tmp_path: Path) -> None:
    from alphaengine.cli.main import app

    r = cli.invoke(app, ["backtest", "run", "fixed_mix", "--start", "2023-06-01"])
    assert r.exit_code == 0, r.output
    assert "after-tax XIRR" in r.output
    run_id = re.search(r"run (bt_\w+):", r.output).group(1)  # type: ignore[union-attr]
    rep = tmp_path / "data" / "runs" / run_id
    assert (rep / "report.html").exists()
    assert (rep / "equity.parquet").exists()
    assert json.loads((rep / "metrics.json").read_text(encoding="utf-8"))["trades"] >= 3

    r2 = cli.invoke(
        app,
        [
            "backtest",
            "run",
            "buy_hold_nifty",
            "--start",
            "2023-06-01",
            "--no-costs",
            "--no-tax",
            "--report",
            str(tmp_path / "out"),
        ],
    )
    assert r2.exit_code == 0, r2.output
    assert (tmp_path / "out" / "report.html").exists()
    run2 = re.search(r"run (bt_\w+):", r2.output).group(1)  # type: ignore[union-attr]

    listing = cli.invoke(app, ["backtest", "list"]).output
    assert run_id in listing
    assert run2 in listing
    shown = cli.invoke(app, ["backtest", "show", run_id]).output
    assert "fixed_mix v1" in shown
    assert "cagr" in shown
    cmp_ = cli.invoke(app, ["backtest", "compare", run_id, run2]).output
    assert "after_tax_xirr" in cmp_

    ctx = build_context()
    rec, eq = get_run(ctx, run_id)
    assert rec.data_snapshot == ctx.equities.snapshot_id()
    assert rec.strategy_config["name"] == "fixed_mix"
    assert len(eq) > 50
    assert "benchmark" in eq.columns
    assert len(list_runs(ctx)) == 2
    with pytest.raises(BacktestError, match="no backtest run"):
        get_run(ctx, "bt_missing")


def test_required_symbols_and_benchmarks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _store(tmp_path, monkeypatch)
    ctx = build_context()
    Base.metadata.create_all(ctx.engine)
    cfg = BacktestConfig(start=date(2023, 6, 1))
    assert required_symbols(build_strategy(load_config("fixed_mix")), ctx, cfg) == [
        "GOLDBEES",
        "JUNIORBEES",
        "NIFTYBEES",
    ]
    liq = required_symbols(build_strategy(load_config("momentum_top20")), ctx, cfg)
    assert liq is not None
    assert set(liq) == {"NIFTYBEES", "JUNIORBEES", "GOLDBEES", "S03", "S04", "S05"}
    blend = required_symbols(build_strategy(load_config("core_satellite")), ctx, cfg)
    assert blend == sorted(set(liq) | {"BANKBEES"})  # dual_momentum's static ETFs + liquidity set
    idx = parse_config({"name": "idx", "universe": {"type": "index", "index": "NIFTY50"}})
    assert required_symbols(build_strategy(idx), ctx, cfg) == []

    md = MarketData.from_bars(
        random_walk_bars(n_symbols=2, n_days=10),
        benchmark_bars=pd.DataFrame(
            {
                "date": pd.bdate_range("2023-01-02", periods=3),
                "symbol": "NIFTY50",
                "close": [1.0, 2, 3],
            }
        ),
    )
    assert benchmark_series(md, "S00").notna().any()
    assert benchmark_series(md, "NIFTY50").iloc[0] == 1.0
    with pytest.raises(BacktestError, match="benchmark"):
        benchmark_series(md, "NOPE")
    run = run_backtest("fixed_mix", BacktestConfig(start=date(2023, 6, 1)), ctx, persist=False)
    assert run.metrics["benchmark"]["cagr"] == pytest.approx(run.metrics["benchmark"]["cagr"])
