from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alphaengine.backtest import metrics as mt
from alphaengine.backtest.report import render_html, write_reports


def equity(
    values: list[float],
    flows: list[float] | None = None,
    start: str = "2020-01-01",
    freq: str = "B",
) -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(values), freq=freq)
    flows = flows or [0.0] * len(values)
    contrib = np.cumsum([values[0], *flows[1:]]).tolist()
    return pd.DataFrame({"value": values, "flow": flows, "contributions": contrib}, index=idx)


class TestReturnMath:
    def test_twr_removes_flows(self) -> None:
        eq = equity([100, 110, 220], [0, 0, 100])  # +10%, then +100 contribution and +9.09%
        r = mt.twr_returns(eq)
        assert r.tolist() == pytest.approx([0.10, 120 / 110 - 1])

    def test_cagr_and_index(self) -> None:
        idx = pd.Series([1.0, 2.0], index=pd.to_datetime(["2020-01-01", "2021-12-31"]))
        assert mt.cagr(idx) == pytest.approx(2 ** (365.25 / 730) - 1)
        assert math.isnan(mt.cagr(idx.iloc[:1]))
        assert mt.growth_index(pd.Series([0.1, -0.5])).tolist() == pytest.approx([1.1, 0.55])

    def test_risk_measures(self) -> None:
        idx = pd.Series(
            [100, 120, 90, 60, 130, 125.0], index=pd.bdate_range("2020-01-01", periods=6)
        )
        dd = mt.max_drawdown(idx)
        assert dd["max_drawdown"] == pytest.approx(-0.5)
        assert dd["peak"] == date(2020, 1, 2)
        assert dd["trough"] == date(2020, 1, 6)
        assert dd["recovery"] == date(2020, 1, 7)
        assert mt.ulcer_index(idx) > 0
        never = mt.max_drawdown(
            pd.Series([100, 80, 70.0], index=pd.bdate_range("2020-01-01", periods=3))
        )
        assert never["recovery"] is None
        r = pd.Series([0.01, -0.01] * 50)
        assert mt.annual_vol(r) == pytest.approx(r.std() * math.sqrt(252))
        assert math.isnan(mt.sharpe(pd.Series([0.001] * 10)))  # zero volatility
        assert mt.sortino(pd.Series([0.01, 0.02])) != mt.sortino(pd.Series([0.01, -0.02]))
        assert math.isnan(mt.sortino(pd.Series([0.01, 0.02])))

    def test_xirr(self) -> None:
        assert mt.xirr([(date(2021, 1, 1), -100.0), (date(2022, 1, 1), 110.0)]) == pytest.approx(
            0.10, abs=1e-6
        )
        sip = [(date(2020, 1, 1), -100.0), (date(2020, 7, 1), -100.0), (date(2021, 1, 1), 210.0)]
        assert 0.06 < mt.xirr(sip) < 0.08
        assert math.isnan(mt.xirr([(date(2020, 1, 1), 100.0), (date(2021, 1, 1), 100.0)]))
        assert math.isnan(mt.xirr([(date(2020, 1, 1), 100.0)]))

    def test_relative_and_yearly(self) -> None:
        rng = np.random.default_rng(0)
        b = pd.Series(
            rng.normal(0.0005, 0.01, 300), index=pd.bdate_range("2020-01-01", periods=300)
        )
        rel = mt.relative(b * 2, b)
        assert rel["beta"] == pytest.approx(2.0)
        assert rel["tracking_error"] > 0
        same = mt.relative(b, b)
        assert same["beta"] == pytest.approx(1.0)
        assert math.isnan(same["information_ratio"])
        assert math.isnan(mt.relative(b.iloc[:5], b.iloc[:5])["beta"])
        idx = pd.Series(
            [100.0, 110.0, 121.0], index=pd.to_datetime(["2020-01-01", "2020-12-31", "2021-12-31"])
        )
        assert mt.yearly_returns(idx).tolist() == pytest.approx([0.10, 0.10])
        long = mt.growth_index(pd.Series(0.0004, index=pd.bdate_range("2015-01-01", periods=800)))
        roll = mt.rolling_cagr(long, 1)
        assert roll["pct_positive"] == 1.0
        assert mt.rolling_cagr(long, 5) == {}

    def test_jsonable(self) -> None:
        out = mt.to_jsonable(
            {"a": float("nan"), "b": [date(2020, 1, 1), 1, "x"], 3: {"c": math.inf}}
        )
        assert out == {"a": None, "b": ["2020-01-01", 1, "x"], "3": {"c": None}}


def test_compute_metrics_and_reports(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    vals = (1_000_000 * np.cumprod(1 + rng.normal(0.0005, 0.01, 800))).tolist()
    eq = equity(vals)
    bench = pd.Series(np.cumprod(1 + rng.normal(0.0004, 0.01, 800)), index=eq.index)
    trades = pd.DataFrame({"value": [1.0] * 10})

    class D:
        def __init__(self, gain: float, lt: bool) -> None:
            self.quantity, self.gain, self.long_term = 10, gain, lt
            self.acquired, self.sold = date(2020, 1, 1), date(2021, 1, 1)

    m = mt.compute_metrics(
        eq,
        trades=trades,
        disposals=[D(5, True), D(-1, False)],
        benchmark=bench,
        rf=0.05,
        after_tax_value=vals[-1] * 0.95,
        totals={"costs": 1000, "capital_gains_tax": 500, "turnover": 4e6},
    )
    assert m["trades"] == 10
    assert m["hit_rate"] == 0.5
    assert m["long_term_share"] == 0.5
    assert m["avg_holding_days"] == 366
    assert m["after_tax_xirr"] < m["xirr"]
    assert "benchmark" in m
    assert m["excess_cagr"] == pytest.approx(m["cagr"] - m["benchmark"]["cagr"])
    assert "rolling_1y" in m
    assert m["turnover"] > 0
    html_path, json_path = write_reports(
        tmp_path / "r",
        m,
        eq,
        bench,
        title="test",
        config={"benchmark": "X"},
        strategy_config={"name": "t"},
        notes=["n1"],
        run_id="bt_1",
    )
    text = html_path.read_text(encoding="utf-8")
    assert "<svg" in text
    assert "Calendar-year returns" in text
    assert "n1" in text
    assert json.loads(json_path.read_text(encoding="utf-8"))["trades"] == 10
    assert "<svg" not in render_html(mt.compute_metrics(eq), eq, None, title="x", config={})[:10]
