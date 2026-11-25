from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from alphaengine.data.adjust import adjustment_factors, apply_adjustment
from alphaengine.data.quality import QualityConfig, run_checks, summarize
from alphaengine.data.universe import liquidity_universe
from tests.helpers import make_bars


def _acts(rows: list[tuple[str, str, str, float | None, float | None]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["symbol", "ex_date", "kind", "ratio", "amount"]).assign(
        source="test"
    )


# Business days from 2024-01-01: Mon 1, Tue 2, Wed 3, Thu 4, Fri 5
SPLIT_BARS = make_bars({"AAA": [100, 102, 51, 52], "BBB": [10, 10, 10, 10]})
SPLIT = _acts([("AAA", "2024-01-03", "SPLIT", 2.0, None)])


class TestAdjustment:
    def test_split_known_answer(self) -> None:
        adj = apply_adjustment(SPLIT_BARS, SPLIT, "split")
        aaa = adj[adj["symbol"] == "AAA"]
        assert list(aaa["close"]) == [50, 51, 51, 52]
        assert list(aaa["volume"]) == [2000, 2000, 1000, 1000]
        assert list(adj[adj["symbol"] == "BBB"]["close"]) == [10, 10, 10, 10]
        # prev_close is recomputed from adjusted closes
        assert aaa["prev_close"].iloc[2] == 51

    def test_bonus_and_dividend_total_return(self) -> None:
        bars = make_bars({"AAA": [100, 100, 98, 98]})
        acts = _acts(
            [
                ("AAA", "2024-01-03", "DIVIDEND", None, 2.0),
                ("AAA", "2024-01-04", "BONUS", 2.0, None),
            ]
        )
        f = adjustment_factors(bars, acts)
        np.testing.assert_allclose(f["split_factor"], [0.5, 0.5, 0.5, 1.0])
        np.testing.assert_allclose(f["tr_factor"], [0.49, 0.49, 0.5, 1.0])
        tr = apply_adjustment(bars, acts, "total")
        rets = tr["close"].pct_change().dropna().to_numpy()
        # Dividend day: total return = (98 + 2) / 100 - 1 = 0. Bonus day: price halves raw.
        np.testing.assert_allclose(rets[1], 0.0, atol=1e-12)

    def test_noop_cases(self) -> None:
        assert apply_adjustment(SPLIT_BARS, SPLIT, "none").equals(SPLIT_BARS)
        f = adjustment_factors(SPLIT_BARS, _acts([]))
        assert (f["split_factor"] == 1).all()
        with pytest.raises(ValueError, match="unknown adjustment"):
            apply_adjustment(SPLIT_BARS, SPLIT, "weird")  # type: ignore[arg-type]

    def test_dividend_after_window_is_ignored(self) -> None:
        bars = make_bars({"AAA": [10, 10, 10]})
        acts = _acts([("AAA", "2030-01-01", "DIVIDEND", None, 50.0)])  # > price, but later
        assert (adjustment_factors(bars, acts)["tr_factor"] == 1).all()

    def test_implausible_or_early_dividend_skipped(self) -> None:
        bars = make_bars({"AAA": [10, 10, 10]})
        acts = _acts(
            [
                ("AAA", "2024-01-02", "DIVIDEND", None, 50.0),  # > price
                ("AAA", "2023-06-01", "DIVIDEND", None, 1.0),  # before any bar
            ]
        )
        assert (adjustment_factors(bars, acts)["tr_factor"] == 1).all()

    @given(
        st.lists(st.floats(min_value=1, max_value=1000), min_size=3, max_size=30),
        st.sampled_from([2.0, 5.0, 10.0, 1.5]),
        st.data(),
    )
    def test_split_adjusted_returns_are_continuous(
        self, prices: list[float], ratio: float, data: st.DataObject
    ) -> None:
        """Simulate a split in raw prices; the adjusted series must equal the unsplit one."""
        k = data.draw(st.integers(min_value=1, max_value=len(prices) - 1))
        raw = [p if i < k else p / ratio for i, p in enumerate(prices)]
        bars = make_bars({"X": raw})
        ex = bars["date"].iloc[k].date().isoformat()
        adj = apply_adjustment(bars, _acts([("X", ex, "SPLIT", ratio, None)]), "split")
        expected = np.array(prices) / ratio
        np.testing.assert_allclose(adj["close"].to_numpy(), expected, rtol=1e-9)


class TestQuality:
    def test_clean_data_has_no_warnings(self) -> None:
        bars = make_bars({"AAA": [100, 101, 102, 103, 104, 105], "BBB": [10] * 6})
        days = sorted(set(bars["date"].dt.date))
        issues = run_checks(bars, days)
        assert [i for i in issues if i.severity != "info"] == []

    def test_detects_problems(self) -> None:
        bars = make_bars(
            {
                "AAA": [100, 102, 51, 52, 53, 54],  # unrecorded split
                "GAP": [10, np.nan, 10, 10, 10, 10],
                "OLD": [5, np.nan, np.nan, np.nan, np.nan, np.nan],
                "ZZZ": [1, 1, 1, 1, 1, 1],
            }
        )
        # Exchange adjusts PREVCLOSE on the ex-date of a split
        ex = bars["date"].iloc[0] + pd.offsets.BDay(2)
        bars.loc[(bars["symbol"] == "AAA") & (bars["date"] == ex), "prev_close"] = 51.0
        bars.loc[bars["symbol"] == "ZZZ", "volume"] = 0
        dupe = bars.iloc[[0]]
        bad = bars.iloc[[1]].assign(high=1.0)
        frame = pd.concat([bars, dupe, bad], ignore_index=True)
        days = sorted(set(bars["date"].dt.date))
        cfg = QualityConfig(stale_days=2)
        checks = {i.check for i in run_checks(frame, days, None, cfg)}
        assert {
            "duplicate_key",
            "ohlc_inconsistent",
            "missing_days",
            "price_spike",
            "prev_close_mismatch",
            "stale_symbol",
            "zero_volume",
        } <= checks

        # Recording the split clears both split-related warnings
        acts = _acts([("AAA", ex.date().isoformat(), "SPLIT", 2.0, None)])
        issues = run_checks(bars, days, acts, cfg)
        aaa = {i.check for i in issues if i.symbol == "AAA"}
        assert not aaa & {"price_spike", "prev_close_mismatch"}
        s = summarize(issues)
        assert set(s) == {"info", "warning", "error"}

    def test_weekend_special_session_is_lenient(self) -> None:
        bars = make_bars({f"S{i:02d}": [10.0] * 10 for i in range(10)})
        sunday = pd.Timestamp("2024-01-07")  # short special session: only S00 traded
        special = bars[bars["symbol"] == "S00"].head(1).assign(date=sunday)
        frame = pd.concat([bars, special], ignore_index=True)
        days = sorted(set(frame["date"].dt.date))
        issues = run_checks(frame, days)
        assert not [i for i in issues if i.check == "missing_days"]
        low = [i for i in issues if i.check == "low_coverage"]
        assert [(i.trade_date, i.severity) for i in low] == [(sunday.date(), "warning")]

    def test_low_coverage(self) -> None:
        spec = {f"S{i:02d}": [10.0] * 10 for i in range(10)}
        bars = make_bars(spec)
        day5 = bars["date"].drop_duplicates().iloc[5]
        bars = bars[~((bars["date"] == day5) & (bars["symbol"] != "S00"))]
        issues = run_checks(bars, sorted(set(bars["date"].dt.date)))
        low = [i for i in issues if i.check == "low_coverage"]
        assert [i.trade_date for i in low] == [day5.date()]
        assert run_checks(bars.iloc[:0], []) == []


class TestLiquidityUniverse:
    def test_ranks_by_median_value_point_in_time(self) -> None:
        bars = make_bars(
            {
                "BIG": [100] * 10,
                "MID": [50] * 10,
                "SMALL": [10] * 10,
                "DELISTED": [500, 500, 500, 500, 500, np.nan, np.nan, np.nan, np.nan, np.nan],
                "NEW": [np.nan] * 8 + [1000, 1000],
            }
        )
        d = bars["date"].drop_duplicates().sort_values()
        early = liquidity_universe(bars, d.iloc[4].date(), top_n=2, lookback_days=5)
        assert early == ["DELISTED", "BIG"]  # survivorship-free: delisted stock was investable
        late = liquidity_universe(bars, d.iloc[9].date(), top_n=3, lookback_days=5)
        assert late == ["BIG", "MID", "SMALL"]  # DELISTED gone, NEW lacks history
        assert liquidity_universe(bars, date(2024, 1, 6), top_n=3) == []  # not a market day
        with pytest.raises(ValueError):  # noqa: PT011
            liquidity_universe(bars, d.iloc[4].date(), top_n=0)
