from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from alphaengine.core.errors import DataError
from alphaengine.data.raw_store import RawStore
from alphaengine.data.schema import BAR_COLUMNS, empty_bars, validate_bars
from alphaengine.data.store import PriceStore
from alphaengine.market.calendar import TradingCalendar, candidate_days
from tests.helpers import make_bars


class TestValidateBars:
    def test_clean_rows_pass_and_symbols_normalized(self) -> None:
        df = make_bars({"RELIANCE": [100, 101]})
        df.loc[0, "symbol"] = "reliance.ns"
        res = validate_bars(df)
        assert res.n_rejected == 0
        assert list(res.clean["symbol"]) == ["RELIANCE", "RELIANCE"]
        assert list(res.clean.columns) == BAR_COLUMNS

    @pytest.mark.parametrize(
        ("mutate", "reason"),
        [
            (lambda d: d.assign(symbol="BAD SYM"), "invalid_symbol"),
            (lambda d: d.assign(close=float("nan")), "missing_price"),
            (lambda d: d.assign(open=0.0), "non_positive_price"),
            (lambda d: d.assign(volume=-5), "missing_or_negative_volume"),
            (lambda d: d.assign(high=1.0), "inconsistent_ohlc"),
        ],
    )
    def test_rejections(self, mutate, reason: str) -> None:  # type: ignore[no-untyped-def]
        res = validate_bars(mutate(make_bars({"TCS": [100]})))
        assert res.clean.empty
        assert list(res.rejected["reason"]) == [reason]

    def test_duplicates_keep_first(self) -> None:
        df = make_bars({"TCS": [100]})
        df2 = pd.concat([df, df.assign(close=100.5)], ignore_index=True)
        res = validate_bars(df2)
        assert len(res.clean) == 1
        assert res.clean["close"].iloc[0] == 100
        assert list(res.rejected["reason"]) == ["duplicate"]

    def test_empty(self) -> None:
        res = validate_bars(empty_bars())
        assert res.clean.empty
        assert res.n_rejected == 0


class TestCalendar:
    def test_rules_outside_observed_range(self, tmp_path: Path) -> None:
        csv = tmp_path / "h.csv"
        csv.write_text(
            "date,kind,description\n2026-01-26,holiday,Republic Day\n"
            "2026-02-01,special_session,Budget\n"
        )
        cal = TradingCalendar.from_csv(csv)
        assert not cal.is_trading_day(date(2026, 1, 26))
        assert cal.is_trading_day(date(2026, 1, 27))
        assert cal.is_trading_day(date(2026, 2, 1))  # Sunday special session
        assert not cal.is_trading_day(date(2026, 1, 31))  # Saturday
        assert cal.next_trading_day(date(2026, 1, 23)) == date(2026, 1, 27)
        assert cal.previous_trading_day(date(2026, 1, 27)) == date(2026, 1, 23)

    def test_observed_days_are_authoritative_inside_range(self) -> None:
        obs = [date(2024, 1, 1), date(2024, 1, 3)]  # Jan 2 "holiday" by observation
        cal = TradingCalendar(observed_trading_days=obs)
        assert cal.trading_days(date(2024, 1, 1), date(2024, 1, 4)) == [
            date(2024, 1, 1),
            date(2024, 1, 3),
            date(2024, 1, 4),
        ]
        assert cal.with_observed([date(2024, 1, 4)]).trading_days(
            date(2024, 1, 4), date(2024, 1, 4)
        ) == [date(2024, 1, 4)]
        assert cal.trading_days(date(2024, 1, 5), date(2024, 1, 1)) == []

    def test_bad_kind(self, tmp_path: Path) -> None:
        csv = tmp_path / "h.csv"
        csv.write_text("date,kind\n2026-01-26,party\n")
        with pytest.raises(ValueError, match="unknown calendar kind"):
            TradingCalendar.from_csv(csv)

    def test_no_trading_day_found(self) -> None:
        cal = TradingCalendar(
            holidays=[date(2026, 1, 1)]
            + [date(2026, 1, d) for d in range(2, 32)]
            + [date(2025, 12, d) for d in range(1, 32)]
        )
        with pytest.raises(ValueError, match="no trading day"):
            cal.next_trading_day(date(2025, 12, 31))
        with pytest.raises(ValueError, match="no trading day"):
            cal.previous_trading_day(date(2026, 1, 2))

    def test_candidate_days(self) -> None:
        assert len(candidate_days(date(2024, 1, 1), date(2024, 1, 7))) == 5
        assert len(candidate_days(date(2024, 1, 1), date(2024, 1, 7), include_weekends=True)) == 7
        assert candidate_days(date(2024, 1, 2), date(2024, 1, 1)) == []


class TestRawStore:
    def test_first_write_wins(self, tmp_path: Path) -> None:
        rs = RawStore(tmp_path)
        p = rs.put("nse", date(2024, 1, 2), "f.zip", b"one")
        rs.put("nse", date(2024, 1, 2), "f.zip", b"two")
        assert p.read_bytes() == b"one"
        assert rs.get("nse", date(2024, 1, 2), "f.zip") == b"one"
        assert rs.get("nse", date(2024, 1, 3), "g.zip") is None


class TestPriceStore:
    def test_empty_store(self, tmp_path: Path) -> None:
        st = PriceStore(tmp_path)
        assert st.read().empty
        assert st.date_range() is None
        assert st.trading_days() == []
        assert st.symbols() == []
        assert st.write(empty_bars()) == 0
        assert len(st.snapshot_id()) == 16

    def test_upsert_across_years_and_filters(self, tmp_path: Path) -> None:
        st = PriceStore(tmp_path)
        bars = make_bars({"AAA": [10, 11, 12], "BBB": [20, 21, 22]}, start="2023-12-28")
        st.write(bars)
        assert [p.parent.name for p in st.files()] == ["year=2023", "year=2024"]
        snap1 = st.snapshot_id()
        st.write(bars)  # idempotent
        assert len(st.read()) == 6
        assert st.snapshot_id() == snap1

        fix = bars[bars["symbol"] == "AAA"].tail(1).assign(close=12.5, high=13.0)
        st.write(fix)
        out = st.read(symbols=["AAA"])
        assert list(out["close"]) == [10, 11, 12.5]
        assert st.snapshot_id() != snap1

        assert st.date_range() == (date(2023, 12, 28), date(2024, 1, 1))
        assert st.symbols() == ["AAA", "BBB"]
        assert len(st.trading_days()) == 3
        sub = st.read(start=date(2023, 12, 29), end=date(2023, 12, 29), columns=["close"])
        assert list(sub.columns) == ["date", "symbol", "close"]
        assert len(sub) == 2
        assert st.read(symbols=[]).empty

    def test_replace_dates_drops_rows_missing_from_new_file(self, tmp_path: Path) -> None:
        st = PriceStore(tmp_path)
        st.write(make_bars({"AAA": [10, 11], "GONE": [5, 5]}))
        day2 = make_bars({"AAA": [10, 11.5]}).tail(1)
        st.write(day2, replace_dates=True)
        out = st.read()
        assert list(zip(out["date"].dt.day, out["symbol"], strict=True)) == [
            (1, "AAA"),
            (1, "GONE"),
            (2, "AAA"),
        ]
        assert out["close"].iloc[-1] == 11.5

    def test_rejects_bad_input(self, tmp_path: Path) -> None:
        st = PriceStore(tmp_path)
        bars = make_bars({"AAA": [10]})
        with pytest.raises(DataError, match="duplicate"):
            st.write(pd.concat([bars, bars]))
        with pytest.raises(ValueError, match="unknown columns"):
            st.read(columns=["nope"])
        with pytest.raises(ValueError, match="dataset"):
            PriceStore(tmp_path, "bad-name")
