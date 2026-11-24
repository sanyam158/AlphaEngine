"""Contract every DailyBarSource must satisfy. Add new adapters to SOURCES."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

import pandas as pd
import pytest

from alphaengine.core.types import Symbol
from alphaengine.data.schema import BAR_COLUMNS, validate_bars
from alphaengine.data.sources.base import DailyBarSource
from alphaengine.data.sources.nse_bhavcopy import NseBhavcopySource
from alphaengine.data.sources.yahoo import YahooSource
from tests.helpers import MockNse, Row

START, END = date(2024, 1, 1), date(2024, 1, 5)
SYMS = [Symbol("NIFTY50")]


def _bhavcopy() -> DailyBarSource:
    nse = MockNse()
    for d, c in [(date(2024, 1, 2), 100.0), (date(2024, 1, 3), 101.0), (date(2024, 1, 4), 99.0)]:
        nse.add(d, [Row("NIFTY50", "INE000000001", c), Row("OTHER", "INE000000002", 10.0)])
    return NseBhavcopySource(nse.fetcher())


def _yahoo() -> DailyBarSource:
    def hist(ticker: str, s: date, e: date) -> pd.DataFrame:
        idx = pd.bdate_range("2023-12-29", "2024-01-08")
        closes = [100.0 + i for i in range(len(idx))]
        return pd.DataFrame(
            {
                "Open": closes,
                "High": [c + 1 for c in closes],
                "Low": [c - 1 for c in closes],
                "Close": closes,
                "Volume": 0,
            },
            index=idx,
        )

    return YahooSource(history=hist, actions=lambda t: pd.DataFrame())


SOURCES: dict[str, Callable[[], DailyBarSource]] = {"nse_bhavcopy": _bhavcopy, "yahoo": _yahoo}


@pytest.fixture(params=sorted(SOURCES))
def source(request: pytest.FixtureRequest) -> DailyBarSource:
    return SOURCES[request.param]()


def test_canonical_schema(source: DailyBarSource) -> None:
    df = source.fetch_bars(START, END, SYMS)
    assert list(df.columns) == BAR_COLUMNS
    assert str(df["date"].dtype) == "datetime64[ns]"
    assert not df.empty


def test_respects_range_and_symbols(source: DailyBarSource) -> None:
    df = source.fetch_bars(START, END, SYMS)
    assert df["date"].min() >= pd.Timestamp(START)
    assert df["date"].max() <= pd.Timestamp(END)
    assert set(df["symbol"]) == {"NIFTY50"}


def test_sorted_unique_and_valid(source: DailyBarSource) -> None:
    df = source.fetch_bars(START, END, SYMS)
    assert df.equals(df.sort_values(["date", "symbol"]).reset_index(drop=True))
    assert not df.duplicated(["date", "symbol"]).any()
    assert validate_bars(df).n_rejected == 0


def test_empty_range(source: DailyBarSource) -> None:
    df = source.fetch_bars(date(2030, 1, 1), date(2030, 1, 3), SYMS)
    assert df.empty
    assert list(df.columns) == BAR_COLUMNS
