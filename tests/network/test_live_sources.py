"""Live smoke tests against NSE and Yahoo. Opt-in only (never in CI):

ALPHAENGINE_RUN_NETWORK=1 pytest tests/network -m network --no-cov
"""

from __future__ import annotations

import os
from datetime import date

import pytest

from alphaengine.core.types import Symbol
from alphaengine.data.schema import validate_bars
from alphaengine.data.sources.nse_bhavcopy import NseBhavcopySource
from alphaengine.data.sources.nse_indices import NseIndexConstituentsSource
from alphaengine.data.sources.yahoo import YahooSource

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(
        os.environ.get("ALPHAENGINE_RUN_NETWORK") != "1", reason="set ALPHAENGINE_RUN_NETWORK=1"
    ),
]


@pytest.mark.parametrize("day", [date(2023, 6, 1), date(2025, 6, 2)], ids=["legacy", "udiff"])
def test_bhavcopy_formats(day: date) -> None:
    src = NseBhavcopySource()
    payload = src.fetch_raw(day)
    assert payload is not None
    res = validate_bars(src.parse(payload))
    assert len(res.clean) > 1000
    assert {"RELIANCE", "TCS", "INFY"} <= set(res.clean["symbol"])
    assert res.clean["isin"].notna().all()


def test_bhavcopy_holiday_returns_none() -> None:
    assert NseBhavcopySource().fetch_raw(date(2025, 8, 15)) is None  # Independence Day


def test_index_constituents() -> None:
    got = NseIndexConstituentsSource().fetch("NIFTY50")
    assert len(got) == 50
    assert all(c.isin for c in got)


def test_yahoo_index_and_actions() -> None:
    yahoo = YahooSource()
    bars = yahoo.fetch_bars(date(2025, 1, 1), date(2025, 1, 31), [Symbol("NIFTY50")])
    assert 15 <= len(bars) <= 23
    acts = yahoo.fetch_actions(Symbol("RELIANCE"))
    assert (acts["kind"] == "DIVIDEND").any()
