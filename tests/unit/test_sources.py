from __future__ import annotations

from datetime import date

import httpx
import numpy as np
import pandas as pd
import pytest

from alphaengine.core.errors import DataError, DataSourceError
from alphaengine.core.types import Symbol
from alphaengine.data.sources.base import RawPayload
from alphaengine.data.sources.http import HttpFetcher
from alphaengine.data.sources.nse_bhavcopy import (
    NseBhavcopySource,
    legacy_name,
    udiff_name,
)
from alphaengine.data.sources.nse_indices import NseIndexConstituentsSource, parse_constituents
from alphaengine.data.sources.yahoo import YahooSource, to_ticker
from tests.helpers import MockNse, Row, legacy_csv, udiff_csv, zip_text

D_OLD = date(2024, 1, 2)
D_NEW = date(2024, 7, 8)
ROWS = [
    Row("RELIANCE", "INE002A01018", 2500.0, prev_close=2490.0),
    Row("M&M", "INE101A01026", 1600.0),
    Row("SOMEBE", "INE999B01011", 50.0, series="BE"),
    Row("SOMEBOND", "INE999C01011", 1000.0, series="N1"),  # filtered series
    Row("ABC-RE1", "INE999D20011", 5.0),  # rights entitlement, excluded
]


class TestBhavcopyParsing:
    def test_legacy(self) -> None:
        src = NseBhavcopySource(MockNse().fetcher())
        payload = RawPayload(
            "nse_bhavcopy",
            D_OLD,
            legacy_name(D_OLD),
            "legacy",
            zip_text(legacy_name(D_OLD), legacy_csv(D_OLD, ROWS)),
        )
        df = src.parse(payload)
        assert list(df["symbol"]) == ["M&M", "RELIANCE", "SOMEBE"]
        rel = df[df["symbol"] == "RELIANCE"].iloc[0]
        assert rel["close"] == 2500.0
        assert rel["prev_close"] == 2490.0
        assert rel["isin"] == "INE002A01018"
        assert rel["date"] == pd.Timestamp(D_OLD)
        assert rel["volume"] == 1000

    def test_pre2012_legacy_without_isin_and_trades(self) -> None:
        day = date(2010, 1, 4)
        src = NseBhavcopySource(MockNse().fetcher())
        text = legacy_csv(day, ROWS, pre2012=True)
        payload = RawPayload("nse_bhavcopy", day, legacy_name(day), "legacy", text.encode())
        df = src.parse(payload)
        assert list(df["symbol"]) == ["M&M", "RELIANCE", "SOMEBE"]
        assert df["isin"].isna().all()
        assert df["trades"].isna().all()
        assert (df["date"] == pd.Timestamp(day)).all()

    def test_udiff_ignores_other_segments(self) -> None:
        src = NseBhavcopySource(MockNse().fetcher())
        payload = RawPayload(
            "nse_bhavcopy", D_NEW, udiff_name(D_NEW), "udiff", udiff_csv(D_NEW, ROWS).encode()
        )  # plain CSV also accepted
        df = src.parse(payload)
        assert list(df["symbol"]) == ["M&M", "RELIANCE", "SOMEBE"]
        assert df["trades"].iloc[0] == 50

    def test_date_mismatch_and_bad_format(self) -> None:
        src = NseBhavcopySource(MockNse().fetcher())
        wrong = RawPayload("x", date(2024, 1, 3), "n", "legacy", legacy_csv(D_OLD, ROWS).encode())
        with pytest.raises(DataError, match="do not match"):
            src.parse(wrong)
        with pytest.raises(DataError, match="unknown bhavcopy format"):
            src.parse(RawPayload("x", D_OLD, "n", "xml", b"a,b\n1,2\n"))
        with pytest.raises(DataError, match="missing columns"):
            src.parse(RawPayload("x", D_OLD, "n", "legacy", b"SYMBOL,SERIES\nA,EQ\n"))

    def test_zip_must_hold_one_csv(self) -> None:
        import io
        import zipfile

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("a.csv", "x")
            zf.writestr("b.csv", "y")
        src = NseBhavcopySource(MockNse().fetcher())
        with pytest.raises(DataError, match="one CSV"):
            src.parse(RawPayload("x", D_OLD, "n", "legacy", buf.getvalue()))


class TestBhavcopyFetching:
    def test_format_preference_and_fallback(self) -> None:
        nse = MockNse()
        nse.add(D_OLD, ROWS, "legacy")
        nse.add(D_NEW, ROWS, "legacy")  # new-era date only available in legacy -> fallback
        src = NseBhavcopySource(nse.fetcher())
        p_old = src.fetch_raw(D_OLD)
        assert p_old is not None and p_old.fmt == "legacy"
        assert len(nse.requests) == 1
        p_new = src.fetch_raw(D_NEW)
        assert p_new is not None and p_new.fmt == "legacy"
        assert "BhavCopy_NSE_CM" in nse.requests[1]  # udiff tried first for new dates
        assert src.fetch_raw(date(2024, 1, 3)) is None  # holiday -> 404 on both

    def test_fetch_bars_range(self) -> None:
        nse = MockNse()
        nse.add(D_OLD, ROWS)
        nse.add(date(2024, 1, 3), ROWS)
        df = NseBhavcopySource(nse.fetcher()).fetch_bars(
            date(2024, 1, 1), date(2024, 1, 5), [Symbol("RELIANCE")]
        )
        assert list(df["date"].dt.day) == [2, 3]
        assert NseBhavcopySource(MockNse().fetcher()).fetch_bars(D_OLD, D_OLD).empty


class TestHttpFetcher:
    def _fetcher(self, handler, sleeps: list[float] | None = None) -> HttpFetcher:  # type: ignore[no-untyped-def]
        s = sleeps if sleeps is not None else []
        clock = iter(np.arange(0, 100, 0.1))
        return HttpFetcher(
            httpx.Client(transport=httpx.MockTransport(handler)),
            min_interval=1.0,
            attempts=3,
            sleep=s.append,
            monotonic=lambda: float(next(clock)),
            warmup_url="https://example.test/",
        )

    def test_retries_5xx_then_succeeds_and_throttles(self) -> None:
        calls: list[str] = []

        def handler(req: httpx.Request) -> httpx.Response:
            calls.append(str(req.url))
            if req.url.path == "/":
                return httpx.Response(200)
            n = sum(1 for c in calls if c.endswith("/f"))
            return httpx.Response(503 if n < 2 else 200, content=b"ok")

        sleeps: list[float] = []
        f = self._fetcher(handler, sleeps)
        assert f.get("https://example.test/f") == b"ok"
        assert calls[0] == "https://example.test/"  # warmup first, once
        assert len(sleeps) >= 2  # backoff + throttle
        f.close()

    def test_errors(self) -> None:
        f = self._fetcher(lambda req: httpx.Response(403))
        with pytest.raises(DataSourceError, match="403"):
            f.get("https://example.test/f")
        f = self._fetcher(lambda req: httpx.Response(500))
        with pytest.raises(DataSourceError, match="after retries"):
            f.get("https://example.test/f")

        def boom(req: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("down")

        f = self._fetcher(boom)  # warmup failure is logged, not raised
        with pytest.raises(DataSourceError, match="after retries"):
            f.get("https://example.test/f")


class TestIndexConstituents:
    CSV = (
        b"Company Name,Industry,Symbol,Series,ISIN Code\n"
        b"Reliance Industries Ltd.,Oil Gas & Consumable Fuels,RELIANCE,EQ,INE002A01018\n"
        b"Mahindra & Mahindra Ltd.,Automobile and Auto Components,M&M,EQ,INE101A01026\n"
    )

    def test_parse_and_fetch(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            if req.url.path.endswith("ind_nifty50list.csv"):
                return httpx.Response(200, content=self.CSV)
            return httpx.Response(404)

        f = HttpFetcher(httpx.Client(transport=httpx.MockTransport(handler)), min_interval=0)
        src = NseIndexConstituentsSource(f)
        got = src.fetch("nifty50")
        assert [c.symbol for c in got] == ["RELIANCE", "M&M"]
        assert got[1].industry == "Automobile and Auto Components"
        with pytest.raises(DataError, match="unknown index"):
            src.url("NIFTYFOO")
        with pytest.raises(DataSourceError):
            src.fetch("NIFTY100")

    def test_bad_files(self) -> None:
        with pytest.raises(DataError, match="Symbol"):
            parse_constituents(b"a,b\n1,2\n")
        with pytest.raises(DataError, match="empty"):
            parse_constituents(b"Symbol\n")


def _yahoo_hist(ticker: str, start: date, end: date) -> pd.DataFrame:
    idx = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-03"], tz="Asia/Kolkata")
    return pd.DataFrame(
        {
            "Open": [100, 101, 102],
            "High": [101, 103, 104],
            "Low": [99, 100, 101],
            "Close": [100.5, 102, 103],
            "Volume": [0, 0, np.nan],
        },
        index=idx,
    )


def _yahoo_actions(ticker: str) -> pd.DataFrame:
    idx = pd.DatetimeIndex(["2020-06-01", "2021-03-01", "2022-08-10"], tz="Asia/Kolkata")
    # A ₹10 dividend before a 1:1 bonus (reported by Yahoo as split 2.0) shows as ₹5.
    return pd.DataFrame({"Dividends": [5.0, 0.0, 3.0], "Stock Splits": [0.0, 2.0, 0.0]}, index=idx)


class TestYahoo:
    def test_tickers(self) -> None:
        assert to_ticker("NIFTY50") == "^NSEI"
        assert to_ticker("INFY") == "INFY.NS"

    def test_bars(self) -> None:
        src = YahooSource(history=_yahoo_hist, actions=_yahoo_actions)
        df = src.fetch_bars(date(2024, 1, 2), date(2024, 1, 3), [Symbol("NIFTY50")])
        assert list(df["close"]) == [102, 103]
        assert df["prev_close"].iloc[0] == 100.5
        assert (df["volume"] == 0).all()
        with pytest.raises(DataError):
            src.fetch_bars(date(2024, 1, 1), date(2024, 1, 2))
        empty = YahooSource(history=lambda *a: pd.DataFrame())
        assert empty.fetch_bars(date(2024, 1, 1), date(2024, 1, 2), [Symbol("X")]).empty

    def test_actions_unadjust_dividends(self) -> None:
        acts = YahooSource(history=_yahoo_hist, actions=_yahoo_actions).fetch_actions(Symbol("ABC"))
        assert list(acts["kind"]) == ["DIVIDEND", "SPLIT", "DIVIDEND"]
        assert acts["amount"].iloc[0] == 10.0
        assert acts["ratio"].iloc[1] == 2.0
        assert acts["amount"].iloc[2] == 3.0
        assert acts["ex_date"].iloc[0] == date(2020, 6, 1)
        assert YahooSource(actions=lambda t: pd.DataFrame()).fetch_actions(Symbol("X")).empty

    def test_failures_wrapped(self) -> None:
        def boom(*_a: object) -> pd.DataFrame:
            raise RuntimeError("yahoo down")

        src = YahooSource(history=boom, actions=boom)
        src._history = boom  # bypass retry delays
        src._actions = boom
        with pytest.raises(DataSourceError, match="history"):
            src.fetch_bars(date(2024, 1, 1), date(2024, 1, 2), [Symbol("X")])
        with pytest.raises(DataSourceError, match="actions"):
            src.fetch_actions(Symbol("X"))
