"""Test builders: synthetic bhavcopy files in both NSE formats, a mock NSE server, bar frames."""

from __future__ import annotations

import io
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

import httpx
import pandas as pd

from alphaengine.data.sources.http import HttpFetcher
from alphaengine.data.sources.nse_bhavcopy import _MONTHS, legacy_name, udiff_name


@dataclass(frozen=True)
class Row:
    symbol: str
    isin: str
    close: float
    series: str = "EQ"
    prev_close: float | None = None
    volume: int = 1000
    open: float | None = None
    high: float | None = None
    low: float | None = None
    trades: int = 50

    def ohlc(self) -> tuple[float, float, float, float]:
        o = self.open if self.open is not None else self.close
        h = self.high if self.high is not None else max(o, self.close) * 1.01
        lo = self.low if self.low is not None else min(o, self.close) * 0.99
        return o, h, lo, self.close


LEGACY_HEADER = (
    "SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,LAST,PREVCLOSE,TOTTRDQTY,TOTTRDVAL,TIMESTAMP,"
    "TOTALTRADES,ISIN,"
)
UDIFF_HEADER = (
    "TradDt,BizDt,Sgmt,Src,FinInstrmTp,FinInstrmId,ISIN,TckrSymb,SctySrs,XpryDt,"
    "FininstrmActlXpryDt,StrkPric,OptnTp,FinInstrmNm,OpnPric,HghPric,LwPric,ClsPric,LastPric,"
    "PrvsClsgPric,UndrlygPric,SttlmPric,OpnIntrst,ChngInOpnIntrst,TtlTradgVol,TtlTrfVal,"
    "TtlNbOfTxsExctd,SsnId,NewBrdLotQty,Rmks,Rsvd1,Rsvd2,Rsvd3,Rsvd4"
)


OLD_LEGACY_HEADER = (
    "SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,LAST,PREVCLOSE,TOTTRDQTY,TOTTRDVAL,TIMESTAMP,"
)


def legacy_csv(day: date, rows: list[Row], *, pre2012: bool = False) -> str:
    """Legacy bhavcopy. ``pre2012`` mimics files before ~Dec 2011: no ISIN/TOTALTRADES columns
    and day-of-month without zero padding (``4-JAN-2010``)."""
    dd = str(day.day) if pre2012 else f"{day:%d}"
    ts = f"{dd}-{_MONTHS[day.month - 1]}-{day:%Y}"
    lines = [OLD_LEGACY_HEADER if pre2012 else LEGACY_HEADER]
    for r in rows:
        o, h, lo, c = r.ohlc()
        pc = r.prev_close if r.prev_close is not None else c
        line = f"{r.symbol},{r.series},{o},{h},{lo},{c},{c},{pc},{r.volume},{r.volume * c},{ts},"
        lines.append(line if pre2012 else f"{line}{r.trades},{r.isin},")
    return "\n".join(lines) + "\n"


def udiff_csv(day: date, rows: list[Row]) -> str:
    d = day.isoformat()
    lines = [UDIFF_HEADER]
    for i, r in enumerate(rows):
        o, h, lo, c = r.ohlc()
        pc = r.prev_close if r.prev_close is not None else c
        lines.append(
            f"{d},{d},CM,NSE,STK,{1000 + i},{r.isin},{r.symbol},{r.series},,,,,{r.symbol} LTD,"
            f"{o},{h},{lo},{c},{c},{pc},,{c},,,{r.volume},{r.volume * c},{r.trades},F1,1,,,,,"
        )
    # A non-CM segment row that must be ignored.
    lines.append(
        f"{d},{d},FO,NSE,STK,9,INE000X01010,IGNORED,EQ,,,,,X,1,1,1,1,1,1,,1,,,1,1,1,F1,1,,,,,"
    )
    return "\n".join(lines) + "\n"


def zip_text(name: str, text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(name.removesuffix(".zip"), text)
    return buf.getvalue()


class MockNse:
    """Serves bhavcopies for configured days (format chosen per day); 404 otherwise."""

    def __init__(self) -> None:
        self.days: dict[date, tuple[str, list[Row]]] = {}
        self.requests: list[str] = []
        self.fail_with: dict[date, int] = {}  # day -> HTTP status
        self.pre2012: set[date] = set()  # days served in the old ISIN-less layout

    def add(self, day: date, rows: list[Row], fmt: str = "legacy") -> None:
        self.days[day] = (fmt, rows)

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.requests.append(url)
        for day, (fmt, rows) in self.days.items():
            if day in self.fail_with and (legacy_name(day) in url or udiff_name(day) in url):
                return httpx.Response(self.fail_with[day])
            if fmt == "legacy" and url.endswith(legacy_name(day)):
                text = legacy_csv(day, rows, pre2012=day in self.pre2012)
                return httpx.Response(200, content=zip_text(legacy_name(day), text))
            if fmt == "udiff" and url.endswith(udiff_name(day)):
                return httpx.Response(200, content=zip_text(udiff_name(day), udiff_csv(day, rows)))
        for day, status in self.fail_with.items():
            if legacy_name(day) in url or udiff_name(day) in url:
                return httpx.Response(status)
        return httpx.Response(404)

    def fetcher(self) -> HttpFetcher:
        client = httpx.Client(transport=httpx.MockTransport(self.handler))
        return HttpFetcher(client, min_interval=0, attempts=2, sleep=lambda _: None)


def make_bars(
    spec: dict[str, list[float]],
    start: str = "2024-01-01",
    *,
    volume: int = 1000,
    isin_of: Callable[[str], str] | None = None,
) -> pd.DataFrame:
    """Bars for each symbol on consecutive business days from ``start`` (NaN close = no bar)."""
    n = max(len(v) for v in spec.values())
    dates = pd.bdate_range(start, periods=n)
    rows = []
    for sym, closes in spec.items():
        prev = None
        for d, c in zip(dates, closes, strict=False):
            if pd.isna(c):
                continue
            rows.append(
                {
                    "date": d,
                    "symbol": sym,
                    "isin": isin_of(sym) if isin_of else f"INE{sym[:6]:X<6}01019"[:12],
                    "series": "EQ",
                    "open": c,
                    "high": c * 1.01,
                    "low": c * 0.99,
                    "close": c,
                    "prev_close": prev if prev is not None else c,
                    "volume": volume,
                    "value": volume * c,
                    "trades": 10,
                }
            )
            prev = c
    from alphaengine.data.schema import coerce_bars

    return coerce_bars(pd.DataFrame(rows))
