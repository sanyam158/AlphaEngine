"""NSE capital-market (cash equity) bhavcopy: the official end-of-day file for every listed
security. It includes delisted stocks for past dates, so it is survivorship-bias free.

Two formats exist:
- **legacy** ``cmDDMONYYYYbhav.csv.zip`` (until early July 2024)
- **UDiFF** ``BhavCopy_NSE_CM_0_0_0_YYYYMMDD_F_0000.csv.zip`` (from 2024-07-08)

The preferred format is chosen by date and the other is tried as a fallback, so the adapter
keeps working if NSE back-fills or retires a format. Prices in the file are **raw**.
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import pandas as pd

from alphaengine.core.errors import DataError
from alphaengine.core.logging import get_logger
from alphaengine.core.types import Symbol
from alphaengine.data.schema import BAR_COLUMNS, coerce_bars, empty_bars, validate_bars
from alphaengine.data.sources.base import RawPayload
from alphaengine.data.sources.http import HttpFetcher
from alphaengine.market.calendar import candidate_days

log = get_logger(__name__)

UDIFF_CUTOVER = date(2024, 7, 8)
ARCHIVE_BASE = "https://nsearchives.nseindia.com"
WARMUP_URL = "https://www.nseindia.com/"
DEFAULT_SERIES = frozenset({"EQ", "BE"})
# Rights entitlements (e.g. ``ABC-RE``, ``ABC-RE1``) trade briefly and are not investable shares.
_RIGHTS_ENTITLEMENT = r"-RE\d*$"
_MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

_LEGACY_MAP = {
    "SYMBOL": "symbol",
    "SERIES": "series",
    "OPEN": "open",
    "HIGH": "high",
    "LOW": "low",
    "CLOSE": "close",
    "PREVCLOSE": "prev_close",
    "TOTTRDQTY": "volume",
    "TOTTRDVAL": "value",
    "TOTALTRADES": "trades",
    "ISIN": "isin",
    "TIMESTAMP": "date",
}
# Seen in legacy files: 04-JAN-2010, 4-JAN-2010, 13-Jul-20 (mid-2020 files use 2-digit years).
_LEGACY_DATE_FORMATS: tuple[str, ...] = ("%d-%b-%Y", "%d-%b-%y")
# Files before ~Dec 2011 have no ISIN / TOTALTRADES columns.
_OPTIONAL = frozenset({"ISIN", "TOTALTRADES", "TtlNbOfTxsExctd"})
_UDIFF_MAP = {
    "TckrSymb": "symbol",
    "SctySrs": "series",
    "OpnPric": "open",
    "HghPric": "high",
    "LwPric": "low",
    "ClsPric": "close",
    "PrvsClsgPric": "prev_close",
    "TtlTradgVol": "volume",
    "TtlTrfVal": "value",
    "TtlNbOfTxsExctd": "trades",
    "ISIN": "isin",
    "TradDt": "date",
}


@dataclass(frozen=True)
class Candidate:
    fmt: str
    name: str
    url: str


def legacy_name(day: date) -> str:
    return f"cm{day:%d}{_MONTHS[day.month - 1]}{day:%Y}bhav.csv.zip"


def udiff_name(day: date) -> str:
    return f"BhavCopy_NSE_CM_0_0_0_{day:%Y%m%d}_F_0000.csv.zip"


class NseBhavcopySource:
    name = "nse_bhavcopy"

    def __init__(
        self,
        fetcher: HttpFetcher | None = None,
        *,
        base_url: str = ARCHIVE_BASE,
        series: frozenset[str] = DEFAULT_SERIES,
        exclude_rights_entitlements: bool = True,
    ) -> None:
        self._fetcher = fetcher or HttpFetcher(warmup_url=WARMUP_URL)
        self._base = base_url.rstrip("/")
        self.series = series
        self.exclude_rights_entitlements = exclude_rights_entitlements

    def candidates(self, day: date) -> list[Candidate]:
        legacy = Candidate(
            "legacy",
            legacy_name(day),
            f"{self._base}/content/historical/EQUITIES/{day:%Y}/{_MONTHS[day.month - 1]}/"
            f"{legacy_name(day)}",
        )
        udiff = Candidate("udiff", udiff_name(day), f"{self._base}/content/cm/{udiff_name(day)}")
        return [udiff, legacy] if day >= UDIFF_CUTOVER else [legacy, udiff]

    def fetch_raw(self, day: date) -> RawPayload | None:
        for cand in self.candidates(day):
            content = self._fetcher.get(cand.url)
            if content is not None:
                return RawPayload(self.name, day, cand.name, cand.fmt, content)
        return None

    def parse(self, payload: RawPayload) -> pd.DataFrame:
        text = _unzip_text(payload.content)
        raw = pd.read_csv(io.StringIO(text), dtype=str, skipinitialspace=True)
        raw.columns = [str(c).strip() for c in raw.columns]
        raw = raw.loc[:, [c for c in raw.columns if c and not c.startswith("Unnamed")]]
        if payload.fmt == "legacy":
            mapping, date_fmts = _LEGACY_MAP, _LEGACY_DATE_FORMATS
        elif payload.fmt == "udiff":
            mapping, date_fmts = _UDIFF_MAP, ("%Y-%m-%d",)
            if "Sgmt" in raw.columns:
                raw = raw[raw["Sgmt"].str.strip() == "CM"]
        else:
            raise DataError(f"unknown bhavcopy format {payload.fmt!r}")
        missing = set(mapping) - set(raw.columns) - _OPTIONAL
        if missing:
            raise DataError(f"{payload.name}: missing columns {sorted(missing)}")
        for col in set(mapping) - set(raw.columns):
            raw[col] = pd.NA
        df = raw[list(mapping)].rename(columns=mapping)
        for col in ("symbol", "series", "isin"):
            df[col] = df[col].str.strip()
        df = df[df["series"].isin(self.series)]
        if self.exclude_rights_entitlements:
            df = df[~df["symbol"].str.contains(_RIGHTS_ENTITLEMENT, regex=True, na=False)]
        df["date"] = _parse_dates(df["date"].str.strip(), date_fmts, payload.name)
        if not df.empty and set(df["date"].dt.date) != {payload.day}:
            raise DataError(f"{payload.name}: file dates do not match requested day {payload.day}")
        return coerce_bars(df).sort_values(["date", "symbol"]).reset_index(drop=True)

    def fetch_bars(
        self, start: date, end: date, symbols: Sequence[Symbol] | None = None
    ) -> pd.DataFrame:
        """Convenience for small ranges (no caching); use the ingest pipeline for backfills."""
        frames = []
        for day in candidate_days(start, end):
            payload = self.fetch_raw(day)
            if payload is None:
                continue
            frames.append(validate_bars(self.parse(payload)).clean)
        if not frames:
            return empty_bars()
        out = pd.concat(frames, ignore_index=True)
        if symbols is not None:
            out = out[out["symbol"].isin(list(symbols))]
        return out[BAR_COLUMNS].reset_index(drop=True)


def _parse_dates(values: pd.Series, formats: Sequence[str], name: str) -> pd.Series:
    """Parse with the first format that matches every value (formats are tried in order)."""
    for fmt in formats:
        parsed = pd.to_datetime(values, format=fmt, errors="coerce")
        if parsed.notna().all():
            return parsed
    sample = values[pd.to_datetime(values, format=formats[0], errors="coerce").isna()].head(1)
    raise DataError(f"{name}: unrecognised date format, e.g. {sample.tolist()}")


def _unzip_text(content: bytes) -> str:
    if content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if len(names) != 1:
                raise DataError(f"expected one CSV in bhavcopy zip, found {names}")
            return zf.read(names[0]).decode("utf-8-sig")
    return content.decode("utf-8-sig")
