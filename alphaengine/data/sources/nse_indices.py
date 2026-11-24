"""Current constituents of NSE indices (``ind_<slug>list.csv``).

NSE publishes only the *current* list. Recording a snapshot regularly (see
``repositories.record_index_snapshot``) builds point-in-time membership going forward; older
history must be imported from reconstitution announcements (``alphaengine import index-members``).
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import pandas as pd

from alphaengine.core.errors import DataError, DataSourceError
from alphaengine.core.types import Symbol, make_symbol
from alphaengine.data.sources.http import HttpFetcher
from alphaengine.data.sources.nse_bhavcopy import ARCHIVE_BASE, WARMUP_URL

INDEX_SLUGS: dict[str, str] = {
    "NIFTY50": "nifty50",
    "NIFTYNEXT50": "niftynext50",
    "NIFTY100": "nifty100",
    "NIFTY200": "nifty200",
    "NIFTY500": "nifty500",
    "NIFTYMIDCAP150": "niftymidcap150",
    "NIFTYSMALLCAP250": "niftysmallcap250",
}


@dataclass(frozen=True)
class Constituent:
    symbol: Symbol
    isin: str | None
    name: str | None
    industry: str | None


class NseIndexConstituentsSource:
    name = "nse_indices"

    def __init__(self, fetcher: HttpFetcher | None = None, *, base_url: str = ARCHIVE_BASE) -> None:
        self._fetcher = fetcher or HttpFetcher(warmup_url=WARMUP_URL)
        self._base = base_url.rstrip("/")

    def url(self, index_name: str) -> str:
        try:
            slug = INDEX_SLUGS[index_name.upper()]
        except KeyError:
            raise DataError(f"unknown index {index_name!r}; known: {sorted(INDEX_SLUGS)}") from None
        return f"{self._base}/content/indices/ind_{slug}list.csv"

    def fetch(self, index_name: str) -> list[Constituent]:
        content = self._fetcher.get(self.url(index_name))
        if content is None:
            raise DataSourceError(f"no constituent file for {index_name}")
        return parse_constituents(content)


def parse_constituents(content: bytes) -> list[Constituent]:
    df = pd.read_csv(io.BytesIO(content), dtype=str)
    df.columns = [c.strip() for c in df.columns]
    if "Symbol" not in df.columns:
        raise DataError(f"constituent file missing 'Symbol' column: {list(df.columns)}")

    def col(row: pd.Series, name: str) -> str | None:
        v = row.get(name)
        return str(v).strip() if isinstance(v, str) and v.strip() else None

    out = [
        Constituent(
            symbol=make_symbol(str(row["Symbol"])),
            isin=col(row, "ISIN Code"),
            name=col(row, "Company Name"),
            industry=col(row, "Industry"),
        )
        for _, row in df.iterrows()
    ]
    if not out:
        raise DataError("constituent file is empty")
    return out
