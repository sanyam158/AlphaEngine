"""Point-in-time market data for strategies: wide (date × symbol) panels.

Panels:
- ``close``: split/bonus-adjusted close, for signals (returns are correct across actions)
- ``tr_close``: total-return close (dividends reinvested), for performance and momentum
- ``raw_close``: as traded, for absolute-price rules (e.g. "price > ₹20")
- ``value``: traded value in ₹, for liquidity
- ``high`` / ``low``: split-adjusted, only when loaded with ``ohlc=True``
- ``raw_open`` / ``raw_high`` / ``raw_low``: as traded, only with ``execution=True`` (backtests)
- ``actions``: corporate actions (the backtester applies them to holdings)
- ``benchmarks``: index levels (NIFTY50, ...)

:meth:`MarketData.truncate` returns a view that ends at a date, which is how look-ahead is
ruled out in tests. Features are cached per :class:`~alphaengine.features.base.FeatureSpec`
and must be causal (row *t* depends only on rows ≤ *t*); every registered feature is checked
for this in the test suite.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
from sqlalchemy import select

from alphaengine.data.adjust import adjustment_factors
from alphaengine.data.db import session_scope
from alphaengine.data.models import IndexMembership, Instrument
from alphaengine.data.prices import load_prices
from alphaengine.data.repositories import load_corporate_actions

if TYPE_CHECKING:
    from alphaengine.app import AppContext
    from alphaengine.data.store import PriceStore
    from alphaengine.features.base import FeatureSpec

DTYPE = np.float32
ACTION_COLUMNS = ["symbol", "ex_date", "kind", "ratio", "amount", "source"]


def _pivot(bars: pd.DataFrame, values: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame({"date": bars["date"], "symbol": bars["symbol"], "v": values.to_numpy()})
    out = frame.pivot(index="date", columns="symbol", values="v").sort_index()
    out.columns = out.columns.astype(str)
    out.columns.name = None
    return out.astype(DTYPE)


@dataclass
class MarketData:
    close: pd.DataFrame
    tr_close: pd.DataFrame
    raw_close: pd.DataFrame
    value: pd.DataFrame
    benchmarks: pd.DataFrame = field(default_factory=pd.DataFrame)
    high: pd.DataFrame | None = None
    low: pd.DataFrame | None = None
    raw_open: pd.DataFrame | None = None
    raw_high: pd.DataFrame | None = None
    raw_low: pd.DataFrame | None = None
    actions: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=ACTION_COLUMNS))
    industries: dict[str, str] = field(default_factory=dict)
    asset_classes: dict[str, str] = field(default_factory=dict)  # symbol -> EQUITY | ETF
    memberships: pd.DataFrame = field(  # index_name, symbol, start_date, end_date
        default_factory=lambda: pd.DataFrame(
            columns=["index_name", "symbol", "start_date", "end_date"]
        )
    )
    _features: dict[FeatureSpec, pd.DataFrame] = field(default_factory=dict, repr=False)

    # ---------------------------------------------------------------- construction
    @classmethod
    def from_bars(
        cls,
        bars: pd.DataFrame,
        actions: pd.DataFrame | None = None,
        benchmark_bars: pd.DataFrame | None = None,
        *,
        ohlc: bool = False,
        execution: bool = False,
        industries: dict[str, str] | None = None,
        memberships: pd.DataFrame | None = None,
        asset_classes: dict[str, str] | None = None,
    ) -> MarketData:
        """Build panels from a raw (unadjusted) long bars frame plus corporate actions."""
        acts = actions if actions is not None else pd.DataFrame(columns=ACTION_COLUMNS)
        bars = bars.sort_values(["date", "symbol"]).reset_index(drop=True)
        f = adjustment_factors(bars, acts)
        split, tr = f["split_factor"], f["tr_factor"]
        bench = pd.DataFrame()
        if benchmark_bars is not None and not benchmark_bars.empty:
            bench = _pivot(benchmark_bars, benchmark_bars["close"])
        md = cls(
            close=_pivot(bars, bars["close"] * split),
            tr_close=_pivot(bars, bars["close"] * tr),
            raw_close=_pivot(bars, bars["close"]),
            value=_pivot(bars, bars["value"].fillna(bars["close"] * bars["volume"])),
            benchmarks=bench,
            high=_pivot(bars, bars["high"] * split) if ohlc else None,
            low=_pivot(bars, bars["low"] * split) if ohlc else None,
            industries=industries or {},
            asset_classes=asset_classes or {},
            raw_open=_pivot(bars, bars["open"]) if execution else None,
            raw_high=_pivot(bars, bars["high"]) if execution else None,
            raw_low=_pivot(bars, bars["low"]) if execution else None,
            actions=acts,
        )
        if memberships is not None:
            md.memberships = memberships
        if not md.benchmarks.empty:
            md.benchmarks = md.benchmarks.reindex(md.close.index).ffill()
        return md

    @classmethod
    def from_store(
        cls,
        ctx: AppContext,
        start: date | None = None,
        end: date | None = None,
        symbols: Sequence[str] | None = None,
        *,
        ohlc: bool = False,
        execution: bool = False,
    ) -> MarketData:
        cols = [
            "close",
            "value",
            "volume",
            *(["high", "low"] if (ohlc or execution) else []),
            *(["open"] if execution else []),
        ]
        bars = load_prices(
            ctx.equities, ctx.session_factory, symbols, start, end, adjust="none", columns=cols
        )
        with session_scope(ctx.session_factory) as s:
            actions = load_corporate_actions(s)
            industries = {
                sym: ind
                for sym, ind in s.execute(select(Instrument.symbol, Instrument.industry))
                if ind
            }
            # Indian ETF / mutual-fund units have ISINs starting with INF (shares: INE).
            asset_classes = {
                sym: "ETF"
                for sym, isin in s.execute(select(Instrument.symbol, Instrument.isin))
                if isin.startswith("INF")
            }
            mem = pd.DataFrame(
                s.execute(
                    select(
                        IndexMembership.index_name,
                        IndexMembership.symbol,
                        IndexMembership.start_date,
                        IndexMembership.end_date,
                    )
                ).all(),
                columns=["index_name", "symbol", "start_date", "end_date"],
            )
        bench = ctx.indices.read(start=start, end=end, columns=["close"])
        return cls.from_bars(
            bars,
            actions,
            bench,
            ohlc=ohlc,
            execution=execution,
            industries=industries,
            memberships=mem,
            asset_classes=asset_classes,
        )

    # ---------------------------------------------------------------- views
    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.close.index)

    @property
    def symbols(self) -> list[str]:
        return [str(c) for c in self.close.columns]

    def truncate(self, as_of: date | pd.Timestamp) -> MarketData:
        """A copy containing only rows on or before ``as_of`` (and an empty feature cache)."""
        ts = pd.Timestamp(as_of)

        def cut(df: pd.DataFrame | None) -> pd.DataFrame | None:
            return None if df is None else df.loc[:ts]

        return MarketData(
            close=self.close.loc[:ts],
            tr_close=self.tr_close.loc[:ts],
            raw_close=self.raw_close.loc[:ts],
            value=self.value.loc[:ts],
            benchmarks=self.benchmarks.loc[:ts] if not self.benchmarks.empty else self.benchmarks,
            high=cut(self.high),
            low=cut(self.low),
            raw_open=cut(self.raw_open),
            raw_high=cut(self.raw_high),
            raw_low=cut(self.raw_low),
            actions=self.actions,
            industries=self.industries,
            asset_classes=self.asset_classes,
            memberships=self.memberships,
        )

    def last_date_on_or_before(self, as_of: date | pd.Timestamp) -> pd.Timestamp | None:
        idx = self.dates
        pos = idx.searchsorted(pd.Timestamp(as_of), side="right") - 1
        return None if pos < 0 else pd.Timestamp(idx[int(pos)])

    def feature(self, spec: FeatureSpec) -> pd.DataFrame:
        """Full-history panel for ``spec`` (cached)."""
        if spec not in self._features:
            from alphaengine.features.base import compute  # noqa: PLC0415 - avoid import cycle

            self._features[spec] = compute(spec, self)
        return self._features[spec]

    def feature_row(self, spec: FeatureSpec, as_of: pd.Timestamp) -> pd.Series:
        panel = self.feature(spec)
        if as_of not in panel.index:
            raise KeyError(f"{as_of.date()} is not a trading day in the data")
        return cast(pd.Series, panel.loc[as_of])

    def asset_class(self, symbol: str) -> str:
        return self.asset_classes.get(symbol, "EQUITY")

    def index_members(self, index_name: str, as_of: date | pd.Timestamp) -> list[str]:
        m = self.memberships
        if m.empty:
            return []
        d = pd.Timestamp(as_of).date()
        sel = m[
            (m["index_name"] == index_name.upper())
            & (m["start_date"] <= d)
            & m["end_date"].map(lambda e: e is None or pd.isna(e) or e > d).astype(bool)
        ]
        return sorted(sel["symbol"].astype(str))


def liquid_symbols(store: PriceStore, start: date, end: date, top_per_year: int) -> list[str]:
    """Symbols ranking in the top ``top_per_year`` by median daily traded value in any calendar
    year of ``[start, end]``. A cheap superset of every liquidity universe over the period, so
    backtests can load ~1k columns instead of the whole market (delisted stocks included)."""
    with store.connect() as con:
        rows = con.execute(
            """
            WITH yearly AS (
                SELECT year(date) AS y, symbol, median(value) AS med
                FROM bars WHERE date BETWEEN ? AND ? GROUP BY 1, 2
            ), ranked AS (
                SELECT symbol, row_number() OVER (PARTITION BY y ORDER BY med DESC) AS rk
                FROM yearly
            )
            SELECT DISTINCT symbol FROM ranked WHERE rk <= ? ORDER BY symbol
            """,
            [pd.Timestamp(start).to_pydatetime(), pd.Timestamp(end).to_pydatetime(), top_per_year],
        ).fetchall()
    return [str(r[0]) for r in rows]
