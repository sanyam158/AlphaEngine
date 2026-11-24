"""High-level price loading for research: curated bars + symbol-rename resolution + adjustment."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from alphaengine.data.adjust import Adjust, apply_adjustment
from alphaengine.data.db import session_scope
from alphaengine.data.models import Instrument, SymbolHistory
from alphaengine.data.repositories import isin_to_current_symbol, load_corporate_actions
from alphaengine.data.schema import KEY
from alphaengine.data.store import PriceStore

_BUFFER = timedelta(days=15)  # history before `start` needed for dividend prev-close lookup


def _symbols_to_read(session: Session, symbols: Sequence[str]) -> list[str]:
    """Requested symbols plus every former symbol of the same ISINs."""
    isins = list(
        session.scalars(select(Instrument.isin).where(Instrument.symbol.in_(list(symbols))))
    )
    former = session.scalars(select(SymbolHistory.symbol).where(SymbolHistory.isin.in_(isins)))
    return sorted(set(symbols) | set(former))


def _earliest_isin_by_symbol(session: Session) -> dict[str, str]:
    """symbol -> ISIN it had when first seen with an ISIN. Rows without an ISIN predate that,
    so this is the best available identity for them (renames before then can't be detected)."""
    out: dict[str, str] = {}
    rows = session.execute(
        select(SymbolHistory.symbol, SymbolHistory.isin).order_by(SymbolHistory.valid_from.desc())
    )
    for sym, isin in rows:
        out[sym] = isin  # descending order: earliest valid_from wins
    return out


def load_prices(
    store: PriceStore,
    session_factory: sessionmaker[Session],
    symbols: Sequence[str] | None = None,
    start: date | None = None,
    end: date | None = None,
    *,
    adjust: Adjust = "split",
    resolve_renames: bool = True,
) -> pd.DataFrame:
    """Long bars frame. With ``resolve_renames`` rows are labelled with the ISIN's *current*
    symbol, so a renamed company has one continuous history."""
    with session_scope(session_factory) as s:
        read_syms = _symbols_to_read(s, symbols) if symbols and resolve_renames else symbols
        mapping = isin_to_current_symbol(s) if resolve_renames else {}
        earliest_isin = _earliest_isin_by_symbol(s) if resolve_renames else {}
        actions = load_corporate_actions(s)
    read_start = start - _BUFFER if start else None
    bars = store.read(read_syms, read_start, end)
    if resolve_renames and not bars.empty and mapping:
        missing_isin = bars["isin"].isna()
        if missing_isin.any():  # pre-2012 files carry no ISIN: infer it from the symbol
            bars.loc[missing_isin, "isin"] = bars.loc[missing_isin, "symbol"].map(earliest_isin)
        current = bars["isin"].map(mapping)
        bars["symbol"] = current.fillna(bars["symbol"]).astype("string")
        bars = bars.drop_duplicates(KEY, keep="last")
        if symbols:
            bars = bars[bars["symbol"].isin(list(symbols))]
    bars = apply_adjustment(bars.reset_index(drop=True), actions, adjust)
    if start is not None:
        bars = bars[bars["date"] >= pd.Timestamp(start)]
    return bars.sort_values(KEY).reset_index(drop=True)


def to_panel(bars: pd.DataFrame, field: str = "close") -> pd.DataFrame:
    """Wide date × symbol frame for vectorized research."""
    return bars.pivot(index="date", columns="symbol", values=field).sort_index()
