"""Persistence helpers for the data layer (symbol master, corporate actions, index membership,
ingest log, quality issues). Functions take an open ``Session``; callers own the transaction."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from alphaengine.core.errors import DataError
from alphaengine.core.logging import get_logger
from alphaengine.core.types import Symbol, make_symbol
from alphaengine.data.models import (
    CorporateAction,
    DataQualityIssue,
    IndexMembership,
    IngestLog,
    Instrument,
    SymbolHistory,
)
from alphaengine.data.sources.base import CORPORATE_ACTION_COLUMNS
from alphaengine.data.sources.nse_indices import Constituent

log = get_logger(__name__)

ACTION_KINDS = frozenset({"SPLIT", "BONUS", "SPLIT_OR_BONUS", "DIVIDEND"})
# Share-count changes. SPLIT_OR_BONUS = source can't tell which (e.g. Yahoo); prices adjust the
# same way, but tax treatment differs, so a classified SPLIT/BONUS replaces it.
SHARE_KINDS = frozenset({"SPLIT", "BONUS", "SPLIT_OR_BONUS"})


# ---------------------------------------------------------------- symbol master
class InstrumentRegistry:
    """In-memory view of the symbol master for fast bulk ingestion.

    Feed trading days in ascending order via :meth:`observe`; changes are tracked on ORM objects
    attached to ``session`` and written when the caller flushes/commits. A changed symbol for a
    known ISIN on a later day is recorded as a rename in ``symbol_history``.
    """

    def __init__(self, session: Session) -> None:
        self._session = session
        self._by_isin: dict[str, Instrument] = {
            i.isin: i for i in session.scalars(select(Instrument))
        }
        self._open_hist: dict[str, SymbolHistory] = {
            h.isin: h
            for h in session.scalars(select(SymbolHistory).where(SymbolHistory.valid_to.is_(None)))
        }
        self.created = 0
        self.renamed = 0

    def __len__(self) -> int:
        return len(self._by_isin)

    def get(self, isin: str) -> Instrument | None:
        return self._by_isin.get(isin)

    def observe(self, day: date, bars: pd.DataFrame) -> None:
        rows = bars.loc[bars["isin"].notna(), ["isin", "symbol", "series"]]
        for isin, symbol, series in rows.drop_duplicates("isin").itertuples(index=False):
            self._observe_one(day, str(isin), str(symbol), None if pd.isna(series) else str(series))

    def _observe_one(self, day: date, isin: str, symbol: str, series: str | None) -> None:
        inst = self._by_isin.get(isin)
        if inst is None:
            inst = Instrument(
                isin=isin,
                symbol=symbol,
                series=series,
                first_seen=day,
                last_seen=day,
                exchange="NSE",
                asset_class="EQUITY",
            )
            hist = SymbolHistory(isin=isin, symbol=symbol, valid_from=day)
            self._session.add_all([inst, hist])
            self._by_isin[isin] = inst
            self._open_hist[isin] = hist
            self.created += 1
            return
        inst.first_seen = min(inst.first_seen, day)
        if day < inst.last_seen:
            return  # out-of-order replay of an older day: never rewrite current identity
        if symbol != inst.symbol:
            old = self._open_hist.get(isin)
            if old is not None:
                old.valid_to = day
            hist = SymbolHistory(isin=isin, symbol=symbol, valid_from=day)
            self._session.add(hist)
            self._open_hist[isin] = hist
            log.info("symbol renamed", isin=isin, old=inst.symbol, new=symbol, on=str(day))
            inst.symbol = symbol
            self.renamed += 1
        inst.last_seen = max(inst.last_seen, day)
        if series and inst.series != series:
            inst.series = series


def isin_to_current_symbol(session: Session) -> dict[str, str]:
    return {isin: sym for isin, sym in session.execute(select(Instrument.isin, Instrument.symbol))}


# ---------------------------------------------------------------- corporate actions
def _validate_action(row: Mapping[Any, Any]) -> tuple[str, date, str, float | None, float | None]:
    symbol = make_symbol(str(row["symbol"]))
    ex_date = pd.Timestamp(row["ex_date"]).date()
    kind = str(row["kind"]).strip().upper()
    if kind not in ACTION_KINDS:
        raise DataError(f"{symbol} {ex_date}: unknown action kind {kind!r}")
    ratio = None if pd.isna(row.get("ratio")) else float(row["ratio"])
    amount = None if pd.isna(row.get("amount")) else float(row["amount"])
    if kind in SHARE_KINDS and (ratio is None or ratio <= 0 or ratio == 1):
        raise DataError(f"{symbol} {ex_date}: {kind} needs ratio > 0 and != 1, got {ratio}")
    if kind == "DIVIDEND" and (amount is None or amount <= 0):
        raise DataError(f"{symbol} {ex_date}: DIVIDEND needs amount > 0, got {amount}")
    return symbol, ex_date, kind, ratio, amount


def upsert_corporate_actions(
    session: Session, actions: pd.DataFrame, source: str | None = None
) -> int:
    """Insert or update actions keyed by (symbol, ex_date, kind). Returns rows written."""
    written = 0
    for row in actions.to_dict("records"):
        symbol, ex_date, kind, ratio, amount = _validate_action(row)
        src = source or str(row.get("source") or "manual")
        if kind in SHARE_KINDS:
            same_day = session.scalars(
                select(CorporateAction).where(
                    CorporateAction.symbol == symbol,
                    CorporateAction.ex_date == ex_date,
                    CorporateAction.kind.in_(SHARE_KINDS - {kind}),
                )
            ).all()
            if kind == "SPLIT_OR_BONUS" and same_day:
                continue  # already classified: keep the specific record
            for other in same_day:  # a classified record replaces an ambiguous one
                session.delete(other)
        existing = session.scalars(
            select(CorporateAction).where(
                CorporateAction.symbol == symbol,
                CorporateAction.ex_date == ex_date,
                CorporateAction.kind == kind,
            )
        ).one_or_none()
        if existing is None:
            session.add(
                CorporateAction(
                    symbol=symbol,
                    ex_date=ex_date,
                    kind=kind,
                    ratio=ratio,
                    amount=amount,
                    source=src,
                )
            )
        else:
            existing.ratio, existing.amount, existing.source = ratio, amount, src
        written += 1
    session.flush()
    return written


def load_corporate_actions(session: Session, symbols: Sequence[str] | None = None) -> pd.DataFrame:
    stmt = select(CorporateAction).order_by(CorporateAction.symbol, CorporateAction.ex_date)
    if symbols is not None:
        stmt = stmt.where(CorporateAction.symbol.in_(list(symbols)))
    rows = [
        {
            "symbol": a.symbol,
            "ex_date": a.ex_date,
            "kind": a.kind,
            "ratio": a.ratio,
            "amount": None if a.amount is None else float(a.amount),
            "source": a.source,
        }
        for a in session.scalars(stmt)
    ]
    return pd.DataFrame(rows, columns=CORPORATE_ACTION_COLUMNS)


# ---------------------------------------------------------------- index membership
@dataclass(frozen=True)
class SnapshotDiff:
    added: frozenset[str]
    removed: frozenset[str]


def record_index_snapshot(
    session: Session,
    index_name: str,
    constituents: Sequence[Constituent],
    as_of: date,
    source: str = "nse_snapshot",
) -> SnapshotDiff:
    """Diff a current-constituents snapshot against open memberships and record changes."""
    index_name = index_name.upper()
    open_rows = {
        m.symbol: m
        for m in session.scalars(
            select(IndexMembership).where(
                IndexMembership.index_name == index_name, IndexMembership.end_date.is_(None)
            )
        )
    }
    latest = session.scalars(
        select(IndexMembership.start_date)
        .where(IndexMembership.index_name == index_name)
        .order_by(IndexMembership.start_date.desc())
        .limit(1)
    ).first()
    if latest is not None and as_of < latest:
        raise DataError(f"snapshot {as_of} is older than recorded membership ({latest})")
    current = {c.symbol: c for c in constituents}
    added = set(current) - set(open_rows)
    removed = set(open_rows) - set(current)
    for sym in removed:
        open_rows[sym].end_date = as_of
    for sym in added:
        c = current[sym]
        session.add(
            IndexMembership(
                index_name=index_name, symbol=sym, isin=c.isin, start_date=as_of, source=source
            )
        )
    for c in constituents:  # enrich symbol master with names/industries
        inst = (
            session.scalars(select(Instrument).where(Instrument.isin == c.isin)).one_or_none()
            if c.isin
            else None
        )
        if inst is not None:
            inst.name = c.name or inst.name
            inst.industry = c.industry or inst.industry
    session.flush()
    return SnapshotDiff(frozenset(added), frozenset(removed))


def import_index_membership(session: Session, rows: pd.DataFrame, source: str = "import") -> int:
    """Import historical membership rows (index_name, symbol, start_date[, end_date, isin])."""
    n = 0
    for r in rows.to_dict("records"):
        start = pd.Timestamp(r["start_date"]).date()
        end_raw = r.get("end_date")
        end = (
            None
            if end_raw is None or pd.isna(end_raw) or end_raw == ""
            else pd.Timestamp(end_raw).date()
        )
        if end is not None and end <= start:
            raise DataError(f"{r['symbol']}: end_date {end} must be after start_date {start}")
        isin = r.get("isin")
        session.add(
            IndexMembership(
                index_name=str(r["index_name"]).upper(),
                symbol=make_symbol(str(r["symbol"])),
                isin=None if isin is None or pd.isna(isin) else str(isin),
                start_date=start,
                end_date=end,
                source=source,
            )
        )
        n += 1
    session.flush()
    return n


def members_as_of(session: Session, index_name: str, as_of: date) -> frozenset[Symbol]:
    stmt = select(IndexMembership.symbol).where(
        IndexMembership.index_name == index_name.upper(),
        IndexMembership.start_date <= as_of,
        (IndexMembership.end_date.is_(None)) | (IndexMembership.end_date > as_of),
    )
    return frozenset(Symbol(s) for s in session.scalars(stmt))


def membership_history_start(session: Session, index_name: str) -> date | None:
    return session.scalars(
        select(IndexMembership.start_date)
        .where(IndexMembership.index_name == index_name.upper())
        .order_by(IndexMembership.start_date)
        .limit(1)
    ).first()


# ---------------------------------------------------------------- ingest log
DONE_STATUSES = ("ok", "no_data")


def completed_days(
    session: Session, source: str, start: date, end: date, *, recheck_after: date | None = None
) -> set[date]:
    """Days not worth fetching again. ``no_data`` days on or after ``recheck_after`` are *not*
    included: a file missing yesterday may simply not have been published yet."""
    stmt = select(IngestLog.trade_date, IngestLog.status).where(
        IngestLog.source == source,
        IngestLog.trade_date >= start,
        IngestLog.trade_date <= end,
        IngestLog.status.in_(DONE_STATUSES),
    )
    return {
        d
        for d, status in session.execute(stmt)
        if not (status == "no_data" and recheck_after is not None and d >= recheck_after)
    }


def record_ingest(
    session: Session,
    *,
    source: str,
    day: date,
    status: str,
    run_id: str,
    rows: int = 0,
    rejected: int = 0,
    message: str | None = None,
) -> None:
    row = session.scalars(
        select(IngestLog).where(IngestLog.source == source, IngestLog.trade_date == day)
    ).one_or_none()
    if row is None:
        row = IngestLog(source=source, trade_date=day)
        session.add(row)
    row.status, row.rows, row.rejected = status, rows, rejected
    row.message, row.run_id = message, run_id


# ---------------------------------------------------------------- quality issues
def save_quality_issues(session: Session, run_id: str, dataset: str, issues: Iterable[Any]) -> int:
    n = 0
    for i in issues:
        session.add(
            DataQualityIssue(
                run_id=run_id,
                dataset=dataset,
                check=i.check,
                severity=i.severity,
                symbol=i.symbol,
                trade_date=i.trade_date,
                detail=i.detail,
            )
        )
        n += 1
    session.flush()
    return n
