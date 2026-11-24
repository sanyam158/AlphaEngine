"""Ingestion pipelines: fetch → raw store → parse → validate → symbol master → curated store.

Pipelines are idempotent and resumable: completed days are recorded in ``ingest_log`` only
after their bars are durably written, so a crash simply redoes the unfinished batch.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date

import pandas as pd
from sqlalchemy.orm import Session, sessionmaker

from alphaengine.core.errors import AlphaEngineError, DataSourceError
from alphaengine.core.ids import new_id
from alphaengine.core.logging import get_logger
from alphaengine.core.types import Symbol
from alphaengine.data.db import session_scope
from alphaengine.data.quality import Issue, QualityConfig, run_checks
from alphaengine.data.raw_store import RawStore
from alphaengine.data.repositories import (
    InstrumentRegistry,
    completed_days,
    load_corporate_actions,
    record_ingest,
    save_quality_issues,
    upsert_corporate_actions,
)
from alphaengine.data.schema import validate_bars
from alphaengine.data.sources.base import DailyBarSource, RawPayload
from alphaengine.data.sources.nse_bhavcopy import NseBhavcopySource
from alphaengine.data.sources.yahoo import YahooSource
from alphaengine.data.store import PriceStore
from alphaengine.market.calendar import candidate_days

log = get_logger(__name__)


@dataclass
class IngestSummary:
    run_id: str
    days_ok: int = 0
    days_no_data: int = 0
    days_skipped: int = 0
    days_error: int = 0
    rows: int = 0
    rejected: int = 0
    new_instruments: int = 0
    renames: int = 0
    errors: list[str] = field(default_factory=list)

    def add(self, r: _DayResult) -> None:
        if r.status == "ok":
            self.days_ok += 1
            self.rows += r.rows
            self.rejected += r.rejected
        elif r.status == "no_data":
            self.days_no_data += 1
        else:
            self.days_error += 1

    def counts(self) -> dict[str, int | str]:
        return {k: v for k, v in vars(self).items() if k != "errors"}


@dataclass
class _DayResult:
    day: date
    status: str  # ok | no_data | error
    bars: pd.DataFrame | None = None
    rows: int = 0
    rejected: int = 0
    message: str | None = None


class BhavcopyIngestor:
    def __init__(
        self,
        source: NseBhavcopySource,
        raw: RawStore,
        store: PriceStore,
        session_factory: sessionmaker[Session],
        *,
        batch_days: int = 20,
        max_consecutive_errors: int = 5,
        on_progress: Callable[[date, str], None] | None = None,
    ) -> None:
        self.source = source
        self.raw = raw
        self.store = store
        self.sf = session_factory
        self.batch_days = batch_days
        self.max_consecutive_errors = max_consecutive_errors
        self.on_progress = on_progress

    def _load_payload(self, day: date) -> RawPayload | None:
        for cand in self.source.candidates(day):
            cached = self.raw.get(self.source.name, day, cand.name)
            if cached is not None:
                return RawPayload(self.source.name, day, cand.name, cand.fmt, cached)
        payload = self.source.fetch_raw(day)
        if payload is not None:
            self.raw.put(payload.source, day, payload.name, payload.content)
        return payload

    def _ingest_day(self, day: date, registry: InstrumentRegistry) -> _DayResult:
        payload = self._load_payload(day)
        if payload is None:
            return _DayResult(day, "no_data")
        result = validate_bars(self.source.parse(payload))
        registry.observe(day, result.clean)
        return _DayResult(day, "ok", result.clean, len(result.clean), result.n_rejected)

    def run(
        self, start: date, end: date, *, force: bool = False, include_weekends: bool = False
    ) -> IngestSummary:
        if end < start:
            raise ValueError("end must be on or after start")
        summary = IngestSummary(run_id=new_id("ing"))
        log.info("bhavcopy ingest start", run_id=summary.run_id, start=str(start), end=str(end))
        with session_scope(self.sf) as s:
            done = set() if force else completed_days(s, self.source.name, start, end)
        candidates = candidate_days(start, end, include_weekends=include_weekends)
        days = [d for d in candidates if d not in done]
        summary.days_skipped = len(candidates) - len(days)

        consecutive_errors = 0
        batch: list[_DayResult] = []
        session = self.sf()
        registry = InstrumentRegistry(session)
        try:
            for day in days:
                try:
                    res = self._ingest_day(day, registry)
                    consecutive_errors = 0
                except AlphaEngineError as exc:
                    consecutive_errors += 1
                    res = _DayResult(day, "error", message=str(exc)[:2000])
                    summary.errors.append(f"{day}: {exc}")
                    log.warning("ingest day failed", day=str(day), error=str(exc))
                summary.add(res)
                batch.append(res)
                if self.on_progress:
                    self.on_progress(day, res.status)
                if consecutive_errors >= self.max_consecutive_errors:
                    self._flush(session, batch, summary.run_id)
                    raise DataSourceError(
                        f"aborting after {consecutive_errors} consecutive failures "
                        f"(last: {res.message}); the source may be blocking requests "
                        "or its format changed"
                    )
                if len(batch) >= self.batch_days:
                    self._flush(session, batch, summary.run_id)
            self._flush(session, batch, summary.run_id)
        except BaseException:
            session.rollback()
            raise
        finally:
            summary.new_instruments = registry.created
            summary.renames = registry.renamed
            session.close()
        log.info("bhavcopy ingest done", **summary.counts())
        return summary

    def _flush(self, session: Session, batch: list[_DayResult], run_id: str) -> None:
        if not batch:
            return
        frames = [r.bars for r in batch if r.bars is not None]
        if frames:
            self.store.write(pd.concat(frames, ignore_index=True), replace_dates=True)
        for r in batch:
            record_ingest(
                session,
                source=self.source.name,
                day=r.day,
                status=r.status,
                run_id=run_id,
                rows=r.rows,
                rejected=r.rejected,
                message=r.message,
            )
        session.commit()
        batch.clear()


def ingest_index_prices(
    source: DailyBarSource, store: PriceStore, symbols: Sequence[Symbol], start: date, end: date
) -> int:
    """Fetch index levels (e.g. NIFTY50) into a separate dataset; returns rows written."""
    result = validate_bars(source.fetch_bars(start, end, symbols))
    if result.n_rejected:
        log.warning("index bars rejected", n=result.n_rejected)
    return store.write(result.clean)


def ingest_corporate_actions(
    source: YahooSource,
    session_factory: sessionmaker[Session],
    symbols: Sequence[Symbol],
    *,
    pause: float = 0.5,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[int, list[str]]:
    """Fetch and upsert actions per symbol. Returns (rows written, symbols that failed)."""
    written, failed = 0, []
    for i, sym in enumerate(symbols):
        try:
            acts = source.fetch_actions(sym)
        except DataSourceError as exc:
            log.warning("actions fetch failed", symbol=sym, error=str(exc))
            failed.append(str(sym))
            continue
        if not acts.empty:
            with session_scope(session_factory) as s:
                written += upsert_corporate_actions(s, acts)
        if pause and i < len(symbols) - 1:
            sleep(pause)
    return written, failed


@dataclass(frozen=True)
class QualityReport:
    run_id: str
    issues: list[Issue]


def run_quality(
    store: PriceStore,
    session_factory: sessionmaker[Session],
    *,
    start: date | None = None,
    end: date | None = None,
    cfg: QualityConfig | None = None,
    persist: bool = True,
) -> QualityReport:
    bars = store.read(start=start, end=end)
    market_days = sorted(set(bars["date"].dt.date)) if not bars.empty else []
    with session_scope(session_factory) as s:
        actions = load_corporate_actions(s)
    issues = run_checks(bars, market_days, actions, cfg)
    run_id = new_id("dq")
    if persist:
        with session_scope(session_factory) as s:
            save_quality_issues(s, run_id, store.dataset, issues)
    return QualityReport(run_id, issues)
