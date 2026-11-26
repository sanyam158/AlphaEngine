"""The end-of-day job and a minimal scheduler loop.

``eod_job`` is idempotent per date. It:

1. ingests bhavcopies up to the run date (``waiting`` if today's file isn't published yet)
2. for every active portfolio: applies corporate actions, fills paper orders at today's open,
   marks to market and snapshots
3. on each portfolio's rebalance day (and on its first day) computes the target and the trade
   list, runs risk checks, and creates orders (paper: filled tomorrow; manual: your checklist)
4. sends one summary message (Telegram or the notifications log)

Run it once a day after ~19:00 IST with ``alphaengine scheduler run-eod`` (Windows Task
Scheduler / cron), or keep ``alphaengine scheduler start`` running.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from sqlalchemy import select

from alphaengine.app import AppContext
from alphaengine.core.clock import IST, Clock, SystemClock
from alphaengine.core.logging import get_logger
from alphaengine.data.db import session_scope
from alphaengine.data.models import JobRun
from alphaengine.data.pipeline import BhavcopyIngestor
from alphaengine.data.sources.nse_bhavcopy import NseBhavcopySource
from alphaengine.notify.notifier import Notifier, build_notifier
from alphaengine.portfolio.service import PortfolioService
from alphaengine.portfolio.store import list_portfolios

log = get_logger(__name__)
JOB = "eod"


@dataclass
class JobOutcome:
    run_date: date
    status: str  # ok | waiting | skipped | error
    message: str = ""
    sections: list[str] = field(default_factory=list)


def _record(ctx: AppContext, run_date: date, status: str, message: str) -> None:
    with session_scope(ctx.session_factory) as s:
        row = s.scalars(
            select(JobRun).where(JobRun.job == JOB, JobRun.run_date == run_date)
        ).one_or_none()
        if row is None:
            row = JobRun(job=JOB, run_date=run_date, status=status, attempts=0)
            s.add(row)
        row.status, row.message, row.attempts = status, message[:4000], (row.attempts or 0) + 1


def _already_done(ctx: AppContext, run_date: date) -> bool:
    with session_scope(ctx.session_factory) as s:
        row = s.scalars(
            select(JobRun).where(JobRun.job == JOB, JobRun.run_date == run_date)
        ).one_or_none()
        return row is not None and row.status == "ok"


def eod_job(
    ctx: AppContext,
    run_date: date | None = None,
    *,
    force: bool = False,
    ingest: bool = True,
    ingestor: BhavcopyIngestor | None = None,
    notifier: Notifier | None = None,
    service: PortfolioService | None = None,
    clock: Clock | None = None,
) -> JobOutcome:
    day = run_date or (clock or SystemClock()).today()
    notify = notifier or build_notifier(ctx.settings)
    if not force and _already_done(ctx, day):
        return JobOutcome(day, "skipped", "already done")
    try:
        if ingest:
            last = (
                ctx.equities.date_range() or (day - timedelta(days=7), day - timedelta(days=7))
            )[1]
            ing = ingestor or BhavcopyIngestor(
                NseBhavcopySource(), ctx.raw, ctx.equities, ctx.session_factory
            )
            ing.run(min(last + timedelta(days=1), day), day, include_weekends=True)
        if day not in set(ctx.equities.trading_days()):
            is_weekday = day.weekday() < 5
            status = "waiting" if is_weekday else "ok"
            msg = (
                "no bhavcopy for today yet (holiday or not published)" if is_weekday else "weekend"
            )
            _record(ctx, day, status, msg)
            return JobOutcome(day, status, msg)

        svc = service or PortfolioService(ctx)
        with session_scope(ctx.session_factory) as s:
            names = [p.name for p in list_portfolios(s, active_only=True)]
        sections = []
        for name in names:
            report = svc.process_day(name, day)
            text = report.describe()
            if svc.is_rebalance_day(name, day):
                res = svc.rebalance(name, day)
                text += "\n" + res.describe()
            sections.append(text)
        message = f"AlphaEngine EOD {day}\n\n" + ("\n\n".join(sections) or "no active portfolios")
        notify.send(message)
        _record(ctx, day, "ok", f"{len(names)} portfolio(s)")
        return JobOutcome(day, "ok", f"{len(names)} portfolio(s)", sections)
    except Exception as exc:
        log.error("eod job failed", day=str(day), error=repr(exc))
        _record(ctx, day, "error", repr(exc))
        try:
            notify.send(f"AlphaEngine EOD {day} FAILED: {exc!r}")
        except Exception as nexc:
            log.error("notification failed", error=repr(nexc))
        return JobOutcome(day, "error", repr(exc))


def next_run(now: datetime, at: str = "19:30") -> datetime:
    """Next weekday at ``at`` (IST) strictly after ``now``."""
    hh, mm = (int(x) for x in at.split(":"))
    cand = now.astimezone(IST).replace(hour=hh, minute=mm, second=0, microsecond=0)
    if cand <= now:
        cand += timedelta(days=1)
    while cand.weekday() >= 5:
        cand += timedelta(days=1)
    return cand


def run_scheduler(
    ctx: AppContext,
    *,
    at: str = "19:30",
    retry_minutes: int = 30,
    give_up_at: str = "23:30",
    clock: Clock | None = None,
    sleep: Callable[[float], None] = time.sleep,
    max_cycles: int | None = None,
    job: Callable[..., JobOutcome] = eod_job,
) -> None:
    """Blocking loop: run the EOD job each weekday evening, retrying while data is pending."""
    clk = clock or SystemClock()
    cycles = 0
    while max_cycles is None or cycles < max_cycles:
        cycles += 1
        target = next_run(clk.now(), at)
        sleep(max((target - clk.now()).total_seconds(), 0))
        gh, gm = (int(x) for x in give_up_at.split(":"))
        deadline = target.replace(hour=gh, minute=gm)
        while True:
            outcome = job(ctx, target.date())
            log.info("eod job", status=outcome.status, message=outcome.message)
            if (
                outcome.status != "waiting"
                or clk.now() + timedelta(minutes=retry_minutes) > deadline
            ):
                break
            sleep(retry_minutes * 60)
