"""CLI: ``alphaengine ingest ...``, ``alphaengine import ...``, ``alphaengine data ...``."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Annotated

import typer

from alphaengine.app import build_context
from alphaengine.core.clock import SystemClock
from alphaengine.core.types import Symbol, make_symbol
from alphaengine.data.db import session_scope
from alphaengine.data.importers import read_actions_csv, read_membership_csv
from alphaengine.data.pipeline import (
    BhavcopyIngestor,
    ingest_corporate_actions,
    ingest_index_prices,
    run_quality,
)
from alphaengine.data.quality import summarize
from alphaengine.data.repositories import (
    import_index_membership,
    members_as_of,
    record_index_snapshot,
    upsert_corporate_actions,
)
from alphaengine.data.sources.nse_bhavcopy import NseBhavcopySource
from alphaengine.data.sources.nse_indices import NseIndexConstituentsSource
from alphaengine.data.sources.yahoo import YahooSource

ingest_app = typer.Typer(help="Download data from external sources.")
import_app = typer.Typer(help="Import data from local CSV files.")
data_app = typer.Typer(help="Inspect and check stored data.")


def _date(value: str | None, default: date | None = None) -> date:
    if value is None:
        if default is None:
            raise typer.BadParameter("date required")
        return default
    return datetime.strptime(value, "%Y-%m-%d").date()  # noqa: DTZ007 - plain calendar date


StartOpt = Annotated[str, typer.Option("--start", help="YYYY-MM-DD")]
EndOpt = Annotated[str | None, typer.Option("--end", help="YYYY-MM-DD (default: today IST)")]


@ingest_app.command("bhavcopy")
def ingest_bhavcopy(
    start: StartOpt,
    end: EndOpt = None,
    force: Annotated[bool, typer.Option(help="Re-ingest days already done")] = False,
    include_weekends: Annotated[
        bool,
        typer.Option(
            "--weekends/--no-weekends",
            help="Also try Sat/Sun: NSE holds special sessions (Budget day, Muhurat) on weekends",
        ),
    ] = True,
) -> None:
    """Backfill / update NSE cash-market bhavcopies (resumable)."""
    ctx = build_context()
    s, e = _date(start), _date(end, SystemClock().today())

    def progress(day: date, status: str) -> None:
        if status != "ok" or day.day == 1:
            typer.echo(f"  {day} {status}")

    summary = BhavcopyIngestor(
        NseBhavcopySource(), ctx.raw, ctx.equities, ctx.session_factory, on_progress=progress
    ).run(s, e, force=force, include_weekends=include_weekends)
    typer.echo(
        f"ok={summary.days_ok} no_data={summary.days_no_data} skipped={summary.days_skipped} "
        f"errors={summary.days_error} rows={summary.rows} rejected={summary.rejected} "
        f"new_instruments={summary.new_instruments} renames={summary.renames}"
    )
    if summary.days_error:
        raise typer.Exit(code=1)


@ingest_app.command("index-prices")
def ingest_index(
    symbols: Annotated[list[str], typer.Argument(help="e.g. NIFTY50 NIFTYBANK")],
    start: StartOpt,
    end: EndOpt = None,
) -> None:
    """Index levels from Yahoo into the 'indices' dataset."""
    ctx = build_context()
    n = ingest_index_prices(
        YahooSource(),
        ctx.indices,
        [make_symbol(x) for x in symbols],
        _date(start),
        _date(end, SystemClock().today()),
    )
    typer.echo(f"wrote {n} index bars")


@ingest_app.command("actions")
def ingest_actions(
    symbols: Annotated[
        list[str] | None, typer.Argument(help="Symbols (default: --index members)")
    ] = None,
    index: Annotated[str | None, typer.Option(help="Use current members of this index")] = None,
) -> None:
    """Splits/bonuses/dividends from Yahoo."""
    ctx = build_context()
    syms: list[Symbol]
    if symbols:
        syms = [make_symbol(x) for x in symbols]
    elif index:
        with session_scope(ctx.session_factory) as s:
            syms = sorted(members_as_of(s, index, SystemClock().today()))
        if not syms:
            raise typer.BadParameter(f"no recorded members for {index}; run 'ingest index-members'")
    else:
        raise typer.BadParameter("give symbols or --index")
    written, failed = ingest_corporate_actions(YahooSource(), ctx.session_factory, syms)
    typer.echo(f"wrote {written} actions for {len(syms) - len(failed)} symbols; failed: {failed}")


@ingest_app.command("index-members")
def ingest_index_members(
    indices: Annotated[list[str], typer.Argument(help="e.g. NIFTY50 NIFTY200")],
) -> None:
    """Snapshot current NSE index constituents (builds point-in-time history over time)."""
    ctx = build_context()
    src = NseIndexConstituentsSource()
    today = SystemClock().today()
    for name in indices:
        constituents = src.fetch(name)
        with session_scope(ctx.session_factory) as s:
            diff = record_index_snapshot(s, name, constituents, today)
        typer.echo(f"{name}: {len(constituents)} members, +{len(diff.added)} -{len(diff.removed)}")


@import_app.command("actions")
def import_actions(path: Path) -> None:
    """Corporate actions CSV: symbol,ex_date,kind,ratio,amount."""
    ctx = build_context()
    with session_scope(ctx.session_factory) as s:
        n = upsert_corporate_actions(s, read_actions_csv(path), source="manual")
    typer.echo(f"imported {n} actions")


@import_app.command("index-members")
def import_members(path: Path) -> None:
    """Historical index membership CSV: index_name,symbol,start_date,end_date[,isin]."""
    ctx = build_context()
    with session_scope(ctx.session_factory) as s:
        n = import_index_membership(s, read_membership_csv(path))
    typer.echo(f"imported {n} membership rows")


@data_app.command("quality")
def data_quality(
    start: Annotated[str | None, typer.Option(help="YYYY-MM-DD")] = None,
    end: EndOpt = None,
    show: Annotated[int, typer.Option(help="Print the first N warnings/errors")] = 20,
) -> None:
    """Run data-quality checks on the equities dataset and store the issues."""
    ctx = build_context()
    report = run_quality(
        ctx.equities,
        ctx.session_factory,
        start=_date(start) if start else None,
        end=_date(end) if end else None,
    )
    for sev, checks in summarize(report.issues).items():
        typer.echo(f"{sev:8} {checks}")
    serious = [i for i in report.issues if i.severity != "info"]
    for i in serious[:show]:
        typer.echo(f"  [{i.severity}] {i.check} {i.symbol or ''} {i.trade_date or ''}: {i.detail}")
    typer.echo(f"run_id={report.run_id}")
    if any(i.severity == "error" for i in report.issues):
        raise typer.Exit(code=1)


@data_app.command("info")
def data_info() -> None:
    """Show stored date ranges, symbol counts and snapshot ids."""
    ctx = build_context()
    for store in (ctx.equities, ctx.indices):
        rng = store.date_range()
        if rng is None:
            typer.echo(f"{store.dataset}: empty")
            continue
        typer.echo(
            f"{store.dataset}: {rng[0]} to {rng[1]}, {len(store.trading_days())} days, "
            f"{len(store.symbols())} symbols, snapshot={store.snapshot_id()}"
        )
