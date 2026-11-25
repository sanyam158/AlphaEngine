"""Paper/manual portfolios, the EOD job and the CLI on a synthetic store (no network)."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from alphaengine.app import AppContext, build_context
from alphaengine.core.clock import IST, FixedClock
from alphaengine.core.types import Side
from alphaengine.data.db import Base, session_scope
from alphaengine.data.models import Instrument
from alphaengine.data.repositories import upsert_corporate_actions
from alphaengine.data.store import PriceStore
from alphaengine.notify.notifier import LogNotifier
from alphaengine.portfolio.service import PortfolioService
from alphaengine.portfolio.store import PortfolioError, get_portfolio, open_orders
from alphaengine.scheduler.jobs import JobOutcome, eod_job, next_run, run_scheduler
from tests.helpers import random_walk_bars

NAMES = {"S00": "NIFTYBEES", "S01": "JUNIORBEES", "S02": "GOLDBEES", "S03": "BANKBEES"}


@pytest.fixture
def ctx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppContext:
    monkeypatch.setenv("ALPHAENGINE_DATABASE_URL", f"sqlite:///{(tmp_path / 'ae.db').as_posix()}")
    monkeypatch.setenv("ALPHAENGINE_DATA_DIR", str(tmp_path / "data"))
    from alphaengine.config.settings import get_settings

    get_settings.cache_clear()
    bars = random_walk_bars(n_symbols=4, n_days=400, seed=5, gap_prob=0.0, start="2023-01-02")
    bars["symbol"] = bars["symbol"].map(lambda s: NAMES[str(s)]).astype("string")
    PriceStore(tmp_path / "data" / "curated", "equities").write(bars)
    c = build_context()
    Base.metadata.create_all(c.engine)
    with session_scope(c.session_factory) as s:  # ETFs are recognised by their INF... ISINs
        for i, sym in enumerate(NAMES.values()):
            s.add(
                Instrument(
                    isin=f"INF00{i}A01011",
                    symbol=sym,
                    first_seen=date(2023, 1, 2),
                    last_seen=date(2024, 7, 1),
                )
            )
    return c


def days(ctx: AppContext) -> list[date]:
    return ctx.equities.trading_days()


def test_paper_portfolio_lifecycle(ctx: AppContext) -> None:
    svc = PortfolioService(ctx)
    svc.create("paper1", "fixed_mix", 1_000_000)
    with pytest.raises(PortfolioError, match="already exists"):
        svc.create("paper1", "fixed_mix", 1)
    d = days(ctx)
    t0, t1 = d[300], d[301]
    assert svc.is_rebalance_day("paper1", t0)  # never traded: start now
    res = svc.rebalance("paper1", t0)
    assert res.orders_created == 3
    assert "NIFTYBEES" in res.describe()
    assert svc.rebalance("paper1", t0).orders_created == 0  # idempotent for the signal date
    # a newer plan replaces orders that are still unfilled
    preview = svc.rebalance("paper1", d[299 + 2], dry_run=True)
    assert any("will be replaced" in f.message for f in preview.findings)
    rep = svc.process_day("paper1", t1)
    assert any("filled BUY" in n for n in rep.notes)
    sm = svc.summary("paper1", t1)
    weights = {h["symbol"]: h["weight"] for h in sm["holdings"]}
    assert weights["NIFTYBEES"] == pytest.approx(0.6, abs=0.02)
    assert sm["cash"] < 20_000
    # mid-quarter day: no rebalance for a quarterly strategy
    mid = next(x for x in d[302:] if x.month in (1, 4, 7, 10) and x.day > 10)
    assert not svc.is_rebalance_day("paper1", mid)
    # dividend applied on its ex-date
    with session_scope(ctx.session_factory) as s:
        upsert_corporate_actions(
            s,
            pd.DataFrame(
                [
                    {
                        "symbol": "GOLDBEES",
                        "ex_date": d[303],
                        "kind": "DIVIDEND",
                        "ratio": None,
                        "amount": 1.0,
                    }
                ]
            ),
        )
    svc.process_day("paper1", d[302])
    rep2 = svc.process_day("paper1", d[303])
    assert any("dividend GOLDBEES" in n for n in rep2.notes)
    assert rep2.day_change is not None
    # kill switch
    svc.set_active("paper1", False)
    with pytest.raises(PortfolioError, match="paused"):
        svc.rebalance("paper1", d[304])
    assert svc.rebalance("paper1", d[304], dry_run=True).orders_created == 0
    svc.set_active("paper1", True)
    # selling realises gains for the tax report
    svc.record_trade("paper1", "NIFTYBEES", Side.SELL, 10, 120.0, d[305])
    assert svc.tax_report("paper1")


def test_manual_portfolio_flow(ctx: AppContext, tmp_path: Path) -> None:
    svc = PortfolioService(ctx)
    svc.create("real", "buy_hold_nifty", 200_000, broker="manual")
    d = days(ctx)
    res = svc.rebalance("real", d[300])
    assert res.orders_created == 1
    rep = svc.process_day("real", d[301])
    assert "waiting for your fills" in rep.notes[0]
    with session_scope(ctx.session_factory) as s:
        oid = open_orders(s, get_portfolio(s, "real"))[0].id
    svc.record_fill(oid, 100, 101.5, d[301], charges=30.0)
    with pytest.raises(PortfolioError, match="no order"):
        svc.record_fill(9999, 1, 1.0, d[301])
    holdings = tmp_path / "h.csv"
    holdings.write_text("Instrument,Qty.\nNIFTYBEES,100\n", encoding="utf-8")
    assert svc.reconcile("real", holdings) == []
    holdings.write_text("Instrument,Qty.\nNIFTYBEES,90\nGOLDBEES,5\n", encoding="utf-8")
    assert svc.reconcile("real", holdings) == [("GOLDBEES", 0, 5), ("NIFTYBEES", 100, 90)]
    n = svc.import_holdings(
        "real",
        [{"symbol": "goldbees", "quantity": 5, "avg_price": 40, "acquired": "2023-06-01"}],
        adjust_cash=True,
    )
    assert n == 1
    assert svc.summary("real")["holdings"][0]["symbol"] in {"NIFTYBEES", "GOLDBEES"}


def test_eod_job(ctx: AppContext, tmp_path: Path) -> None:
    svc = PortfolioService(ctx)
    svc.create("paper1", "fixed_mix", 1_000_000)
    note = LogNotifier(tmp_path / "notes.log")
    d = days(ctx)
    out = eod_job(ctx, d[300], ingest=False, notifier=note, service=svc)
    assert out.status == "ok"
    assert "REBALANCE paper1" in out.sections[0]
    assert "AlphaEngine EOD" in (tmp_path / "notes.log").read_text(encoding="utf-8")
    assert eod_job(ctx, d[300], ingest=False, notifier=note).status == "skipped"
    assert eod_job(ctx, d[301], ingest=False, notifier=note, service=svc).status == "ok"
    weekday_gap = d[-1] + timedelta(days=(7 - d[-1].weekday()) % 7 or 7)  # a later Monday
    assert eod_job(ctx, weekday_gap, ingest=False, notifier=note).status == "waiting"
    sat = next(
        d[-1] + timedelta(days=k) for k in range(1, 8) if (d[-1] + timedelta(days=k)).weekday() == 5
    )
    assert eod_job(ctx, sat, ingest=False, notifier=note).status == "ok"

    class Boom(PortfolioService):
        def process_day(self, name, day):  # type: ignore[no-untyped-def]
            raise RuntimeError("disk full")

    bad = eod_job(ctx, d[302], ingest=False, notifier=note, service=Boom(ctx))
    assert bad.status == "error"
    assert "FAILED" in (tmp_path / "notes.log").read_text(encoding="utf-8")


def test_scheduler_loop() -> None:
    fri = datetime(2026, 3, 6, 20, 0, tzinfo=IST)
    assert next_run(fri) == datetime(2026, 3, 9, 19, 30, tzinfo=IST)  # skips the weekend
    assert next_run(datetime(2026, 3, 9, 10, 0, tzinfo=IST)) == datetime(
        2026, 3, 9, 19, 30, tzinfo=IST
    )
    clock = FixedClock(datetime(2026, 3, 9, 19, 0, tzinfo=IST))
    calls: list[str] = []

    def sleep(seconds: float) -> None:
        clock.advance(timedelta(seconds=seconds))

    def job(ctx, day):  # type: ignore[no-untyped-def]
        calls.append(str(day))
        return JobOutcome(day, "waiting" if len(calls) < 3 else "ok")

    run_scheduler(None, clock=clock, sleep=sleep, max_cycles=1, job=job)  # type: ignore[arg-type]
    assert calls == ["2026-03-09"] * 3  # retried every 30 min until data arrived


def test_cli(ctx: AppContext, tmp_path: Path) -> None:
    from alphaengine.cli.main import app

    r = CliRunner()
    d = days(ctx)
    run = lambda *a: r.invoke(app, list(a))  # noqa: E731
    assert (
        run("portfolio", "create", "p", "--strategy", "fixed_mix", "--capital", "500000").exit_code
        == 0
    )
    assert "p " in run("portfolio", "list").output
    out = run("portfolio", "rebalance", "p", "--date", str(d[300]), "--dry-run").output
    assert "no orders created" in out
    assert "orders created: 3" in run("portfolio", "rebalance", "p", "--date", str(d[300])).output
    chk = tmp_path / "o.csv"
    assert "wrote 3 orders" in run("portfolio", "checklist", "p", "--out", str(chk)).output
    assert "filled" in run("portfolio", "process", "p", "--date", str(d[301])).output
    assert "[FILLED]" in run("portfolio", "orders", "p").output
    assert "NIFTYBEES" in run("portfolio", "show", "p", "--date", str(d[301])).output
    assert (
        "recorded SELL"
        in run(
            "portfolio", "trade", "p", "niftybees", "sell", "5", "110", "--date", str(d[302])
        ).output
    )
    assert (
        "FY" in run("portfolio", "tax", "p").output
        or "no realised" in run("portfolio", "tax", "p").output
    )
    h = tmp_path / "h.csv"
    h.write_text("symbol,quantity\nGOLDBEES,1\n", encoding="utf-8")
    assert run("portfolio", "reconcile", "p", str(h)).exit_code == 1
    lots = tmp_path / "lots.csv"
    lots.write_text(
        "symbol,quantity,avg_price,acquired\nBANKBEES,3,100,2023-02-01\n", encoding="utf-8"
    )
    assert "imported 1" in run("portfolio", "import-holdings", "p", str(lots)).output
    assert "paused" in run("portfolio", "pause", "p").output
    assert "resumed" in run("portfolio", "resume", "p").output
    assert "ok" in run("scheduler", "run-eod", "--date", str(d[303]), "--no-ingest").output
    assert "sent" in run("notify", "test").output
    run("portfolio", "create", "m", "--strategy", "buy_hold_nifty", "--broker", "manual")
    run("portfolio", "rebalance", "m", "--date", str(d[300]))
    with session_scope(ctx.session_factory) as s:
        oid = open_orders(s, get_portfolio(s, "m"))[0].id
    assert (
        "recorded 1"
        in run(
            "portfolio", "fill", str(oid), "--qty", "1", "--price", "100", "--date", str(d[301])
        ).output
    )
    assert "cancelled" in run("portfolio", "cancel", str(oid)).output
