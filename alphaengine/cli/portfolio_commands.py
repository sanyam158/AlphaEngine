"""CLI: ``alphaengine portfolio ...``, ``alphaengine scheduler ...``, ``alphaengine notify test``."""

from __future__ import annotations

import csv
from datetime import date, datetime
from pathlib import Path
from typing import Annotated

import typer

from alphaengine.app import build_context
from alphaengine.core.clock import SystemClock
from alphaengine.core.types import Side
from alphaengine.data.db import session_scope
from alphaengine.notify.notifier import build_notifier
from alphaengine.portfolio.service import PortfolioService
from alphaengine.portfolio.store import get_portfolio, list_portfolios, recent_orders
from alphaengine.scheduler.jobs import eod_job, run_scheduler

portfolio_app = typer.Typer(help="Paper and real (manually executed) portfolios.")
scheduler_app = typer.Typer(help="Daily end-of-day automation.")
notify_app = typer.Typer(help="Notifications.")


def _d(s: str | None) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date() if s else SystemClock().today()  # noqa: DTZ007


def _svc() -> PortfolioService:
    return PortfolioService(build_context())


Name = Annotated[str, typer.Argument(help="Portfolio name")]
On = Annotated[str | None, typer.Option("--date", help="YYYY-MM-DD (default today)")]


@portfolio_app.command("create")
def create(
    name: Name,
    strategy: Annotated[str, typer.Option(help="Built-in name or YAML path")],
    capital: Annotated[float, typer.Option(help="Starting cash (₹)")] = 1_000_000,
    broker: Annotated[
        str, typer.Option(help="paper (simulated fills) or manual (you trade)")
    ] = "paper",
    band: Annotated[float, typer.Option(help="No-trade band on weights")] = 0.0,
    min_trade: Annotated[float, typer.Option(help="Smallest trade (₹)")] = 2_000,
) -> None:
    """Create a portfolio that follows a strategy."""
    _svc().create(
        name, strategy, capital, broker, {"planner": {"band": band, "min_trade_value": min_trade}}
    )
    typer.echo(f"created {name} ({broker}) following {strategy} with ₹{capital:,.0f}")


@portfolio_app.command("list")
def list_cmd() -> None:
    ctx = build_context()
    with session_scope(ctx.session_factory) as s:
        for p in list_portfolios(s):
            typer.echo(
                f"{p.name:<20} {p.broker:<7} {'active' if p.active else 'PAUSED':<7} "
                f"{p.strategy:<20} cash ₹{float(p.cash):,.0f} last {p.last_processed}"
            )


@portfolio_app.command("show")
def show(name: Name, on: On = None) -> None:
    """Holdings, weights, P&L and open orders."""
    sm = _svc().summary(name, _d(on) if on else None)
    typer.echo(
        f"{sm['name']} ({sm['broker']}, {'active' if sm['active'] else 'PAUSED'}) "
        f"{sm['strategy']} as of {sm['as_of']}"
    )
    typer.echo(
        f"  value ₹{sm['value']:,.0f}  cash ₹{sm['cash']:,.0f}  "
        f"since start {sm['value'] / sm['initial_capital'] - 1:+.1%}"
    )
    for h in sm["holdings"]:
        typer.echo(
            f"  {h['symbol']:<14} {h['quantity']:>7} @ ₹{h['price'] or 0:,.2f}  "
            f"{h['weight']:6.1%}  P&L ₹{h['pnl']:,.0f}  LT {h['long_term_qty']}"
        )
    for oid, side, qty, sym, status, px, note in sm["open_orders"]:
        typer.echo(f"  order #{oid}: {side} {qty} {sym} ~₹{px:,.2f} [{status}] {note or ''}")


@portfolio_app.command("rebalance")
def rebalance(
    name: Name,
    on: On = None,
    dry_run: Annotated[bool, typer.Option(help="Show the plan without creating orders")] = False,
) -> None:
    """Compute today's target and trade list now (normally done by the daily job)."""
    res = _svc().rebalance(name, _d(on), dry_run=dry_run)
    typer.echo(res.describe())


@portfolio_app.command("process")
def process(name: Name, on: On = None) -> None:
    """Apply corporate actions, paper fills and mark to market for one day."""
    typer.echo(_svc().process_day(name, _d(on)).describe())


@portfolio_app.command("orders")
def orders(name: Name, limit: int = 30) -> None:
    ctx = build_context()
    with session_scope(ctx.session_factory) as s:
        for o in recent_orders(s, get_portfolio(s, name), limit):
            fill = (
                f" filled {o.filled_quantity} @ ₹{float(o.fill_price):,.2f}" if o.fill_price else ""
            )
            typer.echo(
                f"#{o.id:<5} {o.signal_date} {o.side:<4} {o.quantity:>6} {o.symbol:<14} "
                f"[{o.status}]{fill} {o.reason} {o.note or ''}"
            )


@portfolio_app.command("checklist")
def checklist(
    name: Name, out: Annotated[Path, typer.Option(help="CSV to write")] = Path("orders.csv")
) -> None:
    """Export open orders as a CSV checklist to place at your broker."""
    ctx = build_context()
    with session_scope(ctx.session_factory) as s:
        rows = [
            o
            for o in recent_orders(s, get_portfolio(s, name), 500)
            if o.status in ("NEW", "SUBMITTED", "PARTIAL")
        ]
        with out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["order_id", "side", "symbol", "quantity", "est_price", "note"])
            for o in sorted(rows, key=lambda x: (x.side != "SELL", x.id)):
                w.writerow(
                    [
                        o.id,
                        o.side,
                        o.symbol,
                        o.quantity - o.filled_quantity,
                        float(o.est_price),
                        o.note or "",
                    ]
                )
    typer.echo(f"wrote {len(rows)} orders to {out} (place SELLs first)")


@portfolio_app.command("fill")
def fill(
    order_id: int,
    qty: Annotated[int, typer.Option(help="Shares filled")],
    price: Annotated[float, typer.Option(help="Average fill price")],
    on: On = None,
    charges: Annotated[
        float | None, typer.Option(help="Total charges from the contract note")
    ] = None,
) -> None:
    """Record a fill you got at your broker (manual portfolios)."""
    _svc().record_fill(order_id, qty, price, _d(on), charges)
    typer.echo(f"recorded {qty} @ ₹{price:,.2f} for order #{order_id}")


@portfolio_app.command("cancel")
def cancel_cmd(order_id: int) -> None:
    _svc().cancel_order(order_id)
    typer.echo(f"cancelled order #{order_id}")


@portfolio_app.command("trade")
def trade(
    name: Name,
    symbol: str,
    side: str,
    qty: int,
    price: float,
    on: On = None,
    charges: Annotated[float | None, typer.Option()] = None,
) -> None:
    """Record a trade made outside any order."""
    _svc().record_trade(name, symbol, Side(side.upper()), qty, price, _d(on), charges)
    typer.echo(f"recorded {side.upper()} {qty} {symbol.upper()} @ ₹{price:,.2f}")


@portfolio_app.command("import-holdings")
def import_holdings(
    name: Name,
    path: Path,
    deduct_cash: Annotated[
        bool, typer.Option(help="Pay for them out of the portfolio's cash")
    ] = False,
) -> None:
    """Load existing holdings: CSV with symbol,quantity,avg_price,acquired."""
    with path.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    n = _svc().import_holdings(name, rows, adjust_cash=deduct_cash)
    typer.echo(f"imported {n} lots into {name}")


@portfolio_app.command("reconcile")
def reconcile_cmd(name: Name, path: Path) -> None:
    """Compare the book with your broker's holdings export (symbol,quantity or Instrument,Qty.)."""
    diffs = _svc().reconcile(name, path)
    if not diffs:
        typer.echo("book matches broker holdings")
        return
    for sym, book, broker in diffs:
        typer.echo(f"  {sym:<14} book {book:>7}  broker {broker:>7}  diff {broker - book:+}")
    raise typer.Exit(code=1)


@portfolio_app.command("tax")
def tax(name: Name) -> None:
    """Capital-gains tax estimate per financial year from realised sales."""
    hist = _svc().tax_report(name)
    if not hist:
        typer.echo("no realised sales yet")
    for fy, t in hist.items():
        typer.echo(
            f"FY{fy}-{(fy + 1) % 100:02d}: STCG ₹{t.taxable_stcg:,.0f}  LTCG ₹{t.taxable_ltcg:,.0f} "
            f"(exemption used ₹{t.exemption_used:,.0f})  tax+cess ₹{t.total_tax:,.0f}"
            + (
                f"  carried loss ₹{sum(c.amount for c in t.carried_forward):,.0f}"
                if t.carried_forward
                else ""
            )
        )


@portfolio_app.command("harvest")
def harvest(name: Name, on: On = None) -> None:
    """Tax-loss and LTCG-exemption harvesting ideas for this financial year."""
    ideas = _svc().harvest(name, _d(on) if on else None)
    if not ideas:
        typer.echo("no harvesting opportunities worth the costs right now")
    for i in ideas:
        label = "LOSS" if i.kind == "loss" else "LTCG"
        typer.echo(
            f"  {label} {i.symbol:<14} sell {i.quantity} @ ~₹{i.price:,.2f}  gain ₹{i.gain:,.0f}  "
            f"tax this FY {i.tax_change:+,.0f}  costs ₹{i.round_trip_cost:,.0f}  - {i.note}"
        )


@portfolio_app.command("pause")
def pause(name: Name) -> None:
    """Kill switch: stop generating orders for this portfolio."""
    _svc().set_active(name, False)
    typer.echo(f"{name} paused")


@portfolio_app.command("resume")
def resume(name: Name) -> None:
    _svc().set_active(name, True)
    typer.echo(f"{name} resumed")


@scheduler_app.command("run-eod")
def run_eod(
    on: On = None,
    force: Annotated[bool, typer.Option(help="Run even if already done for the date")] = False,
    no_ingest: Annotated[bool, typer.Option(help="Skip downloading data")] = False,
) -> None:
    """Run the end-of-day job once (schedule this daily ~19:30 IST)."""
    out = eod_job(build_context(), _d(on) if on else None, force=force, ingest=not no_ingest)
    typer.echo(f"{out.run_date}: {out.status} {out.message}")
    for sec in out.sections:
        typer.echo(sec)
    if out.status == "error":
        raise typer.Exit(code=1)


@scheduler_app.command("start")
def start(at: Annotated[str, typer.Option(help="HH:MM IST")] = "19:30") -> None:
    """Keep running and execute the EOD job every weekday evening."""
    typer.echo(f"scheduler running; next run at {at} IST on weekdays (Ctrl+C to stop)")
    run_scheduler(build_context(), at=at)


@notify_app.command("test")
def notify_test() -> None:
    """Send a test message through the configured channel."""
    build_notifier(build_context().settings).send("AlphaEngine test notification")
    typer.echo("sent")
