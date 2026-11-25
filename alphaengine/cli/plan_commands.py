"""CLI: ``alphaengine plan goal``."""

from __future__ import annotations

from datetime import date
from typing import Annotated

import typer

from alphaengine.app import build_context
from alphaengine.planning.goal import Goal, plan_goal
from alphaengine.planning.sources import history_for

plan_app = typer.Typer(help="Financial planning.")


def _amount(text: str) -> float:
    """Accepts 50000, 50k, 12L / 12 lakh, 2cr / 2 crore."""
    t = text.strip().lower().replace(",", "").replace("₹", "")
    for suffix, mult in (("crore", 1e7), ("cr", 1e7), ("lakh", 1e5), ("l", 1e5), ("k", 1e3)):
        if t.endswith(suffix):
            return float(t[: -len(suffix)].strip()) * mult
    return float(t)


@plan_app.command("goal")
def goal(
    target: Annotated[str, typer.Option(help="Goal in today's rupees, e.g. 2cr, 75L")],
    years: Annotated[int, typer.Option()],
    current: Annotated[str, typer.Option(help="Invested today")] = "0",
    sip: Annotated[str, typer.Option(help="Monthly SIP")] = "0",
    step_up: Annotated[float, typer.Option(help="Yearly SIP increase, e.g. 0.1")] = 0.0,
    inflation: Annotated[float, typer.Option()] = 0.06,
    run: Annotated[str | None, typer.Option(help="Use a backtest run's returns")] = None,
    symbol: Annotated[str, typer.Option(help="Else this ETF/index's history")] = "NIFTYBEES",
    success: Annotated[float, typer.Option(help="Confidence for the required-SIP answer")] = 0.8,
    no_tax: Annotated[bool, typer.Option(help="Ignore LTCG on exit")] = False,
) -> None:
    """Probability of reaching a target corpus, and the SIP needed."""
    ctx = build_context()
    hist, source = history_for(ctx, run=run, symbol=symbol, start=date(2010, 1, 1))
    g = Goal(
        target=_amount(target),
        years=years,
        current=_amount(current),
        monthly_sip=_amount(sip),
        step_up=step_up,
        inflation=inflation,
        after_tax=not no_tax,
    )
    r = plan_goal(g, hist, success=success, source=source)
    typer.echo(f"Returns resampled from {source} ({len(hist)} months of history)")
    typer.echo(
        f"  target ₹{g.target:,.0f} today = ₹{r.target_nominal:,.0f} in {years} years "
        f"at {inflation:.0%} inflation"
    )
    typer.echo(
        f"  chance of reaching it: {r.probability:.0%}  (you contribute ₹{r.contributed:,.0f})"
    )
    p = r.final_percentiles
    typer.echo(
        f"  outcome range{' after exit tax' if g.after_tax else ''}: "
        f"bad ₹{p['p10']:,.0f} · typical ₹{p['p50']:,.0f} · good ₹{p['p90']:,.0f}"
    )
    if r.required_sip is not None:
        typer.echo(
            f"  monthly SIP for {success:.0%} confidence: ₹{r.required_sip:,.0f}"
            + (f" (+{step_up:.0%}/yr)" if step_up else "")
        )
