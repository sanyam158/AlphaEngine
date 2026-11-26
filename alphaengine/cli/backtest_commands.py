"""CLI: ``alphaengine backtest run|list|show|compare``."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import typer

from alphaengine.app import build_context
from alphaengine.backtest.config import BacktestConfig
from alphaengine.backtest.report import write_reports
from alphaengine.backtest.runner import get_run, list_runs, run_backtest, run_dir

backtest_app = typer.Typer(help="Backtest strategies and inspect past runs.")


def _d(s: str | None) -> Any:
    return datetime.strptime(s, "%Y-%m-%d").date() if s else None  # noqa: DTZ007


def _pct(x: Any) -> str:
    return "     -" if x is None else f"{x:6.1%}"


@backtest_app.command("run")
def run(
    strategy: Annotated[str, typer.Argument(help="Built-in name or YAML path")],
    start: Annotated[str, typer.Option(help="YYYY-MM-DD")] = "2012-01-01",
    end: Annotated[str | None, typer.Option(help="YYYY-MM-DD (default: latest data)")] = None,
    capital: Annotated[float, typer.Option(help="Initial capital (₹)")] = 1_000_000,
    sip: Annotated[float, typer.Option(help="Monthly contribution (₹)")] = 0,
    broker: Annotated[str, typer.Option(help="zerodha | flat20 | none")] = "zerodha",
    no_costs: Annotated[bool, typer.Option(help="Ignore all charges and slippage")] = False,
    no_tax: Annotated[bool, typer.Option(help="Ignore capital-gains and income tax")] = False,
    cash_rate: Annotated[float, typer.Option(help="Annual yield on idle cash")] = 0.06,
    band: Annotated[float, typer.Option(help="No-trade band on weight changes")] = 0.0,
    benchmark: Annotated[str, typer.Option(help="Benchmark symbol or index")] = "NIFTYBEES",
    report: Annotated[
        Path | None, typer.Option(help="Directory for report.html + metrics.json")
    ] = None,
) -> None:
    """Simulate a strategy with Indian costs and taxes, store the run, write a report."""
    cfg = BacktestConfig(
        start=_d(start),
        end=_d(end),
        initial_capital=capital,
        sip_amount=sip,
        broker="none" if no_costs else broker,
        statutory_costs=not no_costs,
        slippage="fixed" if no_costs else "sqrt",
        slippage_bps=0 if no_costs else 10,
        taxes=not no_tax,
        cash_rate=cash_rate,
        rebalance_band=band,
        benchmark=benchmark,
    )
    ctx = build_context()
    res = run_backtest(strategy, cfg, ctx)
    m = res.metrics
    bm = m.get("benchmark", {})
    typer.echo(f"run {res.run_id}: {res.result.strategy} {m['start']} → {m['end']}")
    typer.echo(
        f"  CAGR {_pct(m['cagr'])}  vs {benchmark} {_pct(bm.get('cagr'))}   "
        f"after-tax XIRR {_pct(m.get('after_tax_xirr'))}"
    )
    typer.echo(
        f"  max DD {_pct(m['max_drawdown'])}  sharpe {m['sharpe']:.2f}  "
        f"turnover {_pct(m['turnover'])}/yr  trades {m.get('trades', 0)}"
    )
    typer.echo(
        f"  final ₹{m['final_value']:,.0f}  after tax & exit ₹{m['after_tax_value']:,.0f}  "
        f"costs ₹{m['costs']:,.0f}  taxes ₹{m['taxes']:,.0f}"
    )
    out = report or run_dir(ctx, res.run_id)
    html_path, _ = write_reports(
        out,
        m,
        res.result.equity,
        res.benchmark,
        title=res.result.strategy,
        config=cfg.model_dump(mode="json"),
        strategy_config=res.result.strategy_config,
        notes=res.result.notes,
        run_id=res.run_id,
    )
    typer.echo(f"  report: {html_path}")


@backtest_app.command("list")
def list_cmd(limit: int = 20) -> None:
    """Recent runs, newest first."""
    ctx = build_context()
    typer.echo(
        f"{'run_id':<30} {'strategy':<18} {'start':<10} {'CAGR':>7} {'bench':>7} "
        f"{'maxDD':>7} {'aftTax':>7}"
    )
    for r in list_runs(ctx, limit):
        m = r.metrics
        typer.echo(
            f"{r.run_id:<30} {r.strategy:<18} {m.get('start')!s:<10} {_pct(m.get('cagr')):>7} "
            f"{_pct((m.get('benchmark') or {}).get('cagr')):>7} {_pct(m.get('max_drawdown')):>7} "
            f"{_pct(m.get('after_tax_xirr')):>7}"
        )


@backtest_app.command("show")
def show(run_id: str) -> None:
    """Key metrics, settings and provenance of one run."""
    ctx = build_context()
    rec, equity = get_run(ctx, run_id)
    m = rec.metrics
    typer.echo(
        f"{rec.run_id} {rec.strategy} v{rec.strategy_version} ({rec.strategy_hash}) "
        f"data={rec.data_snapshot} git={rec.git_sha or '?'} took {rec.duration_s}s"
    )
    for k in (
        "start",
        "end",
        "cagr",
        "xirr",
        "after_tax_xirr",
        "volatility",
        "sharpe",
        "max_drawdown",
        "turnover",
        "cost_drag",
        "tax_drag",
        "avg_holding_days",
    ):
        v = m.get(k)
        typer.echo(f"  {k:<18} {v:.4f}" if isinstance(v, float) else f"  {k:<18} {v}")
    typer.echo(f"  report: {run_dir(ctx, run_id) / 'report.html'}  ({len(equity)} days)")


@backtest_app.command("compare")
def compare(run_ids: list[str]) -> None:
    """Side-by-side metrics for several runs."""
    ctx = build_context()
    recs = [get_run(ctx, r)[0] for r in run_ids]
    keys = [
        "cagr",
        "after_tax_xirr",
        "volatility",
        "sharpe",
        "max_drawdown",
        "turnover",
        "cost_drag",
        "tax_drag",
    ]
    typer.echo(f"{'metric':<16}" + "".join(f"{r.strategy[:16]:>18}" for r in recs))
    for k in keys:
        row = "".join(
            f"{(f'{r.metrics.get(k):.2%}' if k != 'sharpe' else f'{r.metrics.get(k):.2f}') if r.metrics.get(k) is not None else '-':>18}"
            for r in recs
        )
        typer.echo(f"{k:<16}{row}")
