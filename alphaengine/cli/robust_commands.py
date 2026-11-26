"""CLI: ``alphaengine robust run``."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml

from alphaengine.app import build_context
from alphaengine.robustness.analysis import run_analysis
from alphaengine.robustness.report import write

robust_app = typer.Typer(help="Robustness and overfitting checks for a strategy.")


def parse_grid(items: list[str]) -> dict[str, list[Any]]:
    """``["selector.n=10,20,30", "scorer.params.skip=0,21"]`` -> {path: [values]} (YAML-typed)."""
    grid: dict[str, list[Any]] = {}
    for item in items:
        if "=" not in item:
            raise typer.BadParameter(f"grid entry must be path=v1,v2,...: {item!r}")
        path, values = item.split("=", 1)
        grid[path.strip()] = [yaml.safe_load(v.strip()) for v in values.split(",") if v.strip()]
    return grid


@robust_app.command("run")
def run(
    strategy: Annotated[str, typer.Argument(help="Built-in name or YAML path")],
    grid: Annotated[
        list[str] | None, typer.Option("--grid", help="path=v1,v2 (repeatable)")
    ] = None,
    start: Annotated[str, typer.Option(help="YYYY-MM-DD")] = "2012-01-01",
    end: Annotated[str | None, typer.Option(help="YYYY-MM-DD")] = None,
    folds: Annotated[int, typer.Option(help="Walk-forward folds")] = 5,
    benchmark: Annotated[str, typer.Option()] = "NIFTYBEES",
    confirm: Annotated[bool, typer.Option(help="Re-run base config in the full engine")] = True,
    out: Annotated[Path | None, typer.Option(help="Output directory")] = None,
) -> None:
    """Sweep parameters, then report stability, PBO, deflated Sharpe, walk-forward, bootstrap,
    regimes and cost sensitivity, with plain-language warnings."""
    ctx = build_context()
    g = parse_grid(grid or [])
    s = datetime.strptime(start, "%Y-%m-%d").date()  # noqa: DTZ007
    e = datetime.strptime(end, "%Y-%m-%d").date() if end else None  # noqa: DTZ007
    rep = run_analysis(
        ctx,
        strategy,
        g,
        s,
        e,
        folds=folds,
        benchmark=benchmark,
        confirm=confirm,
        on_progress=typer.echo,
    )
    base = rep.sweep.metrics.loc[rep.base_index]
    o = rep.overfitting
    typer.echo(f"\n{rep.strategy}: {len(rep.sweep.params)} variants")
    typer.echo(
        f"  base: CAGR {base['cagr']:.1%} Sharpe {base['sharpe']:.2f} | best Sharpe "
        f"{rep.stability.get('best', float('nan')):.2f} median {rep.stability.get('median', float('nan')):.2f}"
    )
    typer.echo(
        f"  deflated Sharpe {o['base'].get('deflated_sharpe', float('nan')):.0%}  "
        f"PBO {o['pbo'].get('pbo', float('nan')):.0%}  walk-forward OOS Sharpe "
        f"{rep.walk_forward.get('oos_sharpe', float('nan')):.2f} (IS {rep.walk_forward.get('mean_in_sample_sharpe', float('nan')):.2f})"
    )
    typer.echo(
        "  costs 0x/1x/2x: "
        + " / ".join(f"{rep.costs[k]:.1%}" for k in ("0x", "1x", "2x"))
        + f"   benchmark {rep.benchmark['cagr']:.1%}"
    )
    if rep.confirmation:
        c = rep.confirmation
        typer.echo(
            f"  full engine: CAGR {c['cagr']:.1%}  after-tax XIRR {c['after_tax_xirr']:.1%}  "
            f"max DD {c['max_drawdown']:.1%}"
        )
    for w in rep.warnings:
        typer.echo(f"  ! {w}")
    out_dir = out or ctx.settings.data_dir / "robustness" / rep.id
    h, _ = write(rep, out_dir)
    typer.echo(f"  report: {h}")
