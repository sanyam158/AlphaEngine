"""One-call robustness analysis of a strategy: sweep → stability → overfitting statistics →
walk-forward → bootstrap → regimes → cost sensitivity → full-engine confirmation → warnings."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from alphaengine.app import AppContext
from alphaengine.backtest.config import BacktestConfig
from alphaengine.backtest.engine import simulate
from alphaengine.backtest.metrics import compute_metrics
from alphaengine.backtest.runner import benchmark_series, required_symbols
from alphaengine.backtest.vectorized import default_cost_per_side, run_vectorized
from alphaengine.core.ids import new_id
from alphaengine.features.data import MarketData
from alphaengine.robustness import stats
from alphaengine.robustness.sweep import SweepResult, expand_grid, get_path, run_sweep, variant
from alphaengine.strategy.config import AnyConfig, build_strategy, load_config

TRADING_DAYS = 252


@dataclass
class RobustnessReport:
    id: str
    strategy: str
    grid: dict[str, list[Any]]
    base_index: int
    sweep: SweepResult
    stability: dict[str, Any]
    overfitting: dict[str, Any]
    walk_forward: dict[str, Any]
    bootstrap: dict[str, Any]
    regimes: dict[str, Any]
    costs: dict[str, float]
    confirmation: dict[str, Any] | None
    benchmark: dict[str, float]
    warnings: list[str] = field(default_factory=list)


def _sharpe(r: pd.Series) -> float:
    sd = r.std(ddof=1)
    return float(r.mean() / sd * math.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")


def _cagr(r: pd.Series) -> float:
    return stats.annualise(r)


def load_for(
    ctx: AppContext,
    configs: Sequence[AnyConfig],
    start: date,
    end: date | None,
    benchmark: str,
    warmup_days: int = 450,
) -> MarketData:
    """One dataset covering every variant's universe (plus the benchmark)."""
    cfg = BacktestConfig(start=start, end=end, warmup_days=warmup_days)
    symbols: set[str] = {benchmark}
    seen: set[str] = set()
    for c in configs:
        dumped = c.model_dump(mode="json")
        key = json.dumps(dumped.get("universe") or dumped.get("blend"), sort_keys=True)
        if key in seen:  # variants usually share a universe: query it once
            continue
        seen.add(key)
        syms = required_symbols(build_strategy(c), ctx, cfg)
        if syms is None:
            return MarketData.from_store(
                ctx, start - timedelta(days=warmup_days), end, None, execution=True
            )
        symbols |= set(syms)
    return MarketData.from_store(
        ctx, start - timedelta(days=warmup_days), end, sorted(symbols), execution=True
    )


def analyze(
    base: AnyConfig,
    grid: Mapping[str, Sequence[Any]],
    md: MarketData,
    start: date,
    end: date | None = None,
    *,
    benchmark: str = "NIFTYBEES",
    folds: int = 5,
    rf: float = 0.06,
    confirm: bool = True,
    on_progress: Callable[[str], None] | None = None,
) -> RobustnessReport:
    say = on_progress or (lambda _m: None)
    full_grid: dict[str, list[Any]] = {k: list(v) for k, v in grid.items()}
    raw = base.model_dump(mode="json")
    for path, values in full_grid.items():  # the base config's own value must be on the grid
        current = get_path(raw, path)
        if current not in values:
            values.append(current)
            values.sort(key=lambda x: (str(type(x)), x))
    grid = full_grid
    say(f"sweeping {len(expand_grid(grid))} variants")
    sweep = run_sweep(
        base, grid, md, start, end, rf=rf, on_progress=lambda i, n, p: say(f"  {i}/{n} {p}")
    )
    base_params = {p: get_path(raw, p) for p in grid}
    base_index = next(i for i, p in enumerate(sweep.params) if p == base_params)
    rets = sweep.returns_matrix()
    base_r = rets[base_index]
    bench = benchmark_series(md, benchmark).loc[pd.Timestamp(start) :]
    bench_r = bench.pct_change(fill_method=None).reindex(base_r.index).fillna(0.0)

    say("overfitting statistics")
    best_index = int(sweep.metrics["sharpe"].astype(float).to_numpy().argmax())
    overfit = {
        "pbo": stats.pbo(rets),
        "base": stats.deflated_sharpe(base_r, rets),
        "best": stats.deflated_sharpe(rets[best_index], rets),
        "best_index": best_index,
    }
    say("walk-forward")
    wf = stats.walk_forward(rets, n_folds=folds)
    say("bootstrap")
    boot = stats.stationary_bootstrap(base_r)
    regimes = stats.regime_split(base_r, bench)

    say("cost sensitivity")
    base_strategy = build_strategy(variant(base, base_params))
    cps = default_cost_per_side()
    cost_cagr = {}
    for mult in (0.0, 1.0, 2.0):
        res = run_vectorized(
            base_strategy,
            md,
            start,
            end,
            cost_per_side=cps * mult,
            slippage_bps=10 * mult,
            cash_rate=rf,
        )
        cost_cagr[f"{mult:g}x"] = _cagr(res.returns)

    confirmation = None
    if confirm:
        say("confirming with the full engine (integer shares, fills, taxes)")
        bcfg = BacktestConfig(start=start, end=end, benchmark=benchmark, cash_rate=rf)
        full = simulate(base_strategy, md, bcfg)
        m = compute_metrics(
            full.equity,
            trades=full.trades,
            disposals=full.disposals,
            rf=rf,
            after_tax_value=full.liquidation["after_tax_value"],
            totals={k.removeprefix("total_"): v for k, v in full.liquidation.items()},
        )
        confirmation = {
            k: m[k]
            for k in (
                "cagr",
                "after_tax_xirr",
                "max_drawdown",
                "sharpe",
                "turnover",
                "tax_drag",
                "cost_drag",
            )
        }
        confirmation["vectorized_cagr"] = cost_cagr["1x"]

    bm = {"cagr": _cagr(bench_r), "sharpe": _sharpe(bench_r)}
    report = RobustnessReport(
        id=new_id("rb"),
        strategy=base.name,
        grid=full_grid,
        base_index=base_index,
        sweep=sweep,
        stability=_stability(sweep, list(grid)),
        overfitting=overfit,
        walk_forward=wf,
        bootstrap=boot,
        regimes=regimes,
        costs=cost_cagr,
        confirmation=confirmation,
        benchmark=bm,
    )
    report.warnings = warnings_for(report)
    return report


def _stability(sweep: SweepResult, params: list[str]) -> dict[str, Any]:
    from alphaengine.robustness.sweep import stability  # noqa: PLC0415

    return stability(sweep.metrics, params) if params else {}


def warnings_for(r: RobustnessReport) -> list[str]:
    """Plain-language red flags. An empty list doesn't prove a strategy works."""
    w: list[str] = []
    pbo = r.overfitting["pbo"].get("pbo")
    if pbo is not None and math.isfinite(pbo) and pbo > 0.5:
        w.append(
            f"PBO {pbo:.0%}: the in-sample winner usually ranks below median out of sample "
            "(likely overfit parameter choice)."
        )
    dsr = r.overfitting["base"].get("deflated_sharpe")
    if dsr is not None and math.isfinite(dsr) and dsr < 0.95:
        w.append(
            f"Deflated Sharpe {dsr:.0%} < 95%: after {r.overfitting['base']['trials']} "
            "variants, the Sharpe isn't statistically convincing."
        )
    eff = r.walk_forward.get("efficiency")
    if eff is not None and math.isfinite(eff) and eff < 0.5:
        w.append(
            f"Walk-forward efficiency {eff:.2f}: out-of-sample Sharpe is less than half the "
            "in-sample Sharpe."
        )
    base_sharpe = float(r.sweep.metrics["sharpe"].astype(float).iloc[r.base_index])
    med = r.stability.get("median")
    if med is not None and base_sharpe > 0 and med < 0.5 * base_sharpe:
        w.append("Most grid variants do far worse than the chosen parameters (fragile peak).")
    if r.costs.get("2x", 0) < r.benchmark["cagr"]:
        w.append(
            f"With double costs the strategy ({r.costs['2x']:.1%}) trails the benchmark "
            f"({r.benchmark['cagr']:.1%})."
        )
    p5 = r.bootstrap.get("cagr", {}).get("p5")
    if p5 is not None and p5 < 0:
        w.append(f"Bootstrap 5th-percentile CAGR is {p5:.1%}: a bad decade is plausible.")
    if r.confirmation and r.confirmation["after_tax_xirr"] < r.benchmark["cagr"] - 0.005:
        w.append(
            f"After costs and taxes ({r.confirmation['after_tax_xirr']:.1%}) it doesn't beat "
            f"simply holding the benchmark ({r.benchmark['cagr']:.1%} pre-tax)."
        )
    return w


def to_dict(r: RobustnessReport) -> dict[str, Any]:
    wf = {k: v for k, v in r.walk_forward.items() if k != "oos_returns"}
    return {
        "id": r.id,
        "strategy": r.strategy,
        "grid": r.grid,
        "base_index": r.base_index,
        "sweep": r.sweep.metrics.to_dict(orient="records"),
        "stability": r.stability,
        "overfitting": r.overfitting,
        "walk_forward": wf,
        "bootstrap": r.bootstrap,
        "regimes": r.regimes,
        "costs": r.costs,
        "confirmation": r.confirmation,
        "benchmark": r.benchmark,
        "warnings": r.warnings,
    }


def run_analysis(
    ctx: AppContext,
    source: str | Path,
    grid: Mapping[str, Sequence[Any]],
    start: date,
    end: date | None = None,
    **kw: Any,
) -> RobustnessReport:
    base = load_config(source)
    configs = [variant(base, p) for p in expand_grid({k: list(v) for k, v in grid.items()}) or [{}]]
    md = load_for(ctx, [base, *configs], start, end, kw.get("benchmark", "NIFTYBEES"))
    return analyze(base, grid, md, start, end, **kw)
