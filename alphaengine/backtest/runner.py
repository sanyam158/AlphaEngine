"""Load exactly the data a strategy needs, run it, compute metrics, persist the run."""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import select

from alphaengine.app import AppContext
from alphaengine.backtest.config import BacktestConfig
from alphaengine.backtest.engine import BacktestError, BacktestResult, simulate
from alphaengine.backtest.metrics import compute_metrics, to_jsonable
from alphaengine.core.ids import new_id
from alphaengine.core.logging import get_logger
from alphaengine.data.db import session_scope
from alphaengine.data.models import BacktestRunRecord, IndexMembership
from alphaengine.features.data import MarketData, liquid_symbols
from alphaengine.strategy.config import AnyConfig, build_strategy, config_hash, load_config
from alphaengine.strategy.core import Strategy
from alphaengine.strategy.pipeline import Blend, PipelineStrategy
from alphaengine.strategy.stages import IndexUniverse, LiquidityUniverse, StaticUniverse

log = get_logger(__name__)


@dataclass
class BacktestRun:
    run_id: str
    result: BacktestResult
    metrics: dict[str, Any]
    strategy_hash: str
    data_snapshot: str
    benchmark: pd.Series


def required_symbols(  # noqa: PLR0911 - one return per universe type
    strategy: Strategy, ctx: AppContext, cfg: BacktestConfig
) -> list[str] | None:
    """Columns to load: static symbols, a liquidity superset, or index members. ``None``
    means "everything" (unknown universe type)."""
    end = cfg.end or (ctx.equities.date_range() or (cfg.start, cfg.start))[1]
    start = cfg.start - timedelta(days=cfg.warmup_days)
    if isinstance(strategy, Blend):
        out: set[str] = set()
        for child, _ in strategy.parts:
            syms = required_symbols(child, ctx, cfg)
            if syms is None:
                return None
            out |= set(syms)
        return sorted(out)
    if not isinstance(strategy, PipelineStrategy):
        return None
    u = strategy.universe
    if isinstance(u, StaticUniverse):
        return sorted(u.symbols)
    if isinstance(u, LiquidityUniverse):
        return liquid_symbols(ctx.equities, start, end, top_per_year=3 * u.top_n)
    if isinstance(u, IndexUniverse):
        with session_scope(ctx.session_factory) as s:
            rows = s.scalars(
                select(IndexMembership.symbol).where(IndexMembership.index_name == u.index.upper())
            )
            return sorted(set(rows))
    return None


def benchmark_series(md: MarketData, name: str) -> pd.Series:
    if name in md.tr_close.columns:
        return md.tr_close[name].astype(float)
    if not md.benchmarks.empty and name in md.benchmarks.columns:
        return md.benchmarks[name].astype(float)
    raise BacktestError(f"benchmark {name!r} not found in loaded data")


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
            cwd=Path(__file__).resolve().parents[2],
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def run_backtest(
    source: str | Path | AnyConfig, cfg: BacktestConfig, ctx: AppContext, *, persist: bool = True
) -> BacktestRun:
    t0 = time.perf_counter()
    scfg = load_config(source) if isinstance(source, (str, Path)) else source
    strategy = build_strategy(scfg)
    syms = required_symbols(strategy, ctx, cfg)
    if syms is not None and cfg.benchmark not in syms:
        syms = [*syms, cfg.benchmark]
    load_start = cfg.start - timedelta(days=cfg.warmup_days)
    log.info("loading data", symbols="all" if syms is None else len(syms), start=str(load_start))
    md = MarketData.from_store(ctx, load_start, cfg.end, syms, execution=True)
    result = simulate(strategy, md, cfg)
    result.strategy_config = scfg.model_dump(mode="json")
    bench = benchmark_series(md, cfg.benchmark).loc[result.equity.index[0] :]
    metrics = compute_metrics(
        result.equity,
        trades=result.trades,
        disposals=result.disposals,
        benchmark=bench,
        rf=cfg.cash_rate,
        after_tax_value=result.liquidation["after_tax_value"],
        totals={k.removeprefix("total_"): v for k, v in result.liquidation.items()},
    )
    metrics["liquidation"] = {
        k: v for k, v in result.liquidation.items() if not k.startswith("total_")
    }
    metrics["taxes_by_fy"] = {
        f"FY{fy}-{(fy + 1) % 100:02d}": float(t.total_tax) for fy, t in result.taxes.items()
    }
    run = BacktestRun(
        run_id=new_id("bt"),
        result=result,
        metrics=metrics,
        strategy_hash=config_hash(scfg),
        data_snapshot=ctx.equities.snapshot_id(),
        benchmark=bench,
    )
    if persist:
        save_run(ctx, run, scfg, time.perf_counter() - t0)
    return run


def run_dir(ctx: AppContext, run_id: str) -> Path:
    return ctx.settings.data_dir / "runs" / run_id


def save_run(ctx: AppContext, run: BacktestRun, scfg: AnyConfig, duration: float) -> Path:
    out = run_dir(ctx, run.run_id)
    out.mkdir(parents=True, exist_ok=True)
    res = run.result
    eq = res.equity.copy()
    eq["benchmark"] = run.benchmark.reindex(eq.index).ffill()
    eq.to_parquet(out / "equity.parquet")
    res.trades.to_parquet(out / "trades.parquet", index=False)
    pd.DataFrame(
        [
            {"date": t.as_of, "weights": json.dumps(dict(t.weights)), "notes": " | ".join(t.notes)}
            for t in res.targets
        ]
    ).to_parquet(out / "targets.parquet", index=False)
    pd.DataFrame(
        [
            {
                "symbol": d.symbol,
                "quantity": d.quantity,
                "acquired": d.acquired,
                "sold": d.sold,
                "cost": float(d.cost),
                "proceeds": float(d.net_proceeds),
                "gain": float(d.gain),
                "long_term": d.long_term,
                "lot_source": d.lot_source,
            }
            for d in res.disposals
        ],
        columns=[
            "symbol",
            "quantity",
            "acquired",
            "sold",
            "cost",
            "proceeds",
            "gain",
            "long_term",
            "lot_source",
        ],
    ).to_parquet(out / "disposals.parquet", index=False)
    (out / "notes.txt").write_text("\n".join(res.notes), encoding="utf-8")
    with session_scope(ctx.session_factory) as s:
        s.add(
            BacktestRunRecord(
                run_id=run.run_id,
                strategy=scfg.name,
                strategy_version=scfg.version,
                strategy_hash=run.strategy_hash,
                strategy_config=scfg.model_dump(mode="json"),
                backtest_config=res.config.model_dump(mode="json"),
                data_snapshot=run.data_snapshot,
                git_sha=_git_sha(),
                duration_s=round(duration, 2),
                metrics=to_jsonable(run.metrics),
            )
        )
    return out


def list_runs(ctx: AppContext, limit: int = 20) -> list[BacktestRunRecord]:
    with session_scope(ctx.session_factory) as s:
        rows = s.scalars(
            select(BacktestRunRecord).order_by(BacktestRunRecord.created_at.desc()).limit(limit)
        ).all()
        s.expunge_all()
        return list(rows)


def get_run(ctx: AppContext, run_id: str) -> tuple[BacktestRunRecord, pd.DataFrame]:
    with session_scope(ctx.session_factory) as s:
        rec = s.get(BacktestRunRecord, run_id)
        if rec is None:
            raise BacktestError(f"no backtest run {run_id!r}")
        s.expunge(rec)
    equity = pd.read_parquet(run_dir(ctx, run_id) / "equity.parquet")
    return rec, equity
