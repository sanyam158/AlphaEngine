from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from alphaengine.api.deps import Ctx
from alphaengine.api.routers.strategies import parse_text, resolve
from alphaengine.api.schemas import (
    BacktestDetail,
    BacktestJob,
    BacktestRequest,
    BacktestSummary,
    SeriesPoint,
)
from alphaengine.backtest.config import BacktestConfig
from alphaengine.backtest.engine import BacktestError
from alphaengine.backtest.metrics import drawdown, growth_index, twr_returns
from alphaengine.backtest.runner import get_run, list_runs, run_backtest
from alphaengine.core.errors import AlphaEngineError
from alphaengine.core.ids import new_id

router = APIRouter(prefix="/backtests", tags=["backtests"])
MAX_POINTS = 900


class JobBoard:
    """In-process backtest queue: one worker (backtests are memory-hungry)."""

    def __init__(self) -> None:
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="backtest")
        self.jobs: dict[str, BacktestJob] = {}
        self.lock = threading.Lock()

    def set(self, job: BacktestJob) -> None:
        with self.lock:
            self.jobs[job.job_id] = job


def _board(request: Request) -> JobBoard:
    board: JobBoard = request.app.state.backtests
    return board


def _points(s: pd.Series) -> list[SeriesPoint]:
    s = s.dropna()
    step = max(1, len(s) // MAX_POINTS)
    picked = s.iloc[::step]
    if len(s) and picked.index[-1] != s.index[-1]:
        picked = pd.concat([picked, s.iloc[[-1]]])
    return [
        SeriesPoint(date=pd.Timestamp(d).date(), value=float(v))
        for d, v in zip(picked.index, picked.to_numpy(), strict=True)
    ]


def _summary(rec: Any) -> dict[str, Any]:
    m = rec.metrics or {}
    return {
        "run_id": rec.run_id,
        "strategy": rec.strategy,
        "strategy_version": rec.strategy_version,
        "created_at": rec.created_at,
        "start": m.get("start"),
        "end": m.get("end"),
        "cagr": m.get("cagr"),
        "after_tax_xirr": m.get("after_tax_xirr"),
        "max_drawdown": m.get("max_drawdown"),
        "sharpe": m.get("sharpe"),
        "benchmark_cagr": (m.get("benchmark") or {}).get("cagr"),
    }


@router.get("", response_model=list[BacktestSummary])
def list_backtests(ctx: Ctx, limit: int = 50) -> list[dict[str, Any]]:
    return [_summary(r) for r in list_runs(ctx, limit)]


@router.post("", response_model=BacktestJob, status_code=202)
def start_backtest(req: BacktestRequest, ctx: Ctx, request: Request) -> BacktestJob:
    text = req.strategy.strip()
    try:
        source = parse_text(text) if ("\n" in text or text.startswith("{")) else resolve(ctx, text)
    except AlphaEngineError as exc:
        raise HTTPException(422, str(exc)) from exc
    cfg = BacktestConfig(
        start=req.start,
        end=req.end,
        initial_capital=req.capital,
        sip_amount=req.sip,
        broker="zerodha" if req.costs else "none",
        statutory_costs=req.costs,
        slippage="sqrt" if req.costs else "fixed",
        slippage_bps=10 if req.costs else 0,
        taxes=req.taxes,
        benchmark=req.benchmark,
    )
    board = _board(request)
    job = BacktestJob(job_id=new_id("job"), status="queued")
    board.set(job)

    def work() -> None:
        board.set(job.model_copy(update={"status": "running"}))
        try:
            run = run_backtest(source, cfg, ctx)
            board.set(job.model_copy(update={"status": "done", "run_id": run.run_id}))
        except Exception as exc:  # reported to the client, not raised in the worker
            board.set(job.model_copy(update={"status": "error", "error": str(exc)[:500]}))

    board.pool.submit(work)
    return job


@router.get("/jobs/{job_id}", response_model=BacktestJob)
def job_status(job_id: str, request: Request) -> BacktestJob:
    job = _board(request).jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return job


@router.get("/{run_id}", response_model=BacktestDetail)
def get_backtest(run_id: str, ctx: Ctx) -> dict[str, Any]:
    try:
        rec, eq = get_run(ctx, run_id)
    except BacktestError as exc:
        raise HTTPException(404, str(exc)) from exc
    idx = growth_index(pd.concat([pd.Series([0.0], index=[eq.index[0]]), twr_returns(eq)]))
    bench = (
        eq["benchmark"] / eq["benchmark"].dropna().iloc[0]
        if "benchmark" in eq
        else pd.Series(dtype=float)
    )
    return {
        **_summary(rec),
        "metrics": rec.metrics,
        "strategy_config": rec.strategy_config,
        "backtest_config": rec.backtest_config,
        "data_snapshot": rec.data_snapshot,
        "git_sha": rec.git_sha,
        "equity": _points(idx),
        "benchmark": _points(bench),
        "drawdown": _points(drawdown(idx)),
    }
