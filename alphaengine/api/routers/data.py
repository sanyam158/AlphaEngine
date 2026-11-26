from __future__ import annotations

from datetime import date
from typing import Annotated

from fastapi import APIRouter, Query
from sqlalchemy import func, select

from alphaengine.api.deps import Ctx
from alphaengine.api.schemas import DataHealth, DatasetInfo, JobRunOut, QualitySummary, SeriesPoint
from alphaengine.data.db import session_scope
from alphaengine.data.models import DataQualityIssue, JobRun

router = APIRouter(prefix="/data", tags=["data"])


@router.get("/health", response_model=DataHealth)
def data_health(ctx: Ctx) -> DataHealth:
    datasets = []
    for store in (ctx.equities, ctx.indices):
        rng = store.date_range()
        datasets.append(
            DatasetInfo(
                dataset=store.dataset,
                first=rng[0] if rng else None,
                last=rng[1] if rng else None,
                trading_days=len(store.trading_days()),
                symbols=len(store.symbols()),
                snapshot=store.snapshot_id(),
            )
        )
    with session_scope(ctx.session_factory) as s:
        run_id = s.scalars(
            select(DataQualityIssue.run_id).order_by(DataQualityIssue.id.desc()).limit(1)
        ).first()
        counts: dict[str, dict[str, int]] = {"info": {}, "warning": {}, "error": {}}
        examples = []
        if run_id:
            rows = s.execute(
                select(DataQualityIssue.severity, DataQualityIssue.check, func.count())
                .where(DataQualityIssue.run_id == run_id)
                .group_by(DataQualityIssue.severity, DataQualityIssue.check)
            )
            for sev, chk, n in rows:
                counts.setdefault(sev, {})[chk] = int(n)
            for i in s.scalars(
                select(DataQualityIssue)
                .where(DataQualityIssue.run_id == run_id, DataQualityIssue.severity != "info")
                .order_by(DataQualityIssue.trade_date.desc())
                .limit(15)
            ):
                examples.append(
                    {
                        "check": i.check,
                        "severity": i.severity,
                        "symbol": i.symbol,
                        "date": str(i.trade_date),
                        "detail": i.detail,
                    }
                )
        jobs = [
            JobRunOut(
                job=j.job,
                run_date=j.run_date,
                status=j.status,
                attempts=j.attempts,
                message=j.message,
                updated_at=j.updated_at,
            )
            for j in s.scalars(select(JobRun).order_by(JobRun.run_date.desc()).limit(20))
        ]
    return DataHealth(
        datasets=datasets,
        quality=QualitySummary(run_id=run_id, counts=counts, examples=examples),
        jobs=jobs,
    )


@router.get("/prices/{symbol}", response_model=list[SeriesPoint])
def prices(
    symbol: str,
    ctx: Ctx,
    start: Annotated[date | None, Query()] = None,
    end: Annotated[date | None, Query()] = None,
) -> list[SeriesPoint]:
    """Raw closing prices (equities first, then indices)."""
    sym = symbol.upper()
    for store in (ctx.equities, ctx.indices):
        df = store.read([sym], start, end, columns=["close"])
        if not df.empty:
            return [
                SeriesPoint(date=d.date(), value=float(v))
                for d, v in zip(df["date"], df["close"], strict=True)
            ]
    return []
