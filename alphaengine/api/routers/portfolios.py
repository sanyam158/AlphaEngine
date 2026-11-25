from __future__ import annotations

from datetime import date
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import select

from alphaengine.api.deps import Ctx, Service
from alphaengine.api.schemas import (
    CreatePortfolio,
    FillRequest,
    JournalIn,
    JournalOut,
    OrderOut,
    PlannedTradeOut,
    PortfolioDetail,
    PortfolioSummary,
    RebalanceOut,
    SeriesPoint,
    TaxYear,
)
from alphaengine.core.errors import AlphaEngineError
from alphaengine.data.db import session_scope
from alphaengine.data.models import JournalEntry, PortfolioOrder
from alphaengine.portfolio.planner import PlannedTrade
from alphaengine.portfolio.store import get_portfolio, list_portfolios, recent_orders, snapshots

router = APIRouter(tags=["portfolios"])


def _err(exc: AlphaEngineError) -> HTTPException:
    msg = str(exc)
    return HTTPException(404 if msg.startswith("no ") else 409, msg)


def _order(o: PortfolioOrder) -> OrderOut:
    return OrderOut(
        id=o.id,
        signal_date=o.signal_date,
        symbol=o.symbol,
        side=o.side,
        quantity=o.quantity,
        est_price=float(o.est_price),
        status=o.status,
        filled_quantity=o.filled_quantity,
        fill_price=None if o.fill_price is None else float(o.fill_price),
        charges=float(o.charges or 0),
        reason=o.reason,
        note=o.note,
    )


def _trade(t: PlannedTrade) -> PlannedTradeOut:
    return PlannedTradeOut(
        symbol=t.symbol,
        side=t.side.value,
        quantity=t.quantity,
        est_price=t.est_price,
        value=t.value,
        reason=t.reason,
        note=t.note,
    )


@router.get("/portfolios", response_model=list[PortfolioSummary])
def list_all(ctx: Ctx) -> list[PortfolioSummary]:
    out = []
    with session_scope(ctx.session_factory) as s:
        for p in list_portfolios(s):
            snaps = snapshots(s, p)
            value = float(snaps[-1].value) if snaps else None
            change = (
                (float(snaps[-1].value) / float(snaps[-2].value) - 1) if len(snaps) > 1 else None
            )
            out.append(
                PortfolioSummary(
                    name=p.name,
                    strategy=p.strategy,
                    broker=p.broker,
                    active=p.active,
                    initial_capital=float(p.initial_capital),
                    cash=float(p.cash),
                    value=value,
                    last_processed=p.last_processed,
                    day_change=change,
                )
            )
    return out


@router.post("/portfolios", response_model=PortfolioSummary, status_code=201)
def create(req: CreatePortfolio, svc: Service) -> PortfolioSummary:
    try:
        svc.create(
            req.name,
            req.strategy,
            req.capital,
            req.broker,
            {"planner": {"band": req.band, "min_trade_value": req.min_trade_value}},
        )
    except AlphaEngineError as exc:
        raise _err(exc) from exc
    return PortfolioSummary(
        name=req.name,
        strategy=req.strategy,
        broker=req.broker,
        active=True,
        initial_capital=req.capital,
        cash=req.capital,
        value=None,
        last_processed=None,
        day_change=None,
    )


@router.get("/portfolios/{name}", response_model=PortfolioDetail)
def detail(name: str, ctx: Ctx, svc: Service) -> dict[str, Any]:
    try:
        sm = svc.summary(name)
    except AlphaEngineError as exc:
        raise _err(exc) from exc
    with session_scope(ctx.session_factory) as s:
        rec = get_portfolio(s, name)
        snaps = snapshots(s, rec)
        orders = [_order(o) for o in recent_orders(s, rec, 100)]
    equity = [SeriesPoint(date=x.day, value=float(x.value)) for x in snaps]
    change = equity[-1].value / equity[-2].value - 1 if len(equity) > 1 else None
    open_ = [o for o in orders if o.status in ("NEW", "SUBMITTED", "PARTIAL")]
    return {
        **sm,
        "day_change": change,
        "equity": equity,
        "open_orders": open_,
        "recent_orders": orders,
    }


@router.post("/portfolios/{name}/rebalance", response_model=RebalanceOut)
def rebalance(
    name: str,
    svc: Service,
    ctx: Ctx,
    dry_run: Annotated[bool, Query()] = True,
    as_of: Annotated[date | None, Query()] = None,
) -> RebalanceOut:
    day = as_of or (ctx.equities.date_range() or (date.today(), date.today()))[1]  # noqa: DTZ011
    try:
        res = svc.rebalance(name, day, dry_run=dry_run)
    except AlphaEngineError as exc:
        raise _err(exc) from exc
    return RebalanceOut(
        portfolio=name,
        as_of=res.as_of,
        trades=[_trade(t) for t in res.plan.trades],
        deferred=[_trade(t) for t in res.plan.deferred],
        est_charges=res.plan.est_charges,
        findings=[{"level": f.level, "message": f.message} for f in res.findings],
        notes=list(res.target.notes),
        orders_created=res.orders_created,
    )


@router.post("/portfolios/{name}/{action}", response_model=PortfolioSummary)
def set_active(name: str, action: str, svc: Service, ctx: Ctx) -> PortfolioSummary:
    if action not in ("pause", "resume"):
        raise HTTPException(404, "unknown action")
    try:
        svc.set_active(name, action == "resume")
    except AlphaEngineError as exc:
        raise _err(exc) from exc
    return next(p for p in list_all(ctx) if p.name == name)


@router.get("/portfolios/{name}/tax", response_model=list[TaxYear])
def tax(name: str, svc: Service) -> list[TaxYear]:
    try:
        hist = svc.tax_report(name)
    except AlphaEngineError as exc:
        raise _err(exc) from exc
    return [
        TaxYear(
            fy=f"FY{fy}-{(fy + 1) % 100:02d}",
            stcg=float(t.taxable_stcg),
            ltcg=float(t.taxable_ltcg),
            exemption_used=float(t.exemption_used),
            tax=float(t.total_tax),
            carried_forward=float(sum(c.amount for c in t.carried_forward)),
        )
        for fy, t in hist.items()
    ]


@router.post("/orders/{order_id}/fill", response_model=OrderOut)
def fill(order_id: int, req: FillRequest, svc: Service, ctx: Ctx) -> OrderOut:
    try:
        svc.record_fill(order_id, req.quantity, req.price, req.day, req.charges)
    except AlphaEngineError as exc:
        raise _err(exc) from exc
    with session_scope(ctx.session_factory) as s:
        o = s.get(PortfolioOrder, order_id)
        assert o is not None
        return _order(o)


@router.post("/orders/{order_id}/cancel", response_model=OrderOut)
def cancel(order_id: int, svc: Service, ctx: Ctx) -> OrderOut:
    try:
        svc.cancel_order(order_id)
    except AlphaEngineError as exc:
        raise _err(exc) from exc
    with session_scope(ctx.session_factory) as s:
        o = s.get(PortfolioOrder, order_id)
        assert o is not None
        return _order(o)


# ------------------------------------------------------------------ journal
@router.get("/journal", response_model=list[JournalOut], tags=["journal"])
def journal(ctx: Ctx, portfolio: str | None = None, limit: int = 100) -> list[JournalOut]:
    with session_scope(ctx.session_factory) as s:
        stmt = select(JournalEntry).order_by(JournalEntry.id.desc()).limit(limit)
        if portfolio:
            stmt = stmt.where(JournalEntry.portfolio == portfolio)
        return [
            JournalOut(
                id=e.id,
                created_at=e.created_at,
                text=e.text,
                portfolio=e.portfolio,
                symbol=e.symbol,
                tags=e.tags,
            )
            for e in s.scalars(stmt)
        ]


@router.post("/journal", response_model=JournalOut, status_code=201, tags=["journal"])
def add_journal(entry: JournalIn, ctx: Ctx) -> JournalOut:
    with session_scope(ctx.session_factory) as s:
        e = JournalEntry(
            text=entry.text,
            portfolio=entry.portfolio,
            symbol=entry.symbol.upper() if entry.symbol else None,
            tags=entry.tags,
        )
        s.add(e)
        s.flush()
        return JournalOut(
            id=e.id,
            created_at=e.created_at,
            text=e.text,
            portfolio=e.portfolio,
            symbol=e.symbol,
            tags=e.tags,
        )
