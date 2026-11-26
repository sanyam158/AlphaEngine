"""Persistence for portfolios: load a portfolio's state into memory, change it, save it back."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from alphaengine.core.errors import AlphaEngineError
from alphaengine.data.models import (
    Portfolio,
    PortfolioDisposal,
    PortfolioLot,
    PortfolioOrder,
    PortfolioSnapshot,
)
from alphaengine.portfolio.lots import Disposal, LotBook, TaxLot

OPEN_STATUSES = ("NEW", "SUBMITTED", "PARTIAL")


class PortfolioError(AlphaEngineError):
    pass


@dataclass
class PortfolioState:
    """In-memory working copy of one portfolio (lots + cash + new disposals)."""

    record: Portfolio
    lots: LotBook
    cash: Decimal
    new_disposals: list[Disposal] = field(default_factory=list)

    def quantities(self) -> dict[str, int]:
        return {s: self.lots.quantity(s) for s in self.lots.symbols()}


def get_portfolio(session: Session, name: str) -> Portfolio:
    rec = session.scalars(select(Portfolio).where(Portfolio.name == name)).one_or_none()
    if rec is None:
        raise PortfolioError(f"no portfolio named {name!r}")
    return rec


def list_portfolios(session: Session, active_only: bool = False) -> list[Portfolio]:
    stmt = select(Portfolio).order_by(Portfolio.name)
    if active_only:
        stmt = stmt.where(Portfolio.active.is_(True))
    return list(session.scalars(stmt))


def create_portfolio(
    session: Session,
    *,
    name: str,
    strategy: str,
    strategy_hash: str,
    broker: str,
    capital: float,
    settings: dict[str, Any] | None = None,
) -> Portfolio:
    if broker not in ("paper", "manual"):
        raise PortfolioError("broker must be 'paper' or 'manual'")
    if session.scalars(select(Portfolio).where(Portfolio.name == name)).first():
        raise PortfolioError(f"portfolio {name!r} already exists")
    rec = Portfolio(
        name=name,
        strategy=strategy,
        strategy_hash=strategy_hash,
        broker=broker,
        initial_capital=capital,
        cash=capital,
        active=True,
        settings=settings or {},
    )
    session.add(rec)
    session.flush()
    return rec


def load_state(session: Session, rec: Portfolio) -> PortfolioState:
    book = LotBook()
    rows = session.scalars(select(PortfolioLot).where(PortfolioLot.portfolio_id == rec.id))
    book.restore(
        [
            TaxLot(r.symbol, r.quantity, Decimal(str(r.cost)), r.acquired, r.source, r.seq)
            for r in rows
        ]
    )
    return PortfolioState(rec, book, Decimal(str(rec.cash)))


def save_state(session: Session, state: PortfolioState) -> None:
    """Replace the portfolio's lots, update cash, append new disposals."""
    pid = state.record.id
    session.execute(delete(PortfolioLot).where(PortfolioLot.portfolio_id == pid))
    for lot in state.lots:
        session.add(
            PortfolioLot(
                portfolio_id=pid,
                symbol=lot.symbol,
                quantity=lot.quantity,
                cost=lot.cost,
                acquired=lot.acquired,
                source=lot.source,
                seq=lot.seq,
            )
        )
    for d in state.new_disposals:
        session.add(
            PortfolioDisposal(
                portfolio_id=pid,
                symbol=d.symbol,
                quantity=d.quantity,
                acquired=d.acquired,
                sold=d.sold,
                cost=d.cost,
                gross_proceeds=d.gross_proceeds,
                sell_expenses=d.sell_expenses,
                stt=d.stt,
                lot_source=d.lot_source,
            )
        )
    state.new_disposals.clear()
    state.record.cash = float(state.cash.quantize(Decimal("0.01")))
    session.flush()


def load_disposals(session: Session, rec: Portfolio) -> list[Disposal]:
    rows = session.scalars(
        select(PortfolioDisposal)
        .where(PortfolioDisposal.portfolio_id == rec.id)
        .order_by(PortfolioDisposal.sold, PortfolioDisposal.id)
    )
    return [
        Disposal(
            r.symbol,
            r.quantity,
            r.acquired,
            r.sold,
            Decimal(str(r.cost)),
            Decimal(str(r.gross_proceeds)),
            Decimal(str(r.sell_expenses)),
            Decimal(str(r.stt)),
            r.lot_source,
        )
        for r in rows
    ]


def open_orders(session: Session, rec: Portfolio) -> list[PortfolioOrder]:
    return list(
        session.scalars(
            select(PortfolioOrder)
            .where(PortfolioOrder.portfolio_id == rec.id, PortfolioOrder.status.in_(OPEN_STATUSES))
            .order_by(PortfolioOrder.side.desc(), PortfolioOrder.id)  # SELL before BUY
        )
    )


def recent_orders(session: Session, rec: Portfolio, limit: int = 50) -> list[PortfolioOrder]:
    return list(
        session.scalars(
            select(PortfolioOrder)
            .where(PortfolioOrder.portfolio_id == rec.id)
            .order_by(PortfolioOrder.id.desc())
            .limit(limit)
        )
    )


def snapshot(
    session: Session,
    rec: Portfolio,
    day: date,
    cash: Decimal,
    value: Decimal,
    holdings: dict[str, Any],
) -> None:
    row = session.scalars(
        select(PortfolioSnapshot).where(
            PortfolioSnapshot.portfolio_id == rec.id, PortfolioSnapshot.day == day
        )
    ).one_or_none()
    if row is None:
        row = PortfolioSnapshot(portfolio_id=rec.id, day=day)
        session.add(row)
    row.cash, row.value, row.holdings = float(cash), float(value), holdings


def snapshots(session: Session, rec: Portfolio) -> list[PortfolioSnapshot]:
    return list(
        session.scalars(
            select(PortfolioSnapshot)
            .where(PortfolioSnapshot.portfolio_id == rec.id)
            .order_by(PortfolioSnapshot.day)
        )
    )
