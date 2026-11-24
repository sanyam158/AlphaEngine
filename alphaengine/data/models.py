"""ORM models. Every new table needs an Alembic migration (``alembic revision --autogenerate``)."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from alphaengine.data.db import Base


class AuditLog(Base):
    """Append-only record of every signal, order and manual action."""

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
    actor: Mapped[str] = mapped_column(String(64), nullable=False)  # "system", "user", job name
    action: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    entity_type: Mapped[str | None] = mapped_column(String(64))
    entity_id: Mapped[str | None] = mapped_column(String(64), index=True)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON)


class Instrument(Base):
    """Symbol master keyed by ISIN, so history survives symbol renames."""

    __tablename__ = "instruments"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    isin: Mapped[str] = mapped_column(String(12), unique=True, nullable=False)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    exchange: Mapped[str] = mapped_column(String(8), nullable=False, default="NSE")
    asset_class: Mapped[str] = mapped_column(String(16), nullable=False, default="EQUITY")
    series: Mapped[str | None] = mapped_column(String(4))
    name: Mapped[str | None] = mapped_column(String(200))
    industry: Mapped[str | None] = mapped_column(String(100))
    first_seen: Mapped[date] = mapped_column(Date, nullable=False)
    last_seen: Mapped[date] = mapped_column(Date, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class SymbolHistory(Base):
    """Which trading symbol an ISIN used over which period (``valid_to`` is exclusive)."""

    __tablename__ = "symbol_history"
    __table_args__ = (UniqueConstraint("isin", "symbol", "valid_from"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    isin: Mapped[str] = mapped_column(
        String(12), ForeignKey("instruments.isin", ondelete="CASCADE"), nullable=False, index=True
    )
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    valid_from: Mapped[date] = mapped_column(Date, nullable=False)
    valid_to: Mapped[date | None] = mapped_column(Date)


class CorporateAction(Base):
    """Split/bonus (``ratio`` = new shares per old share) or dividend (raw ``amount``/share)."""

    __tablename__ = "corporate_actions"
    __table_args__ = (UniqueConstraint("symbol", "ex_date", "kind"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    ex_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)  # SPLIT | BONUS | DIVIDEND
    ratio: Mapped[float | None] = mapped_column(Float)
    amount: Mapped[float | None] = mapped_column(Numeric(14, 4))
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class IndexMembership(Base):
    """Point-in-time index membership: member during ``[start_date, end_date)``."""

    __tablename__ = "index_membership"
    __table_args__ = (UniqueConstraint("index_name", "symbol", "start_date"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    index_name: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    isin: Mapped[str | None] = mapped_column(String(12))
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date | None] = mapped_column(Date)
    source: Mapped[str] = mapped_column(String(32), nullable=False)


class IngestLog(Base):
    """One row per (source, trade_date) recording the outcome of the last ingest attempt."""

    __tablename__ = "ingest_log"
    __table_args__ = (UniqueConstraint("source", "trade_date"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    trade_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False)  # ok | no_data | error
    rows: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    rejected: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    message: Mapped[str | None] = mapped_column(Text)
    run_id: Mapped[str] = mapped_column(String(40), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class DataQualityIssue(Base):
    __tablename__ = "data_quality_issues"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    dataset: Mapped[str] = mapped_column(String(32), nullable=False)
    check: Mapped[str] = mapped_column(String(40), nullable=False)
    severity: Mapped[str] = mapped_column(String(8), nullable=False)  # info | warning | error
    symbol: Mapped[str | None] = mapped_column(String(20))
    trade_date: Mapped[date | None] = mapped_column(Date)
    detail: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
