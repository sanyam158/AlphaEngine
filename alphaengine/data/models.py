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


class StrategyConfigRecord(Base):
    """Immutable store of strategy configs: (name, version) maps to exactly one content hash."""

    __tablename__ = "strategy_configs"
    __table_args__ = (UniqueConstraint("name", "version"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    config_hash: Mapped[str] = mapped_column(String(16), nullable=False, unique=True)
    config: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class BacktestRunRecord(Base):
    """One backtest: exact strategy + settings + data snapshot + code version, and its metrics.
    Equity curve, trades and targets are stored as files under ``<data_dir>/runs/<run_id>/``."""

    __tablename__ = "backtest_runs"

    run_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    strategy: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    strategy_version: Mapped[int] = mapped_column(Integer, nullable=False)
    strategy_hash: Mapped[str] = mapped_column(String(16), nullable=False)
    strategy_config: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    backtest_config: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    data_snapshot: Mapped[str] = mapped_column(String(16), nullable=False)
    git_sha: Mapped[str | None] = mapped_column(String(40))
    duration_s: Mapped[float] = mapped_column(Float, nullable=False)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )


# ============================================================================ portfolios (Phase 6)
class Portfolio(Base):
    """A paper or real (manually executed) portfolio following one strategy."""

    __tablename__ = "portfolios"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    strategy: Mapped[str] = mapped_column(String(200), nullable=False)  # built-in name or YAML path
    strategy_hash: Mapped[str] = mapped_column(String(16), nullable=False)
    broker: Mapped[str] = mapped_column(String(16), nullable=False)  # paper | manual
    initial_capital: Mapped[float] = mapped_column(Numeric(16, 2), nullable=False)
    cash: Mapped[float] = mapped_column(Numeric(16, 2), nullable=False)
    active: Mapped[bool] = mapped_column(default=True, nullable=False)  # False = kill switch
    settings: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    last_processed: Mapped[date | None] = mapped_column(Date)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class PortfolioLot(Base):
    __tablename__ = "portfolio_lots"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    portfolio_id: Mapped[int] = mapped_column(
        ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False, index=True
    )
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    cost: Mapped[float] = mapped_column(Numeric(18, 6), nullable=False)
    acquired: Mapped[date] = mapped_column(Date, nullable=False)
    source: Mapped[str] = mapped_column(String(8), nullable=False, default="BUY")
    seq: Mapped[int] = mapped_column(Integer, nullable=False)


class PortfolioOrder(Base):
    """Order lifecycle NEW -> SUBMITTED -> PARTIAL/FILLED | CANCELLED | REJECTED | EXPIRED."""

    __tablename__ = "portfolio_orders"
    __table_args__ = (UniqueConstraint("idempotency_key"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    portfolio_id: Mapped[int] = mapped_column(
        ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False, index=True
    )
    idempotency_key: Mapped[str] = mapped_column(String(120), nullable=False)
    signal_date: Mapped[date] = mapped_column(Date, nullable=False)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    side: Mapped[str] = mapped_column(String(4), nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    est_price: Mapped[float] = mapped_column(Numeric(14, 2), nullable=False)
    status: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    filled_quantity: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    fill_price: Mapped[float | None] = mapped_column(Numeric(14, 2))
    charges: Mapped[float] = mapped_column(Numeric(12, 2), nullable=False, default=0)
    reason: Mapped[str] = mapped_column(String(200), nullable=False, default="")
    note: Mapped[str | None] = mapped_column(Text)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    filled_at: Mapped[date | None] = mapped_column(Date)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class PortfolioDisposal(Base):
    """Realized lots, kept for capital-gains reports."""

    __tablename__ = "portfolio_disposals"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    portfolio_id: Mapped[int] = mapped_column(
        ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False, index=True
    )
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    acquired: Mapped[date] = mapped_column(Date, nullable=False)
    sold: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    cost: Mapped[float] = mapped_column(Numeric(18, 6), nullable=False)
    gross_proceeds: Mapped[float] = mapped_column(Numeric(18, 6), nullable=False)
    sell_expenses: Mapped[float] = mapped_column(Numeric(14, 6), nullable=False)
    stt: Mapped[float] = mapped_column(Numeric(14, 6), nullable=False)
    lot_source: Mapped[str] = mapped_column(String(8), nullable=False)


class PortfolioSnapshot(Base):
    __tablename__ = "portfolio_snapshots"
    __table_args__ = (UniqueConstraint("portfolio_id", "day"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    portfolio_id: Mapped[int] = mapped_column(
        ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False, index=True
    )
    day: Mapped[date] = mapped_column(Date, nullable=False)
    cash: Mapped[float] = mapped_column(Numeric(16, 2), nullable=False)
    value: Mapped[float] = mapped_column(Numeric(16, 2), nullable=False)
    holdings: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)


class JobRun(Base):
    """One scheduled job execution per (job, run_date) for idempotency and history."""

    __tablename__ = "job_runs"
    __table_args__ = (UniqueConstraint("job", "run_date"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    job: Mapped[str] = mapped_column(String(40), nullable=False)
    run_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(12), nullable=False)  # ok | waiting | error
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    message: Mapped[str | None] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class JournalEntry(Base):
    """Free-text notes on decisions and trades: the learning log."""

    __tablename__ = "journal_entries"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
    portfolio: Mapped[str | None] = mapped_column(String(64), index=True)
    symbol: Mapped[str | None] = mapped_column(String(20))
    tags: Mapped[str | None] = mapped_column(String(200))
    text: Mapped[str] = mapped_column(Text, nullable=False)
