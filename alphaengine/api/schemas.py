"""API request/response models (these drive the generated TypeScript client)."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class Health(BaseModel):
    status: Literal["ok"] = "ok"
    version: str


class SeriesPoint(BaseModel):
    date: date
    value: float


class DatasetInfo(BaseModel):
    dataset: str
    first: date | None
    last: date | None
    trading_days: int
    symbols: int
    snapshot: str


class JobRunOut(BaseModel):
    job: str
    run_date: date
    status: str
    attempts: int
    message: str | None
    updated_at: datetime | None


class QualitySummary(BaseModel):
    run_id: str | None
    counts: dict[str, dict[str, int]]
    examples: list[dict[str, Any]]


class DataHealth(BaseModel):
    datasets: list[DatasetInfo]
    quality: QualitySummary
    jobs: list[JobRunOut]


# ------------------------------------------------------------------ strategies
class StrategySummary(BaseModel):
    name: str
    version: int
    description: str
    kind: Literal["pipeline", "blend"]
    schedule: str
    source: Literal["builtin", "registered"]
    config_hash: str


class StrategyDetail(StrategySummary):
    config: dict[str, Any]
    yaml: str


class ValidateRequest(BaseModel):
    text: str = Field(description="Strategy config as YAML or JSON")


class ValidateResult(BaseModel):
    ok: bool
    error: str | None = None
    name: str | None = None
    config_hash: str | None = None
    config: dict[str, Any] | None = None


class Position(BaseModel):
    symbol: str
    weight: float
    reasons: list[str]


class SignalsOut(BaseModel):
    strategy: str
    as_of: date
    cash: float
    positions: list[Position]
    notes: list[str]


# ------------------------------------------------------------------ backtests
class BacktestRequest(BaseModel):
    strategy: str = Field(description="Built-in name, or YAML/JSON config text")
    start: date
    end: date | None = None
    capital: float = Field(default=1_000_000, gt=0)
    sip: float = Field(default=0, ge=0)
    costs: bool = True
    taxes: bool = True
    benchmark: str = "NIFTYBEES"


class BacktestJob(BaseModel):
    job_id: str
    status: Literal["queued", "running", "done", "error"]
    run_id: str | None = None
    error: str | None = None


class BacktestSummary(BaseModel):
    run_id: str
    strategy: str
    strategy_version: int
    created_at: datetime | None
    start: str | None
    end: str | None
    cagr: float | None
    after_tax_xirr: float | None
    max_drawdown: float | None
    sharpe: float | None
    benchmark_cagr: float | None


class BacktestDetail(BacktestSummary):
    metrics: dict[str, Any]
    strategy_config: dict[str, Any]
    backtest_config: dict[str, Any]
    data_snapshot: str
    git_sha: str | None
    equity: list[SeriesPoint]
    benchmark: list[SeriesPoint]
    drawdown: list[SeriesPoint]


# ------------------------------------------------------------------ portfolios
class Holding(BaseModel):
    symbol: str
    quantity: int
    price: float | None
    value: float
    cost: float
    pnl: float
    weight: float
    long_term_qty: int


class OrderOut(BaseModel):
    id: int
    signal_date: date
    symbol: str
    side: str
    quantity: int
    est_price: float
    status: str
    filled_quantity: int
    fill_price: float | None
    charges: float
    reason: str
    note: str | None


class PortfolioSummary(BaseModel):
    name: str
    strategy: str
    broker: str
    active: bool
    initial_capital: float
    cash: float
    value: float | None
    last_processed: date | None
    day_change: float | None


class PortfolioDetail(PortfolioSummary):
    as_of: date
    holdings: list[Holding]
    open_orders: list[OrderOut]
    recent_orders: list[OrderOut]
    equity: list[SeriesPoint]


class CreatePortfolio(BaseModel):
    name: str = Field(pattern=r"^[A-Za-z0-9_\-]{2,64}$")
    strategy: str
    capital: float = Field(default=1_000_000, gt=0)
    broker: Literal["paper", "manual"] = "paper"
    band: float = Field(default=0.0, ge=0, lt=1)
    min_trade_value: float = Field(default=2_000, ge=0)


class PlannedTradeOut(BaseModel):
    symbol: str
    side: str
    quantity: int
    est_price: float
    value: float
    reason: str
    note: str


class RebalanceOut(BaseModel):
    portfolio: str
    as_of: date
    trades: list[PlannedTradeOut]
    deferred: list[PlannedTradeOut]
    est_charges: float
    findings: list[dict[str, str]]
    notes: list[str]
    orders_created: int


class FillRequest(BaseModel):
    quantity: int = Field(gt=0)
    price: float = Field(gt=0)
    day: date
    charges: float | None = Field(default=None, ge=0)


class TaxYear(BaseModel):
    fy: str
    stcg: float
    ltcg: float
    exemption_used: float
    tax: float
    carried_forward: float


# ------------------------------------------------------------------ journal
class JournalIn(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    portfolio: str | None = None
    symbol: str | None = None
    tags: str | None = None


class JournalOut(JournalIn):
    id: int
    created_at: datetime | None


# ------------------------------------------------------------------ harvesting & planning
class HarvestIdeaOut(BaseModel):
    kind: Literal["loss", "ltcg_gain"]
    symbol: str
    quantity: int
    price: float
    gain: float
    long_term: bool
    tax_change: float
    round_trip_cost: float
    net_benefit: float
    note: str


class GoalRequest(BaseModel):
    target: float = Field(gt=0, description="In today's rupees")
    years: int = Field(ge=1, le=60)
    current: float = Field(default=0, ge=0)
    monthly_sip: float = Field(default=0, ge=0)
    step_up: float = Field(default=0, ge=0, le=0.5)
    inflation: float = Field(default=0.06, ge=0, le=0.2)
    after_tax: bool = True
    success: float = Field(default=0.8, gt=0, lt=1)
    run_id: str | None = None
    symbol: str = "NIFTYBEES"


class GoalBand(BaseModel):
    year: int
    p10: float
    p50: float
    p90: float


class GoalOut(BaseModel):
    probability: float
    target_nominal: float
    p10: float
    p50: float
    p90: float
    contributed: float
    required_sip: float | None
    source: str
    history_months: int
    bands: list[GoalBand]
