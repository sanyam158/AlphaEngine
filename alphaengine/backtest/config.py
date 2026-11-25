"""Backtest settings. Every default is a conservative, realistic choice for a retail investor
trading equity delivery on NSE; each run stores the exact config it used."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphaengine.market.costs import FLAT_20, NO_COSTS, ZERODHA, BrokerPlan, CostModel
from alphaengine.market.execution_model import (
    FixedBpsSlippage,
    SlippageModel,
    SqrtImpactSlippage,
)

BROKERS: dict[str, BrokerPlan] = {"zerodha": ZERODHA, "flat20": FLAT_20, "none": NO_COSTS}


class BacktestConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    start: date
    end: date | None = None  # default: last day with data
    initial_capital: float = Field(default=1_000_000, gt=0)
    sip_amount: float = Field(
        default=0, ge=0, description="added on each month's first trading day"
    )

    # Costs and fills (signals at close, fills at the next open)
    broker: Literal["zerodha", "flat20", "none"] = "zerodha"
    statutory_costs: bool = True
    slippage: Literal["fixed", "sqrt"] = "sqrt"
    slippage_bps: float = Field(default=10, ge=0, description="for slippage=fixed")
    max_participation: float = Field(default=0.1, gt=0, le=1, description="of 20-day ADV per day")
    max_fill_attempts: int = Field(default=5, ge=1, description="days to retry blocked orders")
    rebalance_band: float = Field(default=0.0, ge=0, lt=1, description="skip |Δweight| below this")
    min_trade_value: float = Field(default=1_000, ge=0)
    cash_buffer: float = Field(default=0.003, ge=0, lt=0.1, description="kept back for costs")

    # Cash, income and tax
    cash_rate: float = Field(default=0.06, ge=0, lt=0.5, description="annual yield on idle cash")
    taxes: bool = True
    income_tax_rate: float = Field(
        default=0.30, ge=0, le=0.5, description="slab rate on dividends and interest"
    )
    ambiguous_share_actions: Literal["bonus", "split"] = Field(
        default="bonus",
        description="how to treat SPLIT_OR_BONUS for stock tax lots (bonus = conservative); "
        "ETFs always split",
    )

    benchmark: str = "NIFTYBEES"  # an equity/ETF symbol (total return) or a loaded index
    warmup_days: int = Field(default=450, ge=0, description="calendar days loaded before start")

    @model_validator(mode="after")
    def _check(self) -> BacktestConfig:
        if self.end is not None and self.end <= self.start:
            raise ValueError("end must be after start")
        return self

    def cost_model(self) -> CostModel:
        return CostModel(BROKERS[self.broker], statutory=self.statutory_costs)

    def slippage_model(self) -> SlippageModel:
        if self.slippage == "fixed":
            return FixedBpsSlippage(self.slippage_bps)
        return SqrtImpactSlippage()

    def config_hash(self) -> str:
        raw = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
