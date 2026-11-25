"""Portfolio operations used by the CLI and the daily job."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from alphaengine.app import AppContext
from alphaengine.backtest.config import BacktestConfig
from alphaengine.backtest.runner import required_symbols
from alphaengine.core.types import Side, make_symbol
from alphaengine.data.db import session_scope
from alphaengine.data.models import Instrument, PortfolioOrder
from alphaengine.data.repositories import load_corporate_actions
from alphaengine.execution.brokers import (
    DayBar,
    ManualBroker,
    PaperBroker,
    apply_fill,
    cancel,
    read_holdings_csv,
    reconcile,
)
from alphaengine.features.data import MarketData
from alphaengine.market.calendar import TradingCalendar
from alphaengine.market.costs import CostModel
from alphaengine.market.tax import FYTax, compute_tax_history, financial_year
from alphaengine.portfolio.lots import TaxLot
from alphaengine.portfolio.planner import Plan, PlannerSettings, plan_rebalance
from alphaengine.portfolio.store import (
    PortfolioError,
    PortfolioState,
    create_portfolio,
    get_portfolio,
    load_disposals,
    load_state,
    open_orders,
    save_state,
    snapshot,
    snapshots,
)
from alphaengine.risk.checks import Finding, RiskLimits, check_plan, drawdown_alert
from alphaengine.strategy.config import build_strategy, config_hash, load_config
from alphaengine.strategy.core import StrategyContext, TargetPortfolio

HISTORY_DAYS = 800  # calendar days of data loaded to compute signals


@dataclass
class DayReport:
    portfolio: str
    day: date
    value: float
    cash: float
    day_change: float | None
    notes: list[str] = field(default_factory=list)

    def describe(self) -> str:
        chg = f" ({self.day_change:+.2%} vs previous)" if self.day_change is not None else ""
        lines = [f"{self.portfolio} {self.day}: ₹{self.value:,.0f}{chg}, cash ₹{self.cash:,.0f}"]
        lines += [f"  {n}" for n in self.notes]
        return "\n".join(lines)


@dataclass
class RebalanceResult:
    portfolio: str
    as_of: date
    target: TargetPortfolio
    plan: Plan
    findings: list[Finding]
    orders_created: int

    def describe(self) -> str:
        head = f"REBALANCE {self.portfolio} (signals at close {self.as_of})"
        risk = [f"  [{f.level}] {f.message}" for f in self.findings]
        status = (
            ("orders created: " + str(self.orders_created))
            if self.orders_created
            else (
                "no orders created"
                + (
                    " (blocked by risk checks)"
                    if any(f.level == "error" for f in self.findings)
                    else ""
                )
            )
        )
        return "\n".join(
            [
                head,
                *[f"  note: {n}" for n in self.target.notes],
                self.plan.describe(),
                *risk,
                status,
            ]
        )


class PortfolioService:
    def __init__(
        self,
        ctx: AppContext,
        cost_model: CostModel | None = None,
        limits: RiskLimits | None = None,
        calendar: TradingCalendar | None = None,
    ) -> None:
        self.ctx = ctx
        self.costs = cost_model or CostModel()
        self.limits = limits or RiskLimits()
        self._calendar = calendar

    # ------------------------------------------------------------------ helpers
    def calendar(self) -> TradingCalendar:
        if self._calendar is None:
            self._calendar = TradingCalendar(observed_trading_days=self.ctx.equities.trading_days())
        return self._calendar

    def _etfs(self, s: Session) -> set[str]:
        return {
            sym
            for sym, isin in s.execute(select(Instrument.symbol, Instrument.isin))
            if isin.startswith("INF")
        }

    def bars(self, symbols: list[str], day: date) -> dict[str, DayBar]:
        if not symbols:
            return {}
        df = self.ctx.equities.read(
            symbols, day - timedelta(days=10), day, columns=["open", "high", "low", "close"]
        )
        out: dict[str, DayBar] = {}
        for sym, grp in df.groupby("symbol"):
            g = grp.sort_values("date")
            today = g[g["date"] == pd.Timestamp(day)]
            if today.empty:
                continue
            prev = g[g["date"] < pd.Timestamp(day)]
            r = today.iloc[0]
            out[str(sym)] = DayBar(
                float(r["open"]),
                float(r["high"]),
                float(r["low"]),
                float(r["close"]),
                float(prev["close"].iloc[-1]) if len(prev) else None,
            )
        return out

    def closes(self, symbols: list[str], day: date) -> dict[str, float]:
        """Last raw close on or before ``day`` per symbol."""
        if not symbols:
            return {}
        df = self.ctx.equities.read(symbols, day - timedelta(days=30), day, columns=["close"])
        last = df.sort_values("date").groupby("symbol")["close"].last()
        return {str(k): float(v) for k, v in last.items()}

    # ------------------------------------------------------------------ lifecycle
    def create(
        self,
        name: str,
        strategy: str,
        capital: float,
        broker: str = "paper",
        settings: dict[str, Any] | None = None,
    ) -> None:
        cfg = load_config(strategy)
        build_strategy(cfg)  # validate now, not at the first rebalance
        with session_scope(self.ctx.session_factory) as s:
            create_portfolio(
                s,
                name=name,
                strategy=strategy,
                strategy_hash=config_hash(cfg),
                broker=broker,
                capital=capital,
                settings=settings,
            )

    def set_active(self, name: str, active: bool) -> None:
        with session_scope(self.ctx.session_factory) as s:
            get_portfolio(s, name).active = active

    # ------------------------------------------------------------------ daily processing
    def process_day(self, name: str, day: date) -> DayReport:
        """Corporate actions since last run, paper fills at today's open, mark to market."""
        with session_scope(self.ctx.session_factory) as s:
            rec = get_portfolio(s, name)
            state = load_state(s, rec)
            notes = self._apply_actions(s, state, rec.last_processed, day)
            orders = open_orders(s, rec)
            if rec.broker == "paper" and orders:
                syms = sorted({o.symbol for o in orders})
                bars = self.bars(syms, day)
                notes += PaperBroker(self.costs).fill(state, orders, day, bars.get)
            elif orders:
                notes.append(f"{len(orders)} open order(s) waiting for your fills")
            held = state.quantities()
            prices = self.closes(list(held), day)
            value = state.cash + sum(
                Decimal(str(prices.get(sym, 0.0))) * q for sym, q in held.items()
            )
            prev = snapshots(s, rec)
            prev_values = [float(x.value) for x in prev if x.day < day]
            snapshot(
                s,
                rec,
                day,
                state.cash,
                value,
                {sym: {"qty": q, "price": prices.get(sym)} for sym, q in held.items()},
            )
            save_state(s, state)
            rec.last_processed = day
            alert = drawdown_alert([*prev_values, float(value)], self.limits)
            if alert:
                notes.append(f"[{alert.level}] {alert.message}")
            change = float(value) / prev_values[-1] - 1 if prev_values else None
            return DayReport(name, day, float(value), float(state.cash), change, notes)

    def _apply_actions(
        self, s: Session, state: PortfolioState, after: date | None, day: date
    ) -> list[str]:
        held = state.quantities()
        if not held:
            return []
        acts = load_corporate_actions(s, list(held))
        if acts.empty:
            return []
        acts["ex_date"] = pd.to_datetime(acts["ex_date"]).dt.date
        # Only actions since the last processed day (just today's on the first run).
        start = after or (day - timedelta(days=1))
        acts = acts[(acts["ex_date"] > start) & (acts["ex_date"] <= day)]
        etfs = self._etfs(s)
        notes = []
        for row in acts.sort_values("ex_date").to_dict("records"):
            sym, ex, kind = str(row["symbol"]), row["ex_date"], str(row["kind"])
            q = state.lots.quantity(sym)
            if q <= 0:
                continue
            if kind == "SPLIT_OR_BONUS":
                kind = "SPLIT" if sym in etfs else "BONUS"
                notes.append(
                    f"{sym}: share change on {ex} treated as {kind.lower()} "
                    "(classify with 'alphaengine import actions' if wrong)"
                )
            if kind == "DIVIDEND":
                amt = Decimal(str(row["amount"])) * q
                state.cash += amt
                notes.append(f"dividend {sym} ₹{amt:,.2f}")
            elif kind == "SPLIT":
                state.lots.apply_split(sym, float(row["ratio"]))
            else:
                state.lots.apply_bonus(sym, float(row["ratio"]), ex)
        return notes

    # ------------------------------------------------------------------ rebalancing
    def is_rebalance_day(self, name: str, day: date) -> bool:
        with session_scope(self.ctx.session_factory) as s:
            rec = get_portfolio(s, name)
            never_traded = (
                s.scalars(
                    select(PortfolioOrder.id).where(PortfolioOrder.portfolio_id == rec.id).limit(1)
                ).first()
                is None
            )
            strategy_src = rec.strategy
        if never_traded:
            return True
        cal = self.calendar()
        if not cal.is_trading_day(day):
            return False
        days = [*cal.trading_days(day - timedelta(days=100), day), cal.next_trading_day(day)]
        strategy = build_strategy(load_config(strategy_src))
        return pd.Timestamp(day) in strategy.rebalance_dates(pd.DatetimeIndex(days))

    def rebalance(
        self,
        name: str,
        as_of: date,
        *,
        dry_run: bool = False,
        planner: PlannerSettings | None = None,
    ) -> RebalanceResult:
        with session_scope(self.ctx.session_factory) as s:
            rec = get_portfolio(s, name)
            if not rec.active and not dry_run:
                raise PortfolioError(f"{name} is paused (kill switch); resume it first")
            strategy_src, pid = rec.strategy, rec.id
            settings = (
                PlannerSettings(**(rec.settings or {}).get("planner", {}))
                if planner is None
                else planner
            )
            state = load_state(s, rec)
            s.expunge(rec)
        cfg = load_config(strategy_src)
        strategy = build_strategy(cfg)
        syms = required_symbols(strategy, self.ctx, BacktestConfig(start=as_of - timedelta(days=1)))
        md = MarketData.from_store(self.ctx, as_of - timedelta(days=HISTORY_DAYS), as_of, syms)
        day = md.last_date_on_or_before(as_of)
        if day is None:
            raise PortfolioError(f"no data on or before {as_of}")
        held = state.quantities()
        prices = self.closes(sorted(set(held) | set(md.symbols)), day.date())
        total = float(state.cash) + sum(q * prices.get(sym, 0.0) for sym, q in held.items())
        current = {
            sym: q * prices[sym] / total for sym, q in held.items() if total and sym in prices
        }
        target = strategy.target(StrategyContext(day, md, current))
        plan = plan_rebalance(
            state.lots,
            float(state.cash),
            {k: v for k, v in prices.items() if k in set(held) | set(target.weights)},
            dict(target.weights),
            day.date(),
            settings,
            self.costs,
        )
        adv = md.value.iloc[-20:].median().to_dict()
        etfs = {s for s in md.symbols if md.asset_class(s) == "ETF"}
        findings = check_plan(
            plan,
            dict(target.weights),
            {str(k): float(v) for k, v in adv.items()},
            self.limits,
            etfs,
        )
        created = 0
        if not dry_run and not any(f.level == "error" for f in findings):
            with session_scope(self.ctx.session_factory) as s:
                rec = get_portfolio(s, name)
                new = []
                for t in plan.trades:
                    key = f"{pid}:{day.date()}:{t.symbol}:{t.side.value}"
                    if s.scalars(
                        select(PortfolioOrder.id).where(PortfolioOrder.idempotency_key == key)
                    ).first():
                        continue  # already created for this signal date
                    o = PortfolioOrder(
                        portfolio_id=pid,
                        idempotency_key=key,
                        signal_date=day.date(),
                        symbol=t.symbol,
                        side=t.side.value,
                        quantity=t.quantity,
                        est_price=t.est_price,
                        status="NEW",
                        reason=t.reason,
                        note=t.note or None,
                    )
                    s.add(o)
                    new.append(o)
                s.flush()
                broker = (
                    PaperBroker(self.costs) if rec.broker == "paper" else ManualBroker(self.costs)
                )
                broker.submit(new)
                created = len(new)
        return RebalanceResult(name, day.date(), target, plan, findings, created)

    # ------------------------------------------------------------------ manual broker
    def record_fill(
        self, order_id: int, qty: int, price: float, day: date, charges: float | None = None
    ) -> None:
        with session_scope(self.ctx.session_factory) as s:
            order = s.get(PortfolioOrder, order_id)
            if order is None:
                raise PortfolioError(f"no order {order_id}")
            from alphaengine.data.models import Portfolio  # noqa: PLC0415

            rec = s.get(Portfolio, order.portfolio_id)
            assert rec is not None
            state = load_state(s, rec)
            ManualBroker(self.costs).record_fill(
                state,
                order,
                qty,
                Decimal(str(price)),
                day,
                None if charges is None else Decimal(str(charges)),
            )
            save_state(s, state)

    def cancel_order(self, order_id: int) -> None:
        with session_scope(self.ctx.session_factory) as s:
            order = s.get(PortfolioOrder, order_id)
            if order is None:
                raise PortfolioError(f"no order {order_id}")
            cancel(order)

    def record_trade(
        self,
        name: str,
        symbol: str,
        side: Side,
        qty: int,
        price: float,
        day: date,
        charges: float | None = None,
    ) -> None:
        """A trade made outside any order (e.g. moving existing holdings in)."""
        sym = make_symbol(symbol)
        with session_scope(self.ctx.session_factory) as s:
            state = load_state(s, get_portfolio(s, name))
            ch = self.costs.charges(side, qty, Decimal(str(price)), day)
            if charges is not None:
                from alphaengine.market.costs import ChargeBreakdown  # noqa: PLC0415

                ch = ChargeBreakdown(
                    ch.turnover,
                    ch.brokerage,
                    ch.stt,
                    ch.exchange_txn,
                    ch.sebi,
                    ch.stamp,
                    ch.gst,
                    ch.dp + (Decimal(str(charges)) - ch.total),
                )
            apply_fill(state, side, sym, qty, Decimal(str(price)), day, ch)
            save_state(s, state)

    def import_holdings(
        self, name: str, rows: list[dict[str, Any]], *, adjust_cash: bool = False
    ) -> int:
        """Load existing holdings as lots: symbol, quantity, avg_price, acquired (YYYY-MM-DD)."""
        with session_scope(self.ctx.session_factory) as s:
            state = load_state(s, get_portfolio(s, name))
            lots = []
            for r in rows:
                qty = int(r["quantity"])
                cost = Decimal(str(r["avg_price"])) * qty
                lots.append(
                    TaxLot(
                        make_symbol(str(r["symbol"])),
                        qty,
                        cost,
                        date.fromisoformat(str(r["acquired"])),
                    )
                )
                if adjust_cash:
                    state.cash -= cost
            state.lots.restore(lots)
            save_state(s, state)
            return len(lots)

    def reconcile(self, name: str, path: Path) -> list[tuple[str, int, int]]:
        with session_scope(self.ctx.session_factory) as s:
            state = load_state(s, get_portfolio(s, name))
        return reconcile(state.quantities(), read_holdings_csv(path))

    # ------------------------------------------------------------------ reporting
    def tax_report(self, name: str) -> dict[int, FYTax]:
        with session_scope(self.ctx.session_factory) as s:
            disposals = load_disposals(s, get_portfolio(s, name))
        return compute_tax_history(disposals)

    def summary(self, name: str, day: date | None = None) -> dict[str, Any]:
        with session_scope(self.ctx.session_factory) as s:
            rec = get_portfolio(s, name)
            state = load_state(s, rec)
            orders = [
                (o.id, o.side, o.quantity, o.symbol, o.status, float(o.est_price), o.note)
                for o in open_orders(s, rec)
            ]
            last_processed = rec.last_processed
            info: dict[str, Any] = {
                "name": rec.name,
                "strategy": rec.strategy,
                "broker": rec.broker,
                "active": rec.active,
                "initial_capital": float(rec.initial_capital),
                "last_processed": rec.last_processed,
            }
        on: date = day or last_processed or date.today()  # noqa: DTZ011
        held = state.quantities()
        prices = self.closes(list(held), on)
        rows: list[dict[str, Any]] = []
        value = float(state.cash)
        for sym in held:
            cost = float(state.lots.cost_basis(sym))
            mv = held[sym] * prices.get(sym, 0.0)
            value += mv
            rows.append(
                {
                    "symbol": sym,
                    "quantity": held[sym],
                    "price": prices.get(sym),
                    "value": mv,
                    "cost": cost,
                    "pnl": mv - cost,
                    "long_term_qty": state.lots.fifo_long_term_quantity(sym, on),
                }
            )
        for row in rows:
            row["weight"] = float(row["value"]) / value if value else 0.0
        return {
            **info,
            "as_of": on,
            "cash": float(state.cash),
            "value": value,
            "holdings": sorted(rows, key=lambda r: -r["value"]),
            "open_orders": orders,
            "fy": financial_year(on),
        }
