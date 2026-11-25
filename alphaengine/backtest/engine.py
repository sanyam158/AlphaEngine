"""Daily portfolio simulation for rebalance-style strategies.

Timeline for each trading day ``d``:

1. corporate actions with ex-date ``d`` change held share counts / pay dividends
2. a monthly SIP contribution (first trading day of each month) is added to cash
3. orders decided at the previous close are filled at today's **open** (sells first), with
   slippage, a participation cap and circuit-lock checks; blocked orders retry for a few days
4. idle cash earns interest
5. the portfolio is marked to the close
6. at a financial-year end, capital-gains tax on that FY's sales is paid from cash
7. on a rebalance date (always including the first day) the strategy sees data up to today's
   close and new orders are queued;
   new SIP cash is put to work by the last target's weights

Prices are raw (as traded) and share counts change on corporate actions, so position sizes,
costs and tax lots are those you'd actually see in a demat account.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd

from alphaengine.backtest.config import BacktestConfig
from alphaengine.core.errors import AlphaEngineError
from alphaengine.core.logging import get_logger
from alphaengine.core.types import Side
from alphaengine.features.data import MarketData
from alphaengine.market.costs import ChargeBreakdown
from alphaengine.market.execution_model import apply_slippage, circuit_locked
from alphaengine.market.tax import CarriedLoss, FYTax, compute_fy_tax, financial_year
from alphaengine.portfolio.lots import Disposal, LotBook
from alphaengine.strategy.core import Strategy, StrategyContext, TargetPortfolio

log = get_logger(__name__)
TRADING_DAYS = 252
_NO_CAP = 10**12


class BacktestError(AlphaEngineError):
    pass


@dataclass
class _Order:
    symbol: str
    side: Side
    quantity: int = 0  # sells
    value: float = 0.0  # buys (rupees to invest)
    reason: str = "rebalance"
    attempts: int = 0


@dataclass
class BacktestResult:
    strategy: str
    config: BacktestConfig
    equity: pd.DataFrame  # indexed by date
    trades: pd.DataFrame
    disposals: list[Disposal]
    targets: list[TargetPortfolio]
    taxes: dict[int, FYTax]
    liquidation: dict[str, float]
    notes: list[str] = field(default_factory=list)
    strategy_config: dict[str, Any] | None = None

    @property
    def final_value(self) -> float:
        return float(self.equity["value"].iloc[-1])


class _Sim:
    def __init__(self, strategy: Strategy, md: MarketData, cfg: BacktestConfig) -> None:
        if md.raw_open is None or md.raw_high is None or md.raw_low is None:
            raise BacktestError("MarketData must be loaded with execution=True")
        self.strategy, self.md, self.cfg = strategy, md, cfg
        self.costs = cfg.cost_model()
        self.slip = cfg.slippage_model()
        start, end = pd.Timestamp(cfg.start), pd.Timestamp(cfg.end) if cfg.end else md.dates[-1]
        self.days = md.dates[(md.dates >= start) & (md.dates <= end)]
        if len(self.days) < 2:
            raise BacktestError("backtest needs at least two trading days of data")
        self.col = {s: i for i, s in enumerate(md.close.columns)}
        self.o = md.raw_open.to_numpy(dtype=float)
        self.h = md.raw_high.to_numpy(dtype=float)
        self.lo = md.raw_low.to_numpy(dtype=float)
        self.c = md.raw_close.to_numpy(dtype=float)
        self.row = {d: i for i, d in enumerate(md.dates)}
        adv = md.value.where(md.value.notna(), 0.0).rolling(20, min_periods=5).median()
        self.adv = adv.shift(1).to_numpy(dtype=float)  # known before today's open
        rets = md.tr_close.pct_change(fill_method=None)
        self.vol = rets.rolling(63, min_periods=20).std().shift(1).to_numpy(dtype=float)
        self.actions = self._index_actions(md.actions)

        self.cash = cfg.initial_capital
        self.shares: dict[str, int] = defaultdict(int)
        self.last_px: dict[str, float] = {}
        self.lots = LotBook()
        self.pending: list[_Order] = []
        self.trades: list[dict[str, Any]] = []
        self.disposals: list[Disposal] = []
        self.targets: list[TargetPortfolio] = []
        self.taxes: dict[int, FYTax] = {}
        self.carried: list[CarriedLoss] = []
        self.notes: list[str] = []
        self.totals: defaultdict[str, float] = defaultdict(float)
        # Invest on day one, then follow the strategy's schedule.
        self.rebalance_days = {self.days[0], *strategy.rebalance_dates(self.days)}
        self.last_target: dict[str, float] = {}
        self.fmv = self._fmv_lookup()

    # ------------------------------------------------------------------ setup helpers
    def _index_actions(
        self, acts: pd.DataFrame
    ) -> dict[pd.Timestamp, list[tuple[str, str, float]]]:
        out: dict[pd.Timestamp, list[tuple[str, str, float]]] = defaultdict(list)
        cols = ["symbol", "ex_date", "kind", "ratio", "amount"]
        for sym, ex, raw_kind, ratio, amount in acts[cols].itertuples(index=False):
            if sym not in self.col:
                continue
            kind = str(raw_kind)
            if kind == "SPLIT_OR_BONUS":
                # ETFs never issue bonuses: their share-count changes are unit splits.
                as_bonus = (
                    self.cfg.ambiguous_share_actions == "bonus"
                    and self.md.asset_class(str(sym)) != "ETF"
                )
                kind = "BONUS" if as_bonus else "SPLIT"
            val = float(amount) if kind == "DIVIDEND" else float(ratio)
            out[pd.Timestamp(ex)].append((str(sym), kind, val))
        return out

    def _fmv_lookup(self) -> Any:
        """Grandfathering FMV (31-Jan-2018 high), divided by later splits held as lots."""
        ts = pd.Timestamp("2018-01-31")
        if ts not in self.row:
            return None
        i = self.row[ts]
        fmv: dict[str, Decimal] = {}
        for sym, j in self.col.items():
            v = self.h[i, j]
            if np.isfinite(v):
                fmv[sym] = Decimal(str(round(v, 2)))
        for ex, events in self.actions.items():
            if ex > ts:
                for sym, kind, ratio in events:
                    if kind == "SPLIT" and sym in fmv:
                        fmv[sym] = fmv[sym] / Decimal(str(ratio))
        return fmv.get

    # ------------------------------------------------------------------ valuation
    def _value(self) -> float:
        return self.cash + sum(q * self.last_px.get(s, 0.0) for s, q in self.shares.items() if q)

    def _weights(self, total: float) -> dict[str, float]:
        if total <= 0:
            return {}
        return {s: q * self.last_px[s] / total for s, q in self.shares.items() if q > 0}

    # ------------------------------------------------------------------ daily steps
    def _corporate_actions(self, d: pd.Timestamp) -> None:
        for sym, kind, val in self.actions.get(d, []):
            held = self.shares.get(sym, 0)
            if held <= 0:
                continue
            if kind == "DIVIDEND":
                gross = held * val
                tax = gross * self.cfg.income_tax_rate if self.cfg.taxes else 0.0
                self.cash += gross - tax
                self.totals["dividends"] += gross
                self.totals["income_tax"] += tax
            elif kind in ("SPLIT", "BONUS"):
                if kind == "SPLIT":
                    self.lots.apply_split(sym, val)
                else:
                    self.lots.apply_bonus(sym, val, d.date())
                # Lots round entitlements per lot (as the depository does); they are the truth.
                self.shares[sym] = self.lots.quantity(sym)

    def _market(self, o: _Order, i: int) -> tuple[float, int] | None:
        """(open price, max fillable shares) for ``o`` today, or ``None`` if it can't trade:
        no print, or locked at a circuit against us."""
        j = self.col[o.symbol]
        px = self.o[i, j]
        if not np.isfinite(px) or px <= 0:
            return None
        prev = self.c[i - 1, j] if i > 0 else np.nan
        if np.isfinite(prev) and circuit_locked(
            o.side,
            open_=px,
            high=self.h[i, j],
            low=self.lo[i, j],
            close=self.c[i, j],
            prev_close=prev,
        ):
            return None
        adv = self.adv[i, j]
        if not np.isfinite(adv):
            return float(px), _NO_CAP  # no volume history yet: don't cap
        return float(px), int(adv * self.cfg.max_participation / px)

    def _price(self, o: _Order, i: int, px: float, qty: int) -> Decimal:
        j = self.col[o.symbol]
        adv, vol = self.adv[i, j], self.vol[i, j]
        bps = self.slip.slippage_bps(
            qty * px,
            float(adv) if np.isfinite(adv) else None,
            float(vol) if np.isfinite(vol) else None,
        )
        return apply_slippage(Decimal(str(round(px, 2))), o.side, bps)

    def _fill(self, d: pd.Timestamp, i: int) -> None:
        if not self.pending:
            return
        retry: list[_Order] = []
        self._fill_sells(d, i, [o for o in self.pending if o.side is Side.SELL], retry)
        self._fill_buys(d, i, [o for o in self.pending if o.side is Side.BUY], retry)
        self.pending = retry

    def _fill_sells(
        self, d: pd.Timestamp, i: int, sells: list[_Order], retry: list[_Order]
    ) -> None:
        sold_today: set[str] = set()
        for o in sells:
            mkt = self._market(o, i)
            qty = min(o.quantity, self.shares.get(o.symbol, 0), mkt[1]) if mkt else 0
            if mkt is None or qty <= 0:
                self._retry(o, retry, d, "no fill (no trade, circuit or no volume)")
                continue
            price = self._price(o, i, mkt[0], qty)
            ch = self.costs.charges(Side.SELL, qty, price, d.date(), dp=o.symbol not in sold_today)
            sold_today.add(o.symbol)
            self.disposals.extend(self.lots.sell(o.symbol, qty, price, d.date(), ch))
            self.shares[o.symbol] -= qty
            self.cash += float(price) * qty - float(ch.total)
            self._record(d, o, qty, price, ch)
            if qty < o.quantity and self.shares[o.symbol] > 0:
                o.quantity -= qty
                self._retry(o, retry, d, "partial fill (participation cap)")

    def _fill_buys(self, d: pd.Timestamp, i: int, buys: list[_Order], retry: list[_Order]) -> None:
        budget = max(0.0, self.cash * (1 - self.cfg.cash_buffer))
        wanted = sum(o.value for o in buys)
        scale = min(1.0, budget / wanted) if wanted > 0 else 0.0
        for o in buys:
            mkt = self._market(o, i)
            if mkt is None or mkt[1] == 0:
                self._retry(o, retry, d, "no fill (no trade, circuit or no volume)")
                continue
            px, cap = mkt
            price = self._price(o, i, px, int(o.value * scale / px))
            qty = min(int(o.value * scale / float(price)), cap)
            if qty <= 0:
                continue
            ch = self.costs.charges(Side.BUY, qty, price, d.date())
            if float(price) * qty + float(ch.total) > self.cash:
                qty = int(self.cash * (1 - self.cfg.cash_buffer) / float(price))
                if qty <= 0:
                    continue
                ch = self.costs.charges(Side.BUY, qty, price, d.date())
            self.cash -= float(price) * qty + float(ch.total)
            self.shares[o.symbol] += qty
            self.lots.buy(o.symbol, qty, price, d.date(), ch)
            self._record(d, o, qty, price, ch)
            remaining = o.value * scale - float(price) * qty
            if qty == cap and remaining > float(price):
                o.value = remaining
                self._retry(o, retry, d, "partial fill (participation cap)")

    def _retry(self, o: _Order, retry: list[_Order], d: pd.Timestamp, why: str) -> None:
        o.attempts += 1
        if o.attempts < self.cfg.max_fill_attempts:
            retry.append(o)
        else:
            self.notes.append(f"{d.date()}: dropped {o.side.value} {o.symbol} ({why})")

    def _record(
        self, d: pd.Timestamp, o: _Order, qty: int, price: Decimal, ch: ChargeBreakdown
    ) -> None:
        total, stt = ch.total, ch.stt
        self.totals["costs"] += float(total)
        self.totals["turnover"] += qty * float(price)
        self.trades.append(
            {
                "date": d,
                "symbol": o.symbol,
                "side": o.side.value,
                "quantity": qty,
                "price": float(price),
                "value": qty * float(price),
                "charges": float(total),
                "stt": float(stt),
                "reason": o.reason,
            }
        )

    def _pay_tax(self, fy: int) -> None:
        res = compute_fy_tax(self.disposals, fy, self.carried, self.fmv)
        self.taxes[fy] = res
        self.carried = res.carried_forward
        if self.cfg.taxes and res.total_tax > 0:
            self.cash -= float(res.total_tax)
            self.totals["capital_gains_tax"] += float(res.total_tax)

    def _raise_cash(self, total: float) -> None:
        """Queue pro-rata sells when tax payments pushed cash below zero."""
        short = -self.cash * 1.02
        for sym, q in self.shares.items():
            if q <= 0:
                continue
            val = q * self.last_px[sym]
            qty = min(q, math.ceil(short * val / max(total - self.cash, 1e-9) / self.last_px[sym]))
            if qty > 0:
                self.pending.append(
                    _Order(sym, Side.SELL, quantity=qty, reason="raise cash for tax")
                )

    def _rebalance(self, d: pd.Timestamp) -> None:
        total = self._value()
        current = self._weights(total)
        target = self.strategy.target(StrategyContext(d, self.md, current))
        self.targets.append(target)
        self.last_target = dict(target.weights)
        band = self.cfg.rebalance_band
        for sym in sorted(set(current) | set(target.weights)):
            tw, cw = target.weights.get(sym, 0.0), current.get(sym, 0.0)
            if sym not in self.col:
                continue
            px = self.last_px.get(sym) or float(self.c[self.row[d], self.col[sym]])
            if not np.isfinite(px) or px <= 0:
                continue
            if tw > 0 and cw > 0 and abs(tw - cw) < band:
                continue
            diff = tw * total - cw * total
            if tw == 0 and self.shares.get(sym, 0) > 0:
                self.pending.append(
                    _Order(sym, Side.SELL, quantity=self.shares[sym], reason="exit")
                )
            elif abs(diff) < self.cfg.min_trade_value:
                continue
            elif diff < 0:
                qty = min(self.shares.get(sym, 0), math.ceil(-diff / px))
                if qty > 0:
                    self.pending.append(_Order(sym, Side.SELL, quantity=qty, reason="trim"))
            else:
                self.pending.append(_Order(sym, Side.BUY, value=diff, reason="rebalance"))

    def _deploy_contribution(self, amount: float) -> None:
        invested = sum(self.last_target.values())
        if invested <= 0:
            return
        for sym, w in self.last_target.items():
            if amount * w >= self.cfg.min_trade_value:
                self.pending.append(_Order(sym, Side.BUY, value=amount * w, reason="sip"))

    # ------------------------------------------------------------------ main loop
    def run(self) -> BacktestResult:
        rows = []
        days = list(self.days)
        last_month = None
        for k, d in enumerate(days):
            i = self.row[d]
            flow = 0.0
            self._corporate_actions(d)
            if self.cfg.sip_amount and k > 0 and (d.year, d.month) != last_month:
                self.cash += self.cfg.sip_amount
                flow = self.cfg.sip_amount
            last_month = (d.year, d.month)
            self._fill(d, i)
            if self.cash > 0 and k > 0:
                interest = self.cash * self.cfg.cash_rate / TRADING_DAYS
                tax = interest * self.cfg.income_tax_rate if self.cfg.taxes else 0.0
                self.cash += interest - tax
                self.totals["interest"] += interest
                self.totals["income_tax"] += tax
            for sym, j in self.col.items():
                px = self.c[i, j]
                if np.isfinite(px) and (self.shares.get(sym) or sym in self.last_target):
                    self.last_px[sym] = float(px)
            nxt = days[k + 1] if k + 1 < len(days) else None
            fy = financial_year(d.date())
            if nxt is None or financial_year(nxt.date()) != fy:
                self._pay_tax(fy)
            total = self._value()
            if self.cash < 0 and nxt is not None:
                self._raise_cash(total)
            if nxt is not None and d in self.rebalance_days:
                self._rebalance(d)
            elif nxt is not None and flow:
                self._deploy_contribution(flow)
            self.totals["sip"] += flow
            rows.append(
                {
                    "date": d,
                    "value": total,
                    "cash": self.cash,
                    "flow": flow,
                    "contributions": self.cfg.initial_capital + self.totals["sip"],
                    "costs": self.totals["costs"],
                    "taxes": self.totals["capital_gains_tax"] + self.totals["income_tax"],
                }
            )
        equity = pd.DataFrame(rows).set_index("date")
        liquidation = self._liquidation(days[-1])
        liquidation.update({f"total_{k}": v for k, v in self.totals.items()})
        return BacktestResult(
            strategy=getattr(self.strategy, "name", "strategy"),
            config=self.cfg,
            equity=equity,
            trades=pd.DataFrame(
                self.trades,
                columns=[
                    "date",
                    "symbol",
                    "side",
                    "quantity",
                    "price",
                    "value",
                    "charges",
                    "stt",
                    "reason",
                ],
            ),
            disposals=self.disposals,
            targets=self.targets,
            taxes=self.taxes,
            liquidation=liquidation,
            notes=self.notes,
            strategy_config=dict(getattr(self.strategy, "config", None) or {}) or None,
        )

    def _liquidation(self, d: pd.Timestamp) -> dict[str, float]:
        """What you'd keep if everything were sold at the final close (charges + tax)."""
        hypothetical: list[Disposal] = []
        charges = 0.0
        for sym, q in self.shares.items():
            if q <= 0:
                continue
            price = Decimal(str(round(self.last_px[sym], 2)))
            ch = self.costs.charges(Side.SELL, q, price, d.date())
            charges += float(ch.total)
            hypothetical.extend(self.lots.preview_sell(sym, q, price, d.date(), ch))
        fy = financial_year(d.date())
        base = self.taxes.get(fy)
        realized_fy = [x for x in self.disposals if financial_year(x.sold) == fy]
        carried_in = self._carried_before(fy)
        with_sale = compute_fy_tax(realized_fy + hypothetical, fy, carried_in, self.fmv)
        extra_tax = float(with_sale.total_tax) - float(base.total_tax if base else 0)
        if not self.cfg.taxes:
            extra_tax = 0.0
        value = self._value()
        return {
            "value": value,
            "sell_charges": charges,
            "tax_on_liquidation": max(extra_tax, 0.0),
            "after_tax_value": value - charges - max(extra_tax, 0.0),
        }

    def _carried_before(self, fy: int) -> list[CarriedLoss]:
        prev = self.taxes.get(fy - 1)
        return prev.carried_forward if prev else []


def simulate(strategy: Strategy, md: MarketData, cfg: BacktestConfig) -> BacktestResult:
    """Run ``strategy`` over ``md`` (which must include warm-up history before ``cfg.start``)."""
    return _Sim(strategy, md, cfg).run()
