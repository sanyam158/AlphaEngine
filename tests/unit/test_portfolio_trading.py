"""Planner, brokers, risk checks, notifications (pure units, no store)."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from alphaengine.core.types import Side
from alphaengine.data.models import Portfolio, PortfolioOrder
from alphaengine.execution.brokers import (
    DayBar,
    ManualBroker,
    PaperBroker,
    cancel,
    read_holdings_csv,
    reconcile,
)
from alphaengine.market.costs import NO_COSTS, CostModel
from alphaengine.notify.notifier import (
    LogNotifier,
    MultiNotifier,
    NotifyError,
    TelegramNotifier,
    _chunks,
)
from alphaengine.portfolio.lots import LotBook
from alphaengine.portfolio.planner import PlannerSettings, plan_rebalance
from alphaengine.portfolio.store import PortfolioError, PortfolioState
from alphaengine.risk.checks import RiskLimits, check_plan, drawdown_alert

FREE = CostModel(NO_COSTS, statutory=False)
D = date(2026, 3, 2)


def book(*lots: tuple[str, int, float, date]) -> LotBook:
    b = LotBook()
    for sym, q, px, acq in lots:
        b.buy(sym, q, Decimal(str(px)), acq)
    return b


class TestPlanner:
    def test_buys_sells_band_and_min_trade(self) -> None:
        b = book(("A", 100, 100.0, date(2024, 1, 1)), ("B", 50, 100.0, date(2024, 1, 1)))
        prices = {"A": 100.0, "B": 100.0, "C": 50.0}
        # total = 10k cash + 10k A + 5k B = 25k
        plan = plan_rebalance(
            b,
            10_000,
            prices,
            {"A": 0.39, "C": 0.4},
            D,
            PlannerSettings(band=0.02, min_trade_value=500),
            FREE,
        )
        kinds = {(t.symbol, t.side, t.reason) for t in plan.trades}
        assert ("B", Side.SELL, "exit") in kinds
        assert ("C", Side.BUY, "buy") in kinds
        assert not any(t.symbol == "A" for t in plan.trades)  # 40% vs 39%: inside the band
        assert plan.total_value == 25_000
        tiny = plan_rebalance(
            b,
            10_000,
            prices,
            {"A": 0.41, "B": 0.2},
            D,
            PlannerSettings(min_trade_value=1_000),
            FREE,
        )
        assert not any(t.symbol == "A" for t in tiny.trades)  # ₹250 change < min trade
        with pytest.raises(ValueError, match="no price"):
            plan_rebalance(b, 0, {"A": 100.0}, {}, D)

    def test_tax_aware_deferral(self) -> None:
        acquired = date(2025, 3, 20)  # becomes long-term 2026-03-20, 18 days after D
        b = book(("A", 100, 50.0, acquired))
        prices = {"A": 100.0}
        trim = plan_rebalance(b, 0, prices, {"A": 0.5}, D, PlannerSettings(min_trade_value=1), FREE)
        assert trim.trades == []
        assert trim.deferred[0].reason == "trim"
        assert "waiting saves" in trim.deferred[0].note
        exit_ = plan_rebalance(b, 0, prices, {}, D, PlannerSettings(), FREE)
        assert exit_.trades[0].reason == "exit"  # exits follow the strategy by default...
        assert "2026-03-20" in exit_.trades[0].note  # ...but carry the tax note
        held = plan_rebalance(b, 0, prices, {}, D, PlannerSettings(defer_exits=True), FREE)
        assert held.trades == []
        far = plan_rebalance(b, 0, prices, {}, D, PlannerSettings(ltcg_wait_days=5), FREE)
        assert far.trades[0].note == ""
        assert "WAIT" in trim.describe()

    def test_cash_budget_scales_buys(self) -> None:
        plan = plan_rebalance(
            LotBook(),
            10_000,
            {"A": 10.0, "B": 10.0},
            {"A": 0.5, "B": 0.5},
            D,
            PlannerSettings(min_trade_value=1, cash_buffer=0.01),
            CostModel(),
        )
        spend = sum(t.value for t in plan.trades)
        assert spend <= 10_000 * 0.99
        assert plan.est_charges > 0


def order(
    oid: int, side: str, sym: str, qty: int, signal: date = date(2026, 3, 2)
) -> PortfolioOrder:
    return PortfolioOrder(
        id=oid,
        portfolio_id=1,
        idempotency_key=str(oid),
        signal_date=signal,
        symbol=sym,
        side=side,
        quantity=qty,
        est_price=100,
        status="NEW",
        filled_quantity=0,
        charges=0,
        attempts=0,
        reason="t",
    )


def state(cash: float, lots: LotBook | None = None) -> PortfolioState:
    return PortfolioState(Portfolio(name="p"), lots or LotBook(), Decimal(str(cash)))


class TestBrokers:
    def test_paper_fills_next_open_sells_first(self) -> None:
        st = state(1_000, book(("A", 10, 100.0, date(2025, 1, 1))))
        orders = [
            order(2, "BUY", "B", 20),
            order(1, "SELL", "A", 10),
            order(3, "BUY", "C", 1, signal=date(2026, 3, 3)),
        ]
        broker = PaperBroker(FREE, slippage_bps=0)
        broker.submit(orders)
        bars = {
            "A": DayBar(110, 111, 109, 110, 108),
            "B": DayBar(50, 51, 49, 50, 50),
            "C": DayBar(10, 10, 10, 10, 10),
        }.get
        notes = broker.fill(st, orders, date(2026, 3, 3), bars)
        assert orders[1].status == "FILLED"
        assert float(orders[1].fill_price) == 110
        assert orders[0].status == "FILLED"  # sale proceeds paid for the buy
        assert st.lots.quantity("B") == 20
        assert orders[2].status == "SUBMITTED"  # decided today: fills tomorrow
        assert st.cash == Decimal("1000") + 1100 - 1000
        assert len(st.new_disposals) == 1
        assert len(notes) == 2

    def test_paper_partial_cash_and_expiry(self) -> None:
        st = state(500)
        buy = order(1, "BUY", "A", 10)
        locked = order(2, "BUY", "L", 1)
        broker = PaperBroker(FREE, slippage_bps=0, max_attempts=2)
        broker.submit([buy, locked])
        bars = {"A": DayBar(100, 101, 99, 100, 100), "L": DayBar(110, 110, 110, 110, 100)}.get
        broker.fill(st, [buy, locked], date(2026, 3, 3), bars)
        assert buy.status == "PARTIAL"
        assert buy.filled_quantity == 4  # ₹500 x 99.5% / ₹100
        broker.fill(st, [locked], date(2026, 3, 4), bars)
        assert locked.status == "CANCELLED"
        assert "expired" in locked.note
        assert (
            broker.fill(state(0), [order(9, "BUY", "Z", 1)], date(2026, 3, 3), lambda s: None) == []
        )

    def test_manual_fills_and_errors(self) -> None:
        st = state(10_000)
        o = order(1, "BUY", "A", 10)
        mb = ManualBroker(CostModel())
        mb.record_fill(st, o, 4, Decimal("100"), date(2026, 3, 3), charges=Decimal("25"))
        assert o.status == "PARTIAL"
        assert float(o.charges) == pytest.approx(25)
        mb.record_fill(st, o, 6, Decimal("102"), date(2026, 3, 3))
        assert o.status == "FILLED"
        assert float(o.fill_price) == pytest.approx((400 + 612) / 10)
        with pytest.raises(PortfolioError, match="FILLED"):
            mb.record_fill(st, o, 1, Decimal("1"), date(2026, 3, 3))
        o2 = order(2, "SELL", "A", 50)
        with pytest.raises(PortfolioError, match=r"1..50"):
            mb.record_fill(st, o2, 51, Decimal("1"), date(2026, 3, 3))
        with pytest.raises(PortfolioError, match="only 10 held"):
            mb.record_fill(st, o2, 20, Decimal("1"), date(2026, 3, 3))
        cancel(o2)
        assert o2.status == "CANCELLED"
        with pytest.raises(PortfolioError, match="illegal"):
            cancel(o)

    def test_reconcile_formats(self, tmp_path: Path) -> None:
        z = tmp_path / "z.csv"
        z.write_text(
            "Instrument,Qty.,Avg. cost\nRELIANCE,10,1200\nM&M,5,2500\nRELIANCE,2,1300\n,,\n",
            encoding="utf-8",
        )
        assert read_holdings_csv(z) == {"RELIANCE": 12, "M&M": 5}
        s = tmp_path / "s.csv"
        s.write_text("symbol,quantity\ntcs.ns,3\n", encoding="utf-8")
        assert read_holdings_csv(s) == {"TCS": 3}
        bad = tmp_path / "b.csv"
        bad.write_text("a,b\n1,2\n", encoding="utf-8")
        with pytest.raises(PortfolioError, match="columns"):
            read_holdings_csv(bad)
        empty = tmp_path / "e.csv"
        empty.write_text("symbol,quantity\n", encoding="utf-8")
        assert read_holdings_csv(empty) == {}
        assert reconcile({"A": 1, "B": 2}, {"A": 1, "B": 3, "C": 1}) == [("B", 2, 3), ("C", 0, 1)]


def test_risk_checks() -> None:
    plan = plan_rebalance(
        LotBook(),
        100_000,
        {"A": 10.0, "B": 10.0},
        {"A": 0.6, "B": 0.4},
        D,
        PlannerSettings(min_trade_value=1),
        FREE,
    )
    found = check_plan(
        plan, {"A": 0.6, "B": 0.4}, {"A": 100_000.0}, RiskLimits(max_position_weight=0.5)
    )
    levels = {(f.level, f.message.split()[0]) for f in found}
    assert ("error", "A") in levels  # 60% > 50% cap
    assert any(f.level == "warning" and "daily traded value" in f.message for f in found)
    etf_ok = check_plan(plan, {"A": 0.6}, {}, RiskLimits(max_position_weight=0.5), {"A"})
    assert not any(f.level == "error" for f in etf_ok)
    churn = check_plan(plan, {}, {}, RiskLimits(max_turnover=0.5))
    assert any("plan trades" in f.message for f in churn)
    assert drawdown_alert([100, 120, 90]) is not None
    assert drawdown_alert([100, 110]) is None
    assert drawdown_alert([100]) is None


class TestNotifier:
    def test_log_and_chunks(self, tmp_path: Path) -> None:
        n = LogNotifier(tmp_path / "n" / "log.txt")
        n.send("hello")
        assert "hello" in (tmp_path / "n" / "log.txt").read_text(encoding="utf-8")
        parts = _chunks("a" * 10 + "\n" + "b" * 25, 12)
        assert "".join(parts) == "a" * 10 + "\n" + "b" * 25
        assert all(len(p) <= 12 for p in parts)
        assert _chunks("", 5) == [""]

    def test_telegram_and_multi(self, tmp_path: Path) -> None:
        sent: list[dict[str, object]] = []

        def ok(req: httpx.Request) -> httpx.Response:
            import json

            sent.append(json.loads(req.content))
            return httpx.Response(200, json={"ok": True})

        tg = TelegramNotifier("TOKEN", "42", httpx.Client(transport=httpx.MockTransport(ok)))
        tg.send("x" * 5000)
        assert len(sent) == 2
        assert sent[0]["chat_id"] == "42"
        bad = TelegramNotifier(
            "T",
            "1",
            httpx.Client(
                transport=httpx.MockTransport(lambda r: httpx.Response(401, text="unauthorized"))
            ),
        )
        with pytest.raises(NotifyError, match="401"):
            bad.send("hi")
        log = LogNotifier(tmp_path / "l.txt")
        MultiNotifier([bad, log]).send("still delivered")
        assert "still delivered" in (tmp_path / "l.txt").read_text(encoding="utf-8")
        with pytest.raises(NotifyError):
            MultiNotifier([bad]).send("nope")
