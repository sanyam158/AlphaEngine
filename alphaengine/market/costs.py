"""Indian equity-delivery transaction costs (NSE), with rates versioned by effective date.

Rates (verified Nov 2026 against Zerodha's published charges page and news coverage of the
2024/2026 budgets). Older history is included where well known:

==========================  ===============================  =======================
Charge                      Rate                              Effective from
==========================  ===============================  =======================
STT (delivery)              0.125% buy & sell                 (before 2013-06-01)
                            0.1% buy & sell                   2013-06-01
Exchange txn (NSE)          0.00322% of turnover              (before 2024-10-01)
                            0.00307% of turnover              2024-10-01 *
SEBI turnover fee           ₹10 per crore                     -
Stamp duty (buy only)       0.015%                            2020-07-01 **
GST / service tax           18% on brokerage+txn+SEBI         2017-07-01 (15% before)
DP charge (sell)            ₹13 + GST per scrip per day       Zerodha, Nov 2026
==========================  ===============================  =======================

\\* NSE cut the charge to 0.00297% from 2024-10-01; Zerodha listed 0.00307% in Nov 2026.
  The date of that later change wasn't verified; the difference is ₹1 per ₹10 lakh traded.
\\*\\* Before the July 2020 Stamp Act change, stamp duty varied by state; 0.015% is used as an
  approximation for earlier dates.

Every component is rounded to the paisa (half-up). STT is rounded to the nearest rupee, as
on exchange contract notes (configurable).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date
from decimal import ROUND_HALF_UP, Decimal

from alphaengine.core.types import Side

PAISA = Decimal("0.01")
RUPEE = Decimal("1")
CRORE = Decimal("10000000")


def _r(x: Decimal, q: Decimal = PAISA) -> Decimal:
    return x.quantize(q, rounding=ROUND_HALF_UP)


@dataclass(frozen=True)
class StatutoryRates:
    """Exchange and government charges, as fractions of turnover unless stated."""

    effective_from: date
    stt_delivery: Decimal  # each side
    exchange_txn: Decimal
    sebi_per_crore: Decimal
    stamp_buy: Decimal
    gst: Decimal


STATUTORY_SCHEDULE: tuple[StatutoryRates, ...] = (
    StatutoryRates(
        date(2000, 1, 1),
        Decimal("0.00125"),
        Decimal("0.0000322"),
        Decimal("10"),
        Decimal("0.00015"),
        Decimal("0.15"),
    ),
    StatutoryRates(
        date(2013, 6, 1),
        Decimal("0.001"),
        Decimal("0.0000322"),
        Decimal("10"),
        Decimal("0.00015"),
        Decimal("0.15"),
    ),
    StatutoryRates(
        date(2017, 7, 1),
        Decimal("0.001"),
        Decimal("0.0000322"),
        Decimal("10"),
        Decimal("0.00015"),
        Decimal("0.18"),
    ),
    StatutoryRates(
        date(2024, 10, 1),
        Decimal("0.001"),
        Decimal("0.0000307"),
        Decimal("10"),
        Decimal("0.00015"),
        Decimal("0.18"),
    ),
)


def rates_on(
    day: date, schedule: tuple[StatutoryRates, ...] = STATUTORY_SCHEDULE
) -> StatutoryRates:
    applicable = [r for r in schedule if r.effective_from <= day]
    if not applicable:
        raise ValueError(f"no statutory rates defined for {day}")
    return max(applicable, key=lambda r: r.effective_from)


@dataclass(frozen=True)
class BrokerPlan:
    """Broker pricing for delivery trades. Brokerage = min(pct × value, cap) + flat, per order."""

    name: str
    brokerage_pct: Decimal = Decimal("0")
    brokerage_cap: Decimal | None = None
    brokerage_flat: Decimal = Decimal("0")
    dp_charge_per_sell: Decimal = Decimal("0")  # before GST, per scrip per day

    def brokerage(self, value: Decimal) -> Decimal:
        pct = value * self.brokerage_pct
        if self.brokerage_cap is not None:
            pct = min(pct, self.brokerage_cap)
        return pct + self.brokerage_flat


ZERODHA = BrokerPlan("zerodha", dp_charge_per_sell=Decimal("13.00"))
FLAT_20 = BrokerPlan(  # typical "₹20 or 0.1%, whichever lower" delivery plan
    "flat20",
    brokerage_pct=Decimal("0.001"),
    brokerage_cap=Decimal("20"),
    dp_charge_per_sell=Decimal("13.00"),
)
NO_COSTS = BrokerPlan("none")


@dataclass(frozen=True)
class ChargeBreakdown:
    turnover: Decimal
    brokerage: Decimal
    stt: Decimal
    exchange_txn: Decimal
    sebi: Decimal
    stamp: Decimal
    gst: Decimal
    dp: Decimal

    @property
    def total(self) -> Decimal:
        return (
            self.brokerage
            + self.stt
            + self.exchange_txn
            + self.sebi
            + self.stamp
            + self.gst
            + self.dp
        )

    @property
    def tax_deductible(self) -> Decimal:
        """Charges that count as cost of acquisition / transfer for capital gains (all but STT)."""
        return self.total - self.stt

    def __add__(self, other: ChargeBreakdown) -> ChargeBreakdown:
        return ChargeBreakdown(*(getattr(self, f) + getattr(other, f) for f in _FIELDS))


_FIELDS = ("turnover", "brokerage", "stt", "exchange_txn", "sebi", "stamp", "gst", "dp")
ZERO_CHARGES = ChargeBreakdown(*(Decimal("0") for _ in _FIELDS))


@dataclass(frozen=True)
class CostModel:
    broker: BrokerPlan = ZERODHA
    schedule: tuple[StatutoryRates, ...] = field(default=STATUTORY_SCHEDULE)
    round_stt_to_rupee: bool = True
    statutory: bool = True  # False -> only brokerage/DP (for "costless" research comparisons)

    def charges(
        self, side: Side, quantity: int, price: Decimal, trade_date: date, *, dp: bool = True
    ) -> ChargeBreakdown:
        """Charges for one order (one scrip, one side, one day). ``dp=False`` when another sell
        of the same scrip already paid the DP charge that day."""
        if quantity <= 0 or price <= 0:
            raise ValueError("quantity and price must be positive")
        value = price * quantity
        rates = rates_on(trade_date, self.schedule)
        brokerage = _r(self.broker.brokerage(value))
        if self.statutory:
            stt = _r(value * rates.stt_delivery, RUPEE if self.round_stt_to_rupee else PAISA)
            txn = _r(value * rates.exchange_txn)
            sebi = _r(value * rates.sebi_per_crore / CRORE)
            stamp = _r(value * rates.stamp_buy) if side is Side.BUY else Decimal("0.00")
        else:
            stt = txn = sebi = stamp = Decimal("0.00")
        dp_fee = self.broker.dp_charge_per_sell if (side is Side.SELL and dp) else Decimal("0")
        gst = _r((brokerage + txn + sebi + dp_fee) * rates.gst)
        return ChargeBreakdown(value, brokerage, stt, txn, sebi, stamp, gst, _r(dp_fee))

    def round_trip_cost_pct(self, value: Decimal, trade_date: date) -> Decimal:
        """Approximate buy+sell cost as a fraction of ``value`` (handy for research)."""
        price = Decimal("100")
        qty = max(int(value / price), 1)
        total = (
            self.charges(Side.BUY, qty, price, trade_date).total
            + self.charges(Side.SELL, qty, price, trade_date).total
        )
        return total / (price * qty)

    def without_statutory(self) -> CostModel:
        return replace(self, statutory=False)
