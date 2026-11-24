"""Capital-gains tax on listed equity (STT paid), resident individual, by financial year.

Rules implemented (verified Nov 2026; Budget 2026 left equity rates unchanged):

=============================  ===============  ===============  =========================
Sale (transfer) date           STCG (≤12 mo)     LTCG (>12 mo)    LTCG exemption per FY
=============================  ===============  ===============  =========================
before 2018-04-01              15%               exempt           -
2018-04-01 .. 2024-07-22       15%               10%              ₹1,00,000
from 2024-07-23                20%               12.5%            ₹1,25,000 (from FY 2024-25)
=============================  ===============  ===============  =========================

- **Grandfathering** (sec 112A): for shares acquired on or before 2018-01-31, the cost for
  LTCG is ``max(actual cost, min(FMV on 2018-01-31, sale value))``. FMV is the day's highest
  traded price; supply it via ``fmv_lookup`` (``None`` means no grandfathering).
- **Set-off**: short-term losses offset STCG, then LTCG; long-term losses offset only LTCG.
  Losses are applied to the highest-rate gains first. Unused losses carry forward 8 FYs.
- **Cess** 4% (3% before FY 2018-19). Surcharge, the unused-basic-exemption adjustment for low
  incomes, and STT/expense rules for F&O are **not** modelled.

This is a planning estimate, not tax advice: confirm figures with your broker's capital-gains
statement or a tax professional.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date
from decimal import ROUND_HALF_UP, Decimal

from alphaengine.portfolio.lots import Disposal

ZERO = Decimal("0")
GRANDFATHER_CUTOFF = date(2018, 1, 31)
CARRY_FORWARD_YEARS = 8

FmvLookup = Callable[[str], Decimal | None]


@dataclass(frozen=True)
class RateRegime:
    effective_from: date
    stcg: Decimal
    ltcg: Decimal | None  # None = exempt


REGIMES: tuple[RateRegime, ...] = (
    RateRegime(date(2004, 10, 1), Decimal("0.10"), None),
    RateRegime(date(2008, 4, 1), Decimal("0.15"), None),
    RateRegime(date(2018, 4, 1), Decimal("0.15"), Decimal("0.10")),
    RateRegime(date(2024, 7, 23), Decimal("0.20"), Decimal("0.125")),
)


def regime_on(d: date) -> RateRegime:
    applicable = [r for r in REGIMES if r.effective_from <= d]
    if not applicable:
        raise ValueError(f"no tax regime defined for {d}")
    return applicable[-1]


def financial_year(d: date) -> int:
    """Indian FY start year: 2024-04-01 .. 2025-03-31 -> 2024 (i.e. FY 2024-25)."""
    return d.year if d.month >= 4 else d.year - 1


def ltcg_exemption(fy: int) -> Decimal:
    if fy >= 2024:
        return Decimal("125000")
    if fy >= 2018:
        return Decimal("100000")
    return ZERO


def cess_rate(fy: int) -> Decimal:
    return Decimal("0.04") if fy >= 2018 else Decimal("0.03")


def taxable_gain(d: Disposal, fmv_lookup: FmvLookup | None = None) -> Decimal:
    """Gain used for tax: applies grandfathering to qualifying long-term disposals."""
    if not d.long_term or d.acquired > GRANDFATHER_CUTOFF or d.sold < date(2018, 4, 1):
        return d.gain
    fmv = fmv_lookup(d.symbol) if fmv_lookup else None
    if fmv is None:
        return d.gain
    deemed_cost = max(d.cost, min(fmv * d.quantity, d.gross_proceeds))
    return d.net_proceeds - deemed_cost


@dataclass(frozen=True)
class CarriedLoss:
    kind: str  # "ST" | "LT"
    amount: Decimal  # positive number
    origin_fy: int


@dataclass
class FYTax:
    fy: int
    stcg: dict[Decimal, Decimal] = field(default_factory=dict)  # rate -> taxable gain
    ltcg: dict[Decimal, Decimal] = field(default_factory=dict)
    exempt_ltcg: Decimal = ZERO  # pre-2018 exempt long-term gains (informational)
    exemption_used: Decimal = ZERO
    losses_used: Decimal = ZERO
    carried_forward: list[CarriedLoss] = field(default_factory=list)
    tax: Decimal = ZERO
    cess: Decimal = ZERO

    @property
    def total_tax(self) -> Decimal:
        return self.tax + self.cess

    @property
    def taxable_stcg(self) -> Decimal:
        return sum(self.stcg.values(), ZERO)

    @property
    def taxable_ltcg(self) -> Decimal:
        return sum(self.ltcg.values(), ZERO)


def _absorb(gains: dict[Decimal, Decimal], loss: Decimal) -> tuple[Decimal, Decimal]:
    """Offset ``loss`` against ``gains`` highest rate first. Returns (used, leftover)."""
    used = ZERO
    for rate in sorted(gains, reverse=True):
        if loss <= 0:
            break
        take = min(gains[rate], loss)
        gains[rate] -= take
        loss -= take
        used += take
    return used, loss


def _bucket(
    disposals: Iterable[Disposal], fy: int, fmv_lookup: FmvLookup | None
) -> tuple[dict[Decimal, Decimal], dict[Decimal, Decimal], Decimal, Decimal, Decimal]:
    """Split the FY's disposals into (ST gains by rate, LT gains by rate, ST loss, LT loss,
    exempt pre-2018 LTCG)."""
    st_gains: dict[Decimal, Decimal] = {}
    lt_gains: dict[Decimal, Decimal] = {}
    st_loss = lt_loss = exempt = ZERO
    for d in disposals:
        if financial_year(d.sold) != fy:
            continue
        g = taxable_gain(d, fmv_lookup)
        reg = regime_on(d.sold)
        lt_rate = reg.ltcg
        if d.long_term:
            if lt_rate is None:
                exempt += g  # exempt gains and losses don't enter set-off
            elif g >= 0:
                lt_gains[lt_rate] = lt_gains.get(lt_rate, ZERO) + g
            else:
                lt_loss -= g
        elif g >= 0:
            st_gains[reg.stcg] = st_gains.get(reg.stcg, ZERO) + g
        else:
            st_loss -= g
    return st_gains, lt_gains, st_loss, lt_loss, exempt


def compute_fy_tax(
    disposals: Iterable[Disposal],
    fy: int,
    carried_in: Iterable[CarriedLoss] = (),
    fmv_lookup: FmvLookup | None = None,
) -> FYTax:
    """Tax for disposals whose sale date falls in FY ``fy``."""
    res = FYTax(fy)
    st_gains, lt_gains, st_loss, lt_loss, res.exempt_ltcg = _bucket(disposals, fy, fmv_lookup)

    # Current-year losses first, then brought-forward (oldest first, expired dropped).
    carried = sorted(
        (c for c in carried_in if fy - c.origin_fy <= CARRY_FORWARD_YEARS and c.amount > 0),
        key=lambda c: c.origin_fy,
    )
    used, st_left = _absorb(st_gains, st_loss)
    res.losses_used += used
    used, st_left = _absorb(lt_gains, st_left)
    res.losses_used += used
    used, lt_left = _absorb(lt_gains, lt_loss)
    res.losses_used += used

    for c in carried:
        pools = [st_gains, lt_gains] if c.kind == "ST" else [lt_gains]
        left = c.amount
        for pool in pools:
            used, left = _absorb(pool, left)
            res.losses_used += used
        if left > 0:
            res.carried_forward.append(CarriedLoss(c.kind, left, c.origin_fy))
    if st_left > 0:
        res.carried_forward.append(CarriedLoss("ST", st_left, fy))
    if lt_left > 0:
        res.carried_forward.append(CarriedLoss("LT", lt_left, fy))

    # LTCG annual exemption, applied to the highest-rate gains first.
    res.exemption_used, _ = _absorb(lt_gains, ltcg_exemption(fy))

    res.stcg = {r: g for r, g in st_gains.items() if g > 0}
    res.ltcg = {r: g for r, g in lt_gains.items() if g > 0}
    tax = sum((r * g for r, g in (*res.stcg.items(), *res.ltcg.items())), ZERO)
    res.tax = tax.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    res.cess = (res.tax * cess_rate(fy)).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    return res


def compute_tax_history(
    disposals: Iterable[Disposal], fmv_lookup: FmvLookup | None = None
) -> dict[int, FYTax]:
    """Tax for every FY with disposals, carrying losses forward year to year."""
    items = list(disposals)
    if not items:
        return {}
    years = range(
        min(financial_year(d.sold) for d in items), max(financial_year(d.sold) for d in items) + 1
    )
    out: dict[int, FYTax] = {}
    carried: list[CarriedLoss] = []
    for fy in years:
        res = compute_fy_tax(items, fy, carried, fmv_lookup)
        out[fy] = res
        carried = res.carried_forward
    return out
