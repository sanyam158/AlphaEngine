"""Tax lots with FIFO disposal (mandatory for demat holdings in India, CBDT circular 768).

- Cost of a lot = purchase value + tax-deductible buy charges (everything except STT).
- Sale proceeds are reduced by tax-deductible sell charges, allocated pro rata across lots.
- SPLIT (ratio r): every lot's quantity × r, same total cost and acquisition date.
- BONUS (ratio r = shares after / shares before): each lot spawns a new lot of
  ``qty × (r − 1)`` shares with **zero cost**, acquired on the allotment (≈ ex) date.
- Fractional entitlements are dropped (exchanges pay them out in cash).
"""

from __future__ import annotations

import calendar
import copy
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from itertools import count

from alphaengine.core.errors import OrderError
from alphaengine.market.costs import PAISA, ZERO_CHARGES, ChargeBreakdown

_seq = count(1)


def _paise(x: Decimal) -> Decimal:
    return x.quantize(PAISA, rounding=ROUND_HALF_UP)


def add_months(d: date, months: int) -> date:
    y, m = divmod(d.month - 1 + months, 12)
    year, month = d.year + y, m + 1
    return date(year, month, min(d.day, calendar.monthrange(year, month)[1]))


def is_long_term(acquired: date, sold: date) -> bool:
    """Listed equity is long-term when held for *more than* 12 months."""
    return sold > add_months(acquired, 12)


@dataclass
class TaxLot:
    symbol: str
    quantity: int
    cost: Decimal  # total, including deductible buy charges
    acquired: date
    source: str = "BUY"  # BUY | BONUS
    seq: int = field(default_factory=lambda: next(_seq))

    @property
    def cost_per_share(self) -> Decimal:
        return self.cost / self.quantity if self.quantity else Decimal("0")


@dataclass(frozen=True)
class Disposal:
    symbol: str
    quantity: int
    acquired: date
    sold: date
    cost: Decimal
    gross_proceeds: Decimal
    sell_expenses: Decimal  # deductible (excludes STT)
    stt: Decimal
    lot_source: str = "BUY"

    @property
    def net_proceeds(self) -> Decimal:
        return self.gross_proceeds - self.sell_expenses

    @property
    def gain(self) -> Decimal:
        """Gain before grandfathering (see ``market.tax`` for the taxable amount)."""
        return self.net_proceeds - self.cost

    @property
    def long_term(self) -> bool:
        return is_long_term(self.acquired, self.sold)


class LotBook:
    def __init__(self) -> None:
        self._lots: dict[str, list[TaxLot]] = {}

    # ---------------------------------------------------------------- queries
    def lots(self, symbol: str) -> list[TaxLot]:
        return list(self._lots.get(symbol, []))

    def symbols(self) -> list[str]:
        return sorted(s for s, lots in self._lots.items() if lots)

    def quantity(self, symbol: str) -> int:
        return sum(lot.quantity for lot in self._lots.get(symbol, []))

    def cost_basis(self, symbol: str) -> Decimal:
        return sum((lot.cost for lot in self._lots.get(symbol, [])), Decimal("0"))

    def __iter__(self) -> Iterator[TaxLot]:
        for s in self.symbols():
            yield from self._lots[s]

    def long_term_quantity(self, symbol: str, on: date) -> int:
        """Shares that would be long-term if sold on ``on`` (FIFO order matters: see
        :meth:`fifo_long_term_quantity`)."""
        return sum(
            lot.quantity for lot in self._lots.get(symbol, []) if is_long_term(lot.acquired, on)
        )

    def fifo_long_term_quantity(self, symbol: str, on: date) -> int:
        """How many shares can be sold on ``on`` before FIFO reaches a short-term lot."""
        n = 0
        for lot in self._lots.get(symbol, []):
            if not is_long_term(lot.acquired, on):
                break
            n += lot.quantity
        return n

    def next_long_term_date(self, symbol: str, on: date) -> date | None:
        """Date after which the oldest short-term lot (FIFO order) becomes long-term."""
        for lot in self._lots.get(symbol, []):
            if not is_long_term(lot.acquired, on):
                return add_months(lot.acquired, 12)
        return None

    # ---------------------------------------------------------------- mutations
    def buy(
        self,
        symbol: str,
        quantity: int,
        price: Decimal,
        day: date,
        charges: ChargeBreakdown = ZERO_CHARGES,
    ) -> TaxLot:
        if quantity <= 0 or price <= 0:
            raise OrderError("buy needs positive quantity and price")
        lot = TaxLot(symbol, quantity, price * quantity + charges.tax_deductible, day)
        self._insert(lot)
        return lot

    def sell(
        self,
        symbol: str,
        quantity: int,
        price: Decimal,
        day: date,
        charges: ChargeBreakdown = ZERO_CHARGES,
    ) -> list[Disposal]:
        if quantity <= 0 or price <= 0:
            raise OrderError("sell needs positive quantity and price")
        held = self.quantity(symbol)
        if quantity > held:
            raise OrderError(f"cannot sell {quantity} {symbol}: only {held} held")
        lots = self._lots[symbol]
        remaining = quantity
        disposals: list[Disposal] = []
        expenses_left = charges.tax_deductible
        stt_left = charges.stt
        while remaining:
            lot = lots[0]
            take = min(remaining, lot.quantity)
            last = take == remaining
            share = Decimal(take) / Decimal(quantity)
            # Allocate in whole paise; the last lot takes the remainder so totals are exact.
            exp = expenses_left if last else _paise(charges.tax_deductible * share)
            stt = stt_left if last else _paise(charges.stt * share)
            expenses_left -= exp
            stt_left -= stt
            cost = lot.cost if take == lot.quantity else lot.cost_per_share * take
            disposals.append(
                Disposal(symbol, take, lot.acquired, day, cost, price * take, exp, stt, lot.source)
            )
            lot.cost -= cost
            lot.quantity -= take
            if lot.quantity == 0:
                lots.pop(0)
            remaining -= take
        return disposals

    def preview_sell(
        self,
        symbol: str,
        quantity: int,
        price: Decimal,
        day: date,
        charges: ChargeBreakdown = ZERO_CHARGES,
    ) -> list[Disposal]:
        """Disposals a sale *would* produce, without changing the book."""
        clone = copy.deepcopy(self)
        return clone.sell(symbol, quantity, price, day, charges)

    def apply_split(self, symbol: str, ratio: float) -> None:
        if ratio <= 0:
            raise ValueError("split ratio must be positive")
        r = Decimal(str(ratio))
        for lot in self._lots.get(symbol, []):
            lot.quantity = int(lot.quantity * r)
        self._lots[symbol] = [lot for lot in self._lots.get(symbol, []) if lot.quantity > 0]

    def apply_bonus(self, symbol: str, ratio: float, ex_date: date) -> list[TaxLot]:
        if ratio <= 1:
            raise ValueError("bonus ratio (shares after / before) must be > 1")
        extra = Decimal(str(ratio)) - 1
        new = [
            TaxLot(symbol, int(lot.quantity * extra), Decimal("0"), ex_date, source="BONUS")
            for lot in self._lots.get(symbol, [])
        ]
        new = [lot for lot in new if lot.quantity > 0]
        for lot in new:
            self._insert(lot)
        return new

    def _insert(self, lot: TaxLot) -> None:
        lots = self._lots.setdefault(lot.symbol, [])
        lots.append(lot)
        lots.sort(key=lambda x: (x.acquired, x.seq))
