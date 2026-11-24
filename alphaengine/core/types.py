"""Core domain types shared by every layer.

Conventions:
- Symbols are NSE trading symbols, upper-case, *without* any ``.NS``/``.BO`` suffix
  (``RELIANCE``, ``M&M``, ``BAJAJ-AUTO``). Vendor suffixes are added/removed in data adapters.
- Market data (``Bar``) uses ``float`` for interop with numpy/pandas.
- Accounting values (orders, fills, positions, cash) use ``Decimal`` so P&L and costs
  reconcile to the paisa.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import NewType

from alphaengine.core.errors import OrderError

Symbol = NewType("Symbol", str)

_SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9&\-_]{0,19}$")
_VENDOR_SUFFIXES = (".NS", ".BO")


def make_symbol(raw: str) -> Symbol:
    """Normalize and validate a trading symbol. Strips vendor suffixes like ``.NS``."""
    s = raw.strip().upper()
    for suffix in _VENDOR_SUFFIXES:
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    if not _SYMBOL_RE.fullmatch(s):
        raise ValueError(f"invalid symbol: {raw!r}")
    return Symbol(s)


class Exchange(StrEnum):
    NSE = "NSE"
    BSE = "BSE"


class AssetClass(StrEnum):
    EQUITY = "EQUITY"
    ETF = "ETF"
    INDEX = "INDEX"


class Side(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(StrEnum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(StrEnum):
    NEW = "NEW"
    SUBMITTED = "SUBMITTED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


TERMINAL_STATUSES = frozenset({OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED})

_ALLOWED_TRANSITIONS: dict[OrderStatus, frozenset[OrderStatus]] = {
    OrderStatus.NEW: frozenset(
        {OrderStatus.SUBMITTED, OrderStatus.CANCELLED, OrderStatus.REJECTED}
    ),
    OrderStatus.SUBMITTED: frozenset(
        {
            OrderStatus.PARTIAL,
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
        }
    ),
    OrderStatus.PARTIAL: frozenset(
        {OrderStatus.PARTIAL, OrderStatus.FILLED, OrderStatus.CANCELLED}
    ),
    OrderStatus.FILLED: frozenset(),
    OrderStatus.CANCELLED: frozenset(),
    OrderStatus.REJECTED: frozenset(),
}


def can_transition(src: OrderStatus, dst: OrderStatus) -> bool:
    return dst in _ALLOWED_TRANSITIONS[src]


@dataclass(frozen=True, slots=True)
class Instrument:
    symbol: Symbol
    exchange: Exchange = Exchange.NSE
    asset_class: AssetClass = AssetClass.EQUITY
    isin: str | None = None
    name: str | None = None


@dataclass(frozen=True, slots=True)
class Bar:
    """One daily OHLCV bar. Prices are adjusted or raw depending on the producing pipeline."""

    symbol: Symbol
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int

    def __post_init__(self) -> None:
        if min(self.open, self.high, self.low, self.close) <= 0:
            raise ValueError(f"{self.symbol} {self.date}: prices must be positive")
        if self.high < max(self.open, self.close, self.low):
            raise ValueError(f"{self.symbol} {self.date}: high below open/close/low")
        if self.low > min(self.open, self.close, self.high):
            raise ValueError(f"{self.symbol} {self.date}: low above open/close/high")
        if self.volume < 0:
            raise ValueError(f"{self.symbol} {self.date}: negative volume")


@dataclass(frozen=True, slots=True)
class Order:
    """An immutable order snapshot. State changes return a new ``Order`` via ``with_status``."""

    id: str
    symbol: Symbol
    side: Side
    quantity: int
    order_type: OrderType = OrderType.MARKET
    limit_price: Decimal | None = None
    status: OrderStatus = OrderStatus.NEW
    filled_quantity: int = 0
    created_at: datetime | None = None
    tag: str | None = None  # e.g. strategy name / rebalance id

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise OrderError("order quantity must be positive")
        if self.order_type is OrderType.LIMIT and (
            self.limit_price is None or self.limit_price <= 0
        ):
            raise OrderError("limit orders need a positive limit_price")
        if self.order_type is OrderType.MARKET and self.limit_price is not None:
            raise OrderError("market orders must not have a limit_price")
        if not 0 <= self.filled_quantity <= self.quantity:
            raise OrderError("filled_quantity out of range")

    @property
    def remaining_quantity(self) -> int:
        return self.quantity - self.filled_quantity

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    def with_status(self, status: OrderStatus, filled_quantity: int | None = None) -> Order:
        if not can_transition(self.status, status):
            raise OrderError(f"illegal order transition {self.status} -> {status}")
        filled = self.filled_quantity if filled_quantity is None else filled_quantity
        if filled < self.filled_quantity:
            raise OrderError("filled quantity cannot decrease")
        if status is OrderStatus.FILLED and filled != self.quantity:
            raise OrderError("FILLED requires filled_quantity == quantity")
        if status is OrderStatus.PARTIAL and not 0 < filled < self.quantity:
            raise OrderError("PARTIAL requires 0 < filled_quantity < quantity")
        return replace(self, status=status, filled_quantity=filled)


@dataclass(frozen=True, slots=True)
class Fill:
    order_id: str
    symbol: Symbol
    side: Side
    quantity: int
    price: Decimal
    timestamp: datetime
    charges: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise OrderError("fill quantity must be positive")
        if self.price <= 0:
            raise OrderError("fill price must be positive")
        if self.charges < 0:
            raise OrderError("charges cannot be negative")

    @property
    def notional(self) -> Decimal:
        return self.price * self.quantity

    @property
    def cash_flow(self) -> Decimal:
        """Signed cash impact: negative for buys, positive for sells, net of charges."""
        sign = Decimal(-1) if self.side is Side.BUY else Decimal(1)
        return sign * self.notional - self.charges


@dataclass(slots=True)
class Position:
    """A long-only position with average-cost basis (tax lots live in the portfolio layer)."""

    symbol: Symbol
    quantity: int = 0
    avg_cost: Decimal = Decimal("0")
    realized_pnl: Decimal = field(default=Decimal("0"))

    def apply(self, fill: Fill) -> None:
        if fill.symbol != self.symbol:
            raise OrderError(f"fill for {fill.symbol} applied to {self.symbol}")
        if fill.side is Side.BUY:
            total_cost = self.avg_cost * self.quantity + fill.notional + fill.charges
            self.quantity += fill.quantity
            self.avg_cost = total_cost / self.quantity
        else:
            if fill.quantity > self.quantity:
                raise OrderError("cannot sell more than held (short selling unsupported)")
            self.realized_pnl += (fill.price - self.avg_cost) * fill.quantity - fill.charges
            self.quantity -= fill.quantity
            if self.quantity == 0:
                self.avg_cost = Decimal("0")

    def market_value(self, price: Decimal) -> Decimal:
        return price * self.quantity

    def unrealized_pnl(self, price: Decimal) -> Decimal:
        return (price - self.avg_cost) * self.quantity
