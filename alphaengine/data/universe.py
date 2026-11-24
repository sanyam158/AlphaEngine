"""Point-in-time investable universes.

- :func:`liquidity_universe` ranks every stock that actually traded (from the survivorship-free
  bhavcopy) by median traded value. It needs no external history and includes stocks that were
  later delisted, which makes it the default research universe.
- :func:`index_universe` uses recorded/imported index membership and warns when the history
  doesn't reach back to the requested date (survivorship bias).
"""

from __future__ import annotations

from datetime import date

import pandas as pd
from sqlalchemy.orm import Session

from alphaengine.core.logging import get_logger
from alphaengine.core.types import Symbol
from alphaengine.data.repositories import members_as_of, membership_history_start

log = get_logger(__name__)


def liquidity_universe(
    bars: pd.DataFrame,
    as_of: date,
    *,
    top_n: int,
    lookback_days: int = 126,
    min_presence: float = 0.8,
    min_median_value: float = 0.0,
) -> list[Symbol]:
    """Top ``top_n`` symbols by median daily traded value (₹) over the last ``lookback_days``
    market days up to and including ``as_of``. Symbols must have traded on at least
    ``min_presence`` of those days and on ``as_of`` itself."""
    if top_n <= 0 or lookback_days <= 0 or not 0 < min_presence <= 1:
        raise ValueError("top_n and lookback_days must be > 0 and 0 < min_presence <= 1")
    ts = pd.Timestamp(as_of)
    hist = bars.loc[bars["date"] <= ts, ["date", "symbol", "value"]]
    days = hist["date"].drop_duplicates().sort_values()
    if days.empty or days.iloc[-1] != ts:
        return []
    window = days.iloc[-lookback_days:]
    w = hist[hist["date"].isin(window)]
    stats = w.groupby("symbol").agg(n=("value", "size"), med=("value", "median"))
    on_day = set(w.loc[w["date"] == ts, "symbol"])
    stats = stats[
        (stats["n"] >= min_presence * len(window))
        & (stats["med"] >= min_median_value)
        & stats.index.isin(on_day)
    ]
    ranked = stats.sort_values(["med"], ascending=False, kind="stable").head(top_n)
    return [Symbol(str(s)) for s in ranked.index]


def index_universe(session: Session, index_name: str, as_of: date) -> frozenset[Symbol]:
    start = membership_history_start(session, index_name)
    if start is None or as_of < start:
        log.warning(
            "index membership history does not cover date; universe may be empty or biased",
            index=index_name,
            as_of=str(as_of),
            history_start=str(start),
        )
    return members_as_of(session, index_name, as_of)
