"""Where the goal planner's return history comes from."""

from __future__ import annotations

from datetime import date

import numpy as np

from alphaengine.app import AppContext
from alphaengine.backtest.metrics import growth_index, twr_returns
from alphaengine.backtest.runner import get_run
from alphaengine.features.data import MarketData
from alphaengine.planning.goal import monthly_returns


def history_for(
    ctx: AppContext,
    *,
    run: str | None = None,
    symbol: str = "NIFTYBEES",
    start: date = date(2010, 1, 1),
) -> tuple[np.ndarray, str]:
    """Monthly returns from a stored backtest (time-weighted, after its costs and taxes) or
    from a symbol's total-return history."""
    if run:
        rec, eq = get_run(ctx, run)
        return monthly_returns(growth_index(twr_returns(eq))), f"backtest {rec.strategy} ({run})"
    md = MarketData.from_store(ctx, start, None, [symbol.upper()])
    if symbol.upper() not in md.tr_close.columns:
        raise ValueError(f"no price history for {symbol}")
    return monthly_returns(md.tr_close[symbol.upper()]), f"{symbol.upper()} total return"
