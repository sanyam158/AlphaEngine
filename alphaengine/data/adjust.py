"""Backward price adjustment for corporate actions.

For every bar dated *before* an action's ex-date:
- **SPLIT / BONUS** with share multiplier ``r`` (2-for-1 split, or a 1:1 bonus, both give r=2):
  prices × 1/r, volume × r.
- **DIVIDEND** of raw amount ``D``: total-return prices × (1 − D / P), where ``P`` is the raw
  close on the last trading day before the ex-date (CRSP-style).

The latest bars are therefore unchanged and history is rescaled. Returns computed from the
adjusted series are unaffected by the rescaling, but price *levels* of old bars depend on later
actions. Use raw prices for any rule based on absolute price (e.g. "price > ₹50").
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from alphaengine.core.logging import get_logger

log = get_logger(__name__)

Adjust = Literal["none", "split", "total"]
_SPLIT_KINDS = ("SPLIT", "BONUS")


def _suffix_factor(dates: np.ndarray, ex_dates: np.ndarray, mults: np.ndarray) -> np.ndarray:
    """factor[i] = Π mults[j] over all j with ex_dates[j] > dates[i]."""
    if ex_dates.size == 0:
        return np.ones(dates.shape[0])
    order = np.argsort(ex_dates, kind="stable")
    ex, m = ex_dates[order], mults[order]
    suffix = np.append(np.cumprod(m[::-1])[::-1], 1.0)
    return suffix[np.searchsorted(ex, dates, side="right")]


def adjustment_factors(bars: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    """Return ``bars[[date, symbol]]`` plus ``split_factor`` and ``tr_factor`` columns.

    ``bars`` must contain raw ``close``; ``actions`` uses CORPORATE_ACTION_COLUMNS.
    """
    out = bars[["date", "symbol"]].copy()
    out["split_factor"] = 1.0
    out["tr_factor"] = 1.0
    if actions.empty or bars.empty:
        return out
    acts = actions.copy()
    acts["ex_date"] = pd.to_datetime(acts["ex_date"]).astype("datetime64[ns]")
    for symbol, idx in bars.groupby("symbol", sort=False).groups.items():
        a = acts[acts["symbol"] == symbol]
        if a.empty:
            continue
        sub = bars.loc[idx].sort_values("date")
        d = sub["date"].to_numpy(dtype="datetime64[ns]")
        closes = sub["close"].to_numpy(dtype=float)

        splits = a[a["kind"].isin(_SPLIT_KINDS)]
        split_f = _suffix_factor(
            d,
            splits["ex_date"].to_numpy(dtype="datetime64[ns]"),
            1.0 / splits["ratio"].to_numpy(dtype=float),
        )

        div_ex, div_m = [], []
        for ex, amt in a.loc[a["kind"] == "DIVIDEND", ["ex_date", "amount"]].itertuples(
            index=False
        ):
            pos = np.searchsorted(d, np.datetime64(ex, "ns"), side="left") - 1
            if pos < 0:
                continue  # no price before ex-date in this window
            prev = closes[pos]
            if not 0 < float(amt) < prev:
                log.warning(
                    "skipping implausible dividend", symbol=symbol, ex_date=str(ex), amount=amt
                )
                continue
            div_ex.append(np.datetime64(ex, "ns"))
            div_m.append(1.0 - float(amt) / prev)
        div_f = _suffix_factor(d, np.array(div_ex, dtype="datetime64[ns]"), np.array(div_m))

        out.loc[sub.index, "split_factor"] = split_f
        out.loc[sub.index, "tr_factor"] = split_f * div_f
    return out


def apply_adjustment(bars: pd.DataFrame, actions: pd.DataFrame, how: Adjust) -> pd.DataFrame:
    """Return a copy of ``bars`` with OHLC (and prev_close) adjusted; volume split-adjusted."""
    if how == "none" or bars.empty:
        return bars.copy()
    if how not in ("split", "total"):
        raise ValueError(f"unknown adjustment {how!r}")
    f = adjustment_factors(bars, actions)
    price_f = f["split_factor"] if how == "split" else f["tr_factor"]
    out = bars.copy()
    for col in ("open", "high", "low", "close", "prev_close"):
        if col in out.columns:
            out[col] = out[col] * price_f
    if "prev_close" in out.columns:
        # prev_close belongs to the previous bar, so it must carry that bar's factor.
        shifted = out.sort_values("date").groupby("symbol")["close"].shift(1)
        out["prev_close"] = shifted.reindex(out.index).fillna(out["prev_close"])
    if "volume" in out.columns:
        out["volume"] = (out["volume"] / f["split_factor"]).round().astype("int64")
    return out
