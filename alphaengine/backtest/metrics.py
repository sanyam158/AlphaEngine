"""Performance metrics.

Returns are **time-weighted** (contributions are not counted as gains), so a strategy with
SIPs can be compared with one without. **XIRR** gives the money-weighted return an investor
actually experienced.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date
from typing import Any, cast

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def twr_returns(equity: pd.DataFrame) -> pd.Series:
    """Daily time-weighted returns from ``value`` and same-day ``flow`` columns."""
    v = equity["value"].astype(float)
    flow = equity.get("flow", pd.Series(0.0, index=equity.index)).astype(float)
    r = (v - flow) / v.shift(1) - 1
    return r.iloc[1:].fillna(0.0)


def growth_index(returns: pd.Series) -> pd.Series:
    return (1 + returns).cumprod()


def cagr(index: pd.Series) -> float:
    if len(index) < 2 or index.iloc[0] <= 0:
        return float("nan")
    years = (index.index[-1] - index.index[0]).days / 365.25
    if years <= 0:
        return float("nan")
    return float((index.iloc[-1] / index.iloc[0]) ** (1 / years) - 1)


def annual_vol(r: pd.Series) -> float:
    return float(r.std(ddof=1) * math.sqrt(TRADING_DAYS)) if len(r) > 1 else float("nan")


def sharpe(r: pd.Series, rf: float = 0.0) -> float:
    ex = r - rf / TRADING_DAYS
    sd = ex.std(ddof=1)
    if not np.isfinite(sd) or sd < 1e-12:  # constant series: undefined
        return float("nan")
    return float(ex.mean() / sd * math.sqrt(TRADING_DAYS))


def sortino(r: pd.Series, rf: float = 0.0) -> float:
    ex = r - rf / TRADING_DAYS
    downside = np.sqrt((np.minimum(ex, 0) ** 2).mean())
    return float(ex.mean() / downside * math.sqrt(TRADING_DAYS)) if downside > 0 else float("nan")


def drawdown(index: pd.Series) -> pd.Series:
    return index / index.cummax() - 1


def max_drawdown(index: pd.Series) -> dict[str, Any]:
    dd = drawdown(index)
    trough = pd.Timestamp(dd.idxmin())  # type: ignore[arg-type]
    peak = pd.Timestamp(index.loc[:trough].idxmax())  # type: ignore[arg-type]
    after = index.loc[trough:]
    recovered = after[after >= index.loc[peak]]
    recovery = pd.Timestamp(recovered.index[0]) if len(recovered) else None
    end = recovery if recovery is not None else pd.Timestamp(index.index[-1])
    return {
        "max_drawdown": float(dd.min()),
        "peak": peak.date(),
        "trough": trough.date(),
        "recovery": recovery.date() if recovery is not None else None,
        "drawdown_days": int((end - peak).days),
    }


def ulcer_index(index: pd.Series) -> float:
    return float(np.sqrt((drawdown(index) ** 2).mean()))


def xirr(flows: Sequence[tuple[date, float]]) -> float:
    """Money-weighted annual return. ``flows``: investor's view (contributions negative,
    final value positive). Actual/365 day count like Excel's XIRR; solved by bisection; NaN
    when no sign change exists."""
    if len(flows) < 2:
        return float("nan")
    t0 = flows[0][0]
    ts = np.array([(d - t0).days / 365.0 for d, _ in flows])  # Excel XIRR day count
    amts = np.array([a for _, a in flows])

    def npv(rate: float) -> float:
        return float((amts / (1 + rate) ** ts).sum())

    lo, hi = -0.9999, 10.0
    f_lo, f_hi = npv(lo), npv(hi)
    if f_lo * f_hi > 0:
        return float("nan")
    for _ in range(200):
        mid = (lo + hi) / 2
        f_mid = npv(mid)
        if abs(f_mid) < 1e-7 or hi - lo < 1e-10:
            return mid
        if f_lo * f_mid < 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    return (lo + hi) / 2


def relative(r: pd.Series, b: pd.Series) -> dict[str, float]:
    """Beta, annualised alpha, tracking error and information ratio vs a benchmark."""
    df = pd.concat([r, b], axis=1, join="inner").dropna()
    if len(df) < 20:
        return {k: float("nan") for k in ("beta", "alpha", "tracking_error", "information_ratio")}
    x, y = df.iloc[:, 1], df.iloc[:, 0]
    var = x.var(ddof=1)
    beta = float(y.cov(x) / var) if var > 0 else float("nan")
    alpha = float((y.mean() - beta * x.mean()) * TRADING_DAYS)
    active = y - x
    te = float(active.std(ddof=1) * math.sqrt(TRADING_DAYS))
    ir = float(active.mean() * TRADING_DAYS / te) if te > 0 else float("nan")
    return {"beta": beta, "alpha": alpha, "tracking_error": te, "information_ratio": ir}


def yearly_returns(index: pd.Series) -> pd.Series:
    ends = index.groupby(pd.DatetimeIndex(index.index).year).last()
    starts = pd.concat([pd.Series([index.iloc[0]], index=[ends.index[0]]), ends.shift(1).iloc[1:]])
    out = cast(pd.Series, ends / starts.to_numpy() - 1)
    out.index.name = "year"
    return out


def rolling_cagr(index: pd.Series, years: int) -> dict[str, float]:
    window = int(years * TRADING_DAYS)
    if len(index) <= window:
        return {}
    roll = (index / index.shift(window)) ** (1 / years) - 1
    roll = roll.dropna()
    return {
        "min": float(roll.min()),
        "median": float(roll.median()),
        "max": float(roll.max()),
        "pct_positive": float((roll > 0).mean()),
    }


def _by_year(s: pd.Series) -> dict[int, float]:
    return {int(str(k)): float(v) for k, v in s.items()}


def _flows(equity: pd.DataFrame, final_value: float) -> list[tuple[date, float]]:
    flows = [(equity.index[0].date(), -float(equity["value"].iloc[0]))]
    flow = equity["flow"]
    flows += [
        (pd.Timestamp(d).date(), -float(f)) for d, f in zip(flow.index, flow, strict=True) if f
    ]
    flows.append((equity.index[-1].date(), final_value))
    return flows


def compute_metrics(
    equity: pd.DataFrame,
    *,
    trades: pd.DataFrame | None = None,
    disposals: Sequence[Any] = (),
    benchmark: pd.Series | None = None,
    rf: float = 0.0,
    after_tax_value: float | None = None,
    totals: dict[str, float] | None = None,
) -> dict[str, Any]:
    r = twr_returns(equity)
    idx = growth_index(pd.concat([pd.Series([0.0], index=[equity.index[0]]), r]))
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    avg_value = float(equity["value"].mean())
    final = float(equity["value"].iloc[-1])
    totals = totals or {}
    m: dict[str, Any] = {
        "start": equity.index[0].date(),
        "end": equity.index[-1].date(),
        "years": round(years, 2),
        "final_value": final,
        "contributions": float(equity["contributions"].iloc[-1]),
        "cagr": cagr(idx),
        "volatility": annual_vol(r),
        "sharpe": sharpe(r, rf),
        "sortino": sortino(r, rf),
        "ulcer_index": ulcer_index(idx),
        "xirr": xirr(_flows(equity, final)),
        **max_drawdown(idx),
    }
    m["calmar"] = m["cagr"] / abs(m["max_drawdown"]) if m["max_drawdown"] < 0 else float("nan")
    if after_tax_value is not None:
        m["after_tax_value"] = after_tax_value
        m["after_tax_xirr"] = xirr(_flows(equity, after_tax_value))
    m["costs"] = float(totals.get("costs", 0.0))
    m["taxes"] = float(totals.get("capital_gains_tax", 0.0) + totals.get("income_tax", 0.0))
    m["cost_drag"] = m["costs"] / avg_value / years
    m["tax_drag"] = m["taxes"] / avg_value / years
    m["turnover"] = float(totals.get("turnover", 0.0)) / 2 / avg_value / years
    if trades is not None:
        m["trades"] = len(trades)
        m["trades_per_year"] = len(trades) / years
    if disposals:
        qty = np.array([d.quantity for d in disposals], dtype=float)
        held = np.array([(d.sold - d.acquired).days for d in disposals], dtype=float)
        m["avg_holding_days"] = float((qty * held).sum() / qty.sum())
        m["hit_rate"] = float(np.mean([d.gain > 0 for d in disposals]))
        m["long_term_share"] = float(np.mean([d.long_term for d in disposals]))
    for yrs in (1, 3, 5):
        roll = rolling_cagr(idx, yrs)
        if roll:
            m[f"rolling_{yrs}y"] = roll
    m["yearly_returns"] = _by_year(yearly_returns(idx))
    if benchmark is not None and not benchmark.dropna().empty:
        b = benchmark.reindex(equity.index).ffill()
        br = b.pct_change(fill_method=None).iloc[1:].fillna(0.0)
        bidx = growth_index(pd.concat([pd.Series([0.0], index=[equity.index[0]]), br]))
        m["benchmark"] = {
            "cagr": cagr(bidx),
            "volatility": annual_vol(br),
            "sharpe": sharpe(br, rf),
            "max_drawdown": float(drawdown(bidx).min()),
            "yearly_returns": _by_year(yearly_returns(bidx)),
            **relative(r, br),
        }
        m["excess_cagr"] = m["cagr"] - m["benchmark"]["cagr"]
    return m


def to_jsonable(obj: Any) -> Any:
    """JSON-safe copy: dates -> ISO strings, NaN/inf -> None."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)
