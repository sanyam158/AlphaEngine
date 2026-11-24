"""Data-quality checks over a bars frame. Each check returns :class:`Issue` records; nothing is
modified. Run after every ingest and before trusting a dataset for research."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from alphaengine.data.adjust import apply_adjustment
from alphaengine.data.schema import KEY, PRICE_COLUMNS

SEVERITIES = ("info", "warning", "error")


@dataclass(frozen=True)
class Issue:
    check: str
    severity: str
    detail: str
    symbol: str | None = None
    trade_date: date | None = None


@dataclass(frozen=True)
class QualityConfig:
    spike_threshold: float = 0.35  # |daily return| on split-adjusted closes
    prev_close_tolerance: float = 0.02  # exchange PREVCLOSE vs our previous close
    stale_days: int = 10  # market days without a print
    coverage_ratio: float = 0.5  # day's symbol count vs 20-day median


def _d(ts: object) -> date:
    return pd.Timestamp(ts).date()  # type: ignore[arg-type]


def check_duplicates(bars: pd.DataFrame) -> list[Issue]:
    dup = bars[bars.duplicated(KEY, keep=False)]
    return [
        Issue("duplicate_key", "error", "duplicate (date, symbol) row", str(s), _d(d))
        for d, s in dup[KEY].drop_duplicates().itertuples(index=False)
    ]


def check_ohlc(bars: pd.DataFrame) -> list[Issue]:
    p = bars[PRICE_COLUMNS]
    bad = (
        (p <= 0).any(axis=1)
        | (bars["high"] < p[["open", "close", "low"]].max(axis=1))
        | (bars["low"] > p[["open", "close", "high"]].min(axis=1))
    )
    return [
        Issue("ohlc_inconsistent", "error", "invalid OHLC relationship", str(s), _d(d))
        for d, s in bars.loc[bad, KEY].itertuples(index=False)
    ]


def check_zero_volume(bars: pd.DataFrame) -> list[Issue]:
    z = bars[bars["volume"] == 0]
    return [
        Issue("zero_volume", "info", "no shares traded", str(s), _d(d))
        for d, s in z[KEY].itertuples(index=False)
    ]


def check_gaps(bars: pd.DataFrame, market_days: Sequence[date]) -> list[Issue]:
    """Symbols missing on market days between their first and last print."""
    if bars.empty or not market_days:
        return []
    md = np.array(sorted(pd.Timestamp(d) for d in market_days), dtype="datetime64[ns]")
    g = bars.groupby("symbol")["date"].agg(["min", "max", "size"])
    lo = np.searchsorted(md, g["min"].to_numpy(dtype="datetime64[ns]"), side="left")
    hi = np.searchsorted(md, g["max"].to_numpy(dtype="datetime64[ns]"), side="right")
    expected = hi - lo
    issues: list[Issue] = []
    for sym, exp, got, a, b in zip(g.index, expected, g["size"], lo, hi, strict=True):
        if got >= exp:
            continue
        have = set(bars.loc[bars["symbol"] == sym, "date"].to_numpy(dtype="datetime64[ns]"))
        missing = [m for m in md[a:b] if m not in have]
        sample = ", ".join(str(_d(m)) for m in missing[:3])
        issues.append(
            Issue(
                "missing_days",
                "warning",
                f"{len(missing)} market day(s) without a bar (first: {sample})",
                str(sym),
                _d(missing[0]),
            )
        )
    return issues


def _with_prev(bars: pd.DataFrame) -> pd.DataFrame:
    b = bars.sort_values(KEY).copy()
    grp = b.groupby("symbol")
    b["prev_own_close"] = grp["close"].shift(1)
    b["prev_own_date"] = grp["date"].shift(1)
    return b


def check_spikes(bars: pd.DataFrame, actions: pd.DataFrame, cfg: QualityConfig) -> list[Issue]:
    """Large moves on split-adjusted prices; usually a missing split/bonus record."""
    adj = _with_prev(apply_adjustment(bars, actions, "split"))
    ret = adj["close"] / adj["prev_own_close"] - 1
    hits = adj[ret.abs() > cfg.spike_threshold]
    return [
        Issue(
            "price_spike",
            "warning",
            f"close moved {r:+.1%} vs previous bar; check for a missing corporate action",
            str(s),
            _d(d),
        )
        for d, s, r in zip(hits["date"], hits["symbol"], ret[hits.index], strict=True)
    ]


def check_prev_close(
    bars: pd.DataFrame, actions: pd.DataFrame, market_days: Sequence[date], cfg: QualityConfig
) -> list[Issue]:
    """NSE adjusts PREVCLOSE on ex-dates. A mismatch with our own previous close on a
    consecutive market day, without a recorded split/bonus, points to a missing action."""
    if bars.empty or "prev_close" not in bars.columns:
        return []
    b = _with_prev(bars)
    md = sorted(pd.Timestamp(d) for d in market_days)
    prev_md = dict(zip(md[1:], md[:-1], strict=False))
    consecutive = b["prev_own_date"].notna() & (b["date"].map(prev_md) == b["prev_own_date"])
    ratio = b["prev_own_close"] / b["prev_close"]
    mism = b[consecutive & b["prev_close"].notna() & ((ratio - 1).abs() > cfg.prev_close_tolerance)]
    known: set[tuple[str, pd.Timestamp]] = set()
    if not actions.empty:
        sa = actions[actions["kind"].isin(["SPLIT", "BONUS"])]
        known = {
            (str(s), pd.Timestamp(e)) for s, e in zip(sa["symbol"], sa["ex_date"], strict=True)
        }
    return [
        Issue(
            "prev_close_mismatch",
            "warning",
            f"exchange prev_close implies adjustment ratio {r:.4f} with no recorded split/bonus",
            str(s),
            _d(d),
        )
        for d, s, r in zip(mism["date"], mism["symbol"], ratio[mism.index], strict=True)
        if (str(s), pd.Timestamp(d)) not in known
    ]


def check_stale(bars: pd.DataFrame, market_days: Sequence[date], cfg: QualityConfig) -> list[Issue]:
    if bars.empty or len(market_days) <= cfg.stale_days:
        return []
    cutoff = pd.Timestamp(sorted(market_days)[-cfg.stale_days - 1])
    last = bars.groupby("symbol")["date"].max()
    stale = last[last <= cutoff]
    return [
        Issue("stale_symbol", "info", f"no bar since {_d(d)} (suspended/delisted?)", str(s), _d(d))
        for s, d in stale.items()
    ]


def check_coverage(bars: pd.DataFrame, cfg: QualityConfig) -> list[Issue]:
    """Days with far fewer symbols than usual indicate a truncated or partial file."""
    counts = bars.groupby("date")["symbol"].size().sort_index()
    if counts.size < 5:
        return []
    baseline = counts.rolling(20, min_periods=5, center=True).median()
    low = counts[counts < cfg.coverage_ratio * baseline]
    typical = baseline[low.index]
    return [
        Issue("low_coverage", "error", f"only {n} symbols vs typical {b:.0f}", None, _d(d))
        for d, n, b in zip(low.index, low.to_numpy(), typical.to_numpy(), strict=True)
    ]


def run_checks(
    bars: pd.DataFrame,
    market_days: Sequence[date],
    actions: pd.DataFrame | None = None,
    cfg: QualityConfig | None = None,
) -> list[Issue]:
    cfg = cfg or QualityConfig()
    acts = (
        actions
        if actions is not None
        else pd.DataFrame(columns=["symbol", "ex_date", "kind", "ratio", "amount", "source"])
    )
    issues = [
        *check_duplicates(bars),
        *check_ohlc(bars),
        *check_coverage(bars, cfg),
        *check_gaps(bars, market_days),
        *check_spikes(bars.drop_duplicates(KEY), acts, cfg),
        *check_prev_close(bars.drop_duplicates(KEY), acts, market_days, cfg),
        *check_stale(bars, market_days, cfg),
        *check_zero_volume(bars),
    ]
    return issues


def summarize(issues: Sequence[Issue]) -> dict[str, dict[str, int]]:
    """{severity: {check: count}}"""
    out: dict[str, dict[str, int]] = {s: {} for s in SEVERITIES}
    for (sev, chk), n in Counter((i.severity, i.check) for i in issues).items():
        out[sev][chk] = n
    return out
