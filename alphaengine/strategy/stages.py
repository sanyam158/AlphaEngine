"""Pipeline stages: Universe → Filters → Scorer → Selector → Weighter → Overlays.

Every stage is a frozen dataclass registered under a ``kind`` and ``type`` so it can be built
from YAML (``{type: top_n, n: 20}``). Stages read data only through ``ctx.data`` at
``ctx.as_of`` and explain their decisions via ``ctx.explain`` / ``ctx.note``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from alphaengine.core.errors import StrategyError
from alphaengine.features.base import FeatureSpec
from alphaengine.strategy.core import StrategyContext

STAGE_KINDS = ("universe", "filter", "scorer", "selector", "weighter", "overlay", "exit")
_COMPONENTS: dict[str, dict[str, type]] = {k: {} for k in STAGE_KINDS}


def component(kind: str, type_name: str) -> Callable[[type], type]:
    if kind not in _COMPONENTS:
        raise ValueError(f"unknown stage kind {kind!r}")

    def deco(cls: type) -> type:
        _COMPONENTS[kind][type_name] = cls
        cls.type_name = type_name  # type: ignore[attr-defined]
        return cls

    return deco


def components(kind: str) -> dict[str, type]:
    return dict(_COMPONENTS[kind])


@dataclass(frozen=True)
class FeatureRef:
    """A feature reference as written in configs: ``{feature: momentum, params: {...}}``."""

    feature: str
    params: dict[str, Any] = field(default_factory=dict)

    def spec(self) -> FeatureSpec:
        return FeatureSpec.of(self.feature, **self.params)


def _row(ctx: StrategyContext, ref: FeatureRef, symbols: list[str]) -> pd.Series:
    row = ctx.data.feature_row(ref.spec(), ctx.as_of)
    return row.reindex(symbols)


# ======================================================================== universes
@component("universe", "static")
@dataclass(frozen=True)
class StaticUniverse:
    """A fixed list of symbols (e.g. ETFs for asset allocation)."""

    symbols: list[str]

    def select(self, ctx: StrategyContext) -> list[str]:
        known = set(ctx.data.symbols)
        missing = [s for s in self.symbols if s not in known]
        if missing:
            ctx.note(f"universe symbols without data: {missing}")
        return [s for s in self.symbols if s in known]


@component("universe", "liquidity")
@dataclass(frozen=True)
class LiquidityUniverse:
    """Top ``top_n`` stocks by median traded value: survivorship-free by construction."""

    top_n: int = 200
    window: int = 126
    exclude: list[str] = field(default_factory=list)
    asset_class: Literal["EQUITY", "ETF", "ALL"] = "EQUITY"

    def select(self, ctx: StrategyContext) -> list[str]:
        adv = ctx.data.feature_row(FeatureSpec.of("adv_value", window=self.window), ctx.as_of)
        today = cast(pd.Series, ctx.data.close.loc[ctx.as_of])
        traded = today.notna().reindex(adv.index, fill_value=False)
        adv = adv[traded & adv.notna() & (adv > 0)]
        adv = adv.drop(labels=[s for s in self.exclude if s in adv.index])
        if self.asset_class != "ALL":
            keep = [s for s in adv.index if ctx.data.asset_class(str(s)) == self.asset_class]
            adv = adv[keep]
        return [
            str(s) for s in adv.sort_values(ascending=False, kind="stable").head(self.top_n).index
        ]


@component("universe", "index")
@dataclass(frozen=True)
class IndexUniverse:
    """Point-in-time members of an index (requires recorded/imported membership history)."""

    index: str

    def select(self, ctx: StrategyContext) -> list[str]:
        members = ctx.data.index_members(self.index, ctx.as_of)
        if not members:
            ctx.note(f"no membership history for {self.index} on {ctx.as_of.date()}")
        known = set(ctx.data.symbols)
        return [s for s in members if s in known]


# ======================================================================== filters
@component("filter", "min_price")
@dataclass(frozen=True)
class MinPrice:
    """Raw (as-traded) close at least ``value`` rupees."""

    value: float

    def apply(self, ctx: StrategyContext, symbols: list[str]) -> list[str]:
        px = ctx.data.raw_close.loc[ctx.as_of].reindex(symbols)
        return [s for s in symbols if px.get(s, np.nan) >= self.value]


@component("filter", "min_history")
@dataclass(frozen=True)
class MinHistory:
    """At least ``days`` trading days of price history."""

    days: int = 252

    def apply(self, ctx: StrategyContext, symbols: list[str]) -> list[str]:
        h = _row(ctx, FeatureRef("history_days"), symbols)
        return [s for s in symbols if h.get(s, 0) >= self.days]


@component("filter", "min_adv")
@dataclass(frozen=True)
class MinAdv:
    """Median daily traded value at least ``value`` rupees."""

    value: float
    window: int = 63

    def apply(self, ctx: StrategyContext, symbols: list[str]) -> list[str]:
        adv = _row(ctx, FeatureRef("adv_value", {"window": self.window}), symbols)
        return [s for s in symbols if adv.get(s, 0) >= self.value]


_OPS: dict[str, Callable[[pd.Series, float], pd.Series]] = {
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
}


@component("filter", "feature")
@dataclass(frozen=True)
class FeatureFilter:
    """Keep symbols whose feature satisfies ``op value`` (e.g. price_to_sma > 0)."""

    feature: str
    op: Literal[">", ">=", "<", "<="]
    value: float
    params: dict[str, Any] = field(default_factory=dict)

    def apply(self, ctx: StrategyContext, symbols: list[str]) -> list[str]:
        v = _row(ctx, FeatureRef(self.feature, self.params), symbols)
        ok = _OPS[self.op](v, self.value).fillna(False)
        return [s for s in symbols if bool(ok.get(s, False))]


# ======================================================================== scorers
@dataclass(frozen=True)
class Factor:
    feature: str
    params: dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    higher_is_better: bool = True


def _zscore(x: pd.Series, clip: float = 3.0) -> pd.Series:
    x = x.astype(float)
    std = x.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return x * 0.0
    return ((x - x.mean()) / std).clip(-clip, clip)


@component("scorer", "feature")
@dataclass(frozen=True)
class FeatureScore:
    """Score = a single feature's value (sign flipped when lower is better)."""

    feature: str
    params: dict[str, Any] = field(default_factory=dict)
    higher_is_better: bool = True

    def score(self, ctx: StrategyContext, symbols: list[str]) -> pd.Series:
        v = _row(ctx, FeatureRef(self.feature, self.params), symbols).astype(float)
        return (v if self.higher_is_better else -v).dropna()


@component("scorer", "composite")
@dataclass(frozen=True)
class CompositeScore:
    """Weighted sum of cross-sectional z-scores (winsorised at ±3). Symbols missing any factor
    are dropped."""

    factors: list[Factor]

    def __post_init__(self) -> None:
        if not self.factors:
            raise StrategyError("composite scorer needs at least one factor")
        object.__setattr__(
            self, "factors", [f if isinstance(f, Factor) else Factor(**f) for f in self.factors]
        )

    def score(self, ctx: StrategyContext, symbols: list[str]) -> pd.Series:
        parts = []
        for f in self.factors:
            v = _row(ctx, FeatureRef(f.feature, f.params), symbols).astype(float)
            parts.append(f.weight * _zscore(v if f.higher_is_better else -v))
        frame = pd.concat(parts, axis=1)
        return frame.dropna().sum(axis=1)


# ======================================================================== selectors
@component("selector", "top_n")
@dataclass(frozen=True)
class TopN:
    """Best ``n`` by score, with a hysteresis ``buffer``: a current holding is kept while it
    ranks within the top ``buffer``. This is the main turnover reducer."""

    n: int
    buffer: int | None = None
    min_score: float | None = None

    def __post_init__(self) -> None:
        if self.n <= 0 or (self.buffer is not None and self.buffer < self.n):
            raise StrategyError("top_n needs n > 0 and buffer >= n")

    def select(self, ctx: StrategyContext, scores: pd.Series) -> list[str]:
        s = scores.dropna()
        if self.min_score is not None:
            s = s[s > self.min_score]
        ranked = list(s.sort_values(ascending=False, kind="stable").index.astype(str))
        rank = {sym: i + 1 for i, sym in enumerate(ranked)}
        keep_limit = self.buffer or self.n
        kept = [sym for sym in ranked if sym in ctx.current_weights and rank[sym] <= keep_limit]
        kept = kept[: self.n]
        chosen = kept + [sym for sym in ranked if sym not in kept][: self.n - len(kept)]
        for sym in chosen:
            held = " (held, within buffer)" if sym in kept and rank[sym] > self.n else ""
            ctx.explain(sym, f"rank {rank[sym]}/{len(ranked)} score {s[sym]:.3f}{held}")
        return chosen


@component("selector", "all")
@dataclass(frozen=True)
class SelectAll:
    """Everything that survived the filters (optionally above ``min_score``)."""

    min_score: float | None = None

    def select(self, ctx: StrategyContext, scores: pd.Series) -> list[str]:
        s = scores.dropna()
        if self.min_score is not None:
            s = s[s > self.min_score]
        return [str(x) for x in s.index]


# ======================================================================== weighters
@component("weighter", "equal")
@dataclass(frozen=True)
class EqualWeight:
    def weights(self, ctx: StrategyContext, selected: list[str]) -> pd.Series:
        if not selected:
            return pd.Series(dtype=float)
        return pd.Series(1.0 / len(selected), index=selected)


@component("weighter", "fixed")
@dataclass(frozen=True)
class FixedWeights:
    """Fixed target weights; weights for unselected symbols stay in cash."""

    weights_by_symbol: dict[str, float]

    def __post_init__(self) -> None:
        if (
            any(w < 0 for w in self.weights_by_symbol.values())
            or sum(self.weights_by_symbol.values()) > 1 + 1e-9
        ):
            raise StrategyError("fixed weights must be >= 0 and sum to <= 1")

    def weights(self, ctx: StrategyContext, selected: list[str]) -> pd.Series:
        return pd.Series({s: self.weights_by_symbol.get(s, 0.0) for s in selected}, dtype=float)


@component("weighter", "inverse_vol")
@dataclass(frozen=True)
class InverseVolatility:
    """Weights ∝ 1/volatility: calmer names get more capital."""

    window: int = 63

    def weights(self, ctx: StrategyContext, selected: list[str]) -> pd.Series:
        if not selected:
            return pd.Series(dtype=float)
        vol = _row(ctx, FeatureRef("volatility", {"window": self.window}), selected)
        inv = (1.0 / vol.where(vol > 0)).dropna()
        if inv.empty:
            return EqualWeight().weights(ctx, selected)
        return inv / inv.sum()


# ======================================================================== overlays
@component("overlay", "max_weight")
@dataclass(frozen=True)
class MaxWeight:
    """Cap each position at ``cap``; the excess is redistributed pro rata among the others
    (or left in cash if every position is capped)."""

    cap: float

    def apply(self, ctx: StrategyContext, w: pd.Series) -> pd.Series:
        if not 0 < self.cap <= 1:
            raise StrategyError("max_weight cap must be in (0, 1]")
        w = w.copy()
        for _ in range(len(w) + 1):
            over = w > self.cap + 1e-12
            if not over.any():
                break
            excess = float((w[over] - self.cap).sum())
            w[over] = self.cap
            room = w[~over & (w < self.cap)]
            if room.empty:
                break
            w[room.index] += excess * room / room.sum()
        return w


@component("overlay", "regime")
@dataclass(frozen=True)
class RegimeFilter:
    """When ``benchmark`` closes below its ``sma``-day average, scale exposure down to
    ``risk_off_exposure`` (the rest goes to cash)."""

    benchmark: str = "NIFTY50"
    sma: int = 200
    risk_off_exposure: float = 0.5

    def apply(self, ctx: StrategyContext, w: pd.Series) -> pd.Series:
        b = ctx.data.benchmarks
        if self.benchmark not in b.columns:
            raise StrategyError(f"benchmark {self.benchmark!r} not loaded")
        series = b[self.benchmark].loc[: ctx.as_of].dropna()
        if len(series) < self.sma:
            ctx.note(f"regime: not enough {self.benchmark} history; staying risk-on")
            return w
        level, avg = float(series.iloc[-1]), float(series.iloc[-self.sma :].mean())
        if level >= avg:
            ctx.note(f"regime: risk-on ({self.benchmark} {level:.0f} >= {self.sma}d avg {avg:.0f})")
            return w
        ctx.note(
            f"regime: risk-off ({self.benchmark} {level:.0f} < {self.sma}d avg {avg:.0f}); "
            f"exposure {self.risk_off_exposure:.0%}"
        )
        return w * self.risk_off_exposure


@component("overlay", "vol_target")
@dataclass(frozen=True)
class VolatilityTarget:
    """Scale exposure so the portfolio's recent volatility is at most ``target`` (never levers)."""

    target: float = 0.15
    window: int = 63

    def apply(self, ctx: StrategyContext, w: pd.Series) -> pd.Series:
        if w.empty:
            return w
        rets = ctx.data.tr_close[list(w.index)].loc[: ctx.as_of].pct_change(fill_method=None)
        rets = rets.iloc[-self.window :].fillna(0.0)
        if len(rets) < self.window // 2:
            return w
        port_vol = float((rets @ w).std() * np.sqrt(252))
        if port_vol <= self.target or port_vol == 0:
            return w
        scale = self.target / port_vol
        ctx.note(
            f"vol target: recent vol {port_vol:.1%} > {self.target:.0%}; exposure x{scale:.2f}"
        )
        return w * scale


@component("overlay", "sector_cap")
@dataclass(frozen=True)
class SectorCap:
    """Limit total weight per industry to ``cap`` (excess to cash). Unknown industry = own group."""

    cap: float = 0.3

    def apply(self, ctx: StrategyContext, w: pd.Series) -> pd.Series:
        sectors = pd.Series({s: ctx.data.industries.get(s, f"?{s}") for s in w.index})
        out = w.copy()
        for sector, total in w.groupby(sectors).sum().items():
            if total > self.cap + 1e-12:
                members = sectors[sectors == sector].index
                out[members] *= self.cap / total
                ctx.note(f"sector cap: {sector} {total:.0%} -> {self.cap:.0%}")
        return out
