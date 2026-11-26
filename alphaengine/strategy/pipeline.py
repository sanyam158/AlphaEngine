"""Composable strategies: a single stage pipeline, or a blend of strategies (core + satellite)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import pandas as pd

from alphaengine.core.errors import StrategyError
from alphaengine.strategy.core import Schedule, Strategy, StrategyContext, TargetPortfolio


class Universe(Protocol):
    def select(self, ctx: StrategyContext) -> list[str]: ...


class Filter(Protocol):
    def apply(self, ctx: StrategyContext, symbols: list[str]) -> list[str]: ...


class Scorer(Protocol):
    def score(self, ctx: StrategyContext, symbols: list[str]) -> pd.Series: ...


class Selector(Protocol):
    def select(self, ctx: StrategyContext, scores: pd.Series) -> list[str]: ...


class Weighter(Protocol):
    def weights(self, ctx: StrategyContext, selected: list[str]) -> pd.Series: ...


class Overlay(Protocol):
    def apply(self, ctx: StrategyContext, w: pd.Series) -> pd.Series: ...


class _ConstantScore:
    def score(self, ctx: StrategyContext, symbols: list[str]) -> pd.Series:
        return pd.Series(1.0, index=symbols, dtype=float)


@dataclass
class PipelineStrategy:
    name: str
    universe: Universe
    weighter: Weighter
    filters: Sequence[Filter] = ()
    scorer: Scorer | None = None
    selector: Selector | None = None
    overlays: Sequence[Overlay] = ()
    schedule: Schedule = field(default_factory=Schedule)
    version: int = 1
    description: str = ""
    config: Mapping[str, Any] | None = None  # the config it was built from, if any
    exits: Sequence[Any] = ()  # ExitRule-like objects; non-empty = rules mode
    max_positions: int | None = None  # rules mode: equal-size slots
    cooldown_days: int = 5  # rules mode: no re-entry this many trading days after an exit

    @property
    def rules_mode(self) -> bool:
        return bool(self.exits)

    def rebalance_dates(self, dates: pd.DatetimeIndex) -> list[pd.Timestamp]:
        return self.schedule.dates(dates)

    def target(self, ctx: StrategyContext) -> TargetPortfolio:
        if ctx.as_of not in ctx.data.close.index:
            raise StrategyError(f"{ctx.as_of.date()} is not a trading day in the data")
        symbols = self.universe.select(ctx)
        n_universe = len(symbols)
        for f in self.filters:
            symbols = f.apply(ctx, symbols)
        scores = (self.scorer or _ConstantScore()).score(ctx, symbols)
        selected = self.selector.select(ctx, scores) if self.selector else list(scores.index)
        w = self.weighter.weights(ctx, selected)
        for o in self.overlays:
            w = o.apply(ctx, w)
        w = w[w > 1e-9].astype(float)
        ctx.notes.insert(
            0, f"universe {n_universe} -> after filters {len(symbols)} -> selected {len(selected)}"
        )
        return TargetPortfolio(
            as_of=ctx.as_of,
            weights={str(k): float(v) for k, v in w.items()},
            rationale={k: list(v) for k, v in ctx.rationale.items() if k in w.index},
            notes=tuple(ctx.notes),
            strategy=self.name,
        )


@dataclass
class Blend:
    """Weighted combination of strategies, e.g. 70% core allocation + 30% momentum satellite.
    Each child is evaluated on the blend's schedule; weights are scaled and summed."""

    name: str
    parts: Sequence[tuple[Strategy, float]]
    schedule: Schedule = field(default_factory=Schedule)
    version: int = 1
    description: str = ""
    config: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        total = sum(w for _, w in self.parts)
        if not self.parts or any(w < 0 for _, w in self.parts) or total > 1 + 1e-9:
            raise StrategyError("blend weights must be >= 0 and sum to <= 1")

    def rebalance_dates(self, dates: pd.DatetimeIndex) -> list[pd.Timestamp]:
        return self.schedule.dates(dates)

    def target(self, ctx: StrategyContext) -> TargetPortfolio:
        combined: dict[str, float] = {}
        rationale: dict[str, list[str]] = {}
        notes: list[str] = []
        for strat, weight in self.parts:
            # Children see their own slice of current holdings (approximation: pro rata).
            child_current = (
                {s: w / weight for s, w in ctx.current_weights.items()} if weight else {}
            )
            child = strat.target(StrategyContext(ctx.as_of, ctx.data, child_current))
            for sym, w in child.weights.items():
                combined[sym] = combined.get(sym, 0.0) + w * weight
                rationale.setdefault(sym, []).extend(
                    f"[{strat.name} {weight:.0%}] {r}" for r in child.rationale.get(sym, [])
                )
            notes.extend(f"[{strat.name}] {n}" for n in child.notes)
        return TargetPortfolio(ctx.as_of, combined, rationale, tuple(notes), self.name)
