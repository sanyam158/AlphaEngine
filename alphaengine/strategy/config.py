"""Declarative strategy configs (YAML/JSON) → strategy objects.

A pipeline config::

    name: momentum_top20
    version: 1
    schedule: monthly
    universe: {type: liquidity, top_n: 200}
    filters:  [{type: min_price, value: 20}]
    scorer:   {type: feature, feature: risk_adj_momentum}
    selector: {type: top_n, n: 20, buffer: 40}
    weighter: {type: equal}
    overlays: [{type: max_weight, cap: 0.1}]

A blend config replaces the stages with ``blend: [{strategy: <name|file|inline>, weight: w}]``.
Stage parameters are validated against each component's fields. ``config_hash`` identifies a
config's exact content, so every backtest/signal can be traced back to it.
"""

from __future__ import annotations

import hashlib
import json
import re
from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, field_validator

import alphaengine.strategy.exits  # noqa: F401 - registers exit components
from alphaengine.core.errors import StrategyError
from alphaengine.strategy.core import Frequency, Schedule, Strategy
from alphaengine.strategy.pipeline import Blend, PipelineStrategy
from alphaengine.strategy.stages import components

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")


class StageConfig(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)
    type: str

    def params(self) -> dict[str, Any]:
        return dict(self.model_extra or {})


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    version: int = Field(default=1, ge=1)
    description: str = ""
    schedule: Frequency = "monthly"

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        if not _NAME_RE.fullmatch(v):
            raise ValueError("name must be lowercase letters, digits and underscores")
        return v


class PipelineConfig(_Base):
    universe: StageConfig
    filters: list[StageConfig] = Field(default_factory=list)
    scorer: StageConfig | None = None
    selector: StageConfig | None = None
    weighter: StageConfig = StageConfig(type="equal")
    overlays: list[StageConfig] = Field(default_factory=list)
    exits: list[StageConfig] = Field(default_factory=list)  # rules mode (swing trading)
    max_positions: int | None = Field(default=None, ge=1, le=200)
    cooldown_days: int = Field(default=5, ge=0, le=250)


class BlendPart(BaseModel):
    model_config = ConfigDict(extra="forbid")
    strategy: str | dict[str, Any]  # built-in name, YAML path, or inline config
    weight: float = Field(ge=0, le=1)


class BlendConfig(_Base):
    blend: list[BlendPart]


AnyConfig = PipelineConfig | BlendConfig


def config_hash(cfg: AnyConfig) -> str:
    canonical = json.dumps(cfg.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def parse_config(raw: dict[str, Any]) -> AnyConfig:
    try:
        return (
            BlendConfig.model_validate(raw)
            if "blend" in raw
            else PipelineConfig.model_validate(raw)
        )
    except ValidationError as exc:
        raise StrategyError(f"invalid strategy config: {exc}") from exc


def _build_stage(kind: str, cfg: StageConfig, where: str) -> Any:
    registry = components(kind)
    cls = registry.get(cfg.type)
    if cls is None:
        raise StrategyError(f"{where}: unknown {kind} type {cfg.type!r}; known: {sorted(registry)}")
    try:
        return TypeAdapter(cls).validate_python(cfg.params())
    except ValidationError as exc:
        raise StrategyError(f"{where} ({kind} {cfg.type!r}): {exc}") from exc
    except StrategyError as exc:
        raise StrategyError(f"{where} ({kind} {cfg.type!r}): {exc}") from exc


def build_strategy(cfg: AnyConfig, *, _depth: int = 0) -> Strategy:
    if _depth > 5:
        raise StrategyError("blend nesting too deep (cycle?)")
    raw = cfg.model_dump(mode="json")
    if isinstance(cfg, BlendConfig):
        parts = []
        for i, part in enumerate(cfg.blend):
            child_cfg = (
                load_config(part.strategy)
                if isinstance(part.strategy, str)
                else parse_config(part.strategy)
            )
            try:
                parts.append((build_strategy(child_cfg, _depth=_depth + 1), part.weight))
            except StrategyError as exc:
                raise StrategyError(f"blend[{i}]: {exc}") from exc
        return Blend(cfg.name, parts, Schedule(cfg.schedule), cfg.version, cfg.description, raw)
    return PipelineStrategy(
        name=cfg.name,
        universe=_build_stage("universe", cfg.universe, "universe"),
        filters=[_build_stage("filter", f, f"filters[{i}]") for i, f in enumerate(cfg.filters)],
        scorer=_build_stage("scorer", cfg.scorer, "scorer") if cfg.scorer else None,
        selector=_build_stage("selector", cfg.selector, "selector") if cfg.selector else None,
        weighter=_build_stage("weighter", cfg.weighter, "weighter"),
        overlays=[_build_stage("overlay", o, f"overlays[{i}]") for i, o in enumerate(cfg.overlays)],
        schedule=Schedule(cfg.schedule),
        version=cfg.version,
        description=cfg.description,
        config=raw,
        exits=[_build_stage("exit", x, f"exits[{i}]") for i, x in enumerate(cfg.exits)],
        max_positions=cfg.max_positions or (10 if cfg.exits else None),
        cooldown_days=cfg.cooldown_days,
    )


def builtin_names() -> list[str]:
    root = resources.files("alphaengine.strategy") / "builtin"
    return sorted(p.name.removesuffix(".yaml") for p in root.iterdir() if p.name.endswith(".yaml"))


def load_config(source: str | Path) -> AnyConfig:
    """Load from a YAML/JSON file path or a built-in strategy name."""
    path = Path(source)
    if path.suffix in {".yaml", ".yml", ".json"}:
        if not path.exists():
            raise StrategyError(f"strategy file not found: {path}")
        text = path.read_text(encoding="utf-8")
    else:
        res = resources.files("alphaengine.strategy") / "builtin" / f"{source}.yaml"
        if not res.is_file():
            raise StrategyError(f"unknown strategy {source!r}; built-ins: {builtin_names()}")
        text = res.read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise StrategyError(f"{source}: config must be a mapping")
    return parse_config(raw)


def load_strategy(source: str | Path) -> Strategy:
    return build_strategy(load_config(source))
