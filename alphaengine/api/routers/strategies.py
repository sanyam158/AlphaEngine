from __future__ import annotations

import json
from datetime import date, timedelta
from typing import Annotated, Any

import pandas as pd
import yaml
from fastapi import APIRouter, HTTPException, Query

from alphaengine.api.deps import Ctx
from alphaengine.api.schemas import (
    Position,
    SignalsOut,
    StrategyDetail,
    StrategySummary,
    ValidateRequest,
    ValidateResult,
)
from alphaengine.backtest.config import BacktestConfig
from alphaengine.backtest.runner import required_symbols
from alphaengine.core.errors import StrategyError
from alphaengine.data.db import session_scope
from alphaengine.features.data import MarketData
from alphaengine.strategy.config import (
    AnyConfig,
    BlendConfig,
    build_strategy,
    builtin_names,
    config_hash,
    load_config,
    parse_config,
)
from alphaengine.strategy.core import StrategyContext
from alphaengine.strategy.registry import get_config, list_configs

router = APIRouter(prefix="/strategies", tags=["strategies"])


def _summary(cfg: AnyConfig, source: str) -> dict[str, Any]:
    return {
        "name": cfg.name,
        "version": cfg.version,
        "description": cfg.description,
        "kind": "blend" if isinstance(cfg, BlendConfig) else "pipeline",
        "schedule": cfg.schedule,
        "source": source,
        "config_hash": config_hash(cfg),
    }


def parse_text(text: str) -> AnyConfig:
    try:
        raw = json.loads(text) if text.lstrip().startswith("{") else yaml.safe_load(text)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise StrategyError(f"could not parse: {exc}") from exc
    if not isinstance(raw, dict):
        raise StrategyError("config must be a mapping")
    cfg = parse_config(raw)
    build_strategy(cfg)
    return cfg


def resolve(ctx: Ctx, name: str) -> AnyConfig:
    if name in builtin_names():
        return load_config(name)
    with session_scope(ctx.session_factory) as s:
        try:
            return get_config(s, name)
        except StrategyError as exc:
            raise HTTPException(404, str(exc)) from exc


@router.get("", response_model=list[StrategySummary])
def list_strategies(ctx: Ctx) -> list[dict[str, Any]]:
    out = [_summary(load_config(n), "builtin") for n in builtin_names()]
    with session_scope(ctx.session_factory) as s:
        latest: dict[str, int] = {}
        for name, version, _ in list_configs(s):
            latest[name] = max(version, latest.get(name, 0))
        for name, version in latest.items():
            out.append(_summary(get_config(s, name, version), "registered"))
    return out


@router.get("/{name}", response_model=StrategyDetail)
def get_strategy(name: str, ctx: Ctx) -> dict[str, Any]:
    cfg = resolve(ctx, name)
    raw = cfg.model_dump(mode="json")
    source = "builtin" if name in builtin_names() else "registered"
    return {**_summary(cfg, source), "config": raw, "yaml": yaml.safe_dump(raw, sort_keys=False)}


@router.post("/validate", response_model=ValidateResult)
def validate(req: ValidateRequest) -> ValidateResult:
    try:
        cfg = parse_text(req.text)
    except StrategyError as exc:
        return ValidateResult(ok=False, error=str(exc))
    return ValidateResult(
        ok=True, name=cfg.name, config_hash=config_hash(cfg), config=cfg.model_dump(mode="json")
    )


@router.get("/{name}/signals", response_model=SignalsOut)
def signals(name: str, ctx: Ctx, as_of: Annotated[date | None, Query()] = None) -> SignalsOut:
    cfg = resolve(ctx, name)
    strategy = build_strategy(cfg)
    rng = ctx.equities.date_range()
    if rng is None:
        raise HTTPException(409, "no price data yet")
    day = as_of or rng[1]
    syms = required_symbols(strategy, ctx, BacktestConfig(start=day - timedelta(days=1)))
    md = MarketData.from_store(ctx, day - timedelta(days=800), day, syms)
    last = md.last_date_on_or_before(day)
    if last is None:
        raise HTTPException(404, f"no trading day on or before {day}")
    t = strategy.target(StrategyContext(pd.Timestamp(last), md, {}))
    return SignalsOut(
        strategy=cfg.name,
        as_of=last.date(),
        cash=t.cash,
        positions=[
            Position(symbol=s, weight=w, reasons=list(t.rationale.get(s, [])))
            for s, w in sorted(t.weights.items(), key=lambda kv: -kv[1])
        ],
        notes=list(t.notes),
    )
