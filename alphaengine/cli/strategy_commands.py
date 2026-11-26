"""CLI: ``alphaengine strategy list|validate|show|register|signals``."""

from __future__ import annotations

import csv
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
import yaml

from alphaengine.app import build_context
from alphaengine.core.errors import StrategyError
from alphaengine.data.db import session_scope
from alphaengine.features.data import MarketData
from alphaengine.strategy.config import (
    build_strategy,
    builtin_names,
    config_hash,
    load_config,
)
from alphaengine.strategy.core import Strategy, StrategyContext
from alphaengine.strategy.pipeline import Blend, PipelineStrategy
from alphaengine.strategy.registry import list_configs, register_config
from alphaengine.strategy.stages import StaticUniverse

strategy_app = typer.Typer(help="Create, inspect and run strategies.")

Source = Annotated[str, typer.Argument(help="Built-in name or path to a YAML file")]


def static_symbols(strategy: Strategy) -> set[str] | None:
    """Symbols a strategy can ever use, if its universes are all static (else ``None``)."""
    if isinstance(strategy, PipelineStrategy):
        u = strategy.universe
        return set(u.symbols) if isinstance(u, StaticUniverse) else None
    if isinstance(strategy, Blend):
        out: set[str] = set()
        for child, _ in strategy.parts:
            syms = static_symbols(child)
            if syms is None:
                return None
            out |= syms
        return out
    return None


@strategy_app.command("list")
def list_strategies() -> None:
    """Built-in strategies and configs registered in the database."""
    for name in builtin_names():
        cfg = load_config(name)
        typer.echo(f"{name:<20} v{cfg.version}  {cfg.description}")
    ctx = build_context()
    with session_scope(ctx.session_factory) as s:
        rows = list_configs(s)
    if rows:
        typer.echo("\nregistered:")
        for n, v, h in rows:
            typer.echo(f"  {n} v{v} ({h})")


@strategy_app.command("validate")
def validate(source: Source) -> None:
    """Parse and build a strategy config, reporting any errors."""
    try:
        cfg = load_config(source)
        build_strategy(cfg)
    except StrategyError as exc:
        typer.echo(f"invalid: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"ok: {cfg.name} v{cfg.version} hash={config_hash(cfg)}")


@strategy_app.command("show")
def show(source: Source) -> None:
    """Print the fully resolved config (with defaults) as YAML."""
    cfg = load_config(source)
    typer.echo(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))


@strategy_app.command("register")
def register(source: Source) -> None:
    """Store the config immutably (same name+version must keep the same content)."""
    cfg = load_config(source)
    build_strategy(cfg)
    ctx = build_context()
    with session_scope(ctx.session_factory) as s:
        rec = register_config(s, cfg)
        typer.echo(f"registered {rec.name} v{rec.version} ({rec.config_hash})")


def _read_current(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    with path.open(newline="", encoding="utf-8") as fh:
        return {r["symbol"].strip().upper(): float(r["weight"]) for r in csv.DictReader(fh)}


@strategy_app.command("signals")
def signals(
    source: Source,
    as_of: Annotated[str | None, typer.Option(help="YYYY-MM-DD (default: latest data)")] = None,
    current: Annotated[
        Path | None, typer.Option(help="CSV of current holdings: symbol,weight")
    ] = None,
    history_days: Annotated[int, typer.Option(help="Calendar days of history to load")] = 800,
) -> None:
    """Compute the target portfolio for a date, with the reasons for each position."""
    strategy = build_strategy(load_config(source))
    ctx = build_context()
    rng = ctx.equities.date_range()
    if rng is None:
        raise typer.BadParameter("no price data; run 'alphaengine ingest bhavcopy' first")
    target_day: date = (
        datetime.strptime(as_of, "%Y-%m-%d").date()  # noqa: DTZ007 - calendar date
        if as_of
        else rng[1]
    )
    syms = static_symbols(strategy)
    data = MarketData.from_store(
        ctx,
        target_day - timedelta(days=history_days),
        target_day,
        sorted(syms) if syms is not None else None,
    )
    day = data.last_date_on_or_before(target_day)
    if day is None:
        raise typer.BadParameter(f"no trading day on or before {target_day}")
    target = strategy.target(StrategyContext(pd.Timestamp(day), data, _read_current(current)))
    typer.echo(target.describe())
