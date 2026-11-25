"""Parameter sweeps: run every combination of a grid on one shared dataset.

Parameters are addressed by dotted paths into the strategy config, with list indices as numbers:
``selector.n``, ``scorer.params.lookback``, ``filters.1.value``, ``overlays.0.risk_off_exposure``,
``blend.0.weight``.
"""

from __future__ import annotations

import copy
import itertools
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from alphaengine.backtest.metrics import compute_metrics
from alphaengine.backtest.vectorized import VectorResult, run_vectorized
from alphaengine.core.errors import StrategyError
from alphaengine.core.logging import get_logger
from alphaengine.features.data import MarketData
from alphaengine.strategy.config import AnyConfig, build_strategy, parse_config

log = get_logger(__name__)

Grid = Mapping[str, Sequence[Any]]


def set_path(cfg: dict[str, Any], path: str, value: Any) -> dict[str, Any]:
    """Return a deep copy of ``cfg`` with ``path`` set to ``value`` (path must already exist,
    except the final key of a mapping, which may be new)."""
    out = copy.deepcopy(cfg)
    node: Any = out
    parts = path.split(".")
    for i, part in enumerate(parts):
        last = i == len(parts) - 1
        key: Any = int(part) if isinstance(node, list) else part
        if isinstance(node, list) and not 0 <= key < len(node):
            raise StrategyError(f"{path}: index {key} out of range")
        if isinstance(node, dict) and not last and key not in node:
            raise StrategyError(f"{path}: no key {key!r}")
        if last:
            node[key] = value
        else:
            node = node[key]
            if not isinstance(node, (dict, list)):
                raise StrategyError(f"{path}: {part!r} is not a mapping or list")
    return out


def get_path(cfg: Mapping[str, Any], path: str) -> Any:
    node: Any = cfg
    for part in path.split("."):
        node = node[int(part)] if isinstance(node, list) else node[part]
    return node


def expand_grid(grid: Grid) -> list[dict[str, Any]]:
    keys = list(grid)
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*grid.values())]


def variant(base: AnyConfig, params: Mapping[str, Any]) -> AnyConfig:
    raw = base.model_dump(mode="json")
    for path, value in params.items():
        raw = set_path(raw, path, value)
    label = "_".join(f"{p.split('.')[-1]}{v}" for p, v in params.items()) or "base"
    label = re.sub(r"[^a-z0-9_]", "", label.lower().replace(".", "p").replace("-", "m"))
    raw["name"] = f"{base.name}__{label}"[:64]
    return parse_config(raw)


@dataclass
class SweepResult:
    params: list[dict[str, Any]]
    results: list[VectorResult]
    metrics: pd.DataFrame  # one row per combo: params + metrics

    def returns_matrix(self) -> pd.DataFrame:
        """Daily returns, one column per combo (for PBO / deflated Sharpe)."""
        return pd.concat({i: r.returns for i, r in enumerate(self.results)}, axis=1).fillna(0.0)


def run_sweep(
    base: AnyConfig,
    grid: Grid,
    md: MarketData,
    start: date,
    end: date | None = None,
    *,
    rf: float = 0.06,
    on_progress: Callable[[int, int, dict[str, Any]], None] | None = None,
    **vector_kw: Any,
) -> SweepResult:
    combos = expand_grid(grid) or [{}]
    results, rows = [], []
    for n, params in enumerate(combos):
        strategy = build_strategy(variant(base, params))
        res = run_vectorized(strategy, md, start, end, cash_rate=rf, **vector_kw)
        m = compute_metrics(
            res.equity,
            rf=rf,
            totals={
                "costs": res.total_costs,
                "turnover": float(
                    (res.turnover * res.equity["value"].reindex(res.turnover.index)).sum()
                ),
            },
        )
        rows.append(
            {
                **params,
                **{
                    k: m[k]
                    for k in (
                        "cagr",
                        "volatility",
                        "sharpe",
                        "max_drawdown",
                        "calmar",
                        "turnover",
                        "cost_drag",
                    )
                },
            }
        )
        results.append(res)
        if on_progress:
            on_progress(n + 1, len(combos), params)
    return SweepResult(combos, results, pd.DataFrame(rows))


def stability(
    metrics: pd.DataFrame, params: Sequence[str], objective: str = "sharpe"
) -> dict[str, Any]:
    """How fragile is the best combo? Compares it with its grid neighbours (one step away on
    one parameter) and with the whole grid."""
    if metrics.empty:
        return {}
    vals = metrics[objective].astype(float).to_numpy()
    pos = int(vals.argmax())
    best_val = float(vals[pos])
    best = metrics.iloc[pos]
    levels = {p: sorted(metrics[p].unique().tolist()) for p in params}
    neighbours = []
    for k in range(len(metrics)):
        if k == pos:
            continue
        row = metrics.iloc[k]
        steps = [abs(levels[p].index(row[p]) - levels[p].index(best[p])) for p in params]
        if sum(steps) == 1:
            neighbours.append(float(vals[k]))
    return {
        "best_params": {p: best[p] for p in params},
        "best": best_val,
        "median": float(np.median(vals)),
        "worst": float(vals.min()),
        "neighbour_mean": float(np.mean(neighbours)) if neighbours else float("nan"),
        "share_within_80pct_of_best": float((vals >= 0.8 * best_val).mean())
        if best_val > 0
        else float("nan"),
    }
