"""Goal planner: how likely is a corpus target, and what SIP does it take?

Monte Carlo over monthly returns resampled in 12-month blocks from real history (a backtest
run, or an index/ETF's total-return series), so crashes and recoveries keep their sequence.
Targets are in **today's rupees** and grown with inflation. Contributions are made at the start
of each month and can step up every year. An optional exit tax charges LTCG (12.5% + 4% cess)
on gains above contributions at the end, which is a simple approximation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

LTCG_EFFECTIVE = 0.125 * 1.04


@dataclass(frozen=True)
class Goal:
    target: float  # in today's rupees
    years: int
    current: float = 0.0
    monthly_sip: float = 0.0
    step_up: float = 0.0  # yearly increase in the SIP, e.g. 0.10
    inflation: float = 0.06
    after_tax: bool = True

    def __post_init__(self) -> None:
        if self.years < 1 or self.years > 60:
            raise ValueError("years must be between 1 and 60")
        if self.target <= 0 or self.current < 0 or self.monthly_sip < 0:
            raise ValueError("target must be > 0; current and SIP must be >= 0")


@dataclass
class GoalResult:
    probability: float
    target_nominal: float
    final_percentiles: dict[str, float]  # p10/p50/p90, nominal ₹
    contributed: float
    yearly_bands: list[dict[str, float]] = field(default_factory=list)  # year, p10, p50, p90
    required_sip: float | None = None  # for the requested success probability
    source: str = ""


def monthly_returns(prices: pd.Series) -> np.ndarray:
    """Month-end to month-end returns of a (total-return) price or equity series."""
    s = prices.dropna()
    s.index = pd.DatetimeIndex(s.index)
    m = s.groupby(s.index.to_period("M")).last()
    r = m.pct_change().dropna().to_numpy(dtype=float)
    if len(r) < 36:
        raise ValueError("need at least 3 years of history to resample")
    return r


def _paths(
    hist: np.ndarray, months: int, n: int, rng: np.random.Generator, block: int = 12
) -> np.ndarray:
    """n x months matrix of resampled monthly returns (circular 12-month blocks)."""
    n_blocks = -(-months // block)
    starts = rng.integers(0, len(hist), size=(n, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % len(hist)
    out: np.ndarray = hist[idx].reshape(n, n_blocks * block)[:, :months]
    return out


def _final_values(
    goal: Goal, returns: np.ndarray, sip: float
) -> tuple[np.ndarray, np.ndarray, float]:
    n, months = returns.shape
    value = np.full(n, goal.current, dtype=float)
    contributed = goal.current
    yearly = np.empty((goal.years, n))
    for m in range(months):
        contrib = sip * (1 + goal.step_up) ** (m // 12)
        value = (value + contrib) * (1 + returns[:, m])
        contributed += contrib
        if (m + 1) % 12 == 0:
            yearly[(m + 1) // 12 - 1] = value
    if goal.after_tax:
        gain = np.maximum(value - contributed, 0.0)
        value = value - gain * LTCG_EFFECTIVE
    return value, yearly, contributed


def _band(year: int, q: np.ndarray) -> dict[str, float]:
    return {"year": year, "p10": float(q[0]), "p50": float(q[1]), "p90": float(q[2])}


def plan_goal(
    goal: Goal,
    history: np.ndarray,
    *,
    n_paths: int = 5000,
    seed: int = 0,
    success: float = 0.8,
    source: str = "",
) -> GoalResult:
    rng = np.random.default_rng(seed)
    months = goal.years * 12
    rets = _paths(history, months, n_paths, rng)  # common random numbers for the SIP search
    target_nominal = goal.target * (1 + goal.inflation) ** goal.years
    final, yearly, contributed = _final_values(goal, rets, goal.monthly_sip)
    prob = float((final >= target_nominal).mean())

    def prob_for(sip: float) -> float:
        return float((_final_values(goal, rets, sip)[0] >= target_nominal).mean())

    required: float | None = None
    hi = max(goal.monthly_sip, 1_000.0)
    while prob_for(hi) < success and hi < 1e8:
        hi *= 2
    if hi < 1e8:
        lo = 0.0
        if prob_for(lo) >= success:
            required = 0.0
        else:
            for _ in range(40):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if prob_for(mid) < success else (lo, mid)
            required = float(np.ceil(hi / 100) * 100)
    q = np.percentile(final, [10, 50, 90])
    bands = [
        {
            "year": y + 1,
            **dict(
                zip(
                    ("p10", "p50", "p90"),
                    map(float, np.percentile(yearly[y], [10, 50, 90])),
                    strict=True,
                )
            ),
        }
        for y in range(goal.years)
    ]
    return GoalResult(
        probability=prob,
        target_nominal=float(target_nominal),
        final_percentiles={"p10": float(q[0]), "p50": float(q[1]), "p90": float(q[2])},
        contributed=float(contributed),
        yearly_bands=bands,
        required_sip=required,
        source=source,
    )
