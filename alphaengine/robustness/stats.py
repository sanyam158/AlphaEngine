"""Statistics for "is this backtest real?".

- **Probabilistic / Deflated Sharpe ratio** (Bailey & López de Prado, 2012/2014): the chance the
  true Sharpe exceeds a benchmark, allowing for short samples, fat tails, and, for the deflated
  version, how many variants were tried before picking this one.
- **PBO** (probability of backtest overfitting, via combinatorially symmetric cross-validation):
  how often the variant that looks best in-sample ranks below median out-of-sample.
- **Stationary bootstrap** (Politis & Romano) of daily returns: ranges for CAGR and drawdown.
- **Regime split**: performance when the benchmark is above vs below its 200-day average.
"""

from __future__ import annotations

import itertools
import math
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

TRADING_DAYS = 252
EULER_GAMMA = 0.5772156649015329
_N = NormalDist()


def _moments(r: pd.Series) -> tuple[float, float, float, int]:
    x = r.dropna().to_numpy(dtype=float)
    sd = x.std(ddof=1)
    if len(x) < 3 or sd == 0:
        return float("nan"), 0.0, 3.0, len(x)
    sr = x.mean() / sd
    z = (x - x.mean()) / x.std(ddof=0)
    return float(sr), float((z**3).mean()), float((z**4).mean()), len(x)


def annualise(r: pd.Series) -> float:
    """Geometric annual return of daily returns ``r``."""
    x = r.to_numpy(dtype=float)
    if len(x) == 0:
        return float("nan")
    return float(np.prod(1 + x) ** (TRADING_DAYS / len(x)) - 1)


def probabilistic_sharpe(r: pd.Series, sr_benchmark: float = 0.0) -> float:
    """P(true per-period Sharpe > ``sr_benchmark``) given skew/kurtosis of ``r`` (daily)."""
    sr, skew, kurt, n = _moments(r)
    if not math.isfinite(sr) or n < 3:
        return float("nan")
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr**2
    if denom <= 0:
        return float("nan")
    return _N.cdf((sr - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(denom))


def expected_max_sharpe(sharpes: np.ndarray) -> float:
    """Expected maximum of N per-period Sharpe ratios under the null of zero true Sharpe."""
    n = len(sharpes)
    if n < 2:
        return 0.0
    var = float(np.var(sharpes, ddof=1))
    return math.sqrt(var) * (
        (1 - EULER_GAMMA) * _N.inv_cdf(1 - 1 / n) + EULER_GAMMA * _N.inv_cdf(1 - 1 / (n * math.e))
    )


def deflated_sharpe(r: pd.Series, trial_returns: pd.DataFrame) -> dict[str, float]:
    """DSR of ``r`` given all the variants tried (columns of ``trial_returns``)."""
    trial_sr = np.array([_moments(trial_returns[c])[0] for c in trial_returns.columns], dtype=float)
    trial_sr = trial_sr[np.isfinite(trial_sr)]
    sr0 = expected_max_sharpe(trial_sr)
    return {
        "trials": len(trial_sr),
        "sharpe_annual": _moments(r)[0] * math.sqrt(TRADING_DAYS),
        "expected_max_sharpe_annual": sr0 * math.sqrt(TRADING_DAYS),
        "psr_vs_zero": probabilistic_sharpe(r, 0.0),
        "deflated_sharpe": probabilistic_sharpe(r, sr0),
    }


def pbo(returns: pd.DataFrame, n_blocks: int = 10, metric: str = "sharpe") -> dict[str, Any]:
    """Probability of backtest overfitting via CSCV. ``returns``: T x N (one column per variant).
    Splits time into ``n_blocks`` blocks; for every half/half split, picks the best variant
    in-sample and records its out-of-sample rank."""
    m = returns.dropna(how="all").fillna(0.0).to_numpy(dtype=float)
    t, n = m.shape
    if n < 2 or t < n_blocks * 5:
        return {"pbo": float("nan"), "splits": 0}
    n_blocks -= n_blocks % 2
    blocks = np.array_split(np.arange(t), n_blocks)

    def score(x: np.ndarray) -> np.ndarray:
        if metric == "sharpe":
            sd = x.std(axis=0, ddof=1)
            out: np.ndarray = np.where(sd > 0, x.mean(axis=0) / np.where(sd > 0, sd, 1), -np.inf)
            return out
        mean: np.ndarray = x.mean(axis=0)
        return mean

    logits = []
    for is_blocks in itertools.combinations(range(n_blocks), n_blocks // 2):
        is_idx = np.concatenate([blocks[b] for b in is_blocks])
        oos_idx = np.concatenate([blocks[b] for b in range(n_blocks) if b not in is_blocks])
        best = int(np.argmax(score(m[is_idx])))
        oos = score(m[oos_idx])
        rank = (oos < oos[best]).sum() + 0.5 * ((oos == oos[best]).sum() - 1)
        w = (rank + 0.5) / n  # relative rank in (0, 1)
        logits.append(math.log(w / (1 - w)))
    arr = np.array(logits)
    return {
        "pbo": float((arr <= 0).mean()),
        "splits": len(arr),
        "median_logit": float(np.median(arr)),
    }


def _stationary_indices(n: int, p: float, rng: np.random.Generator) -> np.ndarray:
    """Indices for one stationary-bootstrap path: blocks of geometric length (mean 1/p) starting
    at random points, wrapping around the end."""
    jumps = rng.random(n) < p
    jumps[0] = True
    seg = np.cumsum(jumps) - 1  # block id for each position
    block_starts = rng.integers(0, n, size=int(seg[-1]) + 1)
    first_pos = np.flatnonzero(jumps)  # position where each block begins
    offset = np.arange(n) - first_pos[seg]
    idx: np.ndarray = (block_starts[seg] + offset) % n
    return idx


def stationary_bootstrap(
    r: pd.Series, n_paths: int = 1000, mean_block: int = 20, seed: int = 0
) -> dict[str, dict[str, float]]:
    """Percentiles of CAGR, max drawdown and Sharpe over resampled return paths."""
    x = r.dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 30:
        return {}
    rng = np.random.default_rng(seed)
    p = 1.0 / mean_block
    cagr, mdd, shp = np.empty(n_paths), np.empty(n_paths), np.empty(n_paths)
    for k in range(n_paths):
        path = x[_stationary_indices(n, p, rng)]
        growth = np.cumprod(1 + path)
        cagr[k] = growth[-1] ** (TRADING_DAYS / n) - 1
        mdd[k] = float((growth / np.maximum.accumulate(growth) - 1).min())
        sd = path.std(ddof=1)
        shp[k] = path.mean() / sd * math.sqrt(TRADING_DAYS) if sd > 0 else np.nan

    def pct(a: np.ndarray) -> dict[str, float]:
        q = np.nanpercentile(a, [5, 50, 95])
        return {"p5": float(q[0]), "p50": float(q[1]), "p95": float(q[2])}

    return {
        "cagr": pct(cagr),
        "max_drawdown": pct(mdd),
        "sharpe": pct(shp),
        "prob_cagr_below_zero": {"value": float((cagr < 0).mean())},
    }


def regime_split(r: pd.Series, benchmark: pd.Series, sma: int = 200) -> dict[str, dict[str, float]]:
    """Annualised return/vol of ``r`` when the benchmark was above vs below its moving average
    (decided on the previous day's close, so no look-ahead)."""
    b = benchmark.reindex(r.index).ffill()
    up = (b > b.rolling(sma, min_periods=sma).mean()).shift(1)
    out = {}
    for name, mask in (("benchmark_uptrend", up == True), ("benchmark_downtrend", up == False)):  # noqa: E712
        sub = r[mask.fillna(False).astype(bool)]
        if len(sub) < 20:
            continue
        bs = b.pct_change(fill_method=None).reindex(sub.index)
        out[name] = {
            "days": float(len(sub)),
            "return_annual": annualise(sub),
            "benchmark_annual": annualise(bs.fillna(0.0)),
            "volatility": float(sub.std(ddof=1) * math.sqrt(TRADING_DAYS)),
        }
    return out


def walk_forward(
    returns: pd.DataFrame, n_folds: int = 5, objective: str = "sharpe", min_train_days: int = 504
) -> dict[str, Any]:
    """Anchored walk-forward on precomputed variant returns (T x N): for each fold, pick the
    variant with the best ``objective`` on all data so far, then record its next-period returns.
    Returns stitched out-of-sample returns and the in-sample vs out-of-sample Sharpe."""
    m = returns.dropna(how="all").fillna(0.0)
    t = len(m)
    test_len = (t - min_train_days) // n_folds
    if test_len < 40:
        return {}
    oos_parts, folds = [], []

    def sharpe(x: pd.Series) -> float:
        sd = x.std(ddof=1)
        return float(x.mean() / sd * math.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")

    for k in range(n_folds):
        train_end = min_train_days + k * test_len
        train, test = m.iloc[:train_end], m.iloc[train_end : train_end + test_len]
        scores = train.apply(sharpe) if objective == "sharpe" else train.mean()
        best = scores.idxmax()
        oos_parts.append(test[best])
        folds.append(
            {
                "train_end": str(train.index[-1].date()),
                "test_end": str(test.index[-1].date()),
                "chosen": int(best) if isinstance(best, (int, np.integer)) else str(best),
                "in_sample_sharpe": float(scores.max()),
                "out_of_sample_sharpe": sharpe(test[best]),
            }
        )
    oos = pd.concat(oos_parts)
    is_mean = float(np.nanmean([f["in_sample_sharpe"] for f in folds]))
    oos_sharpe = sharpe(oos)
    return {
        "folds": folds,
        "oos_returns": oos,
        "oos_sharpe": oos_sharpe,
        "oos_cagr": float((1 + oos).prod() ** (TRADING_DAYS / len(oos)) - 1),
        "mean_in_sample_sharpe": is_mean,
        "efficiency": oos_sharpe / is_mean if is_mean and math.isfinite(is_mean) else float("nan"),
    }
