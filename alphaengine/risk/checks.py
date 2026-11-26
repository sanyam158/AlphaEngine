"""Pre-trade and portfolio risk checks. Findings are advisory (``warning``) or blocking
(``error``); the planner's trades are only submitted when there are no errors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from alphaengine.portfolio.planner import Plan


@dataclass(frozen=True)
class RiskLimits:
    max_position_weight: float = 0.25  # single stocks; ETFs are exempt
    max_adv_participation: float = 0.10  # of 20-day median traded value per order
    max_turnover: float = 1.5  # one plan may trade at most 150% of the portfolio (buys+sells)
    drawdown_alert: float = 0.15


@dataclass(frozen=True)
class Finding:
    level: str  # warning | error
    message: str


def check_plan(
    plan: Plan,
    target: dict[str, float],
    adv_value: dict[str, float],
    limits: RiskLimits | None = None,
    diversified: set[str] | frozenset[str] = frozenset(),
) -> list[Finding]:
    """``diversified`` symbols (index ETFs) are exempt from the single-position cap."""
    lim = limits or RiskLimits()
    out: list[Finding] = []
    for sym, w in target.items():
        if sym not in diversified and w > lim.max_position_weight + 1e-9:
            out.append(
                Finding(
                    "error",
                    f"{sym} target {w:.0%} exceeds max position {lim.max_position_weight:.0%}",
                )
            )
    for t in plan.trades:
        adv = adv_value.get(t.symbol)
        if adv and t.value > lim.max_adv_participation * adv:
            out.append(
                Finding(
                    "warning",
                    f"{t.side.value} {t.symbol} ₹{t.value:,.0f} is "
                    f"{t.value / adv:.0%} of daily traded value; expect slippage",
                )
            )
    traded = sum(t.value for t in plan.trades)
    if plan.total_value > 0 and traded / plan.total_value > lim.max_turnover:
        out.append(
            Finding(
                "error",
                f"plan trades {traded / plan.total_value:.0%} of the portfolio "
                f"(limit {lim.max_turnover:.0%})",
            )
        )
    return out


def drawdown_alert(values: Sequence[float], limits: RiskLimits | None = None) -> Finding | None:
    lim = limits or RiskLimits()
    if len(values) < 2:
        return None
    peak, last = max(values), values[-1]
    dd = last / peak - 1 if peak > 0 else 0.0
    if dd <= -lim.drawdown_alert:
        return Finding("warning", f"portfolio is {dd:.1%} below its peak")
    return None
