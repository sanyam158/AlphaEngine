from __future__ import annotations

from datetime import date

from fastapi import APIRouter, HTTPException

from alphaengine.api.deps import Ctx
from alphaengine.api.schemas import GoalBand, GoalOut, GoalRequest
from alphaengine.core.errors import AlphaEngineError
from alphaengine.planning.goal import Goal, plan_goal
from alphaengine.planning.sources import history_for

router = APIRouter(prefix="/plan", tags=["planning"])


@router.post("/goal", response_model=GoalOut)
def goal(req: GoalRequest, ctx: Ctx) -> GoalOut:
    try:
        hist, source = history_for(ctx, run=req.run_id, symbol=req.symbol, start=date(2010, 1, 1))
        g = Goal(
            target=req.target,
            years=req.years,
            current=req.current,
            monthly_sip=req.monthly_sip,
            step_up=req.step_up,
            inflation=req.inflation,
            after_tax=req.after_tax,
        )
    except (ValueError, AlphaEngineError) as exc:
        raise HTTPException(422, str(exc)) from exc
    r = plan_goal(g, hist, success=req.success, source=source)
    return GoalOut(
        probability=r.probability,
        target_nominal=r.target_nominal,
        contributed=r.contributed,
        required_sip=r.required_sip,
        source=source,
        history_months=len(hist),
        bands=[GoalBand(**b) for b in r.yearly_bands],
        **r.final_percentiles,
    )
