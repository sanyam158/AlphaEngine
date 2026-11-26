"""Harvesting assistant: realise losses (to offset gains) or long-term gains (to use the yearly
exemption), then buy back to keep the same exposure.

- **Tax-loss harvesting:** selling lots that are under water this FY offsets gains already
  realised. Short-term losses offset STCG and LTCG; long-term losses offset only LTCG; unused
  losses carry forward 8 years.
- **LTCG gain harvesting:** listed-equity LTCG up to ₹1.25 lakh per FY is tax-free. Selling
  long-term winners up to the unused exemption and buying them back resets the cost basis
  higher, so future tax is lower, for nothing but transaction costs.

Each suggestion compares this FY's tax with and without the sale (using the real set-off
rules). Indian law has no explicit wash-sale rule, but sell-and-immediately-rebuy trades purely
for tax can be questioned under GAAR; check with a tax professional for large amounts.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from alphaengine.core.types import Side
from alphaengine.market.costs import CostModel
from alphaengine.market.tax import CarriedLoss, compute_fy_tax, financial_year, ltcg_exemption
from alphaengine.portfolio.lots import Disposal, LotBook


@dataclass(frozen=True)
class HarvestIdea:
    kind: str  # "loss" | "ltcg_gain"
    symbol: str
    quantity: int
    price: float
    gain: float  # realised gain (negative for losses), before costs
    long_term: bool
    tax_change: float  # this FY's tax with the sale minus without (negative = saving)
    round_trip_cost: float  # estimated charges to sell and buy back
    note: str

    @property
    def net_benefit(self) -> float:
        return -self.tax_change - self.round_trip_cost


def _tax(disposals: list[Disposal], fy: int, carried: list[CarriedLoss]) -> float:
    return float(compute_fy_tax(disposals, fy, carried).total_tax)


def harvest_ideas(
    lots: LotBook,
    prices: dict[str, float],
    on: date,
    realised: list[Disposal],
    carried: list[CarriedLoss] | None = None,
    costs: CostModel | None = None,
    min_benefit: float = 500.0,
) -> list[HarvestIdea]:
    cm = costs or CostModel()
    fy = financial_year(on)
    this_fy = [d for d in realised if financial_year(d.sold) == fy]
    carried = carried or []
    base_tax = _tax(this_fy, fy, carried)
    ideas: list[HarvestIdea] = []
    used_exemption = float(compute_fy_tax(this_fy, fy, carried).exemption_used)
    room = float(ltcg_exemption(fy)) - used_exemption

    for sym in lots.symbols():
        px = prices.get(sym)
        if not px:
            continue
        qty = lots.quantity(sym)
        price = Decimal(str(px))
        sell = cm.charges(Side.SELL, qty, price, on)
        buy = cm.charges(Side.BUY, qty, price, on)
        rt_cost = float(sell.total + buy.total)
        preview = lots.preview_sell(sym, qty, price, on, sell)
        gain = float(sum((d.gain for d in preview), Decimal(0)))
        losses = [d for d in preview if d.gain < 0]
        if losses and gain < 0:
            change = _tax(this_fy + preview, fy, carried) - base_tax
            idea = HarvestIdea(
                "loss",
                sym,
                qty,
                px,
                gain,
                all(d.long_term for d in preview),
                change,
                rt_cost,
                "offsets gains realised this FY"
                if change < 0
                else "no gains to offset this FY: the loss carries forward 8 years",
            )
            if idea.net_benefit >= min_benefit or (change == 0 and abs(gain) >= 10 * min_benefit):
                ideas.append(idea)
        # LTCG harvesting: sell only the long-term lots, up to the unused exemption.
        lt_qty = lots.fifo_long_term_quantity(sym, on)
        if room > 0 and lt_qty > 0:
            lt_preview = lots.preview_sell(
                sym, lt_qty, price, on, cm.charges(Side.SELL, lt_qty, price, on)
            )
            lt_gain = float(sum((d.gain for d in lt_preview), Decimal(0)))
            if lt_gain > 0:
                # scale down to fit the remaining exemption
                q = lt_qty if lt_gain <= room else max(int(lt_qty * room / lt_gain), 0)
                if q > 0:
                    sell_q = cm.charges(Side.SELL, q, price, on)
                    part = lots.preview_sell(sym, q, price, on, sell_q)
                    g = float(sum((d.gain for d in part), Decimal(0)))
                    rt = float(
                        cm.charges(Side.SELL, q, price, on).total
                        + cm.charges(Side.BUY, q, price, on).total
                    )
                    future_saving = g * 0.125  # tax avoided later on this gain at today's LTCG rate
                    if future_saving - rt >= min_benefit:
                        ideas.append(
                            HarvestIdea(
                                "ltcg_gain",
                                sym,
                                q,
                                px,
                                g,
                                True,
                                0.0,
                                rt,
                                f"uses ₹{g:,.0f} of the ₹{ltcg_exemption(fy):,.0f} LTCG exemption; "
                                f"buy back to reset cost basis (saves ~₹{future_saving:,.0f})",
                            )
                        )
                        room -= g
    return sorted(ideas, key=lambda x: -x.net_benefit if x.kind == "loss" else -x.gain)
