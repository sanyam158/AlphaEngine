# Market model guide (Phase 2)

Costs and taxes are where low-turnover investing actually wins, so the backtester and paper
trader use these models for every simulated trade.

## Transaction costs: `alphaengine.market.costs`
```python
from datetime import date
from decimal import Decimal
from alphaengine.core.types import Side
from alphaengine.market.costs import CostModel, ZERODHA, FLAT_20

m = CostModel(ZERODHA)
c = m.charges(Side.BUY, 100, Decimal("1000"), date(2026, 1, 15))
c.total          # 118.74 -> STT 100, txn 3.07, SEBI 0.10, stamp 15.00, GST 0.57
c.tax_deductible # 18.74  -> everything except STT counts toward cost basis
```
- Rates are versioned by effective date (`STATUTORY_SCHEDULE`), so a 2012 backtest pays the
  0.125% STT of the time. The table at the top of `costs.py` lists every rate and its source.
- A delivery round trip costs about **0.24%** on Zerodha (STT dominates). At 12 rebalances a
  year with 50% turnover that's about 1.4% a year, which is why the strategies favour low turnover.
- `BrokerPlan` covers other brokers (percentage with cap, flat fee, DP charge).
- Not modelled: NSE IPFT (₹0.01/crore, negligible) and the ₹0.01 minimum per contract note.

## Fills: `alphaengine.market.execution_model`
- `FixedBpsSlippage`, or `SqrtImpactSlippage` (base + k·σ·√(order/ADV)). Prices move against
  you and are rounded to the ₹0.05 tick.
- `max_fill_quantity` caps a day's fill at a share of average volume (default 10%).
- `circuit_locked` blocks buys at an upper circuit and sells at a lower one (single-price bars).
- `settlement_date`: T+1 from 2023-01-27, T+2 before.

## Tax lots: `alphaengine.portfolio.lots`
- FIFO, as required for demat holdings. `LotBook.buy/sell` returns `Disposal`s with cost,
  proceeds, expenses and holding period.
- **Split**: lot quantities are multiplied, with the same cost and date. **Bonus**: new ₹0-cost
  lots dated on the allotment date. So bonus shares sold within a year are **short-term**.
- `fifo_long_term_quantity` and `next_long_term_date` support waiting a few days to sell at
  the LTCG rate instead of STCG.

## Capital-gains tax: `alphaengine.market.tax`
| Sale date | STCG | LTCG | Exemption/FY |
|---|---|---|---|
| before 2018-04-01 | 15% (10% before Apr 2008) | exempt | - |
| 2018-04-01 to 2024-07-22 | 15% | 10% | ₹1,00,000 |
| from 2024-07-23 | 20% | 12.5% | ₹1,25,000 |

- Grandfathering for shares bought on or before 2018-01-31, using
  `data.prices.grandfather_fmv_lookup` (the 31-Jan-2018 high, adjusted for later splits).
- Set-off: ST losses offset STCG then LTCG; LT losses only LTCG; highest rate first. Unused
  losses carry forward 8 years (`compute_tax_history` threads them through). Cess is 4%.
- Not modelled: surcharge, the unused-basic-exemption adjustment, and dividend income tax
  (dividends are taxed at your slab rate).

> These are planning estimates, not tax advice. Check them against your broker's capital-gains
> report.

## Bonus vs split classification
Yahoo reports bonuses as splits, so its share-count events are stored as **`SPLIT_OR_BONUS`**.
That's fine for prices, since both adjust the same way. For tax, classify the events that matter
to your holdings with a CSV (`alphaengine import actions file.csv`, kind `BONUS` or `SPLIT`). A
classified row replaces the ambiguous one. Until then, grandfathering is skipped for that
symbol, so tax errs high, and a warning is logged.
