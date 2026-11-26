# Swing trading, harvesting and goal planning (Phase 8)

## Swing trading (rules mode)
A strategy with `exits` runs in **rules mode**:
- It opens new positions only into free slots (`max_positions`, equal size).
- It never trims or rebalances existing positions.
- It holds each position until an exit rule fires. Rules are checked on every close, and the
  sell happens at the next open.
- A stock that was just exited can't be bought back for `cooldown_days` trading days (default 5).
  Otherwise a stop-out followed by an immediate re-buy would just churn.

```yaml
schedule: daily
max_positions: 10
cooldown_days: 5
exits:
  - {type: trailing_stop, atr_mult: 3.0}   # or pct: 0.15
  - {type: stop_loss, pct: 0.08}           # or atr_mult: 2
  - {type: take_profit, pct: 0.25}
  - {type: time_stop, days: 60}
  - {type: feature, feature: rsi, params: {window: 2}, op: ">", value: 70}
```
Built-ins: `swing_breakout` (52-week-high breakouts in uptrends) and `swing_pullback`
(RSI(2) < 10 dips in uptrends). Swing strategies need the full engine (`alphaengine backtest run`).
The fast sweep engine and paper portfolios reject them for now.

### After-tax comparison, 2012 to Nov 2026, ₹10 lakh start
| Strategy | After costs & tax | No costs or tax | Max DD | Turnover/yr | Trades |
|---|---|---|---|---|---|
| swing_breakout | 3.9% | 6.7% | −46% | 313% | 977 |
| swing_pullback | **−4.7%** | 9.8% | −59% | 3,365% | 10,024 |
| low_vol_momentum | **17.5%** | 20.7% | −32% | 103% | 1,935 |
| fixed_mix (ETFs) | 12.8% | — | −29% | 8% | 179 |
| buy_hold_nifty | 12.3% | — | −36% | 1% | 2 |

`swing_pullback`'s 10,024 trades cost about ₹9.9 lakh in charges and slippage, almost the
starting capital, plus about ₹1.8 lakh of short-term tax. That turns a 9.8% gross edge into a loss.
For a part-time investor in India, **costs and short-term tax make frequent trading a losing
game**. Keep swing strategies as research.

## Harvesting assistant
```bash
alphaengine portfolio harvest <name>     # also on each portfolio's dashboard page
```
- **Tax-loss harvesting:** sell positions that are under water to offset gains realised this FY,
  using the real set-off rules. With no gains to offset, the loss carries forward 8 years.
- **LTCG gain harvesting:** long-term gains up to **₹1.25 lakh per FY are tax-free**. Selling
  long-term winners up to the unused exemption and buying them back resets the cost basis.
- Each idea shows the tax change this FY and the round-trip costs. Ideas whose benefit doesn't
  beat the costs are hidden.
- India has no explicit wash-sale rule, but large sell-and-rebuy trades done only for tax can be
  questioned. Check with a tax professional.

## Goal planner
```bash
alphaengine plan goal --target 2cr --years 15 --current 10L --sip 50k --step-up 0.05
alphaengine plan goal --target 1cr --years 10 --sip 40k --run <backtest run id>
```
The planner runs 5,000 simulated futures built from 12-month blocks of real monthly returns: a
stored backtest's, or Nifty's total return. The target is in today's rupees and grown with
inflation. The output is the chance of reaching it, bad / typical / good outcomes after exit
tax, and the monthly SIP needed for your chosen confidence. The dashboard has a **Goal planner**
page with a chart of the range year by year.

Example from Nifty history: ₹2 Cr in today's money is ₹4.79 Cr in 15 years at 6% inflation.
Starting from ₹10 L with a ₹50k SIP (+5%/yr), the chance is **12%**. About **₹1.08 L/month** gives
80% confidence.
