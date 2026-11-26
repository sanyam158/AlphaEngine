# Backtest guide (Phase 4)

```bash
alphaengine backtest run momentum_top20 --start 2012-01-01
alphaengine backtest run core_satellite --start 2012-01-01 --capital 100000 --sip 25000
alphaengine backtest run my_strategy.yaml --no-costs --no-tax       # how much do frictions cost?
alphaengine backtest list
alphaengine backtest show <run_id>
alphaengine backtest compare <run_id> <run_id> ...
```
Each run is stored with the strategy config and hash, the backtest settings, the data snapshot id
and the git commit. Its equity curve, trades, targets, tax lots sold, `report.html` and
`metrics.json` go under `var/data/runs/<run_id>/`.

## What the simulation does
1. Splits, bonuses and dividends change your held shares or pay cash on the ex-date.
2. A monthly SIP is added on the first trading day of each month and invested by the latest
   target weights.
3. Orders decided at a close are filled at the **next day's open**, sells before buys:
   - with slippage that grows with order size relative to typical volume, and prices rounded to
     the ₹0.05 tick;
   - with each day's fill capped at 10% of 20-day average traded value;
   - with no buying at an upper circuit or selling at a lower one;
   - with Zerodha delivery charges (STT, exchange charges, SEBI fee, stamp duty, GST, DP charge).
4. Idle cash earns interest (6% a year by default), taxed at your slab rate (30% by default).
5. At each **financial-year end**, capital-gains tax on that year's sales (FIFO lots, STCG/LTCG,
   exemption, set-off, carry-forward) is paid from the portfolio. If cash runs short, holdings are
   sold pro rata.
6. Day one is always a rebalance, and then the strategy's own schedule applies.

## Reading the metrics
- **CAGR (time-weighted):** the strategy's own return. Contributions don't count as gains.
- **XIRR:** the money-weighted return your actual SIP cash flows earned.
- **After-tax XIRR:** as if you sold everything on the last day, after exit charges and tax.
  This is the number to compare.
- **Turnover:** one-way, as a share of the average portfolio per year. **Cost drag** and **tax drag**
  are costs and taxes as a share of the average portfolio per year.
- **Rolling 1/3/5-year CAGR:** the worst, median and best windows, which shows what a bad stretch
  could feel like.

## Known limitations
- Dividends are credited on the ex-date (in reality they're paid a few weeks later), and dividend
  tax is applied at a flat slab rate.
- For tax lots, Yahoo `SPLIT_OR_BONUS` events are treated as **bonuses**. That's conservative: it
  produces more short-term gains. Classify your holdings' events for exact results (see MARKET.md).
- No surcharge, no advance-tax timing, and no shorting, leverage or intraday trading.
- Survivorship: the liquidity universe includes delisted stocks, but a stock that left the
  bhavcopy universe (suspended) stays valued at its last price until it trades again.
- A backtest is a hypothesis test, not a forecast. Phase 5 adds walk-forward tests, parameter
  stability checks and overfitting checks before any of this deserves trust.
