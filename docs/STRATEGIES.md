# Strategy engine guide (Phase 3)

A strategy turns market data at the close of a day into **target weights** plus the reasons for
each one. The same strategy object will drive the backtester (Phase 4) and paper trading
(Phase 6).

> Built-in strategies are research hypotheses to backtest, not investment advice.

## Try it
```bash
alphaengine strategy list
alphaengine strategy signals dual_momentum                 # latest close
alphaengine strategy signals momentum_top20 --as-of 2020-03-31
alphaengine strategy signals momentum_top20 --current holdings.csv   # symbol,weight (enables hysteresis)
alphaengine strategy validate my_strategy.yaml
alphaengine strategy show momentum_top20                   # resolved config with defaults
alphaengine strategy register my_strategy.yaml             # immutable store; bump version to change
```

## Built-ins
| Name | Idea | Trades |
|---|---|---|
| `buy_hold_nifty` | 100% NIFTYBEES | never |
| `fixed_mix` | 60/20/20 Nifty 50 / Next 50 / Gold ETFs | quarterly rebalance |
| `dual_momentum` | strongest of Nifty/Next 50/Bank/Gold ETFs by 12-mo return; cash if none beats ~6.5% | ~monthly, usually no change |
| `momentum_top20` | top 20 of the 200 most liquid stocks by risk-adjusted 12-1 momentum, hold while in the top 40, half exposure below Nifty's 200-day average | monthly |
| `low_vol_momentum` | momentum + low volatility, inverse-vol weights, sector cap 30% | quarterly |
| `core_satellite` | 70% `dual_momentum` + 30% `momentum_top20` | monthly |

## Writing a strategy
```yaml
name: my_trend            # lowercase, digits, underscores
version: 1                # bump whenever you change anything
description: ...
schedule: monthly         # daily | weekly | monthly | quarterly | yearly (last trading day)
universe: {type: liquidity, top_n: 200}
filters:
  - {type: min_price, value: 20}
  - {type: feature, feature: price_to_sma, params: {window: 200}, op: ">", value: 0}
scorer:
  type: composite          # sum of winsorised cross-sectional z-scores
  factors:
    - {feature: momentum, params: {lookback: 252, skip: 21}, weight: 1}
    - {feature: volatility, params: {window: 126}, weight: 0.5, higher_is_better: false}
selector: {type: top_n, n: 15, buffer: 30}
weighter: {type: inverse_vol, window: 63}
overlays:
  - {type: regime, benchmark: NIFTY50, sma: 200, risk_off_exposure: 0.3}
  - {type: max_weight, cap: 0.1}
```
Blends: `blend: [{strategy: dual_momentum, weight: 0.6}, {strategy: ./my_trend.yaml, weight: 0.4}]`.

### Components
- **universe**: `static {symbols}`, `liquidity {top_n, window, exclude, asset_class: EQUITY|ETF|ALL}`, `index {index}`
- **filter**: `min_price {value}` (raw ₹), `min_history {days}`, `min_adv {value, window}`, `feature {feature, op, value, params}`
- **scorer**: `feature {feature, params, higher_is_better}`, `composite {factors}`
- **selector**: `top_n {n, buffer, min_score}`, `all {min_score}`
- **weighter**: `equal`, `fixed {weights_by_symbol}`, `inverse_vol {window}`
- **overlay**: `max_weight {cap}`, `regime {benchmark, sma, risk_off_exposure}`, `vol_target {target, window}`, `sector_cap {cap}`

Unallocated weight is **cash**. Phase 4 will credit it with interest.

### Features
`returns`, `momentum`, `risk_adj_momentum`, `volatility`, `sma`, `price_to_sma`, `high_proximity`,
`adv_value`, `history_days`, `rsi`, `atr` (needs OHLC), `zscore`, `max_abs_return`.
Add one with `@feature("name", history=lambda p: ..., param=default)` in
`alphaengine/features/library.py`, plus a `SMALL` entry in `tests/unit/test_features.py`. Every
feature must pass the look-ahead tests: values at day *t* must not change when later data is
removed or shuffled.

## Guardrails built in
- **No look-ahead:** each feature and an end-to-end strategy are checked against truncated history.
- **Survivorship-free universe:** `liquidity` ranks what actually traded, delisted stocks included.
- **ETFs are kept out of stock universes:** ISINs starting with `INF` are ETFs or fund units.
- **Bad-data filter:** `max_abs_return < 0.35` drops stocks with an unexplained >35% one-day move,
  which is almost always a missing split record.
- **Turnover control:** the `top_n` buffer means a holding is sold only when it drops well down
  the ranking.

## Limitations (planned work)
- **No fundamentals yet:** value and quality factors (P/E, ROCE, debt) need a fundamentals source.
- **Blend approximation:** a blend splits current holdings among its children pro rata, which is
  approximate when two children hold the same symbol.
- **Index universes:** these need membership history. NSE snapshots only start from your first
  `ingest index-members` run.
