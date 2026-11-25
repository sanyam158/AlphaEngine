# Robustness guide (Phase 5)

A good backtest is easy to make by accident. This phase answers whether a strategy's edge is
real, stable and big enough to survive costs.

```bash
alphaengine robust run momentum_top20 --start 2012-01-01 \
    --grid selector.n=10,20,30 --grid scorer.params.lookback=126,252
```
Grid paths point into the strategy config: `selector.n`, `scorer.params.lookback`,
`filters.1.value`, `overlays.0.risk_off_exposure`, `scorer.factors.1.weight`,
`blend.0.weight`. Your config's own value is always added to the grid. Output goes to
`var/data/robustness/<id>/robustness.html` and `robustness.json`.

## What it runs
| Check | Question it answers | Comfortable when |
|---|---|---|
| **Parameter heatmap / stability** | Do nearby settings work about as well, or is it one lucky cell? | broad plateau; median ≥ half the best |
| **Deflated Sharpe** (Bailey & López de Prado) | Is the Sharpe significant, given fat tails and the number of variants tried? | ≥ 95% |
| **PBO** (CSCV) | How often is the in-sample winner below median out of sample? | < ~30% (50% = noise) |
| **Walk-forward** | If parameters were picked using only the past, what happened next? | OOS Sharpe ≥ half IS Sharpe |
| **Stationary bootstrap** | What range of CAGR and drawdown is plausible? | 5th-percentile CAGR > 0 |
| **Regimes** | How does it behave when Nifty is above vs below its 200-day average? | no hidden collapse |
| **Costs ×0 / ×1 / ×2** | Does the edge survive higher frictions? | 2× still beats the benchmark |
| **Full-engine confirmation** | After integer shares, fills and **taxes**? | after-tax XIRR > benchmark |

The report ends with plain-language **warnings**. No warnings is necessary, not sufficient.

## Engines
- `backtest.vectorized.run_vectorized` works on weights with the same close→next-open timing,
  turnover-based costs and no taxes. It's about 5× faster and used for sweeps. A parity test keeps
  it within 1% of the full engine without frictions, with identical targets.
- `backtest.engine.simulate` is the full day-by-day engine (shares, lots, taxes, fills), used for
  confirmation and for anything you'd actually trade.

## Good practice
1. Decide the grid **before** looking at results, and keep it small. Every extra variant lowers the
   deflated Sharpe.
2. Choose parameters from the middle of a stable region, not the single best cell.
3. Hold out recent years: run the analysis to 2021, freeze the choice, then look at 2022 onward
   once (`--end`).
4. Change a strategy, bump its `version`, and re-run. Runs record config hashes.
