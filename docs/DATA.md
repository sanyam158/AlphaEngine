# Data layer guide (Phase 1)

## Where data comes from
| Data | Source | Notes |
|---|---|---|
| Daily OHLCV, all NSE stocks & ETFs | NSE cash-market **bhavcopy** | Official, free, includes delisted stocks (survivorship-free). Legacy format until 2024-07-05, UDiFF from 2024-07-08; both handled. Series `EQ`,`BE`; rights entitlements (`*-RE`) excluded. |
| Index levels (NIFTY50, NIFTYBANK, NIFTY500, SENSEX) | Yahoo (`.[yahoo]` extra) | Price index, not TRI. |
| Splits / bonuses / dividends | Yahoo, or your CSV | Yahoo reports bonuses as splits; dividends are converted back to raw ₹/share. |
| Index constituents | NSE `ind_<index>list.csv` | Current list only; snapshots build point-in-time history going forward. Also stores industry. |

## Storage
- `var/data/raw/<source>/<yyyy>/…`: immutable downloaded files. Re-ingests read from here, not the network.
- `var/data/curated/<dataset>/year=YYYY/data.parquet`: raw (unadjusted) bars, queried via DuckDB.
  A bhavcopy re-ingest replaces whole days.
- Database: symbol master (ISIN-keyed, with rename history), corporate actions, index membership,
  ingest log, data-quality issues.

## Everyday commands
```bash
alphaengine db upgrade
alphaengine ingest bhavcopy --start 2010-01-01           # resumable; rerun any time to catch up
alphaengine ingest index-members NIFTY50 NIFTY200        # run monthly to build PIT membership
alphaengine ingest index-prices NIFTY50 NIFTY500 --start 2010-01-01
alphaengine ingest actions --index NIFTY200              # or list symbols
alphaengine import actions my_actions.csv                # symbol,ex_date,kind,ratio,amount
alphaengine import index-members history.csv             # index_name,symbol,start_date,end_date
alphaengine data quality
alphaengine data info
```
A full backfill from 2010 is about 4,000 requests. At the polite default of about 3 requests/second
(slower in practice), allow 30–60 minutes. It stops by itself after 5 failures in a row, which
usually means NSE is throttling; rerun later and it resumes where it stopped.

## Using prices in research
```python
from alphaengine.app import build_context
from alphaengine.data.prices import load_prices, to_panel

ctx = build_context()
bars = load_prices(ctx.equities, ctx.session_factory, ["RELIANCE", "TCS"], adjust="total")
close = to_panel(bars)  # date x symbol
```
- `adjust="split"` handles splits and bonuses, for signals. `"total"` also adds dividends back, for
  returns. `"none"` gives raw prices, for absolute price rules like "price > ₹50".
- Renamed companies come back as one continuous series under the current symbol.
- `data.universe.liquidity_universe(bars, as_of, top_n=200)` gives a survivorship-free
  "top-N by traded value" universe, which is the default research universe.

## Quality warnings and what to do
- `price_spike` / `prev_close_mismatch` usually mean a split, bonus or demerger that isn't
  recorded. Run `ingest actions SYMBOL`, or add it to a CSV and `import actions`.
- `missing_days`: the stock was suspended or moved to an unsupported series. Usually harmless.
- `low_coverage` (error): a day's file looks truncated. Re-ingest that day with `--force`.

## Known gaps (planned)
- Demergers (e.g. Raymond, July 2024) aren't modelled as an action type yet. Add them as a
  `SPLIT` with the price ratio as a stopgap.
- Index membership before your first snapshot needs a CSV import. Until then, use the liquidity
  universe for backtests.
- Nifty TRI (total-return index) isn't available from Yahoo. NSE's index API will be added later.
- NSE holiday calendar: future holidays come from a CSV you maintain (`date,kind,description`)
  from NSE's annual circular. Past trading days come from the observed bhavcopies.
