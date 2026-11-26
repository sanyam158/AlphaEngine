# AlphaEngine — Master Plan

## Context
You're a software engineer who wants to learn investing and build a personal system that grows net worth **with as little trading and time as possible**, not as a full-time trader. The repo `AlphaEngine` is empty (README only). (An old memory note describes earlier foundation code. It isn't in the repo, so we start fresh.)

Decisions so far:
- **Market:** India, NSE/BSE (equities + ETFs), IST, Indian costs and taxes.
- **Execution:** signals + paper trading. The system produces rebalance lists and tracks a paper portfolio, and **you place the real orders**. A broker adapter (Zerodha Kite / Upstox) comes later behind an interface.
- **Styles:** monthly factor rotation, trend-following allocation, core + satellite, swing trading. All four are supported by one strategy framework. The low-turnover styles come first, and swing trading comes last.
- **Stack:** Python 3.12 core, FastAPI service, React dashboard, Postgres + Parquet/DuckDB.

> Note: This plan covers building research and decision-support software. Strategies listed here are hypotheses to backtest, not investment advice. I'm not a licensed advisor. Validate everything yourself, and paper trade before you commit real capital.

---

## 1. Guiding principles (how the system makes money *and* saves time)
1. **Low turnover by design.** Monthly or quarterly rebalances, no-trade bands, and tax-aware selling. In India, STT, stamp duty and STCG at 20% (vs LTCG at 12.5% above ₹1.25L) make the >1-year holding period worth a lot.
2. **Correctness over speed.** Point-in-time data, no look-ahead, survivorship-bias-free universes, and realistic costs. Most "great" backtests are bugs.
3. **One strategy, many runtimes.** The same strategy code runs in the vectorized backtest, the event-driven backtest, paper trading and (later) live trading.
4. **Overfitting guardrails are built in.** Walk-forward testing, out-of-sample holdouts, parameter-stability checks and the deflated Sharpe ratio.
5. **Production hygiene.** Typed code, tests at every layer, reproducible runs (each run records data snapshot + config hash + git SHA), structured logs, audit trail.

---

## 2. System architecture (components)

```
            ┌──────────────── React Dashboard ────────────────┐
            │ Research · Backtests · Portfolio · Signals · Jobs│
            └───────────────────────┬─────────────────────────┘
                                    │ REST/WebSocket
                         ┌──────────▼──────────┐
                         │   FastAPI service    │  auth, run mgmt
                         └──────────┬──────────┘
     ┌──────────────┬───────────────┼───────────────┬──────────────┐
 ┌───▼────┐   ┌─────▼─────┐   ┌─────▼─────┐   ┌─────▼─────┐  ┌─────▼─────┐
 │ Data   │──▶│ Feature / │──▶│ Strategy  │──▶│ Portfolio │─▶│ Execution │
 │ layer  │   │ factor lib│   │ engine    │   │ & risk    │  │ (paper/OMS)│
 └───┬────┘   └───────────┘   └─────┬─────┘   └───────────┘  └─────┬─────┘
     │                        ┌─────▼─────┐                        │
     │                        │ Backtest  │  vectorized + event    │
     │                        │ engine    │                        │
     │                        └─────┬─────┘                        │
 ┌───▼──────────────────────────────▼──────────────────────────────▼───┐
 │ Storage: Postgres (metadata, runs, orders, portfolio) + Parquet/DuckDB│
 └──────────────────────────────────────────────────────────────────────┘
   Scheduler/worker (EOD jobs) · Notifications (Telegram/email) · Observability
```

### 2.1 Data layer (`alphaengine/data`)
- **Sources (adapters behind a `DataSource` protocol):**
  - NSE bhavcopy (daily OHLCV, official, free) is the primary EOD source.
  - `yfinance` is the fallback and is used for indices, ETFs and backfill.
  - NSE index constituent history is used for survivorship-free Nifty 50/100/200/500 universes.
  - Corporate actions (splits, bonuses, dividends) produce adjustment factors.
  - Fundamentals (quarterly P&L, balance sheet, ratios) come from a pluggable source. Every record is stored with an **as-of / publish date** so the data stays point-in-time.
  - Macro and other assets: gold ETF, liquid/gilt ETF and the risk-free rate (91-day T-bill), for trend-allocation strategies.
  - Kite historical API is added later as a high-quality source.
- **Pipeline:** fetch → raw (immutable Parquet) → validate → normalize/adjust → curated (Parquet partitioned by date) → DuckDB views.
- **Data quality checks:** gaps vs the trading calendar, price spikes over N σ without a corporate action, zero/negative volume, duplicates, stale symbols. Results go to a `data_quality_report` table.
- **Symbol master:** ISIN-keyed so it survives renames and symbol changes. Internal symbols carry no `.NS` suffix.
- **Snapshots:** each curated dataset version gets a hash, and backtests record which snapshot they used.

### 2.2 Market model (`alphaengine/market`)
- **NSE trading calendar:** holidays and special sessions.
- **Indian cost model** (configurable and versioned by effective date): brokerage, STT (delivery 0.1% both sides), exchange transaction charges, SEBI fee, stamp duty (0.015% on buy), GST 18% on brokerage + transaction charges, and DP charges on sell.
- **Slippage model:** fixed bps, plus a volume-participation model based on % of ADV.
- **Constraints:** circuit limits (no fill at an upper or lower circuit), lot rounding to whole shares, T+1 settlement cash.
- **Tax model:** STCG/LTCG with FIFO lot tracking and the LTCG exemption threshold. It produces *after-tax* returns, which is the metric you actually care about.

### 2.3 Feature / factor library (`alphaengine/features`)
- Pure, vectorized functions over a panel (date × symbol), each declaring its **lookback** and its **data dependency** so the system can check for look-ahead.
- **Starter set:**
  - Momentum: 12-1, 6-1, risk-adjusted.
  - Volatility, ATR.
  - Moving averages and trend filters (200-DMA, dual momentum).
  - 52-week-high proximity.
  - Value: E/P, B/P.
  - Quality: ROE, ROCE, debt/equity, earnings stability.
  - Low volatility.
  - Liquidity: ADV in rupees.
  - Technical indicators for swing trading: RSI, breakouts, Bollinger.
- A feature cache in Parquet, keyed by feature + params + data snapshot.

### 2.4 Strategy engine (`alphaengine/strategy`)
A strategy is a **declarative pipeline** of pluggable stages, each with a clear interface:

```
Universe → Filters → Signals/Scores → Selection → Weighting → Risk overlay → Target portfolio
```
- `Universe`: e.g. Nifty 200 point-in-time, plus a liquidity filter.
- `Filter`: e.g. price > 200-DMA, ADV > ₹5 Cr, exclude ASM/GSM stocks.
- `Scorer`: one factor or a z-scored multi-factor composite with weights.
- `Selector`: top-N, top-quantile, or a threshold with a hysteresis buffer (e.g. buy in the top 20, sell only when a stock drops out of the top 40). Hysteresis is the key turnover reducer.
- `Weighter`: equal weight, inverse volatility, risk parity, market-cap capped.
- `RiskOverlay`: market-regime filter (e.g. Nifty below its 200-DMA → move X% to a liquid ETF), max position %, sector caps, volatility targeting.
- `Schedule`: monthly, quarterly, weekly, daily (swing trading), or event-based.
- Each strategy has two forms. In **Python**, you subclass the `Strategy` base or compose stages. In **YAML/JSON**, a Pydantic-validated config builds a strategy from registered components, so the UI can create strategies without code.
- **Strategy registry:** a name and a semantic version. Stored configs are immutable, so results are always traceable to an exact config.
- **Output:** `TargetPortfolio(date, weights, rationale)`. The rationale records why each name was chosen, so you learn as you go.

### 2.5 Backtest engine (`alphaengine/backtest`)
Two engines share the same strategy, cost and market model:
1. **Vectorized rebalance engine:** fast, for research and parameter sweeps on monthly and quarterly strategies. Signals are computed at close on day T and filled at the open of T+1 (configurable).
2. **Event-driven engine:** a bar-by-bar loop with an order book, partial fills, stops and intraperiod risk. It's needed for swing trading and serves as the "truth" check for the vectorized engine.
- **Features:**
  - Cash accounting, dividends, corporate actions.
  - Benchmark comparison (Nifty 50 TRI, Nifty 500 TRI, a 60/40 mix).
  - No-trade bands.
  - Tax lots and after-tax equity curve.
  - Start/end date, initial capital, monthly SIP contributions. SIPs model your real life of investing monthly.
- **Metrics:**
  - Return and risk: CAGR, volatility, Sharpe, Sortino, Calmar, max drawdown and its duration, ulcer index.
  - Trading: turnover, number of trades per year, average holding period, cost drag, tax drag, hit rate.
  - Relative to benchmark: alpha, beta, information ratio, tracking error.
  - Rolling returns: 1, 3 and 5 years.
- **Robustness tools:**
  - Walk-forward optimization.
  - Parameter grid sweeps with a stability heatmap.
  - Bootstrap / Monte Carlo of returns.
  - Deflated Sharpe ratio and a probability-of-backtest-overfitting estimate.
  - Sub-period analysis by regime.
  - Cost-sensitivity (2× costs).
- **Run artifacts:** config, data snapshot hash, git SHA, equity curve, trades, holdings, metrics. Stored in Postgres + Parquet and viewable in the UI.

### 2.6 Portfolio & risk (`alphaengine/portfolio`, `alphaengine/risk`)
- **Portfolio book:** positions, tax lots, cash, cost basis, realized and unrealized P&L. The same model serves backtest, paper and real (manually entered) portfolios.
- **Rebalance planner:** current holdings + target weights produce a **minimal trade list**. It respects no-trade bands and minimum trade size (₹), and it prefers selling lots already in LTCG. It also flags a trade if waiting N days would turn it from STCG into LTCG.
- **Multi-strategy allocation** (core + satellite), e.g. 70% core (index ETF + trend allocation) and 30% satellite (factor rotation, swing trading), with a combined risk view.
- **Risk checks** (pre-trade and daily): concentration, sector exposure, drawdown alerts, correlation between strategies.

### 2.7 Execution / OMS (`alphaengine/execution`)
- An order model with a state machine: `NEW → SUBMITTED → PARTIAL → FILLED / CANCELLED / REJECTED`.
- A `Broker` protocol. The first implementation is `PaperBroker`, which fills at the next open plus slippage using the cost model. `ManualBroker` generates a rebalance checklist (CSV/UI) and you mark the fills you placed yourself. `KiteBroker` comes later.
- **Reconciliation:** compare the system book with the broker or manual holdings, and flag drift.
- **Kill switch and idempotency keys,** so the system is safe for the later live phase.

### 2.8 Orchestration (`alphaengine/scheduler`)
- APScheduler runs in a worker container (Prefect can come later if needed).
- **EOD job (around 19:00 IST on trading days):** ingest → quality checks → features → signals for every active strategy → paper fills → portfolio mark-to-market → notifications.
- **Rebalance-day job:** produces the rebalance list for your manual execution.
- Jobs are idempotent, keyed by date, and record their run history for the UI.

### 2.9 Notifications (`alphaengine/notify`)
- A Telegram bot (primary) and email for daily summaries, rebalance lists, risk alerts and data-quality failures.

### 2.10 API (`alphaengine/api`, FastAPI)
- **Endpoints:**
  - Strategies: CRUD, validate config.
  - Backtests: launch (async job), status, results, compare runs.
  - Portfolios: holdings, P&L, tax lots.
  - Signals and rebalance plans.
  - Data: symbols, prices, quality reports.
  - Jobs.
- Single-user auth (JWT or API key). OpenAPI-generated TypeScript client for the frontend.

### 2.11 Dashboard (`web/`, React + Vite + TypeScript)
- **Stack:** TanStack Query, TanStack Table, lightweight-charts (prices) and Recharts (equity curves).
- **Pages:**
  - **Home:** net worth, today's signals, upcoming rebalances, alerts.
  - **Strategy builder:** a form over the YAML schema, plus code-strategy listing.
  - **Backtest lab:** run, compare runs side by side, parameter heatmaps, drawdown chart, trades table.
  - **Portfolio:** holdings, allocation, tax lots, realized gains report for ITR.
  - **Rebalance:** trade checklist with "mark as executed."
  - **Journal:** notes per trade/decision. This is your learning log.
  - **Data health.**

### 2.12 Cross-cutting
- **Config:** Pydantic Settings + `.env`. Secrets are never committed.
- **Logging:** structlog JSON with run_id and job_id correlation. A Prometheus metrics endpoint is optional.
- **DB:** SQLAlchemy 2.0 ORM + Alembic migrations.
- **Audit log** of every signal, order and manual action.

---

## 3. Repository layout
```
AlphaEngine/
  pyproject.toml            # uv-managed; ruff, mypy(strict), pytest config
  alphaengine/
    core/        # types (Symbol, Bar, Order, Position), clock, errors, ids
    config/      # settings, universe definitions
    data/        # sources/, pipeline/, quality/, store/ (parquet+duckdb), models (ORM)
    market/      # calendar, costs, slippage, tax
    features/    # factor functions + registry + cache
    strategy/    # base, stages/, registry, yaml schema, builtin/ strategies
    backtest/    # vectorized/, event/, metrics, robustness/, results store
    portfolio/   # book, tax lots, rebalance planner, allocation
    risk/        # checks, overlays
    execution/   # orders, OMS, brokers/ (paper, manual, kite-later)
    scheduler/   # jobs, runner
    notify/      # telegram, email
    api/         # FastAPI app, routers, schemas, deps
    cli/         # typer CLI: ingest, backtest, signals, rebalance
  migrations/    # alembic
  web/           # React app
  tests/
    unit/  property/  integration/  golden/  e2e/
  infra/
    docker-compose.yml  # postgres, api, worker, web
    Dockerfile.api  Dockerfile.web
  .github/workflows/ci.yml
  docs/          # architecture, ADRs, strategy notes, learning notes
  CLAUDE.md      # conventions for future sessions
```

---

## 4. Testing & quality strategy (production level)
- **Unit tests:** every factor, cost formula, tax calculation, order state transition and calendar rule. Cost and tax tests are hand-calculated against real contract-note examples.
- **Property-based tests (Hypothesis):**
  - Portfolio value = cash + Σ positions at all times.
  - Weights sum to ≤ 1.
  - No negative cash unless margin is enabled.
  - Rebalancing to the current weights generates zero trades.
  - Cost is ≥ 0 and monotonic in notional.
- **Look-ahead tests:**
  - Truncate future data, and signals for past dates must not change.
  - Shuffle future bars, and past signals must stay identical.
- **Golden / regression backtests:** fixed synthetic and real datasets with stored expected metrics. CI fails if they drift without an explicit snapshot update.
- **Engine parity test:** the vectorized and event-driven engines must agree within tolerance on rebalance strategies.
- **Known-answer tests:** buy-and-hold must equal the index return minus costs. A constant-weight strategy must match its analytic result.
- **Integration tests:** Postgres via testcontainers, the pipeline on recorded HTTP fixtures (VCR / respx, with no live network in CI), and the API through the FastAPI TestClient.
- **Contract tests** for every `DataSource` and `Broker` implementation (a shared test suite).
- **Frontend tests:** Vitest + React Testing Library for components. A few Playwright e2e flows: run a backtest, view results, mark a rebalance executed.
- **Tooling:**
  - ruff (lint + format), mypy strict, pre-commit.
  - Coverage gate: 85% core, 70% overall.
  - GitHub Actions: lint → typecheck → unit → integration → golden → web build/test → docker build.

---

## 5. Strategy ideas to implement and research (in priority order)
1. **Benchmark baselines (must-have):** Nifty 50 buy-and-hold, a Nifty 50 / gold / gilt fixed mix, monthly SIP into an index. Every strategy must beat these after costs and taxes, or it isn't worth running.
2. **Dual-momentum asset allocation (trend, around 1 trade/month):** rotate between Nifty 50 ETF, Nifty Midcap ETF, gold ETF and a liquid/gilt ETF, based on 6–12 month absolute + relative momentum.
3. **Momentum factor rotation (satellite):** Nifty 200 universe, 12-1 risk-adjusted momentum, top 20, monthly rebalance with a top-40 hold buffer, and a regime filter (Nifty 200 above its 200-DMA).
4. **Quality + value composite:** quarterly rebalance. High ROCE, low debt, reasonable P/E. Low turnover and long holding periods suit LTCG.
5. **Low-volatility / momentum blend,** a defensive equity sleeve.
6. **Core + satellite combiner:** 60–80% core (index ETF + dual momentum) and 20–40% satellite (strategies 3–4), with volatility targeting at the total level.
7. **Swing trading (phase 7, optional):** a 52-week-high breakout with an ATR stop, or a pullback-in-uptrend (RSI(2) on stocks above their 200-DMA). Built on the event-driven engine. Expect higher costs and STCG tax, and judge it on after-tax results.

Other ideas for later:
- A tax-loss harvesting assistant.
- An "insider/promoter buying" signal.
- An earnings-surprise drift signal.
- A goal planner (target corpus, SIP needed, Monte Carlo of outcomes).
- An LLM-assisted quarterly-results summarizer for your holdings.

---

## 6. Phased roadmap (each phase is shippable and tested)

| Phase | Deliverable | Done when |
|---|---|---|
| **0. Foundations** | Repo scaffold, uv/pyproject, ruff/mypy/pytest, pre-commit, CI, docker-compose (Postgres), settings, logging, core types, Alembic, CLAUDE.md | CI green on an empty skeleton; `docker compose up` brings up the DB |
| **1. Data layer** | Symbol master, NSE calendar, bhavcopy + yfinance adapters, corporate-action adjustment, Parquet/DuckDB store, quality checks, index-constituent history, CLI `ingest` | 10+ years of Nifty 500 EOD loaded; quality report clean; contract tests pass |
| **2. Market model** | Cost model, slippage, tax lots + STCG/LTCG | Costs match real Zerodha contract-note examples to the paisa |
| **3. Strategy engine + features** | Factor library, pipeline stages, registry, YAML schema, baseline + dual-momentum + momentum strategies | Strategies produce target portfolios; look-ahead tests pass |
| **4. Backtest engine** | Vectorized engine, metrics, benchmark comparison, SIP support, run store, CLI `backtest`, HTML/JSON report | Golden tests + known-answer tests pass; first real research results |
| **5. Robustness + event-driven engine** | Walk-forward, parameter sweeps, deflated Sharpe, Monte Carlo; event-driven engine + parity test | Parity within tolerance; robustness report per strategy |
| **6. Portfolio, paper trading, scheduler** | Book, rebalance planner (bands, tax-aware), PaperBroker, ManualBroker, EOD jobs, Telegram notifications | System runs unattended daily; you receive rebalance lists |
| **7. API + Dashboard** | FastAPI routers, generated TS client, React pages (Home, Backtest Lab, Strategy Builder, Portfolio, Rebalance, Journal, Data Health) | Full workflow done from the UI; Playwright e2e passes |
| **8. Swing trading & extras** | Swing strategies on the event engine, stops, tax-loss harvesting, goal planner | After-tax comparison vs low-turnover strategies |
| **9. Broker integration (optional)** | KiteBroker (holdings sync first, then approval-gated orders), reconciliation, kill switch | Holdings auto-reconcile; orders only after explicit approval |

**Suggested learning path alongside the build.** Phase 1–2 teaches market mechanics, costs and taxes. Phase 3–4 teaches factors and their evidence. Phase 5 teaches why backtests lie. Phase 6 onward is about running paper trading for at least 3–6 months before committing real capital. Recommended reading:
- Antti Ilmanen, *Expected Returns* (selectively).
- Gary Antonacci, *Dual Momentum Investing*.
- Wesley Gray and Jack Vogel, *Quantitative Momentum*.
- Marcos López de Prado, *Advances in Financial Machine Learning* (chapters on backtest overfitting).
- NSE's factor index methodology documents (Nifty200 Momentum 30, Nifty Alpha Low-Vol).

---

## 7. Verification (end-to-end)
- `uv run pytest` (unit, property, integration and golden) plus `uv run mypy alphaengine` and `ruff check` are green locally and in CI.
- `alphaengine ingest --from 2010-01-01` loads the data, and `alphaengine data-quality` reports no critical issues.
- `alphaengine backtest strategies/dual_momentum.yaml --report out/` produces metrics and an equity curve vs Nifty 50 TRI, both pre- and after-tax.
- Sanity check: the buy-and-hold Nifty backtest matches the published index CAGR within the cost tolerance.
- `docker compose up` brings up the API, worker and web. Launching a backtest from the dashboard and viewing it works, and so does the Playwright e2e suite.
- Paper portfolio: the EOD job runs on a trading day, and the Telegram summary plus the rebalance checklist arrive.

## 8. Immediate next step after approval
Execute **Phase 0** (scaffold, tooling, CI, docker-compose, core types, CLAUDE.md), then **Phase 1** (data layer), committing per phase on feature branches.
