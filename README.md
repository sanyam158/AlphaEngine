# AlphaEngine

Personal research, backtesting and paper-trading system for Indian equities (NSE/BSE),
designed for **low-turnover, after-tax** investing that needs little time to run.

> Research and decision-support software, not investment advice. Paper trade before
> committing real capital.

See [docs/PLAN.md](docs/PLAN.md) for the architecture and the phased roadmap.

## Status
- [x] Phase 0: foundations (tooling, CI, settings, logging, core types, DB + migrations)
- [x] Phase 1: data layer ([guide](docs/DATA.md))
- [x] Phase 2: market model: costs, fills, tax lots, capital-gains tax ([guide](docs/MARKET.md))
- [x] Phase 3: strategy engine and feature library ([guide](docs/STRATEGIES.md))
- [x] Phase 4: backtest engine: costs, taxes, SIP, metrics, reports ([guide](docs/BACKTEST.md))
- [x] Phase 5: robustness: sweeps, walk-forward, deflated Sharpe, PBO, bootstrap, fast engine ([guide](docs/ROBUSTNESS.md))
- [x] Phase 6: paper/manual portfolios, tax-aware planner, daily job, Telegram ([guide](docs/PAPER_TRADING.md))
- [x] Phase 7: FastAPI + React dashboard, journal ([guide](docs/DASHBOARD.md))
- [ ] Phase 8–9: swing trading, broker integration

## Quick start
```bash
python -m venv .venv
.venv/Scripts/pip install -e ".[dev,yahoo]"   # Windows; use .venv/bin on macOS/Linux
cp .env.example .env
docker compose -f infra/docker-compose.yml up -d   # optional: Postgres (SQLite works too)
alphaengine db upgrade
alphaengine ingest bhavcopy --start 2024-01-01
alphaengine data quality
pytest -m "not integration"
```
