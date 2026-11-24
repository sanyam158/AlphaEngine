# AlphaEngine

Personal research, backtesting and paper-trading system for Indian equities (NSE/BSE),
designed for **low-turnover, after-tax** investing that needs little time to run.

> Research and decision-support software, not investment advice. Paper trade before
> committing real capital.

See [docs/PLAN.md](docs/PLAN.md) for the architecture and the phased roadmap.

## Status
- [x] Phase 0: foundations (tooling, CI, settings, logging, core types, DB + migrations)
- [ ] Phase 1: data layer
- [ ] Phase 2: market model (costs, taxes)
- [ ] Phase 3: strategy engine + features
- [ ] Phase 4: backtest engine
- [ ] Phase 5: robustness + event-driven engine
- [ ] Phase 6: portfolio, paper trading, scheduler
- [ ] Phase 7: API + dashboard
- [ ] Phase 8–9: swing trading, broker integration

## Quick start
```bash
python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"      # Windows; use .venv/bin on macOS/Linux
cp .env.example .env
docker compose -f infra/docker-compose.yml up -d   # optional: Postgres (SQLite works too)
alphaengine db upgrade
pytest -m "not integration"
```
