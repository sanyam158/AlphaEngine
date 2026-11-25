# AlphaEngine — conventions

Personal research, backtesting and paper-trading system for Indian equities (NSE/BSE).
The master plan and phase roadmap live in `docs/PLAN.md`; keep it updated when scope changes.

## Commands
- Setup: `python -m venv .venv && .venv/Scripts/pip install -e ".[dev,yahoo]"` (uv-compatible `pyproject.toml`)
- Lint/format: `ruff check . && ruff format --check .`
- Types: `mypy` (strict, configured in pyproject)
- Tests: `pytest -m "not integration"`; integration needs `ALPHAENGINE_TEST_POSTGRES_URL`
  (`docker compose -f infra/docker-compose.yml up -d`)
- Migrations: `alembic revision --autogenerate -m "..."`, apply with `alphaengine db upgrade`

## Rules
- Python ≥ 3.11, fully typed, `mypy --strict` clean. Coverage gate 85%.
- Symbols are NSE symbols without `.NS`/`.BO` suffix; normalize with `core.types.make_symbol`.
- Time: aware datetimes only, market tz is IST (`core.clock.IST`). Depend on the `Clock`
  protocol, never call `datetime.now()` in domain code.
- Money/accounting uses `Decimal`; market-data/research math uses float/numpy/pandas.
- Persistence via SQLAlchemy 2.0 ORM only; every schema change ships with an Alembic migration.
- Logging via `core.logging.get_logger` (structlog); no `print`.
- External API calls (data vendors, brokers) wrap in `core.retry.retry` and raise `DataSourceError`
  / domain errors from `core.errors`.
- No look-ahead: any feature/signal must only use data available at its as-of date; add a
  truncation test for every new feature.
- Tests never hit the network: use `tests/helpers.py` (MockNse, synthetic bars). Live smoke
  tests live in `tests/network` and run only with `ALPHAENGINE_RUN_NETWORK=1`.
- New price sources implement `DailyBarSource` and must be added to `tests/contract`.
- Curated prices are raw; adjust only at read time (`data.prices.load_prices`). See docs/DATA.md.
- New features go in `features/library.py` and must get a `SMALL` entry in
  `tests/unit/test_features.py` (causality test). New stages register via `@component`.
- Work on feature branches per phase; never commit `.env` or data under `var/`.
