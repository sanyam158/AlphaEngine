"""``alphaengine`` command-line entry point. Later phases add ingest/backtest/signals/rebalance."""

from __future__ import annotations

from pathlib import Path

import typer
from alembic import command
from alembic.config import Config

from alphaengine import __version__
from alphaengine.cli.data_commands import data_app, import_app, ingest_app
from alphaengine.config.settings import get_settings
from alphaengine.core.logging import configure_logging, get_logger

app = typer.Typer(help="AlphaEngine: research, backtesting and paper trading for NSE/BSE.")
db_app = typer.Typer(help="Database migrations.")
app.add_typer(db_app, name="db")
app.add_typer(ingest_app, name="ingest")
app.add_typer(import_app, name="import")
app.add_typer(data_app, name="data")

_ROOT = Path(__file__).resolve().parents[2]


@app.callback()
def _main() -> None:
    s = get_settings()
    configure_logging(s.log_level, s.log_json)


@app.command()
def version() -> None:
    """Print the AlphaEngine version."""
    typer.echo(__version__)


def _alembic_config() -> Config:
    cfg = Config(str(_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(_ROOT / "migrations"))
    cfg.set_main_option("sqlalchemy.url", get_settings().database_url)
    return cfg


@db_app.command("upgrade")
def db_upgrade(revision: str = "head") -> None:
    """Apply migrations up to ``revision``."""
    command.upgrade(_alembic_config(), revision)
    get_logger(__name__).info("db upgraded", revision=revision)


@db_app.command("downgrade")
def db_downgrade(revision: str) -> None:
    """Revert migrations down to ``revision``."""
    command.downgrade(_alembic_config(), revision)


if __name__ == "__main__":  # pragma: no cover
    app()
