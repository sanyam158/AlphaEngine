"""Runs migrations and a round-trip against a real Postgres.

Set ALPHAENGINE_TEST_POSTGRES_URL (CI does; locally use infra/docker-compose.yml)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import select

from alphaengine.data.db import make_engine, make_session_factory, session_scope
from alphaengine.data.models import AuditLog

pytestmark = pytest.mark.integration

URL = os.environ.get("ALPHAENGINE_TEST_POSTGRES_URL")
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(not URL, reason="ALPHAENGINE_TEST_POSTGRES_URL not set")
def test_postgres_migrate_and_roundtrip() -> None:
    assert URL
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    cfg.set_main_option("sqlalchemy.url", URL)
    cfg.attributes["configure_logger"] = False
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")

    engine = make_engine(URL)
    factory = make_session_factory(engine)
    with session_scope(factory) as s:
        s.add(AuditLog(actor="ci", action="roundtrip", payload={"ok": True}))
    with session_scope(factory) as s:
        row = s.scalars(select(AuditLog).where(AuditLog.action == "roundtrip")).one()
        assert row.payload == {"ok": True}
        assert row.created_at.tzinfo is not None
    engine.dispose()
    command.downgrade(cfg, "base")
