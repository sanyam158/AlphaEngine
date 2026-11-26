from __future__ import annotations

from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, select
from sqlalchemy.orm import Session, sessionmaker

from alphaengine.data.db import make_engine, session_scope
from alphaengine.data.models import AuditLog

ROOT = Path(__file__).resolve().parents[2]


def test_session_scope_commits(session_factory: sessionmaker[Session]) -> None:
    with session_scope(session_factory) as s:
        s.add(AuditLog(actor="system", action="test", payload={"k": 1}))
    with session_scope(session_factory) as s:
        row = s.scalars(select(AuditLog)).one()
        assert row.payload == {"k": 1}
        assert row.created_at is not None


def test_session_scope_rolls_back(session_factory: sessionmaker[Session]) -> None:
    def write_then_fail() -> None:
        with session_scope(session_factory) as s:
            s.add(AuditLog(actor="system", action="test"))
            s.flush()
            raise RuntimeError("fail")

    with pytest.raises(RuntimeError):
        write_then_fail()
    with session_scope(session_factory) as s:
        assert s.scalars(select(AuditLog)).all() == []


def test_migrations_upgrade_and_downgrade(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'mig.db'}"
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    cfg.set_main_option("sqlalchemy.url", url)
    cfg.attributes["configure_logger"] = False

    command.upgrade(cfg, "head")
    engine = make_engine(url)
    assert "audit_log" in inspect(engine).get_table_names()
    command.downgrade(cfg, "base")
    assert "audit_log" not in inspect(engine).get_table_names()
    engine.dispose()
