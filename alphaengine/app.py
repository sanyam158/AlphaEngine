"""Composition root: builds the shared services from settings (used by the CLI, later the API
and scheduler)."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from alphaengine.config.settings import Settings, get_settings
from alphaengine.data.db import make_engine, make_session_factory
from alphaengine.data.raw_store import RawStore
from alphaengine.data.store import PriceStore


@dataclass(frozen=True)
class AppContext:
    settings: Settings
    engine: Engine
    session_factory: sessionmaker[Session]
    raw: RawStore
    equities: PriceStore
    indices: PriceStore


def build_context(settings: Settings | None = None) -> AppContext:
    s = settings or get_settings()
    engine = make_engine(s.database_url)
    return AppContext(
        settings=s,
        engine=engine,
        session_factory=make_session_factory(engine),
        raw=RawStore(s.raw_dir),
        equities=PriceStore(s.curated_dir, "equities"),
        indices=PriceStore(s.curated_dir, "indices"),
    )
