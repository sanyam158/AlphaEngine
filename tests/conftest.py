from __future__ import annotations

from collections.abc import Iterator

import pytest
from hypothesis import HealthCheck, settings
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from alphaengine.config.settings import get_settings
from alphaengine.data.db import Base, make_engine, make_session_factory

# DataFrame-building property tests can exceed Hypothesis's 200ms per-example deadline on a busy
# machine or CI runner; correctness, not speed, is what they check.
settings.register_profile(
    "alphaengine", deadline=None, suppress_health_check=[HealthCheck.too_slow]
)
settings.load_profile("alphaengine")


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory):
    """Never let tests read the developer's .env or touch real data dirs."""
    monkeypatch.setenv("ALPHAENGINE_ENV", "test")
    monkeypatch.setenv("ALPHAENGINE_DATA_DIR", str(tmp_path_factory.mktemp("data")))
    monkeypatch.chdir(tmp_path_factory.mktemp("cwd"))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def engine() -> Iterator[Engine]:
    eng = make_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture
def session_factory(engine: Engine) -> sessionmaker[Session]:
    return make_session_factory(engine)
