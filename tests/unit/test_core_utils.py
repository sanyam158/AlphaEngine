from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

import alphaengine.core.ids as ids_mod
from alphaengine.config.settings import Environment, Settings, get_settings
from alphaengine.core.clock import IST, FixedClock, SystemClock, to_ist
from alphaengine.core.ids import new_id
from alphaengine.core.logging import configure_logging, get_logger
from alphaengine.core.retry import retry


class TestClock:
    def test_to_ist(self) -> None:
        ts = to_ist(datetime(2026, 1, 5, 3, 45, tzinfo=UTC))
        assert (ts.hour, ts.minute) == (9, 15)
        with pytest.raises(ValueError, match="naive"):
            to_ist(datetime(2026, 1, 5))  # noqa: DTZ001

    def test_system_clock_is_ist(self) -> None:
        now = SystemClock().now()
        assert now.utcoffset() == timedelta(hours=5, minutes=30)
        assert isinstance(SystemClock().today(), date)

    def test_fixed_clock_monotonic(self) -> None:
        c = FixedClock(datetime(2026, 1, 5, 9, 15, tzinfo=IST))
        c.advance(timedelta(days=1))
        assert c.today() == date(2026, 1, 6)
        c.set(datetime(2026, 1, 7, tzinfo=IST))
        assert c.now().day == 7
        with pytest.raises(ValueError, match="backwards"):
            c.advance(timedelta(seconds=-1))
        with pytest.raises(ValueError, match="backwards"):
            c.set(datetime(2026, 1, 1, tzinfo=IST))


class TestIds:
    def test_format_and_uniqueness(self) -> None:
        ids = [new_id("run") for _ in range(500)]
        assert len(set(ids)) == 500
        assert all(i.startswith("run_") and len(i) == 30 for i in ids)

    def test_sortable_over_time(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ids_mod.time, "time_ns", lambda: 1_000_000_000)
        a = new_id("ord")
        monkeypatch.setattr(ids_mod.time, "time_ns", lambda: 2_000_000_000)
        b = new_id("ord")
        assert a < b

    @pytest.mark.parametrize("bad", ["", "Run", "run-x", "1run"])
    def test_bad_prefix(self, bad: str) -> None:
        with pytest.raises(ValueError, match="prefix"):
            new_id(bad)


class TestRetry:
    def test_retries_then_succeeds(self) -> None:
        calls, sleeps = [], []

        @retry(attempts=3, base_delay=1, exceptions=(ConnectionError,), sleep=sleeps.append)
        def flaky() -> str:
            calls.append(1)
            if len(calls) < 3:
                raise ConnectionError("boom")
            return "ok"

        assert flaky() == "ok"
        assert len(calls) == 3
        assert len(sleeps) == 2
        assert 0.5 <= sleeps[0] <= 1.0
        assert 1.0 <= sleeps[1] <= 2.0

    def test_reraises_after_last_attempt(self) -> None:
        @retry(attempts=2, sleep=lambda _: None)
        def always() -> None:
            raise TimeoutError("nope")

        with pytest.raises(TimeoutError):
            always()

    def test_does_not_catch_other_errors(self) -> None:
        calls = []

        @retry(attempts=5, exceptions=(ConnectionError,), sleep=lambda _: None)
        def bad() -> None:
            calls.append(1)
            raise ValueError

        with pytest.raises(ValueError):  # noqa: PT011
            bad()
        assert len(calls) == 1

    def test_invalid_attempts(self) -> None:
        with pytest.raises(ValueError, match="attempts"):
            retry(attempts=0)


class TestSettings:
    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ALPHAENGINE_LOG_LEVEL", "debug")
        monkeypatch.setenv("ALPHAENGINE_DATA_DIR", "/tmp/ae")
        s = get_settings()
        assert s.env is Environment.TEST
        assert s.log_level == "DEBUG"
        assert s.raw_dir == Path("/tmp/ae/raw")
        assert s.curated_dir == Path("/tmp/ae/curated")
        assert get_settings() is s

    def test_invalid_values(self) -> None:
        with pytest.raises(ValueError, match="log level"):
            Settings(log_level="loud")
        with pytest.raises(ValueError):  # noqa: PT011
            Settings(initial_capital=0)


def test_logging_smoke(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", json=True)
    get_logger("t").info("hello", run_id="run_1")
    err = capsys.readouterr().err
    assert '"event": "hello"' in err
    assert '"run_id": "run_1"' in err
