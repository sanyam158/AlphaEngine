"""End-to-end ingest against a mock NSE server (no network): fetch → raw → curated → DB."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker
from typer.testing import CliRunner

from alphaengine.core.errors import DataSourceError
from alphaengine.core.types import Symbol
from alphaengine.data.db import session_scope
from alphaengine.data.models import DataQualityIssue, IngestLog, Instrument
from alphaengine.data.pipeline import (
    BhavcopyIngestor,
    ingest_corporate_actions,
    ingest_index_prices,
    run_quality,
)
from alphaengine.data.prices import load_prices, to_panel
from alphaengine.data.raw_store import RawStore
from alphaengine.data.repositories import upsert_corporate_actions
from alphaengine.data.sources.nse_bhavcopy import NseBhavcopySource
from alphaengine.data.sources.yahoo import YahooSource
from alphaengine.data.store import PriceStore
from tests.helpers import MockNse, Row

SF = sessionmaker[Session]

# Mon 2024-07-01 .. Fri 2024-07-12 spans the legacy -> UDiFF cutover (2024-07-08).
DAYS = [date(2024, 7, d) for d in (1, 2, 3, 4, 5, 8, 9, 10, 11, 12)]
HOLIDAY = date(2024, 7, 4)


def _universe(i: int, day: date) -> list[Row]:
    rows = [
        # OLDNAME renames to NEWNAME on 2024-07-09 (same ISIN)
        Row("OLDNAME" if day < date(2024, 7, 9) else "NEWNAME", "INE100A01011", 200.0 + i),
        # SPLITCO does a 2-for-1 split on 2024-07-10; exchange PREVCLOSE adjusted that day
        Row(
            "SPLITCO",
            "INE200A01011",
            (1000.0 + i) if day < date(2024, 7, 10) else (1000.0 + i) / 2,
            prev_close=(1000.0 + i - 1) / 2 if day == date(2024, 7, 10) else None,
        ),
        Row("STEADY", "INE300A01011", 50.0, volume=10_000),
    ]
    if day < date(2024, 7, 8):
        rows.append(Row("GONE", "INE400A01011", 10.0))  # delisted after first week
    return rows


@pytest.fixture
def nse() -> MockNse:
    m = MockNse()
    for i, d in enumerate(DAYS):
        if d != HOLIDAY:
            m.add(d, _universe(i, d), "udiff" if d >= date(2024, 7, 8) else "legacy")
    return m


def _ingestor(nse: MockNse, tmp_path: Path, sf: SF, **kw: object) -> BhavcopyIngestor:
    return BhavcopyIngestor(
        NseBhavcopySource(nse.fetcher()),
        RawStore(tmp_path / "raw"),
        PriceStore(tmp_path / "curated"),
        sf,
        batch_days=3,
        **kw,  # type: ignore[arg-type]
    )


def test_full_ingest_is_idempotent_and_resumable(
    nse: MockNse, tmp_path: Path, session_factory: SF
) -> None:
    progress: list[tuple[date, str]] = []
    ing = _ingestor(
        nse, tmp_path, session_factory, on_progress=lambda d, s: progress.append((d, s))
    )
    summary = ing.run(DAYS[0], DAYS[-1])
    assert (summary.days_ok, summary.days_no_data, summary.days_error) == (9, 1, 0)
    assert summary.new_instruments == 4 and summary.renames == 1
    assert (HOLIDAY, "no_data") in progress

    store = ing.store
    assert len(store.trading_days()) == 9
    assert store.date_range() == (date(2024, 7, 1), date(2024, 7, 12))
    snap = store.snapshot_id()

    # Second run: everything skipped, no HTTP traffic, data unchanged
    n_req = len(nse.requests)
    again = ing.run(DAYS[0], DAYS[-1])
    assert again.days_skipped == 10 and again.days_ok == 0
    assert len(nse.requests) == n_req
    assert store.snapshot_id() == snap

    # Forced run re-reads from the immutable raw cache, still no HTTP traffic
    forced = ing.run(DAYS[0], DAYS[-1], force=True)
    assert forced.days_ok == 9
    assert len(nse.requests) == n_req + 2  # only the holiday is re-requested (both formats)
    assert store.snapshot_id() == snap

    with session_scope(session_factory) as s:
        assert s.scalar(select(func.count()).select_from(IngestLog)) == 10
        inst = s.scalars(select(Instrument).where(Instrument.isin == "INE100A01011")).one()
        assert inst.symbol == "NEWNAME"

    # Rename resolution: one continuous history under the current symbol
    bars = load_prices(store, session_factory, ["NEWNAME"], adjust="none")
    assert len(bars) == 9 and set(bars["symbol"]) == {"NEWNAME"}
    raw = load_prices(store, session_factory, ["OLDNAME"], adjust="none", resolve_renames=False)
    assert len(raw) == 5

    # Quality: unrecorded split is flagged; recording it clears the warning
    report = run_quality(store, session_factory)
    flagged = {(i.check, i.symbol) for i in report.issues}
    assert ("prev_close_mismatch", "SPLITCO") in flagged
    assert ("price_spike", "SPLITCO") in flagged
    with session_scope(session_factory) as s:
        assert s.scalar(select(func.count()).select_from(DataQualityIssue)) == len(report.issues)
        upsert_corporate_actions(
            s,
            pd.DataFrame(
                [
                    {
                        "symbol": "SPLITCO",
                        "ex_date": "2024-07-10",
                        "kind": "SPLIT",
                        "ratio": 2.0,
                        "amount": None,
                    }
                ]
            ),
            source="test",
        )
    report2 = run_quality(store, session_factory, persist=False)
    assert not {(i.check, i.symbol) for i in report2.issues} & {
        ("prev_close_mismatch", "SPLITCO"),
        ("price_spike", "SPLITCO"),
    }
    adj = load_prices(store, session_factory, ["SPLITCO"], start=date(2024, 7, 2))
    closes = to_panel(adj)["SPLITCO"]
    assert closes.pct_change().abs().max() < 0.01  # continuous after adjustment
    assert closes.index.min() == pd.Timestamp("2024-07-02")


def test_aborts_after_consecutive_errors_but_keeps_progress(
    nse: MockNse, tmp_path: Path, session_factory: SF
) -> None:
    for d in DAYS[2:]:
        nse.fail_with[d] = 403
    ing = _ingestor(nse, tmp_path, session_factory, max_consecutive_errors=3)
    with pytest.raises(DataSourceError, match="consecutive"):
        ing.run(DAYS[0], DAYS[-1])
    with session_scope(session_factory) as s:
        statuses = dict(s.execute(select(IngestLog.trade_date, IngestLog.status)).all())
    assert statuses[DAYS[0]] == "ok" and statuses[DAYS[1]] == "ok"
    assert statuses[DAYS[2]] == "error"
    assert len(ing.store.trading_days()) == 2

    # After the source recovers, a rerun retries only the failed/missing days
    nse.fail_with.clear()
    summary = ing.run(DAYS[0], DAYS[-1])
    assert summary.days_skipped == 2 and summary.days_ok == 7


def test_pre2012_rows_join_renamed_history(tmp_path: Path, session_factory: SF) -> None:
    """Old files have no ISIN; prices must still attach to the right company after a rename."""
    nse = MockNse()
    old_days = [date(2011, 11, 1), date(2011, 11, 2)]
    for d in old_days:
        nse.add(d, [Row("OLDNAME", "INE100A01011", 100.0)])
        nse.pre2012.add(d)
    nse.add(date(2011, 12, 1), [Row("OLDNAME", "INE100A01011", 101.0)])
    nse.add(date(2011, 12, 2), [Row("NEWNAME", "INE100A01011", 102.0)])
    ing = _ingestor(nse, tmp_path, session_factory)
    summary = ing.run(old_days[0], date(2011, 12, 2))
    assert summary.days_error == 0
    assert summary.days_ok == 4
    bars = load_prices(ing.store, session_factory, ["NEWNAME"], adjust="none")
    assert list(bars["close"]) == [100.0, 100.0, 101.0, 102.0]
    assert set(bars["symbol"]) == {"NEWNAME"}


def test_unexpected_parse_failure_is_a_day_error(
    nse: MockNse, tmp_path: Path, session_factory: SF, monkeypatch: pytest.MonkeyPatch
) -> None:
    ing = _ingestor(nse, tmp_path, session_factory)
    real_parse = ing.source.parse

    def flaky(payload):  # type: ignore[no-untyped-def]
        if payload.day == DAYS[1]:
            raise ValueError("weird file")
        return real_parse(payload)

    monkeypatch.setattr(ing.source, "parse", flaky)
    summary = ing.run(DAYS[0], DAYS[3])
    assert summary.days_error == 1
    assert "could not parse" in summary.errors[0]
    assert summary.days_ok == 2


def test_bad_range(nse: MockNse, tmp_path: Path, session_factory: SF) -> None:
    with pytest.raises(ValueError, match="end"):
        _ingestor(nse, tmp_path, session_factory).run(DAYS[1], DAYS[0])


def test_index_prices_and_actions(tmp_path: Path, session_factory: SF) -> None:
    def hist(ticker: str, s: date, e: date) -> pd.DataFrame:
        idx = pd.bdate_range("2024-01-01", periods=3)
        return pd.DataFrame(
            {
                "Open": [1.0, 2, 3],
                "High": [1.5, 2.5, 3.5],
                "Low": [0.5, 1.5, 2.5],
                "Close": [1.0, 2, 3],
                "Volume": 0,
            },
            index=idx,
        )

    def actions(ticker: str) -> pd.DataFrame:
        if ticker == "BAD.NS":
            raise RuntimeError("nope")
        return pd.DataFrame(
            {"Dividends": [2.0], "Stock Splits": [0.0]}, index=pd.DatetimeIndex(["2024-01-02"])
        )

    yahoo = YahooSource(history=hist, actions=actions)
    yahoo._actions = actions  # skip retry backoff in tests
    store = PriceStore(tmp_path, "indices")
    assert (
        ingest_index_prices(yahoo, store, [Symbol("NIFTY50")], date(2024, 1, 1), date(2024, 1, 3))
        == 3
    )
    written, failed = ingest_corporate_actions(
        yahoo, session_factory, [Symbol("GOOD"), Symbol("BAD")], sleep=lambda _: None
    )
    assert (written, failed) == (1, ["BAD"])


def test_cli_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nse: MockNse) -> None:
    from alphaengine.cli import data_commands
    from alphaengine.cli.main import app

    monkeypatch.setenv("ALPHAENGINE_DATABASE_URL", f"sqlite:///{(tmp_path / 'ae.db').as_posix()}")
    monkeypatch.setenv("ALPHAENGINE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(
        data_commands, "NseBhavcopySource", lambda: NseBhavcopySource(nse.fetcher())
    )
    runner = CliRunner()
    assert runner.invoke(app, ["db", "upgrade"]).exit_code == 0
    r = runner.invoke(app, ["data", "info"])
    assert "equities: empty" in r.output
    r = runner.invoke(app, ["ingest", "bhavcopy", "--start", "2024-07-01", "--end", "2024-07-12"])
    assert r.exit_code == 0, r.output
    assert "ok=9 no_data=3" in r.output  # holiday + one weekend (Sat/Sun 6-7 Jul)
    r = runner.invoke(app, ["data", "info"])
    assert "2024-07-01 to 2024-07-12, 9 days" in r.output
    r = runner.invoke(app, ["data", "quality"])
    assert "prev_close_mismatch" in r.output
    acts = tmp_path / "a.csv"
    acts.write_text("symbol,ex_date,kind,ratio,amount\nSPLITCO,2024-07-10,SPLIT,2,\n")
    r = runner.invoke(app, ["import", "actions", str(acts)])
    assert "imported 1 actions" in r.output
    mem = tmp_path / "m.csv"
    mem.write_text("index_name,symbol,start_date,end_date\nNIFTY50,STEADY,2024-01-01,\n")
    assert "imported 1" in runner.invoke(app, ["import", "index-members", str(mem)]).output
