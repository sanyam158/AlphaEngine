from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from alphaengine.core.errors import DataError
from alphaengine.data.db import session_scope
from alphaengine.data.importers import read_actions_csv, read_membership_csv
from alphaengine.data.models import Instrument, SymbolHistory
from alphaengine.data.repositories import (
    InstrumentRegistry,
    completed_days,
    import_index_membership,
    load_corporate_actions,
    members_as_of,
    record_index_snapshot,
    record_ingest,
    upsert_corporate_actions,
)
from alphaengine.data.sources.nse_indices import Constituent
from alphaengine.data.universe import index_universe

SF = sessionmaker[Session]


def _day_bars(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["isin", "symbol"]).assign(series="EQ")


class TestInstrumentRegistry:
    def test_create_and_rename(self, session_factory: SF) -> None:
        with session_scope(session_factory) as s:
            reg = InstrumentRegistry(s)
            reg.observe(date(2024, 1, 1), _day_bars([("INE1", "OLDCO"), ("INE2", "OTHER")]))
            reg.observe(date(2024, 1, 2), _day_bars([("INE1", "OLDCO")]))
            reg.observe(date(2024, 1, 3), _day_bars([("INE1", "NEWCO")]))
            reg.observe(date(2023, 12, 29), _day_bars([("INE1", "ANCIENT")]))  # out of order
            assert (reg.created, reg.renamed, len(reg)) == (2, 1, 2)
        with session_scope(session_factory) as s:
            inst = s.scalars(select(Instrument).where(Instrument.isin == "INE1")).one()
            assert inst.symbol == "NEWCO"
            assert (inst.first_seen, inst.last_seen) == (date(2023, 12, 29), date(2024, 1, 3))
            hist = s.scalars(select(SymbolHistory).order_by(SymbolHistory.valid_from)).all()
            assert [(h.isin, h.symbol, h.valid_from, h.valid_to) for h in hist] == [
                ("INE1", "OLDCO", date(2024, 1, 1), date(2024, 1, 3)),
                ("INE2", "OTHER", date(2024, 1, 1), None),
                ("INE1", "NEWCO", date(2024, 1, 3), None),
            ]
            # A fresh registry reloads state from the DB
            reg2 = InstrumentRegistry(s)
            assert reg2.get("INE1") is not None
            reg2.observe(date(2024, 1, 4), _day_bars([("INE1", "NEWCO")]))
            assert reg2.renamed == 0


class TestCorporateActions:
    def test_upsert_and_load(self, session_factory: SF) -> None:
        acts = pd.DataFrame(
            [
                {
                    "symbol": "abc",
                    "ex_date": "2024-01-03",
                    "kind": "split",
                    "ratio": 2,
                    "amount": None,
                },
                {
                    "symbol": "ABC",
                    "ex_date": "2024-02-01",
                    "kind": "DIVIDEND",
                    "ratio": None,
                    "amount": 5,
                },
            ]
        )
        with session_scope(session_factory) as s:
            assert upsert_corporate_actions(s, acts, source="t") == 2
            upd = acts.iloc[[1]].assign(amount=6.5)
            upsert_corporate_actions(s, upd, source="t2")
        with session_scope(session_factory) as s:
            df = load_corporate_actions(s, ["ABC"])
            assert list(df["kind"]) == ["SPLIT", "DIVIDEND"]
            assert df["amount"].iloc[1] == 6.5
            assert df["source"].iloc[1] == "t2"
            assert load_corporate_actions(s, ["NONE"]).empty

    def test_classified_split_or_bonus_wins(self, session_factory: SF) -> None:
        def row(kind: str) -> pd.DataFrame:
            return pd.DataFrame(
                [
                    {
                        "symbol": "ABC",
                        "ex_date": "2024-10-28",
                        "kind": kind,
                        "ratio": 2,
                        "amount": None,
                    }
                ]
            )

        with session_scope(session_factory) as s:
            upsert_corporate_actions(s, row("SPLIT_OR_BONUS"), source="yahoo")
            upsert_corporate_actions(s, row("BONUS"), source="manual")  # replaces ambiguous
            upsert_corporate_actions(s, row("SPLIT_OR_BONUS"), source="yahoo")  # ignored
        with session_scope(session_factory) as s:
            df = load_corporate_actions(s, ["ABC"])
            assert list(zip(df["kind"], df["source"], strict=True)) == [("BONUS", "manual")]

    @pytest.mark.parametrize(
        ("row", "msg"),
        [
            ({"kind": "MERGER", "ratio": 2, "amount": None}, "unknown action kind"),
            ({"kind": "SPLIT", "ratio": 1, "amount": None}, "ratio"),
            ({"kind": "BONUS", "ratio": None, "amount": None}, "ratio"),
            ({"kind": "DIVIDEND", "ratio": None, "amount": 0}, "amount"),
        ],
    )
    def test_validation(self, session_factory: SF, row: dict[str, object], msg: str) -> None:
        df = pd.DataFrame([{"symbol": "ABC", "ex_date": "2024-01-03", **row}])
        with pytest.raises(DataError, match=msg), session_scope(session_factory) as s:
            upsert_corporate_actions(s, df)


class TestIndexMembership:
    def _c(self, *syms: str) -> list[Constituent]:
        return [Constituent(s, f"ISIN{s}", f"{s} Ltd", "Tech") for s in syms]  # type: ignore[arg-type]

    def test_snapshots_build_point_in_time_history(self, session_factory: SF) -> None:
        d1, d2 = date(2024, 1, 1), date(2024, 4, 1)
        with session_scope(session_factory) as s:
            s.add(Instrument(isin="ISINA", symbol="A", first_seen=d1, last_seen=d1))
            diff = record_index_snapshot(s, "nifty50", self._c("A", "B"), d1)
            assert diff.added == {"A", "B"} and not diff.removed
            diff = record_index_snapshot(s, "NIFTY50", self._c("A", "C"), d2)
            assert diff.added == {"C"} and diff.removed == {"B"}
            with pytest.raises(DataError, match="older"):
                record_index_snapshot(s, "NIFTY50", self._c("A"), date(2023, 1, 1))
        with session_scope(session_factory) as s:
            assert members_as_of(s, "NIFTY50", date(2023, 12, 31)) == frozenset()
            assert members_as_of(s, "NIFTY50", date(2024, 3, 31)) == {"A", "B"}
            assert members_as_of(s, "NIFTY50", d2) == {"A", "C"}
            inst = s.scalars(select(Instrument).where(Instrument.isin == "ISINA")).one()
            assert inst.industry == "Tech"
            assert index_universe(s, "NIFTY50", date(2020, 1, 1)) == frozenset()  # warns

    def test_import_from_csv(self, session_factory: SF, tmp_path: Path) -> None:
        p = tmp_path / "m.csv"
        p.write_text(
            "index_name,symbol,start_date,end_date\n"
            "NIFTY50,AAA,2015-01-01,2018-06-29\nNIFTY50,BBB,2015-01-01,\n"
        )
        with session_scope(session_factory) as s:
            assert import_index_membership(s, read_membership_csv(p)) == 2
        with session_scope(session_factory) as s:
            assert members_as_of(s, "NIFTY50", date(2017, 1, 1)) == {"AAA", "BBB"}
            assert members_as_of(s, "NIFTY50", date(2018, 6, 29)) == {"BBB"}
        bad = tmp_path / "bad.csv"
        bad.write_text("index_name,symbol,start_date,end_date\nN,AAA,2015-01-01,2014-01-01\n")
        with pytest.raises(DataError, match="after"), session_scope(session_factory) as s:
            import_index_membership(s, read_membership_csv(bad))
        missing = tmp_path / "missing.csv"
        missing.write_text("symbol\nAAA\n")
        with pytest.raises(DataError, match="missing columns"):
            read_membership_csv(missing)


def test_actions_csv(tmp_path: Path) -> None:
    p = tmp_path / "a.csv"
    p.write_text(
        "symbol,ex_date,kind,ratio,amount\nABC,2024-01-03,BONUS,2,\nABC,2024-02-01,DIVIDEND,,4.5\n"
    )
    df = read_actions_csv(p)
    assert df["ratio"].iloc[0] == 2
    assert df["amount"].iloc[1] == 4.5


def test_ingest_log(session_factory: SF) -> None:
    with session_scope(session_factory) as s:
        record_ingest(s, source="x", day=date(2024, 1, 1), status="ok", run_id="r", rows=5)
        record_ingest(s, source="x", day=date(2024, 1, 2), status="error", run_id="r")
        record_ingest(s, source="x", day=date(2024, 1, 2), status="no_data", run_id="r2")
    with session_scope(session_factory) as s:
        assert completed_days(s, "x", date(2024, 1, 1), date(2024, 1, 31)) == {
            date(2024, 1, 1),
            date(2024, 1, 2),
        }
        # a recent "no data" day is fetched again (the file may not have been published yet)
        assert completed_days(
            s, "x", date(2024, 1, 1), date(2024, 1, 31), recheck_after=date(2024, 1, 2)
        ) == {date(2024, 1, 1)}
