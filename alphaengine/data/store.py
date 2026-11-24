"""Curated price store: Parquet files partitioned by year, queried through DuckDB.

Layout: ``<root>/<dataset>/year=YYYY/data.parquet``. Writes are upserts keyed on
(date, symbol) and are atomic per file (write temp file, then ``os.replace``).
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

from alphaengine.core.errors import DataError
from alphaengine.data.schema import BAR_COLUMNS, KEY, coerce_bars, empty_bars

_FILE = "data.parquet"


class PriceStore:
    def __init__(self, root: Path, dataset: str = "equities") -> None:
        if not dataset.isidentifier():
            raise ValueError(f"invalid dataset name {dataset!r}")
        self.root = root
        self.dataset = dataset
        self.base = root / dataset

    # ------------------------------------------------------------------ write
    def write(self, bars: pd.DataFrame, *, replace_dates: bool = False) -> int:
        """Upsert bars: new rows replace existing rows with the same (date, symbol).

        With ``replace_dates`` every existing row on a date present in ``bars`` is dropped
        first. Use it for whole-market daily files (bhavcopy), so a re-ingest also removes rows
        that a newer parse no longer produces."""
        if bars.empty:
            return 0
        new = coerce_bars(bars)
        if new.duplicated(KEY).any():
            raise DataError("bars to write contain duplicate (date, symbol) keys")
        for year, chunk in new.groupby(new["date"].dt.year):
            path = self._year_path(int(year))
            if path.exists():
                old = pd.read_parquet(path)
                if replace_dates:
                    old = old[~old["date"].isin(chunk["date"].unique())]
                else:
                    old = old.set_index(KEY)
                    old = old[~old.index.isin(chunk.set_index(KEY).index)].reset_index()
                merged = pd.concat([coerce_bars(old), chunk], ignore_index=True)
            else:
                merged = chunk
            merged = merged.sort_values(KEY).reset_index(drop=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".parquet.tmp")
            merged.to_parquet(tmp, index=False)
            os.replace(tmp, path)
        return len(new)

    # ------------------------------------------------------------------ read
    def files(self) -> list[Path]:
        return sorted(self.base.glob(f"year=*/{_FILE}")) if self.base.exists() else []

    @contextmanager
    def connect(self) -> Iterator[duckdb.DuckDBPyConnection]:
        """DuckDB connection with a ``bars`` view over the dataset (empty view if no data)."""
        con = duckdb.connect()
        try:
            files = self.files()
            if files:
                listing = ", ".join(f"'{p.as_posix()}'" for p in files)
                con.execute(f"CREATE VIEW bars AS SELECT * FROM read_parquet([{listing}])")
            else:
                con.register("_empty", empty_bars())
                con.execute("CREATE VIEW bars AS SELECT * FROM _empty")
            yield con
        finally:
            con.close()

    def read(
        self,
        symbols: Sequence[str] | None = None,
        start: date | None = None,
        end: date | None = None,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        cols = list(columns) if columns else BAR_COLUMNS
        unknown = set(cols) - set(BAR_COLUMNS)
        if unknown:
            raise ValueError(f"unknown columns {sorted(unknown)}")
        for k in reversed(KEY):
            if k not in cols:
                cols.insert(0, k)
        where: list[str] = []
        params: list[object] = []
        if symbols is not None:
            if not symbols:
                return empty_bars()[cols]
            where.append(f"symbol IN ({', '.join('?' for _ in symbols)})")
            params.extend(symbols)
        if start is not None:
            where.append("date >= ?")
            params.append(pd.Timestamp(start).to_pydatetime())
        if end is not None:
            where.append("date <= ?")
            params.append(pd.Timestamp(end).to_pydatetime())
        sql = f"SELECT {', '.join(cols)} FROM bars"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY date, symbol"
        with self.connect() as con:
            df = con.execute(sql, params).df()
        if df.empty:
            return empty_bars()[cols]
        if set(cols) == set(BAR_COLUMNS):
            return coerce_bars(df)
        df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
        return df

    def trading_days(self) -> list[date]:
        with self.connect() as con:
            rows = con.execute("SELECT DISTINCT date FROM bars ORDER BY date").fetchall()
        return [pd.Timestamp(r[0]).date() for r in rows]

    def date_range(self) -> tuple[date, date] | None:
        with self.connect() as con:
            lo, hi = con.execute("SELECT min(date), max(date) FROM bars").fetchone() or (None, None)
        if lo is None:
            return None
        return pd.Timestamp(lo).date(), pd.Timestamp(hi).date()

    def symbols(self) -> list[str]:
        with self.connect() as con:
            rows = con.execute("SELECT DISTINCT symbol FROM bars ORDER BY symbol").fetchall()
        return [str(r[0]) for r in rows]

    def snapshot_id(self) -> str:
        """Content hash of the dataset; recorded by backtests for reproducibility."""
        h = hashlib.sha256(self.dataset.encode())
        for path in self.files():
            h.update(path.relative_to(self.base).as_posix().encode())
            h.update(hashlib.sha256(path.read_bytes()).digest())
        return h.hexdigest()[:16]

    def _year_path(self, year: int) -> Path:
        return self.base / f"year={year}" / _FILE
