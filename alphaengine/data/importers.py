"""CSV importers for data that has no free API (historical index changes, manual actions).

Corporate actions CSV: ``symbol,ex_date,kind,ratio,amount``
  - SPLIT/BONUS ``ratio`` = shares held after per share before (2-for-1 split → 2; 1:1 bonus → 2;
    3:2 bonus i.e. 3 new for every 2 held → 2.5)
  - DIVIDEND ``amount`` = raw ₹ per share as declared

Index membership CSV: ``index_name,symbol,start_date,end_date[,isin]`` (end_date blank = current)
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd

from alphaengine.core.errors import DataError

ACTION_REQUIRED = {"symbol", "ex_date", "kind"}
MEMBERSHIP_REQUIRED = {"index_name", "symbol", "start_date"}


def _read(path: Path, required: set[str]) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = required - set(df.columns)
    if missing:
        raise DataError(f"{path.name}: missing columns {sorted(missing)}")
    df = df.apply(lambda c: c.str.strip())
    return cast(pd.DataFrame, df.replace({"": None}))


def read_actions_csv(path: Path) -> pd.DataFrame:
    df = _read(path, ACTION_REQUIRED)
    for col in ("ratio", "amount"):
        df[col] = pd.to_numeric(df[col], errors="raise") if col in df.columns else None
    return df


def read_membership_csv(path: Path) -> pd.DataFrame:
    return _read(path, MEMBERSHIP_REQUIRED)
