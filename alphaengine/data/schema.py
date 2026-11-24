"""Canonical daily-bar schema shared by every price source and the Parquet store.

A *bars frame* is a long DataFrame with one row per (date, symbol) and the columns in
:data:`BAR_COLUMNS`. ``date`` is ``datetime64[ns]`` normalized to midnight (no tz; dates are
IST trading dates). Prices are **raw** (unadjusted) unless a loader says otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alphaengine.core.types import make_symbol

BAR_COLUMNS: list[str] = [
    "date",
    "symbol",
    "isin",
    "series",
    "open",
    "high",
    "low",
    "close",
    "prev_close",
    "volume",
    "value",
    "trades",
]
PRICE_COLUMNS = ["open", "high", "low", "close"]
KEY = ["date", "symbol"]

_DTYPES: dict[str, str] = {
    "symbol": "string",
    "isin": "string",
    "series": "string",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "prev_close": "float64",
    "volume": "int64",
    "value": "float64",
    "trades": "Int64",
}


def empty_bars() -> pd.DataFrame:
    df = pd.DataFrame({c: pd.Series(dtype=_DTYPES.get(c, "object")) for c in BAR_COLUMNS})
    df["date"] = pd.Series(dtype="datetime64[ns]")
    return df[BAR_COLUMNS]


def coerce_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Add missing optional columns, coerce dtypes and column order. Does not validate."""
    out = df.copy()
    for col in BAR_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[BAR_COLUMNS]
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
    out["date"] = out["date"].astype("datetime64[ns]")
    for col in ("open", "high", "low", "close", "prev_close", "value"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    vol = pd.to_numeric(out["volume"], errors="coerce")
    out["volume"] = vol.fillna(-1).round().astype("int64")  # -1 marks missing -> rejected
    out["trades"] = pd.to_numeric(out["trades"], errors="coerce").round().astype("Int64")
    for col in ("symbol", "isin", "series"):
        out[col] = out[col].astype("string")
    return out


@dataclass(frozen=True)
class ValidationResult:
    clean: pd.DataFrame
    rejected: pd.DataFrame  # BAR_COLUMNS + "reason"

    @property
    def n_rejected(self) -> int:
        return len(self.rejected)


def _normalize_symbol(raw: object) -> str | None:
    if raw is None or raw is pd.NA or (isinstance(raw, float) and np.isnan(raw)):
        return None
    try:
        return make_symbol(str(raw))
    except ValueError:
        return None


def validate_bars(df: pd.DataFrame) -> ValidationResult:
    """Coerce and validate a bars frame. Invalid rows are returned in ``rejected`` with a reason;
    the first occurrence of a duplicated (date, symbol) is kept."""
    bars = coerce_bars(df)
    reason = pd.Series(pd.NA, index=bars.index, dtype="string")

    def flag(mask: pd.Series, why: str) -> None:
        reason[mask & reason.isna()] = why

    bars["symbol"] = bars["symbol"].map(_normalize_symbol).astype("string")
    flag(bars["symbol"].isna(), "invalid_symbol")
    flag(bars["date"].isna(), "missing_date")
    flag(bars[PRICE_COLUMNS].isna().any(axis=1), "missing_price")
    flag((bars[PRICE_COLUMNS] <= 0).any(axis=1), "non_positive_price")
    flag(bars["volume"] < 0, "missing_or_negative_volume")
    hi_bad = bars["high"] < bars[["open", "close", "low"]].max(axis=1)
    lo_bad = bars["low"] > bars[["open", "close", "high"]].min(axis=1)
    flag(hi_bad | lo_bad, "inconsistent_ohlc")
    flag(bars.duplicated(KEY, keep="first") & reason.isna(), "duplicate")

    bad = reason.notna()
    rejected = bars[bad].assign(reason=reason[bad]).reset_index(drop=True)
    clean = bars[~bad].sort_values(KEY).reset_index(drop=True)
    return ValidationResult(clean=clean, rejected=rejected)
