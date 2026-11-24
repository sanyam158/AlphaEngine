"""Yahoo Finance adapter (optional extra: ``pip install -e .[yahoo]``).

Used for **index levels** (Nifty 50 etc.) and **corporate actions** (splits/bonuses and
dividends). Yahoo reports bonus issues as splits, so share-count events are recorded as
``SPLIT_OR_BONUS``: fine for price adjustment, but they need classifying for tax lots.

Not used for equity prices: Yahoo's ``Close`` is already split-adjusted, while the curated
store holds raw exchange prices and applies adjustments itself.

Yahoo reports dividends adjusted for *later* splits; :meth:`YahooSource.fetch_actions`
converts them back to the raw per-share amount declared on the ex-date.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date, timedelta
from typing import cast

import numpy as np
import pandas as pd

from alphaengine.core.errors import DataError, DataSourceError
from alphaengine.core.retry import retry
from alphaengine.core.types import Symbol
from alphaengine.data.schema import BAR_COLUMNS, coerce_bars, empty_bars
from alphaengine.data.sources.base import CORPORATE_ACTION_COLUMNS

INDEX_TICKERS: dict[str, str] = {
    "NIFTY50": "^NSEI",
    "NIFTYBANK": "^NSEBANK",
    "NIFTY500": "^CRSLDX",
    "SENSEX": "^BSESN",
}

HistoryFn = Callable[[str, date, date], pd.DataFrame]
ActionsFn = Callable[[str], pd.DataFrame]


def to_ticker(symbol: str) -> str:
    return INDEX_TICKERS.get(symbol, f"{symbol}.NS")


def _yf_history(ticker: str, start: date, end: date) -> pd.DataFrame:  # pragma: no cover - network
    import yfinance as yf  # noqa: PLC0415 - optional dependency

    return cast(
        pd.DataFrame,
        yf.Ticker(ticker).history(
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),  # yfinance end is exclusive
            auto_adjust=False,
            actions=True,
        ),
    )


def _yf_actions(ticker: str) -> pd.DataFrame:  # pragma: no cover - network
    import yfinance as yf  # noqa: PLC0415 - optional dependency

    return cast(pd.DataFrame, yf.Ticker(ticker).actions)


class YahooSource:
    name = "yahoo"

    def __init__(self, history: HistoryFn | None = None, actions: ActionsFn | None = None) -> None:
        wrap = retry(attempts=3, base_delay=2.0)
        self._history: HistoryFn = wrap(history or _yf_history)
        self._actions: ActionsFn = wrap(actions or _yf_actions)

    def fetch_bars(
        self, start: date, end: date, symbols: Sequence[Symbol] | None = None
    ) -> pd.DataFrame:
        if not symbols:
            raise DataError("YahooSource.fetch_bars requires explicit symbols")
        frames = []
        for sym in symbols:
            try:
                hist = self._history(to_ticker(sym), start, end)
            except Exception as exc:
                raise DataSourceError(f"yahoo history failed for {sym}: {exc!r}") from exc
            if hist is None or hist.empty:
                continue
            idx = pd.DatetimeIndex(hist.index)
            if idx.tz is not None:
                idx = idx.tz_localize(None)
            df = pd.DataFrame(
                {
                    "date": idx.normalize(),
                    "symbol": sym,
                    "open": hist["Open"].to_numpy(),
                    "high": hist["High"].to_numpy(),
                    "low": hist["Low"].to_numpy(),
                    "close": hist["Close"].to_numpy(),
                    "volume": hist["Volume"].fillna(0).to_numpy(),
                }
            )
            df["prev_close"] = df["close"].shift(1)
            frames.append(df)
        if not frames:
            return empty_bars()
        out = coerce_bars(pd.concat(frames, ignore_index=True))
        out = out[(out["date"] >= pd.Timestamp(start)) & (out["date"] <= pd.Timestamp(end))]
        return out.sort_values(["date", "symbol"]).reset_index(drop=True)[BAR_COLUMNS]

    def fetch_actions(self, symbol: Symbol) -> pd.DataFrame:
        """Corporate actions with raw dividend amounts. Columns: CORPORATE_ACTION_COLUMNS."""
        try:
            raw = self._actions(to_ticker(symbol))
        except Exception as exc:
            raise DataSourceError(f"yahoo actions failed for {symbol}: {exc!r}") from exc
        if raw is None or raw.empty:
            return pd.DataFrame(columns=CORPORATE_ACTION_COLUMNS)
        idx = pd.DatetimeIndex(raw.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        ex = idx.normalize()
        splits = raw.get("Stock Splits", pd.Series(0.0, index=raw.index)).fillna(0).to_numpy()
        divs = raw.get("Dividends", pd.Series(0.0, index=raw.index)).fillna(0).to_numpy()

        rows: list[dict[str, object]] = []
        split_dates = ex[splits > 0].to_numpy()
        split_ratios = splits[splits > 0].astype(float)
        for d, r in zip(split_dates, split_ratios, strict=True):
            rows.append({"ex_date": d, "kind": "SPLIT_OR_BONUS", "ratio": float(r), "amount": None})
        for d, amt in zip(ex.to_numpy(), divs, strict=True):
            if amt <= 0:
                continue
            later = split_ratios[split_dates > d]
            raw_amount = float(amt) * float(np.prod(later)) if later.size else float(amt)
            rows.append(
                {"ex_date": d, "kind": "DIVIDEND", "ratio": None, "amount": round(raw_amount, 4)}
            )
        out = pd.DataFrame(rows, columns=["ex_date", "kind", "ratio", "amount"])
        out.insert(0, "symbol", str(symbol))
        out["source"] = self.name
        out["ex_date"] = pd.to_datetime(out["ex_date"]).dt.date
        return out.sort_values(["ex_date", "kind"]).reset_index(drop=True)[CORPORATE_ACTION_COLUMNS]
