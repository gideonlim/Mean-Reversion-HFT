"""Alpaca historical-bars wrapper. All bar timestamps are normalized to ET dates."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from config import ET, et_today


def fetch_daily_bars(
    symbol: str,
    start: date,
    end: date | None,
    client: StockHistoricalDataClient,
) -> pd.DataFrame:
    """Fetch daily bars from Alpaca. Returns a single-index DataFrame keyed by ET date.

    Columns include: open, high, low, close, volume.
    Today's partial bar (if returned mid-session) is *not* filtered here — callers
    that need only fully-closed bars should use `last_two_closed_bars` or filter on
    `< et_today()` themselves.
    """
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=datetime.combine(start, datetime.min.time()),
        end=datetime.combine(end, datetime.min.time()) if end else None,
    )
    bars = client.get_stock_bars(req)
    df = bars.df
    if df.empty:
        return df
    df = df.reset_index()
    df["et_date"] = df["timestamp"].dt.tz_convert(ET).dt.date
    df = df.set_index("et_date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df.drop(columns=["symbol", "timestamp"], errors="ignore")


@dataclass
class ClosedBars:
    prev_prev_date: date
    prev_date: date
    prev_prev_close: float
    prev_close: float

    @property
    def log_return_lag_1(self) -> float:
        return float(np.log(self.prev_close / self.prev_prev_close))


def last_two_closed_bars(
    symbol: str,
    client: StockHistoricalDataClient,
    today_et: date | None = None,
) -> ClosedBars:
    """Return the two most recent fully-closed daily bars (date < ET today).

    Pulls a 14-day lookback to safely span weekends and holidays.
    Raises ValueError if fewer than 2 closed bars are available.

    `today_et` override is for testing; production calls leave it None.
    """
    if today_et is None:
        today_et = et_today()
    start = today_et - timedelta(days=14)
    df = fetch_daily_bars(symbol, start=start, end=today_et, client=client)
    return last_two_closed_bars_from_df(df, today_et)


def last_two_closed_bars_from_df(df: pd.DataFrame, today_et: date) -> ClosedBars:
    """Pure helper: extract the last two closed bars from a fetched DataFrame.

    Split out so tests can drive it without a live Alpaca client.
    """
    closed = df[df.index < today_et]
    if len(closed) < 2:
        raise ValueError(
            f"Need >= 2 closed daily bars, got {len(closed)}. "
            "Symbol may be new or a long holiday window may have skipped bars."
        )
    prev_prev_date, prev_date = closed.index[-2], closed.index[-1]
    return ClosedBars(
        prev_prev_date=prev_prev_date,
        prev_date=prev_date,
        prev_prev_close=float(closed["close"].iloc[-2]),
        prev_close=float(closed["close"].iloc[-1]),
    )
