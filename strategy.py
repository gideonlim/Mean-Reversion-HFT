"""Pure math for the lag-1 mean-reversion strategy.

No I/O, no Alpaca imports. Operates on pandas Series/DataFrames.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def add_signal_columns(df: pd.DataFrame, close_col: str = "close") -> pd.DataFrame:
    """Add log_return, lag_1, dir_lag_1, signal, trade_log_return columns.

    Returns a new DataFrame; does not mutate input.
    """
    out = df.copy()
    out["log_return"] = np.log(out[close_col] / out[close_col].shift(1))
    out["log_return_lag_1"] = out["log_return"].shift(1)
    out["dir_lag_1"] = out["log_return_lag_1"].map(
        lambda x: 1 if x > 0 else -1
    )
    out["signal"] = -1 * out["dir_lag_1"]
    out["trade_log_return"] = out["signal"] * out["log_return"]
    return out


def signal_from_lag(log_return_lag_1: float) -> int:
    """Live-execution helper: bet against yesterday's direction.

    Returns +1 if yesterday went down (or was zero/NaN), -1 if yesterday went up.
    Mirrors the lambda in add_signal_columns: NaN > 0 is False, so dir = -1, signal = +1.
    """
    return -1 if log_return_lag_1 > 0 else 1


def groupby_lag_dir(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate today's log_return grouped by yesterday's direction.

    Returns a DataFrame with sum / mean / count for each direction bucket.
    """
    return df.groupby("dir_lag_1").aggregate(
        {"log_return": ["sum", "mean", "count"]}
    )


@dataclass
class ValidationResult:
    full: pd.DataFrame
    in_sample: pd.DataFrame
    out_sample: pd.DataFrame
    edge_holds: bool


def validate_in_out_sample(df: pd.DataFrame, split: float = 0.75) -> ValidationResult:
    """Run the lag-direction groupby on full / in-sample / out-of-sample slices.

    The edge holds iff both in-sample AND out-of-sample show:
        dir_lag_1 == -1 -> mean log_return > 0   (yesterday down -> today up)
        dir_lag_1 == +1 -> mean log_return < 0   (yesterday up   -> today down)
    """
    df = df.dropna(subset=["log_return", "log_return_lag_1"])
    i = int(len(df) * split)
    in_sample, out_sample = df.iloc[:i], df.iloc[i:]

    full_tbl = groupby_lag_dir(df)
    in_tbl = groupby_lag_dir(in_sample)
    out_tbl = groupby_lag_dir(out_sample)

    edge_holds = _has_mean_reverting_signs(in_tbl) and _has_mean_reverting_signs(out_tbl)
    return ValidationResult(full_tbl, in_tbl, out_tbl, edge_holds)


def _has_mean_reverting_signs(tbl: pd.DataFrame) -> bool:
    """Both buckets must exist with the expected mean signs."""
    means = tbl[("log_return", "mean")]
    if -1 not in means.index or 1 not in means.index:
        return False
    return bool(means.loc[-1] > 0 and means.loc[1] < 0)


@dataclass
class Stats:
    win_rate: float
    gross_compound_return: float
    daily_sharpe: float
    annualized_sharpe: float
    n_trades: int


def compute_stats(df: pd.DataFrame, n: int = 252) -> Stats:
    """Compute win rate, gross compound return, and Sharpe ratios from trade_log_return.

    `n` is the number of trading periods per year (252 for US equities, 365 for crypto).
    """
    trades = df["trade_log_return"].dropna()
    is_won = trades > 0
    win_rate = float(is_won.mean()) if len(trades) else 0.0
    gross_compound_return = float(np.exp(trades.sum()) - 1) if len(trades) else 0.0
    mu = float(trades.mean()) if len(trades) else 0.0
    sigma = float(trades.std()) if len(trades) > 1 else 0.0
    # Guard against numerical-noise std on degenerate series (identical values).
    daily_sharpe = mu / sigma if sigma > 1e-12 else 0.0
    annualized_sharpe = float(daily_sharpe * np.sqrt(n))
    return Stats(
        win_rate=win_rate,
        gross_compound_return=gross_compound_return,
        daily_sharpe=daily_sharpe,
        annualized_sharpe=annualized_sharpe,
        n_trades=len(trades),
    )
