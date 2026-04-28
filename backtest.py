"""Backtest the lag-1 mean-reversion strategy on Alpaca daily bars.

Pulls LOOKBACK_YEARS of daily bars for SETTINGS.SYMBOL, computes the validation
groupby on full / in-sample / out-of-sample slices, prints stats, saves the
equity curve PNG. Exits non-zero if the in/out-of-sample edge gate fails.
"""
from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from dotenv import load_dotenv

import strategy
from config import SETTINGS, et_today
from data import fetch_daily_bars


def main() -> int:
    load_dotenv()
    api_key = os.environ.get("APCA_API_KEY_ID")
    api_secret = os.environ.get("APCA_API_SECRET_KEY")
    if not api_key or not api_secret:
        print("ERROR: APCA_API_KEY_ID and APCA_API_SECRET_KEY must be set in .env", file=sys.stderr)
        return 2

    today = et_today()
    try:
        start = date(today.year - SETTINGS.LOOKBACK_YEARS, today.month, today.day)
    except ValueError:
        start = date(today.year - SETTINGS.LOOKBACK_YEARS, today.month, today.day - 1)
    print(f"Fetching daily bars for {SETTINGS.SYMBOL} from {start} to {today}...")

    client = StockHistoricalDataClient(api_key, api_secret)
    bars = fetch_daily_bars(SETTINGS.SYMBOL, start=start, end=today, client=client)
    if bars.empty:
        print("ERROR: no bars returned", file=sys.stderr)
        return 2
    print(f"Got {len(bars)} bars: {bars.index.min()} -> {bars.index.max()}")

    df = strategy.add_signal_columns(bars, close_col="close")
    result = strategy.validate_in_out_sample(df, split=SETTINGS.IN_SAMPLE_SPLIT)
    stats = strategy.compute_stats(df, n=SETTINGS.ANNUALIZATION_N)

    pd.options.display.float_format = "{:.6f}".format
    print("\n=== Lag-direction groupby (FULL) ===")
    print(result.full)
    print(f"\n=== IN-SAMPLE (first {SETTINGS.IN_SAMPLE_SPLIT:.0%}) ===")
    print(result.in_sample)
    print(f"\n=== OUT-OF-SAMPLE (last {1 - SETTINGS.IN_SAMPLE_SPLIT:.0%}) ===")
    print(result.out_sample)

    print("\n=== Stats ===")
    print(f"win_rate              = {stats.win_rate:.4f}")
    print(f"gross_compound_return = {stats.gross_compound_return:.4f}  ({stats.gross_compound_return * 100:.2f}%)")
    print(f"daily_sharpe          = {stats.daily_sharpe:.4f}")
    print(f"annualized_sharpe     = {stats.annualized_sharpe:.4f}  (N={SETTINGS.ANNUALIZATION_N})")
    print(f"n_trades              = {stats.n_trades}")

    print(f"\nEdge gate (in & out both mean-reverting): {'PASS' if result.edge_holds else 'FAIL'}")

    log_dir = Path(SETTINGS.LOG_DIR)
    log_dir.mkdir(exist_ok=True)
    plot_path = log_dir / f"equity_curve_{SETTINGS.SYMBOL}_{today.isoformat()}.png"
    cum = df["trade_log_return"].cumsum()
    fig, ax = plt.subplots(figsize=(10, 5))
    cum.plot(ax=ax)
    ax.set_title(f"{SETTINGS.SYMBOL} mean-reversion equity (cumulative log-return)")
    ax.set_xlabel("date")
    ax.set_ylabel("cumulative log-return")
    ax.axhline(0, color="grey", linewidth=0.5)
    split_idx = int(len(df) * SETTINGS.IN_SAMPLE_SPLIT)
    if split_idx < len(df):
        ax.axvline(df.index[split_idx], color="red", linewidth=0.7, linestyle="--", label="in/out split")
        ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    print(f"\nEquity curve saved to {plot_path}")

    return 0 if result.edge_holds else 1


if __name__ == "__main__":
    sys.exit(main())
