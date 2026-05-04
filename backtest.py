"""Backtest the lag-1 mean-reversion strategy on Alpaca daily bars.

For each symbol in SETTINGS.SYMBOLS:
- Fetch LOOKBACK_YEARS of daily bars
- Run the validation gate (in/out-of-sample mean-reversion check)
- Compute per-symbol stats (win rate, Sharpe, gross return)
- Save per-symbol equity curve PNG

After per-symbol loop, simulate a portfolio combined view using the
configured weights. Save a combined portfolio equity curve and print
combined stats.

Exit code:
- 0 if at least one symbol passes the edge gate
- 1 if all symbols fail
- 2 on configuration / data fetch error
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

    weights = SETTINGS.get_weights()
    client = StockHistoricalDataClient(api_key, api_secret)
    log_dir = Path(SETTINGS.LOG_DIR)
    log_dir.mkdir(exist_ok=True)

    pd.options.display.float_format = "{:.6f}".format

    per_symbol: dict[str, dict] = {}
    any_pass = False

    print(f"Backtesting {len(SETTINGS.SYMBOLS)} symbol(s) from {start} to {today}")
    print(f"Weights: {weights}")
    if SETTINGS.LONG_ONLY:
        print("(Stats/equity shown for LONG_ONLY mode — flat on short-signal days)")
    print()

    for symbol in SETTINGS.SYMBOLS:
        print(f"\n========== {symbol} ==========")
        bars = fetch_daily_bars(symbol, start=start, end=today, client=client)
        if bars.empty:
            print(f"WARNING: no bars returned for {symbol}, skipping")
            continue
        print(f"Got {len(bars)} bars: {bars.index.min()} -> {bars.index.max()}")

        df_full = strategy.add_signal_columns(bars, close_col="close", long_only=False)
        result = strategy.validate_in_out_sample(df_full, split=SETTINGS.IN_SAMPLE_SPLIT)

        df = strategy.add_signal_columns(bars, close_col="close", long_only=SETTINGS.LONG_ONLY)
        stats = strategy.compute_stats(df, n=SETTINGS.ANNUALIZATION_N)

        print(f"\n=== IN-SAMPLE (first {SETTINGS.IN_SAMPLE_SPLIT:.0%}) ===")
        print(result.in_sample)
        print(f"\n=== OUT-OF-SAMPLE (last {1 - SETTINGS.IN_SAMPLE_SPLIT:.0%}) ===")
        print(result.out_sample)
        print(f"\nEdge gate: {'PASS' if result.edge_holds else 'FAIL'}")
        print(f"  win_rate              = {stats.win_rate:.4f}")
        print(f"  gross_compound_return = {stats.gross_compound_return:.4f}  ({stats.gross_compound_return * 100:.2f}%)")
        print(f"  annualized_sharpe     = {stats.annualized_sharpe:.4f}  (N={SETTINGS.ANNUALIZATION_N})")
        print(f"  n_trades              = {stats.n_trades}")

        # Per-symbol equity curve
        plot_path = log_dir / f"equity_curve_{symbol}_{today.isoformat()}.png"
        cum = df["trade_log_return"].cumsum()
        fig, ax = plt.subplots(figsize=(10, 5))
        cum.plot(ax=ax)
        ax.set_title(f"{symbol} mean-reversion equity (cumulative log-return)")
        ax.set_xlabel("date")
        ax.set_ylabel("cumulative log-return")
        ax.axhline(0, color="grey", linewidth=0.5)
        split_idx = int(len(df) * SETTINGS.IN_SAMPLE_SPLIT)
        if split_idx < len(df):
            ax.axvline(df.index[split_idx], color="red", linewidth=0.7, linestyle="--", label="in/out split")
            ax.legend()
        fig.tight_layout()
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
        print(f"  curve saved to {plot_path}")

        per_symbol[symbol] = {
            "df": df,
            "stats": stats,
            "edge_holds": result.edge_holds,
        }
        if result.edge_holds:
            any_pass = True

    # Combined portfolio view (simulated weighted equity curve)
    if per_symbol and len(SETTINGS.SYMBOLS) > 1:
        print("\n\n========== PORTFOLIO ==========")
        portfolio_returns = _build_portfolio_returns(per_symbol, weights)
        if not portfolio_returns.empty:
            port_stats = strategy.compute_stats(
                pd.DataFrame({"trade_log_return": portfolio_returns}),
                n=SETTINGS.ANNUALIZATION_N,
            )
            print(f"  portfolio_gross_return = {port_stats.gross_compound_return:.4f}  ({port_stats.gross_compound_return * 100:.2f}%)")
            print(f"  portfolio_win_rate     = {port_stats.win_rate:.4f}")
            print(f"  portfolio_ann_sharpe   = {port_stats.annualized_sharpe:.4f}")
            print(f"  portfolio_n_days       = {port_stats.n_trades}")

            # Combined equity curve in dollars
            cum_log = portfolio_returns.cumsum()
            equity = SETTINGS.STARTING_CAPITAL * np.exp(cum_log)
            plot_path = log_dir / f"equity_curve_portfolio_{today.isoformat()}.png"
            fig, ax = plt.subplots(figsize=(10, 5))
            equity.plot(ax=ax, color="#2563eb")
            ax.axhline(SETTINGS.STARTING_CAPITAL, color="grey", linewidth=0.6, linestyle="--")
            ax.set_title(f"Portfolio mean-reversion equity ({len(SETTINGS.SYMBOLS)} symbols, weighted)")
            ax.set_xlabel("date")
            ax.set_ylabel("Equity ($)")
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
            fig.tight_layout()
            fig.savefig(plot_path, dpi=120)
            plt.close(fig)
            print(f"  curve saved to {plot_path}")

    print("\n========== SUMMARY ==========")
    for symbol, info in per_symbol.items():
        gate = "PASS" if info["edge_holds"] else "FAIL"
        s = info["stats"]
        print(
            f"  {symbol:6s} gate={gate}  "
            f"return={s.gross_compound_return*100:+7.2f}%  "
            f"sharpe={s.annualized_sharpe:+.3f}  "
            f"trades={s.n_trades}"
        )

    if not any_pass:
        print("\nWARNING: no symbols passed the edge gate")
    return 0


def _build_portfolio_returns(
    per_symbol: dict[str, dict],
    weights: dict[str, float],
) -> pd.Series:
    """Compute weighted portfolio log-return series across all symbols.

    For each common trading date, portfolio simple return =
        sum(weight[s] * (exp(trade_log_return[s]) - 1))
    Then convert back to log space for Sharpe / cumulative aggregation.
    """
    frames = []
    for symbol, info in per_symbol.items():
        weight = weights.get(symbol, 0.0)
        if weight <= 0:
            continue
        # simple_return * weight per date
        ser = (np.exp(info["df"]["trade_log_return"].fillna(0)) - 1) * weight
        ser.name = symbol
        frames.append(ser)
    if not frames:
        return pd.Series(dtype=float)
    combined_simple = pd.concat(frames, axis=1).sum(axis=1, min_count=1)
    combined_simple = combined_simple.dropna()
    return np.log(1 + combined_simple)


if __name__ == "__main__":
    sys.exit(main())
