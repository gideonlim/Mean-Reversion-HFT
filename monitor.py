"""Snapshot account state, positions, and open orders to a JSON log file.

Called before and after live.py in the GHA workflow to track position changes.
Also runs as standalone market-open / market-close workflows.
Appends one entry per invocation; the log file is committed back to the repo.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest
from dotenv import load_dotenv

from config import SETTINGS, et_now, et_today

LOG_FILE = Path(SETTINGS.LOG_DIR) / "trade_log.json"


def _enum_value(e) -> str:
    """Robustly extract a string from an Alpaca enum (e.g. OrderSide.BUY -> 'buy').

    Prefers .value (which alpaca-py exposes), falls back to repr parsing.
    """
    if hasattr(e, "value"):
        return str(e.value).lower()
    return str(e).split(".")[-1].lower()


def snapshot(client: TradingClient, label: str) -> dict:
    """Build a snapshot dict of current account state."""
    now = et_now()
    acct = client.get_account()
    positions = client.get_all_positions()

    pos_list = []
    for p in positions:
        pos_list.append({
            "symbol": p.symbol,
            "side": _enum_value(p.side),
            "qty": int(float(p.qty)),
            "avg_entry_price": float(p.avg_entry_price),
            "market_value": float(p.market_value),
            "unrealized_pl": float(p.unrealized_pl),
            "unrealized_plpc": float(p.unrealized_plpc) if p.unrealized_plpc else 0.0,
        })

    # Only open/pending orders — filled/cancelled are historical, not state.
    # Report.py fetches its own full order history directly from Alpaca.
    try:
        orders = client.get_orders(filter=GetOrdersRequest(
            status=QueryOrderStatus.OPEN,
            symbols=list(SETTINGS.SYMBOLS),
            limit=50,
        ))
    except Exception:
        orders = []

    order_list = []
    for o in sorted(orders, key=lambda x: x.created_at):
        order_list.append({
            "created_at": str(o.created_at)[:19],
            "symbol": o.symbol,
            "side": _enum_value(o.side),
            "qty": int(float(o.qty)),
            "filled_qty": int(float(o.filled_qty)) if o.filled_qty else 0,
            "time_in_force": _enum_value(o.time_in_force),
            "status": _enum_value(o.status),
            "client_order_id": o.client_order_id,
        })

    return {
        "timestamp": now.isoformat(),
        "et_date": et_today().isoformat(),
        "label": label,
        "account": {
            "equity": float(acct.equity),
            "cash": float(acct.cash),
            "buying_power": float(acct.buying_power),
            "portfolio_value": float(acct.portfolio_value) if acct.portfolio_value else float(acct.equity),
        },
        "positions": pos_list,
        "open_orders": order_list,
    }


def _print_summary(snap: dict) -> None:
    """Print a formatted monitor summary to console."""
    label = snap["label"]
    acct = snap["account"]
    positions = snap["positions"]
    orders = snap["open_orders"]

    sep = "=" * 60
    dash = "-" * 60

    print(sep)
    print(f"  MONITOR SNAPSHOT ({label})")
    print(sep)
    print(f"  Account equity: ${acct['equity']:,.2f}  "
          f"Cash: ${acct['cash']:,.2f}  "
          f"Positions: {len(positions)}")

    if positions:
        print(dash)
        for p in positions:
            side_label = p["side"].upper()
            pnl = p["unrealized_pl"]
            pnl_pct = p["unrealized_plpc"] * 100
            sign = "+" if pnl >= 0 else ""
            print(f"  {p['symbol']}: {side_label} {p['qty']} shares "
                  f"@ ${p['avg_entry_price']:.2f}  "
                  f"P&L: {sign}${pnl:,.2f} ({sign}{pnl_pct:.2f}%)")

    print(dash)
    if orders:
        print(f"  Open orders: {len(orders)}")
        for o in orders:
            print(f"    {o['symbol']} {o['side'].upper()} {o['qty']} "
                  f"- {o['status']} ({o['time_in_force']})  "
                  f"coid: {o['client_order_id']}")
    else:
        print("  Open orders: none")

    print(sep)


def load_log() -> list[dict]:
    if LOG_FILE.exists():
        try:
            return json.loads(LOG_FILE.read_text())
        except (json.JSONDecodeError, ValueError):
            return []
    return []


def save_log(entries: list[dict]) -> None:
    LOG_FILE.parent.mkdir(exist_ok=True)
    LOG_FILE.write_text(json.dumps(entries, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Snapshot account/positions to trade_log.json")
    parser.add_argument(
        "label",
        choices=["pre", "post", "market_open", "market_close"],
        help="Snapshot label: 'pre'/'post' (around live trade), 'market_open'/'market_close' (standalone)",
    )
    args = parser.parse_args(argv)

    load_dotenv()
    api_key = os.environ.get("APCA_API_KEY_ID")
    api_secret = os.environ.get("APCA_API_SECRET_KEY")
    if not api_key or not api_secret:
        print("ERROR: APCA_API_KEY_ID and APCA_API_SECRET_KEY must be set", file=sys.stderr)
        return 2

    client = TradingClient(api_key, api_secret, paper=True)
    snap = snapshot(client, args.label)

    entries = load_log()
    entries.append(snap)
    save_log(entries)

    _print_summary(snap)
    return 0


if __name__ == "__main__":
    sys.exit(main())
