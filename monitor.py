"""Snapshot account state, positions, and recent orders to a JSON log file.

Called before and after live.py in the GHA workflow to track position changes.
Appends one entry per invocation; the log file is committed back to the repo.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest
from dotenv import load_dotenv

from config import SETTINGS, et_now, et_today

LOG_FILE = Path(SETTINGS.LOG_DIR) / "trade_log.json"


def snapshot(client: TradingClient, label: str) -> dict:
    """Build a snapshot dict of current account state."""
    now = et_now()
    acct = client.get_account()
    positions = client.get_all_positions()

    pos_list = []
    for p in positions:
        pos_list.append({
            "symbol": p.symbol,
            "side": str(p.side).split(".")[-1].lower(),
            "qty": int(float(p.qty)),
            "avg_entry_price": float(p.avg_entry_price),
            "market_value": float(p.market_value),
            "unrealized_pl": float(p.unrealized_pl),
            "unrealized_plpc": float(p.unrealized_plpc) if p.unrealized_plpc else 0.0,
        })

    # Recent orders for our symbol (last 2 days)
    since = (now - timedelta(days=2)).isoformat()
    try:
        orders = client.get_orders(filter=GetOrdersRequest(
            status=QueryOrderStatus.ALL,
            after=since,
            symbols=[SETTINGS.SYMBOL],
            limit=10,
        ))
    except Exception:
        orders = []

    order_list = []
    for o in sorted(orders, key=lambda x: x.created_at):
        order_list.append({
            "created_at": str(o.created_at)[:19],
            "side": str(o.side).split(".")[-1].lower(),
            "qty": int(float(o.qty)),
            "filled_qty": int(float(o.filled_qty)) if o.filled_qty else 0,
            "time_in_force": str(o.time_in_force).split(".")[-1].lower(),
            "status": str(o.status).split(".")[-1].lower(),
            "filled_avg_price": float(o.filled_avg_price) if o.filled_avg_price else None,
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
        "recent_orders": order_list,
    }


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
    parser.add_argument("label", choices=["pre", "post"], help="Snapshot label: 'pre' or 'post' trade")
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

    print(f"[{snap['label']}] equity=${snap['account']['equity']:,.2f}  "
          f"positions={len(snap['positions'])}  "
          f"orders={len(snap['recent_orders'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
