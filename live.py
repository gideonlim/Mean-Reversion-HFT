"""Daily live paper-trading runner.

State machine:
  1. Idempotency check (ET date marker)        - exit 0 if already ran today
  2. Pre-flight: market open + trade window     - exit 0 if outside window
  3. Asset tradability + shortability
  4. Cancel any orphan orders for the symbol
  5. Compute signal from last 2 fully-closed daily bars (today's partial bar excluded)
  6. Read current position + account cash
  7. Decide transition (single net-delta order, or close+open flip)
  8. Submit MOC orders with deterministic ET-dated client_order_id
  9. Log JSON DECISION line + write marker file

Run with --dry-run to skip submissions and the marker file.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler
from pathlib import Path

from alpaca.common.exceptions import APIError
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide
from dotenv import load_dotenv

from broker import Broker
from config import SETTINGS, et_now, et_today
from data import last_two_closed_bars
from strategy import signal_from_lag


@dataclass
class IntendedOrder:
    side: OrderSide
    qty: int
    action: str  # "open" | "close" | "scale" | "flip_close" | "flip_open" | "shortable_skip_close"


@dataclass
class Decision:
    transition: str  # "no_op" | "open" | "scale" | "flip" | "shortable_skip"
    orders: list[IntendedOrder] = field(default_factory=list)
    target_qty: int = 0
    delta: int = 0


def decide_transition(
    *,
    signal: int,
    current_qty_signed: int,
    target_abs_qty: int,
    shortable: bool,
) -> Decision:
    """Pure function mapping (signal, current position, target size, shortability)
    to a list of intended orders. Tested directly.
    """
    if signal == -1 and not shortable:
        if current_qty_signed > 0:
            return Decision(
                transition="shortable_skip",
                orders=[IntendedOrder(OrderSide.SELL, current_qty_signed, "shortable_skip_close")],
                target_qty=0,
                delta=-current_qty_signed,
            )
        return Decision(transition="shortable_skip", orders=[], target_qty=0, delta=-current_qty_signed)

    target_qty_signed = signal * target_abs_qty
    delta = target_qty_signed - current_qty_signed

    if delta == 0:
        return Decision(transition="no_op", target_qty=target_qty_signed, delta=0)

    if current_qty_signed * target_qty_signed >= 0:
        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
        action = "open" if current_qty_signed == 0 else "scale"
        return Decision(
            transition=action,
            orders=[IntendedOrder(side, abs(delta), action)],
            target_qty=target_qty_signed,
            delta=delta,
        )

    close_side = OrderSide.SELL if current_qty_signed > 0 else OrderSide.BUY
    open_side = OrderSide.BUY if target_qty_signed > 0 else OrderSide.SELL
    return Decision(
        transition="flip",
        orders=[
            IntendedOrder(close_side, abs(current_qty_signed), "flip_close"),
            IntendedOrder(open_side, abs(target_qty_signed), "flip_open"),
        ],
        target_qty=target_qty_signed,
        delta=delta,
    )


def setup_logging() -> logging.Logger:
    log_dir = Path(SETTINGS.LOG_DIR)
    log_dir.mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    rotating = RotatingFileHandler(log_dir / "live.log", maxBytes=10 * 1024 * 1024, backupCount=12)
    rotating.setFormatter(fmt)
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not root.handlers:
        root.addHandler(rotating)
        root.addHandler(stream)
    return logging.getLogger("live")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mean-reversion live paper-trade runner.")
    parser.add_argument("--dry-run", action="store_true", help="Compute decision; skip order submission and marker.")
    args = parser.parse_args(argv)

    log = setup_logging()
    load_dotenv()

    api_key = os.environ.get("APCA_API_KEY_ID")
    api_secret = os.environ.get("APCA_API_SECRET_KEY")
    if not api_key or not api_secret:
        log.error("APCA_API_KEY_ID and APCA_API_SECRET_KEY must be set in .env")
        return 2

    et_date = et_today()
    marker_path = Path(SETTINGS.LOG_DIR) / f"last_run_{et_date.isoformat()}.json"

    # 1. Idempotency
    if not args.dry_run and marker_path.exists():
        log.info("already ran today (ET=%s); exiting", et_date)
        _write_gha_skip(f"Already ran today (ET={et_date})")
        return 0

    trading_client = TradingClient(api_key, api_secret, paper=True)
    data_client = StockHistoricalDataClient(api_key, api_secret)
    broker = Broker(trading_client)

    # 2. Pre-flight: market open + trade window
    window = broker.todays_trade_window()
    if window is None:
        log.info("market closed today (ET=%s); exiting", et_date)
        _write_gha_skip(f"Market closed (ET={et_date})")
        return 0

    now = et_now()
    if not args.dry_run:
        if now < window.start:
            log.info("too early; window starts %s, now %s; will retry on next scheduled run", window.start, now)
            _write_gha_skip(f"Too early — window opens {window.start.strftime('%H:%M ET')}, now {now.strftime('%H:%M ET')}")
            return 0
        if now > window.end:
            log.warning("missed today's window; window ended %s, now %s", window.end, now)
            _write_gha_skip(f"Missed window — ended {window.end.strftime('%H:%M ET')}, now {now.strftime('%H:%M ET')}")
            return 0

    # 3. Asset tradability
    asset = broker.get_asset(SETTINGS.SYMBOL)
    if not asset.tradable:
        log.warning("symbol %s not tradable today; exiting", SETTINGS.SYMBOL)
        return 0

    # 4. Cancel orphan orders
    cancelled = broker.cancel_open_orders(SETTINGS.SYMBOL)
    if cancelled:
        log.info("cancelled %d orphan order(s) for %s", cancelled, SETTINGS.SYMBOL)

    # 5. Compute signal from closed bars only
    try:
        closed = last_two_closed_bars(SETTINGS.SYMBOL, data_client, today_et=et_date)
    except ValueError as e:
        log.error("could not get closed bars: %s", e)
        return 1
    log_return_lag_1 = closed.log_return_lag_1
    signal = signal_from_lag(log_return_lag_1)

    # 6. Read current state (read once, reuse)
    current_qty = broker.get_position_signed_qty(SETTINGS.SYMBOL)
    equity = broker.get_account_equity()
    target_abs_qty = max(int(equity * SETTINGS.POSITION_FRACTION / closed.prev_close), 0)

    # 7. Decide
    shortable = asset.shortable and asset.easy_to_borrow and not SETTINGS.LONG_ONLY
    decision = decide_transition(
        signal=signal,
        current_qty_signed=current_qty,
        target_abs_qty=target_abs_qty,
        shortable=shortable,
    )

    if decision.transition == "shortable_skip":
        log.warning("symbol %s not currently shortable (shortable=%s, etb=%s); staying flat", SETTINGS.SYMBOL, asset.shortable, asset.easy_to_borrow)

    # 8. Submit orders
    submitted_ids: list[str] = []
    submitted_coids: list[str] = []
    for intended in decision.orders:
        coid = f"meanrev-{et_date.isoformat()}-{SETTINGS.SYMBOL}-{intended.action}"
        if args.dry_run:
            log.info("DRY_RUN would submit: side=%s qty=%d sym=%s coid=%s", intended.side.value, intended.qty, SETTINGS.SYMBOL, coid)
            submitted_coids.append(coid)
            continue
        try:
            o = broker.submit_moc(SETTINGS.SYMBOL, intended.qty, intended.side, coid)
            submitted_ids.append(o.id)
            submitted_coids.append(o.client_order_id)
            log.info("submitted: side=%s qty=%d sym=%s coid=%s id=%s", o.side.value, o.qty, o.symbol, o.client_order_id, o.id)
        except APIError as e:
            if "client_order_id" in str(e).lower() or "duplicate" in str(e).lower():
                log.info("order already submitted today (duplicate coid=%s); idempotent exit", coid)
                return 0
            log.error("MOC submission failed (action=%s, qty=%d): %s", intended.action, intended.qty, e)
            return 1

    # 9. Decision JSON line + marker
    record = {
        "et_run_date": et_date.isoformat(),
        "et_run_time": now.isoformat(),
        "symbol": SETTINGS.SYMBOL,
        "prev_prev_date": closed.prev_prev_date.isoformat(),
        "prev_date": closed.prev_date.isoformat(),
        "prev_prev_close": closed.prev_prev_close,
        "prev_close": closed.prev_close,
        "log_return_lag1": log_return_lag_1,
        "signal": signal,
        "current_qty": current_qty,
        "target_qty": decision.target_qty,
        "delta": decision.delta,
        "transition_type": decision.transition,
        "intended_orders": [
            {"side": o.side.value, "qty": o.qty, "action": o.action} for o in decision.orders
        ],
        "order_ids": submitted_ids,
        "client_order_ids": submitted_coids,
        "account_equity": equity,
        "target_abs_qty": target_abs_qty,
        "shortable": shortable,
        "dry_run": args.dry_run,
    }
    log.info("DECISION %s", json.dumps(record))

    if not args.dry_run:
        marker_path.write_text(json.dumps(record, indent=2))

    _write_gha_summary(record, args.dry_run)
    return 0


def _write_gha_skip(reason: str) -> None:
    """Write a short skip reason to GHA step summary."""
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    try:
        with open(summary_path, "a") as f:
            f.write(f"**Skipped:** {reason}\n")
    except OSError:
        pass


def _write_gha_summary(record: dict, dry_run: bool) -> None:
    """Write a markdown summary to GitHub Actions step summary (if running in GHA)."""
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    signal_label = "LONG (buy)" if record["signal"] == 1 else "SHORT (sell)"
    mode = "DRY RUN" if dry_run else record["transition_type"].upper()
    orders_str = ", ".join(
        f"{o['side']} {o['qty']} ({o['action']})" for o in record["intended_orders"]
    ) or "none"
    md = f"""## {record['symbol']} — {record['et_run_date']}

| | |
|---|---|
| **Signal** | {signal_label} |
| **Transition** | {mode} |
| **Orders** | {orders_str} |
| **Position** | {record['current_qty']} → {record['target_qty']} (delta {record['delta']:+d}) |
| **Equity** | ${record['account_equity']:,.2f} |
| **Target qty** | {record['target_abs_qty']} shares |
| **Prev close** | ${record['prev_close']:.2f} ({record['prev_date']}) |
| **Log return lag-1** | {record['log_return_lag1']:+.6f} |
"""
    try:
        with open(summary_path, "a") as f:
            f.write(md)
    except OSError:
        pass


if __name__ == "__main__":
    sys.exit(main())
