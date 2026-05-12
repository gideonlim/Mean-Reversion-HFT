"""Alpaca trading I/O wrapper.

All trading-side calls live here so the orchestrator (live.py) can be tested with
a mock Broker, and so future broker swaps don't reach into other modules.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, PositionIntent, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest

from config import ET

log = logging.getLogger(__name__)


# MOC submission window is [close - 150min, close - 11min].
# Wide start (2.5h) because GitHub Actions throttles cron to ~hourly fires;
# a narrow window misses every run. Submitting MOC orders early is safe —
# they queue for the closing auction regardless of when submitted.
# Alpaca rejects CLS orders after close-10min, so we keep 1-min safety buffer on the end.
MOC_WINDOW_START_MIN_BEFORE_CLOSE = 150
MOC_WINDOW_END_MIN_BEFORE_CLOSE = 11


@dataclass
class TradeWindow:
    start: datetime  # ET-aware
    end: datetime    # ET-aware
    close: datetime  # ET-aware


@dataclass
class AssetInfo:
    symbol: str
    tradable: bool
    shortable: bool
    easy_to_borrow: bool


@dataclass
class PositionState:
    """Snapshot of a position used by the live runner.

    signed_qty is the actual position size — positive long, negative short, 0 flat.
    qty_available is the absolute share count free to transact right now: it goes
    to 0 when shares are held_for_orders by a pending close, even though the
    position itself is unchanged. Used as a pre-flight gate so we don't submit a
    close-side order the broker is guaranteed to reject.
    """
    signed_qty: int
    qty_available: int


@dataclass
class SubmittedOrder:
    id: str
    client_order_id: str
    symbol: str
    qty: int
    side: OrderSide


class Broker:
    def __init__(self, client: TradingClient):
        self._client = client

    # ---- Market state ---------------------------------------------------

    def todays_trade_window(self) -> TradeWindow | None:
        """Compute today's MOC submission window from Alpaca's clock.

        Returns None if the market is not open today. Works automatically on
        early-close days because next_close shifts to the actual session close.
        """
        clock = self._client.get_clock()
        if not clock.is_open:
            return None
        close_et = clock.next_close.astimezone(ET)
        return TradeWindow(
            start=close_et - timedelta(minutes=MOC_WINDOW_START_MIN_BEFORE_CLOSE),
            end=close_et - timedelta(minutes=MOC_WINDOW_END_MIN_BEFORE_CLOSE),
            close=close_et,
        )

    def get_asset(self, symbol: str) -> AssetInfo:
        a = self._client.get_asset(symbol)
        return AssetInfo(
            symbol=a.symbol,
            tradable=bool(a.tradable),
            shortable=bool(a.shortable),
            easy_to_borrow=bool(a.easy_to_borrow),
        )

    # ---- Account & position --------------------------------------------

    def get_account_cash(self) -> float:
        return float(self._client.get_account().cash)

    def get_account_equity(self) -> float:
        return float(self._client.get_account().equity)

    def get_position_state(self, symbol: str) -> PositionState:
        """Read the full position state — signed size and available-to-trade qty.

        Both come from a single get_open_position call. Handles 404 (no position)
        cleanly. signed_qty reflects the actual position size (positive long,
        negative short); qty_available is whatever the broker reports as free to
        transact (pinned to 0 when pending close orders hold the shares).
        """
        try:
            pos = self._client.get_open_position(symbol)
        except APIError as e:
            if "position does not exist" in str(e).lower() or "404" in str(e):
                return PositionState(signed_qty=0, qty_available=0)
            raise
        qty = int(float(pos.qty))
        qty_available = (
            int(float(pos.qty_available)) if hasattr(pos, "qty_available") else qty
        )
        if str(getattr(pos, "side", "")).lower() == "short":
            qty = -qty
        return PositionState(signed_qty=qty, qty_available=qty_available)

    def get_position_signed_qty(self, symbol: str) -> int:
        """Convenience: signed position size only. See get_position_state for the full state."""
        return self.get_position_state(symbol).signed_qty

    # ---- Orders --------------------------------------------------------

    def cancel_open_orders(self, symbol: str, *, keep_today: str | None = None) -> int:
        """Cancel orphan open orders for the given symbol. Returns count cancelled.

        If keep_today is set (e.g. "meanrev-2026-05-01"), orders whose
        client_order_id starts with that prefix are skipped — they belong to
        today's run and must not be cancelled.
        """
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
        open_orders = self._client.get_orders(filter=req)
        n = 0
        for o in open_orders:
            if keep_today and o.client_order_id and o.client_order_id.startswith(keep_today):
                log.debug("keeping today's order %s (coid=%s)", o.id, o.client_order_id)
                continue
            try:
                self._client.cancel_order_by_id(o.id)
                n += 1
            except APIError as e:
                log.debug("cancel_order_by_id(%s) failed: %s", o.id, e)
        return n

    def submit_moc(
        self,
        symbol: str,
        qty: int,
        side: OrderSide,
        client_order_id: str,
        position_intent: PositionIntent | None = None,
    ) -> SubmittedOrder:
        """Submit a market-on-close order. qty must be a positive whole number.

        position_intent disambiguates close-vs-open on same-side flips so the
        broker doesn't reject the open leg with held_for_orders=existing_qty.
        """
        if qty <= 0:
            raise ValueError(f"submit_moc qty must be > 0, got {qty}")
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.CLS,
            client_order_id=client_order_id,
            position_intent=position_intent,
        )
        o = self._client.submit_order(order_data=req)
        return SubmittedOrder(
            id=str(o.id),
            client_order_id=o.client_order_id,
            symbol=o.symbol,
            qty=int(float(o.qty)),
            side=o.side,
        )
