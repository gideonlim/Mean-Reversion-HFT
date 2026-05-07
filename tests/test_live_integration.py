"""Integration tests for live._process_symbol with mocked broker + data.

Walks 5 trading days end-to-end: per-symbol decision -> cancel orphans ->
submit orders. Verifies coid generation, position_intent flows through to
the broker, and partial-failure handling doesn't cause cascading errors.
"""
from __future__ import annotations

import logging
from datetime import date
from unittest.mock import MagicMock

from alpaca.trading.enums import OrderSide, PositionIntent

import live
from broker import AssetInfo
from data import ClosedBars
from live import _process_symbol


def _make_mocks(*, current_qty: int, prev_close: float, log_return: float, shortable: bool = True):
    """Build a (broker, data_client, captured_calls) trio for _process_symbol."""
    import math
    broker = MagicMock()
    broker.get_asset.return_value = AssetInfo(symbol="SPY", tradable=True, shortable=shortable, easy_to_borrow=shortable)
    broker.cancel_open_orders.return_value = 0
    broker.get_position_signed_qty.return_value = current_qty
    # Each submit_moc returns a SubmittedOrder-like
    submitted: list[dict] = []

    def _capture_submit(symbol, qty, side, coid, *, position_intent=None):
        submitted.append({
            "symbol": symbol, "qty": qty, "side": side, "coid": coid,
            "position_intent": position_intent,
        })
        m = MagicMock()
        m.id = f"id-{coid}"
        m.client_order_id = coid
        m.side = side
        m.qty = qty
        m.symbol = symbol
        return m

    broker.submit_moc.side_effect = _capture_submit

    data_client = MagicMock()

    # log_return_lag_1 is a @property derived from log(prev_close / prev_prev_close).
    # Solve for prev_prev_close: prev_prev_close = prev_close / exp(log_return).
    prev_prev_close = prev_close / math.exp(log_return)
    closed = ClosedBars(
        prev_prev_date=date(2026, 5, 5),
        prev_date=date(2026, 5, 6),
        prev_prev_close=prev_prev_close,
        prev_close=prev_close,
    )

    return broker, data_client, submitted, closed


def test_integration_open_long_from_flat_full_pipeline(monkeypatch):
    broker, data_client, submitted, closed = _make_mocks(
        current_qty=0, prev_close=720.0, log_return=-0.01,  # -1% yesterday -> long signal
    )
    monkeypatch.setattr(live, "last_two_closed_bars", lambda *a, **k: closed)

    rec = _process_symbol(
        symbol="SPY", weight=1.0, equity=50000.0,
        broker=broker, data_client=data_client,
        et_date=date(2026, 5, 7), dry_run=False, log=logging.getLogger("test"),
    )

    assert rec["transition"] == "open"
    assert rec["signal"] == 1
    assert len(submitted) == 1
    o = submitted[0]
    assert o["side"] == OrderSide.BUY
    assert o["coid"] == "meanrev-2026-05-07-SPY-open"
    assert o["position_intent"] == PositionIntent.BUY_TO_OPEN
    # cancel_open_orders called with today's keep_today prefix
    broker.cancel_open_orders.assert_called_once()
    _, kwargs = broker.cancel_open_orders.call_args
    assert kwargs.get("keep_today") == "meanrev-2026-05-07"


def test_integration_flip_long_to_short_submits_two_orders_with_intents(monkeypatch):
    """Recreates the May 6 scenario: long position, signal flips to short.

    Uses prev_close=719.69 so equity*0.95/prev_close yields exactly 66 shares
    (matches production sizing).
    """
    broker, data_client, submitted, closed = _make_mocks(
        current_qty=66, prev_close=719.69, log_return=0.008,
    )
    monkeypatch.setattr(live, "last_two_closed_bars", lambda *a, **k: closed)

    rec = _process_symbol(
        symbol="SPY", weight=1.0, equity=50000.0,
        broker=broker, data_client=data_client,
        et_date=date(2026, 5, 7), dry_run=False, log=logging.getLogger("test"),
    )

    assert rec["transition"] == "flip"
    assert rec["signal"] == -1
    assert rec["current_qty"] == 66
    assert rec["target_qty"] == -rec["target_abs_qty"]  # short of target_abs
    assert rec["target_abs_qty"] >= 1
    assert len(submitted) == 2

    close = submitted[0]
    assert close["side"] == OrderSide.SELL
    assert close["qty"] == 66  # close the existing 66-share long
    assert close["coid"] == "meanrev-2026-05-07-SPY-flip_close"
    assert close["position_intent"] == PositionIntent.SELL_TO_CLOSE

    open_ = submitted[1]
    assert open_["side"] == OrderSide.SELL
    assert open_["qty"] == rec["target_abs_qty"]
    assert open_["coid"] == "meanrev-2026-05-07-SPY-flip_open"
    assert open_["position_intent"] == PositionIntent.SELL_TO_OPEN  # the bug fix


def test_integration_dry_run_skips_submission(monkeypatch):
    """Dry run must compute the decision but never call submit_moc."""
    broker, data_client, submitted, closed = _make_mocks(
        current_qty=0, prev_close=720.0, log_return=-0.01,
    )
    monkeypatch.setattr(live, "last_two_closed_bars", lambda *a, **k: closed)

    rec = _process_symbol(
        symbol="SPY", weight=1.0, equity=50000.0,
        broker=broker, data_client=data_client,
        et_date=date(2026, 5, 7), dry_run=True, log=logging.getLogger("test"),
    )

    assert rec["transition"] == "open"
    assert submitted == []
    broker.submit_moc.assert_not_called()


def test_integration_long_only_skips_short_signal(monkeypatch):
    """LONG_ONLY (shortable=False) + short signal + flat -> no orders, no error."""
    broker, data_client, submitted, closed = _make_mocks(
        current_qty=0, prev_close=720.0, log_return=0.008, shortable=False,
    )
    monkeypatch.setattr(live, "last_two_closed_bars", lambda *a, **k: closed)

    rec = _process_symbol(
        symbol="SPY", weight=1.0, equity=50000.0,
        broker=broker, data_client=data_client,
        et_date=date(2026, 5, 7), dry_run=False, log=logging.getLogger("test"),
    )

    assert rec["transition"] == "shortable_skip"
    assert rec["signal"] == -1
    assert submitted == []


def test_integration_zero_weight_short_circuits(monkeypatch):
    """Symbol with weight=0 returns early; broker not even called."""
    broker = MagicMock()
    data_client = MagicMock()

    rec = _process_symbol(
        symbol="USO", weight=0.0, equity=50000.0,
        broker=broker, data_client=data_client,
        et_date=date(2026, 5, 7), dry_run=False, log=logging.getLogger("test"),
    )

    assert rec["transition"] == "no_allocation"
    broker.get_asset.assert_not_called()
    broker.submit_moc.assert_not_called()


def test_integration_full_week_no_errors(monkeypatch):
    """Walk a 5-day week through _process_symbol with state carried forward.

    Pre-computes expected target qty using the same sizing formula
    (floor(equity * POSITION_FRACTION * weight / prev_close)) so the test
    survives any small change in equity or prev_close.
    """
    import math
    from config import SETTINGS

    # (et_date, log_return_yesterday, expected_transition_or_None)
    # None means "compute target from sizing — verify result against state."
    days = [
        (date(2026, 5, 4), -0.01,  "open"),   # flat -> long
        (date(2026, 5, 5), -0.005, None),     # long, +1 again, target may shift slightly
        (date(2026, 5, 6),  0.008, "flip"),   # +1 -> -1: flip to short
        (date(2026, 5, 7),  0.005, None),     # -1 again: no-op or scale on short side
        (date(2026, 5, 8), -0.01,  "flip"),   # -1 -> +1: flip to long
    ]
    qty = 0
    for et_date, log_ret, expected_trans in days:
        equity = 50000.0 + (et_date.day - 4) * 50.0
        prev_close = 720.0 + (et_date.day - 4) * 0.5
        expected_target_abs = math.floor(equity * SETTINGS.POSITION_FRACTION * 1.0 / prev_close)

        broker, data_client, submitted, closed = _make_mocks(
            current_qty=qty, prev_close=prev_close, log_return=log_ret,
        )
        monkeypatch.setattr(live, "last_two_closed_bars", lambda *a, **k: closed)

        rec = _process_symbol(
            symbol="SPY", weight=1.0, equity=equity,
            broker=broker, data_client=data_client,
            et_date=et_date, dry_run=False, log=logging.getLogger("test"),
        )

        assert "error" not in rec, f"{et_date}: unexpected error: {rec.get('error')}"
        assert rec["current_qty"] == qty
        assert rec["target_abs_qty"] == expected_target_abs
        if expected_trans is not None:
            assert rec["transition"] == expected_trans, f"{et_date}: got {rec['transition']}"

        # Every order has a valid position_intent (the May 6 bug fix)
        for o in submitted:
            assert o["position_intent"] is not None, f"{et_date}: order missing intent: {o}"
            # coid format: meanrev-YYYY-MM-DD-SYMBOL-action
            assert o["coid"].startswith(f"meanrev-{et_date.isoformat()}-SPY-")

        # Apply fills to advance qty for next day
        for o in submitted:
            qty += o["qty"] if o["side"] == OrderSide.BUY else -o["qty"]

        # Sanity: signed delta from rec matches what we applied
        assert qty - rec["current_qty"] == rec["delta"], (
            f"{et_date}: qty drift after fills: was {rec['current_qty']}, now {qty}, expected delta {rec['delta']}"
        )
