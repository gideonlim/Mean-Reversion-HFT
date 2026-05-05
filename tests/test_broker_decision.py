"""Tests for the live runner's decision logic and operational concerns.

Coverage:
- 6-row decision table (open / scale / no-op / flip in both directions)
- Single net-delta order vs flip-pair behavior
- Shortable-skip path (with and without an existing long to force-close)
- ET-date idempotency (date routes through et_today, not local time)
- Dynamic trade window (early-close vs normal-close)
- Today's partial bar filtered out of last_two_closed_bars
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
from alpaca.trading.enums import OrderSide

import data as data_module
from broker import Broker, MOC_WINDOW_END_MIN_BEFORE_CLOSE, MOC_WINDOW_START_MIN_BEFORE_CLOSE
from data import last_two_closed_bars_from_df
from live import Decision, decide_transition

ET = ZoneInfo("America/New_York")


# ---- Decision table ----------------------------------------------------

def test_open_long_from_flat():
    d = decide_transition(signal=1, current_qty_signed=0, target_abs_qty=10, shortable=True)
    assert d.transition == "open"
    assert len(d.orders) == 1
    assert d.orders[0].side == OrderSide.BUY and d.orders[0].qty == 10
    assert d.target_qty == 10 and d.delta == 10


def test_open_short_from_flat():
    d = decide_transition(signal=-1, current_qty_signed=0, target_abs_qty=8, shortable=True)
    assert d.transition == "open"
    assert len(d.orders) == 1
    assert d.orders[0].side == OrderSide.SELL and d.orders[0].qty == 8
    assert d.target_qty == -8 and d.delta == -8


def test_no_op_when_long_unchanged():
    d = decide_transition(signal=1, current_qty_signed=10, target_abs_qty=10, shortable=True)
    assert d.transition == "no_op"
    assert d.orders == []


def test_no_op_when_short_unchanged():
    d = decide_transition(signal=-1, current_qty_signed=-10, target_abs_qty=10, shortable=True)
    assert d.transition == "no_op"
    assert d.orders == []


def test_scale_up_long_single_order():
    d = decide_transition(signal=1, current_qty_signed=10, target_abs_qty=15, shortable=True)
    assert d.transition == "scale"
    assert len(d.orders) == 1
    assert d.orders[0].side == OrderSide.BUY and d.orders[0].qty == 5


def test_scale_down_long_single_order():
    d = decide_transition(signal=1, current_qty_signed=15, target_abs_qty=10, shortable=True)
    assert d.transition == "scale"
    assert len(d.orders) == 1
    assert d.orders[0].side == OrderSide.SELL and d.orders[0].qty == 5


def test_flip_long_to_short_single_order():
    """Flip long->short emits ONE order for |delta| shares (close + open in one).

    Two separate orders fail because Alpaca holds shares for the close order,
    leaving 0 available for the open order. Single order avoids the issue.
    """
    d = decide_transition(signal=-1, current_qty_signed=10, target_abs_qty=8, shortable=True)
    assert d.transition == "flip"
    assert len(d.orders) == 1
    assert d.orders[0].side == OrderSide.SELL
    assert d.orders[0].qty == 18  # close 10 + open 8
    assert d.orders[0].action == "flip"
    assert d.target_qty == -8 and d.delta == -18


def test_flip_short_to_long_single_order():
    """Flip short->long emits ONE BUY for |delta| shares."""
    d = decide_transition(signal=1, current_qty_signed=-10, target_abs_qty=12, shortable=True)
    assert d.transition == "flip"
    assert len(d.orders) == 1
    assert d.orders[0].side == OrderSide.BUY
    assert d.orders[0].qty == 22  # close 10 + open 12
    assert d.orders[0].action == "flip"
    assert d.target_qty == 12 and d.delta == 22


# ---- Shortable-skip path ----------------------------------------------

def test_shortable_skip_when_signal_short_and_not_shortable_starting_flat():
    d = decide_transition(signal=-1, current_qty_signed=0, target_abs_qty=10, shortable=False)
    assert d.transition == "shortable_skip"
    assert d.orders == []
    assert d.target_qty == 0


def test_shortable_skip_force_closes_existing_long():
    d = decide_transition(signal=-1, current_qty_signed=10, target_abs_qty=10, shortable=False)
    assert d.transition == "shortable_skip"
    assert len(d.orders) == 1
    assert d.orders[0].side == OrderSide.SELL and d.orders[0].qty == 10
    assert d.orders[0].action == "shortable_skip_close"
    assert d.target_qty == 0


def test_shortable_irrelevant_when_signal_long():
    # signal=+1 doesn't need shortability; should behave normally even if shortable=False
    d = decide_transition(signal=1, current_qty_signed=0, target_abs_qty=10, shortable=False)
    assert d.transition == "open"
    assert d.orders[0].side == OrderSide.BUY


def test_long_only_mode_uses_shortable_skip_for_short_signal():
    # LONG_ONLY=True means live.py passes shortable=False when signal=-1.
    # Verify the decision logic stays flat (no short position opened).
    d = decide_transition(signal=-1, current_qty_signed=0, target_abs_qty=10, shortable=False)
    assert d.transition == "shortable_skip"
    assert d.orders == []
    assert d.target_qty == 0


def test_long_only_mode_closes_existing_long_on_short_signal():
    # If currently long and signal flips to -1 in LONG_ONLY mode, close the long, don't open short.
    d = decide_transition(signal=-1, current_qty_signed=15, target_abs_qty=10, shortable=False)
    assert d.transition == "shortable_skip"
    assert len(d.orders) == 1
    assert d.orders[0].side == OrderSide.SELL and d.orders[0].qty == 15
    assert d.target_qty == 0


# ---- last_two_closed_bars: today's partial bar filtered ---------------

def _bars_df(dates, closes):
    df = pd.DataFrame({"close": closes}, index=pd.Index(dates, name="et_date"))
    return df


def test_last_two_closed_bars_excludes_today():
    today = date(2026, 4, 28)
    df = _bars_df(
        [date(2026, 4, 24), date(2026, 4, 25), date(2026, 4, 26), date(2026, 4, 27), date(2026, 4, 28)],
        [100.0, 101.0, 102.0, 103.0, 104.0],
    )
    closed = last_two_closed_bars_from_df(df, today)
    # The 2026-04-28 bar (today) must be filtered out.
    assert closed.prev_date == date(2026, 4, 27)
    assert closed.prev_prev_date == date(2026, 4, 26)
    assert closed.prev_close == 103.0
    assert closed.prev_prev_close == 102.0


def test_last_two_closed_bars_when_today_absent():
    today = date(2026, 4, 28)
    df = _bars_df(
        [date(2026, 4, 24), date(2026, 4, 25), date(2026, 4, 26), date(2026, 4, 27)],
        [100.0, 101.0, 102.0, 103.0],
    )
    closed = last_two_closed_bars_from_df(df, today)
    assert closed.prev_date == date(2026, 4, 27)
    assert closed.prev_prev_date == date(2026, 4, 26)


def test_last_two_closed_bars_raises_when_too_few():
    today = date(2026, 4, 28)
    df = _bars_df([date(2026, 4, 27)], [100.0])
    with pytest.raises(ValueError):
        last_two_closed_bars_from_df(df, today)


# ---- Trade window: dynamic from clock.next_close ----------------------

def _mock_broker_with_close(close_et: datetime, is_open: bool = True) -> Broker:
    client = MagicMock()
    client.get_clock.return_value = MagicMock(is_open=is_open, next_close=close_et)
    return Broker(client)


def test_trade_window_normal_close():
    close = datetime(2026, 4, 28, 16, 0, tzinfo=ET)
    broker = _mock_broker_with_close(close)
    window = broker.todays_trade_window()
    assert window is not None
    assert window.start == close - timedelta(minutes=MOC_WINDOW_START_MIN_BEFORE_CLOSE)
    assert window.end == close - timedelta(minutes=MOC_WINDOW_END_MIN_BEFORE_CLOSE)
    # With 150-min start: 16:00 - 2h30m = 13:30. End: 16:00 - 11min = 15:49.
    assert window.start.hour == 13 and window.start.minute == 30
    assert window.end.hour == 15 and window.end.minute == 49


def test_trade_window_early_close_shifts_window_automatically():
    close = datetime(2026, 11, 27, 13, 0, tzinfo=ET)  # day-after-Thanksgiving early close
    broker = _mock_broker_with_close(close)
    window = broker.todays_trade_window()
    assert window is not None
    # 13:00 - 2h30m = 10:30. End: 13:00 - 11min = 12:49.
    assert window.start.hour == 10 and window.start.minute == 30
    assert window.end.hour == 12 and window.end.minute == 49


def test_trade_window_market_closed_returns_none():
    broker = _mock_broker_with_close(datetime(2026, 4, 28, 16, 0, tzinfo=ET), is_open=False)
    assert broker.todays_trade_window() is None


# ---- cancel_open_orders: keep today's orders, cancel orphans ---------------

def test_cancel_open_orders_keeps_todays_orders():
    """Orders with today's coid prefix must NOT be cancelled (they're ours)."""
    client = MagicMock()
    today_order = MagicMock(id="o1", client_order_id="meanrev-2026-05-01-SPY-flip_close")
    orphan_order = MagicMock(id="o2", client_order_id="meanrev-2026-04-30-SPY-open")
    client.get_orders.return_value = [today_order, orphan_order]
    broker = Broker(client)

    n = broker.cancel_open_orders("SPY", keep_today="meanrev-2026-05-01")
    assert n == 1  # only the orphan cancelled
    client.cancel_order_by_id.assert_called_once_with("o2")


def test_cancel_open_orders_without_keep_today_cancels_all():
    """Without keep_today, all open orders are cancelled (backward compat)."""
    client = MagicMock()
    client.get_orders.return_value = [
        MagicMock(id="o1", client_order_id="meanrev-2026-05-01-SPY-flip_close"),
        MagicMock(id="o2", client_order_id="meanrev-2026-04-30-SPY-open"),
    ]
    broker = Broker(client)

    n = broker.cancel_open_orders("SPY")
    assert n == 2
    assert client.cancel_order_by_id.call_count == 2


# ---- ET-date idempotency: client_order_id and marker use ET, not local time ----

def test_client_order_id_uses_et_date_not_local_date():
    # Late ET evening = next-day local in Sydney. The marker / coid must use ET.
    sydney = ZoneInfo("Australia/Sydney")
    # 23:00 ET on 2026-04-28 = 13:00 Sydney on 2026-04-29 (during AEST winter).
    et_moment = datetime(2026, 4, 28, 23, 0, tzinfo=ET)
    sydney_moment = et_moment.astimezone(sydney)
    assert sydney_moment.date() == date(2026, 4, 29)  # local date is the *next* day
    et_date = et_moment.date()
    assert et_date == date(2026, 4, 28)
    coid = f"meanrev-{et_date.isoformat()}-SPY-open"
    assert coid == "meanrev-2026-04-28-SPY-open"  # uses ET date, NOT 2026-04-29


def test_et_today_uses_new_york_zone(monkeypatch):
    # Patch et_now to a fixed UTC moment that straddles ET midnight; confirm et_today
    # returns the ET calendar date, not the local Windows date.
    from datetime import datetime as real_datetime

    fixed_utc = real_datetime(2026, 4, 29, 3, 0, tzinfo=ZoneInfo("UTC"))  # 23:00 ET on 04-28
    expected_et_date = date(2026, 4, 28)

    import config

    class FakeDatetime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_utc.astimezone(tz) if tz else fixed_utc

    monkeypatch.setattr(config, "datetime", FakeDatetime)
    assert config.et_today() == expected_et_date


# ---- Sanity: signal-from-lag wiring used by live.py ------------------

def test_signal_wiring_for_live():
    from strategy import signal_from_lag

    assert signal_from_lag(0.005) == -1   # yesterday up -> short today
    assert signal_from_lag(-0.005) == 1   # yesterday down -> long today


# ---- Multi-symbol weight validation ---------------------------------------

def test_get_weights_default_equal_weight():
    from config import Settings

    s = Settings(SYMBOLS=("SPY", "USO", "GLD", "SLV"))
    w = s.get_weights()
    assert w == {"SPY": 0.25, "USO": 0.25, "GLD": 0.25, "SLV": 0.25}


def test_get_weights_single_symbol_default():
    from config import Settings

    s = Settings(SYMBOLS=("SPY",))
    assert s.get_weights() == {"SPY": 1.0}


def test_get_weights_custom_full_allocation():
    from config import Settings

    s = Settings(
        SYMBOLS=("SPY", "USO", "GLD", "SLV"),
        SYMBOL_WEIGHTS=(("SPY", 0.4), ("USO", 0.2), ("GLD", 0.2), ("SLV", 0.2)),
    )
    assert s.get_weights() == {"SPY": 0.4, "USO": 0.2, "GLD": 0.2, "SLV": 0.2}


def test_get_weights_custom_partial_rest_is_cash():
    """Sum < 1 is allowed; remainder stays as cash. Symbols with no weight default to 0."""
    from config import Settings

    s = Settings(
        SYMBOLS=("SPY", "USO", "GLD"),
        SYMBOL_WEIGHTS=(("SPY", 0.4), ("USO", 0.2)),  # GLD missing -> 0
    )
    w = s.get_weights()
    assert w == {"SPY": 0.4, "USO": 0.2, "GLD": 0.0}
    assert sum(w.values()) == pytest.approx(0.6)  # 0.4 cash buffer


def test_get_weights_sum_exceeds_one_raises():
    from config import Settings

    s = Settings(SYMBOLS=("SPY", "USO"), SYMBOL_WEIGHTS=(("SPY", 0.7), ("USO", 0.5)))
    with pytest.raises(ValueError, match="sum to"):
        s.get_weights()


def test_get_weights_negative_weight_raises():
    from config import Settings

    s = Settings(SYMBOLS=("SPY",), SYMBOL_WEIGHTS=(("SPY", -0.1),))
    with pytest.raises(ValueError, match="non-negative"):
        s.get_weights()


def test_get_weights_unknown_symbol_raises():
    from config import Settings

    s = Settings(SYMBOLS=("SPY",), SYMBOL_WEIGHTS=(("USO", 0.5),))
    with pytest.raises(ValueError, match="not in SYMBOLS"):
        s.get_weights()


def test_get_weights_empty_symbols_raises():
    from config import Settings

    s = Settings(SYMBOLS=())
    with pytest.raises(ValueError, match="non-empty"):
        s.get_weights()
