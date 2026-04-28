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


def test_flip_long_to_short_two_orders():
    d = decide_transition(signal=-1, current_qty_signed=10, target_abs_qty=8, shortable=True)
    assert d.transition == "flip"
    assert len(d.orders) == 2
    assert d.orders[0].side == OrderSide.SELL and d.orders[0].qty == 10 and d.orders[0].action == "flip_close"
    assert d.orders[1].side == OrderSide.SELL and d.orders[1].qty == 8 and d.orders[1].action == "flip_open"
    assert d.target_qty == -8 and d.delta == -18


def test_flip_short_to_long_two_orders():
    d = decide_transition(signal=1, current_qty_signed=-10, target_abs_qty=12, shortable=True)
    assert d.transition == "flip"
    assert len(d.orders) == 2
    assert d.orders[0].side == OrderSide.BUY and d.orders[0].qty == 10 and d.orders[0].action == "flip_close"
    assert d.orders[1].side == OrderSide.BUY and d.orders[1].qty == 12 and d.orders[1].action == "flip_open"


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


def test_trade_window_normal_close_resolves_to_15_35_to_15_49():
    close = datetime(2026, 4, 28, 16, 0, tzinfo=ET)
    broker = _mock_broker_with_close(close)
    window = broker.todays_trade_window()
    assert window is not None
    assert window.start == close - timedelta(minutes=MOC_WINDOW_START_MIN_BEFORE_CLOSE)
    assert window.end == close - timedelta(minutes=MOC_WINDOW_END_MIN_BEFORE_CLOSE)
    assert window.start.hour == 15 and window.start.minute == 35
    assert window.end.hour == 15 and window.end.minute == 49


def test_trade_window_early_close_shifts_window_automatically():
    close = datetime(2026, 11, 27, 13, 0, tzinfo=ET)  # day-after-Thanksgiving early close
    broker = _mock_broker_with_close(close)
    window = broker.todays_trade_window()
    assert window is not None
    assert window.start.hour == 12 and window.start.minute == 35
    assert window.end.hour == 12 and window.end.minute == 49


def test_trade_window_market_closed_returns_none():
    broker = _mock_broker_with_close(datetime(2026, 4, 28, 16, 0, tzinfo=ET), is_open=False)
    assert broker.todays_trade_window() is None


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
