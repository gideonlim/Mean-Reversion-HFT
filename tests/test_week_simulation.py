"""Week-long simulations to verify decision logic across realistic scenarios.

Each test walks through 5 trading days. After each day's orders, we apply the
fills to update current_qty (simulating MOC fills at close), then move to the
next day. This catches end-to-end errors that single-day tests miss:
  - position_intent set correctly for every order type in every direction
  - state transitions match across the full week
  - quantities reconcile (no drift, no off-by-one)
"""
from __future__ import annotations

from alpaca.trading.enums import OrderSide, PositionIntent

from live import IntendedOrder, decide_transition


def _apply_fills(current_qty: int, orders: list[IntendedOrder]) -> int:
    """Update signed qty after orders fill. BUY adds, SELL subtracts."""
    qty = current_qty
    for o in orders:
        delta = o.qty if o.side == OrderSide.BUY else -o.qty
        qty += delta
    return qty


# ---- Scenario 1: bidirectional week with both flip directions -----------

def test_week_open_noop_scaleup_flip_both_directions():
    """Mon: open long. Tue: no-op. Wed: scale up. Thu: flip-close to flat. Fri: open short."""
    qty = 0

    # Mon — flat -> long
    d = decide_transition(signal=1, current_qty_signed=qty, target_abs_qty=66, shortable=True)
    assert d.transition == "open"
    assert [(o.side, o.qty, o.action, o.position_intent) for o in d.orders] == [
        (OrderSide.BUY, 66, "open", PositionIntent.BUY_TO_OPEN),
    ]
    qty = _apply_fills(qty, d.orders)
    assert qty == 66

    # Tue — long unchanged
    d = decide_transition(signal=1, current_qty_signed=qty, target_abs_qty=66, shortable=True)
    assert d.transition == "no_op" and d.orders == []
    qty = _apply_fills(qty, d.orders)
    assert qty == 66

    # Wed — scale up long
    d = decide_transition(signal=1, current_qty_signed=qty, target_abs_qty=80, shortable=True)
    assert d.transition == "scale"
    assert [(o.side, o.qty, o.action, o.position_intent) for o in d.orders] == [
        (OrderSide.BUY, 14, "scale", PositionIntent.BUY_TO_OPEN),
    ]
    qty = _apply_fills(qty, d.orders)
    assert qty == 80

    # Thu — flip long -> short: only the close lands today; short opens next session.
    d = decide_transition(signal=-1, current_qty_signed=qty, target_abs_qty=66, shortable=True)
    assert d.transition == "flip"
    assert [(o.side, o.qty, o.action, o.position_intent) for o in d.orders] == [
        (OrderSide.SELL, 80, "flip_close", PositionIntent.SELL_TO_CLOSE),
    ]
    qty = _apply_fills(qty, d.orders)
    assert qty == 0

    # Fri — flat -> short (the deferred open from yesterday's flip)
    d = decide_transition(signal=-1, current_qty_signed=qty, target_abs_qty=66, shortable=True)
    assert d.transition == "open"
    assert [(o.side, o.qty, o.action, o.position_intent) for o in d.orders] == [
        (OrderSide.SELL, 66, "open", PositionIntent.SELL_TO_OPEN),
    ]
    qty = _apply_fills(qty, d.orders)
    assert qty == -66


# ---- Scenario 2: LONG_ONLY week (shortable=False the whole time) --------

def test_week_long_only_short_signals_keep_flat():
    """In LONG_ONLY mode, short signals must close existing long and stay flat."""
    qty = 0

    # Mon — open long
    d = decide_transition(signal=1, current_qty_signed=qty, target_abs_qty=66, shortable=False)
    assert d.transition == "open"
    assert d.orders[0].position_intent == PositionIntent.BUY_TO_OPEN
    qty = _apply_fills(qty, d.orders)
    assert qty == 66

    # Tue — short signal, not shortable -> close existing long
    d = decide_transition(signal=-1, current_qty_signed=qty, target_abs_qty=66, shortable=False)
    assert d.transition == "shortable_skip"
    assert [(o.side, o.qty, o.action, o.position_intent) for o in d.orders] == [
        (OrderSide.SELL, 66, "shortable_skip_close", PositionIntent.SELL_TO_CLOSE),
    ]
    qty = _apply_fills(qty, d.orders)
    assert qty == 0

    # Wed — short signal, flat, not shortable -> stay flat (no orders)
    d = decide_transition(signal=-1, current_qty_signed=qty, target_abs_qty=66, shortable=False)
    assert d.transition == "shortable_skip" and d.orders == []
    qty = _apply_fills(qty, d.orders)
    assert qty == 0

    # Thu — long signal, open new long
    d = decide_transition(signal=1, current_qty_signed=qty, target_abs_qty=70, shortable=False)
    assert d.transition == "open"
    assert d.orders[0].position_intent == PositionIntent.BUY_TO_OPEN
    qty = _apply_fills(qty, d.orders)
    assert qty == 70

    # Fri — long signal, scale up
    d = decide_transition(signal=1, current_qty_signed=qty, target_abs_qty=75, shortable=False)
    assert d.transition == "scale"
    assert d.orders[0].position_intent == PositionIntent.BUY_TO_OPEN
    qty = _apply_fills(qty, d.orders)
    assert qty == 75


# ---- Scenario 3: scaling in both directions, including reducing positions

def test_week_scale_up_and_down_both_directions():
    """Exercise scale up/down on long AND short positions."""
    qty = 0

    # Mon — open large long
    d = decide_transition(signal=1, current_qty_signed=qty, target_abs_qty=100, shortable=True)
    assert d.transition == "open"
    qty = _apply_fills(qty, d.orders)
    assert qty == 100

    # Tue — scale DOWN long (target smaller than current)
    d = decide_transition(signal=1, current_qty_signed=qty, target_abs_qty=80, shortable=True)
    assert d.transition == "scale"
    assert [(o.side, o.qty, o.action, o.position_intent) for o in d.orders] == [
        (OrderSide.SELL, 20, "scale", PositionIntent.SELL_TO_CLOSE),  # reducing long = closing
    ]
    qty = _apply_fills(qty, d.orders)
    assert qty == 80

    # Wed — flip long -> short: close only; short opens next session
    d = decide_transition(signal=-1, current_qty_signed=qty, target_abs_qty=50, shortable=True)
    assert d.transition == "flip"
    qty = _apply_fills(qty, d.orders)
    assert qty == 0

    # Thu — flat -> short (deferred open from yesterday's flip)
    d = decide_transition(signal=-1, current_qty_signed=qty, target_abs_qty=50, shortable=True)
    assert d.transition == "open"
    assert [(o.side, o.qty, o.action, o.position_intent) for o in d.orders] == [
        (OrderSide.SELL, 50, "open", PositionIntent.SELL_TO_OPEN),
    ]
    qty = _apply_fills(qty, d.orders)
    assert qty == -50

    # Fri — scale DOWN short (less negative, partial close)
    d = decide_transition(signal=-1, current_qty_signed=qty, target_abs_qty=40, shortable=True)
    assert d.transition == "scale"
    assert [(o.side, o.qty, o.action, o.position_intent) for o in d.orders] == [
        (OrderSide.BUY, 10, "scale", PositionIntent.BUY_TO_CLOSE),  # reducing short = closing
    ]
    qty = _apply_fills(qty, d.orders)
    assert qty == -40


# ---- Scenario 4: choppy week (signal flips every day) -------------------

def test_week_choppy_signal_flips_every_day():
    """Worst-case mean-reversion: signal flips every single day. Each flip closes
    today and the new direction opens the next session, so the position
    alternates between flat and one-side every day rather than ping-ponging."""
    qty = 0

    # Mon — open long
    d = decide_transition(signal=1, current_qty_signed=qty, target_abs_qty=50, shortable=True)
    assert d.transition == "open"
    qty = _apply_fills(qty, d.orders)
    assert qty == 50

    # Tue — flip to short signal: close long today, short opens next session
    d = decide_transition(signal=-1, current_qty_signed=qty, target_abs_qty=50, shortable=True)
    assert d.transition == "flip"
    assert len(d.orders) == 1
    qty = _apply_fills(qty, d.orders)
    assert qty == 0

    # Wed — signal flipped back to long: but we're flat, so this is just open long
    d = decide_transition(signal=1, current_qty_signed=qty, target_abs_qty=50, shortable=True)
    assert d.transition == "open"
    assert len(d.orders) == 1
    qty = _apply_fills(qty, d.orders)
    assert qty == 50

    # Thu — flip to short again: close long today, short opens next session
    d = decide_transition(signal=-1, current_qty_signed=qty, target_abs_qty=50, shortable=True)
    assert d.transition == "flip"
    qty = _apply_fills(qty, d.orders)
    assert qty == 0

    # Fri — long signal again, flat -> open long
    d = decide_transition(signal=1, current_qty_signed=qty, target_abs_qty=50, shortable=True)
    assert d.transition == "open"
    qty = _apply_fills(qty, d.orders)
    assert qty == 50


# ---- Scenario 5: realistic SPY week recreating the bug we just fixed ----

def test_week_recreates_held_for_orders_bug_scenario():
    """Recreate the May 11 production state: long 65, signal flips to short.

    The original design tried two orders (close + flip_open with SELL_TO_OPEN),
    but Alpaca rejects the open leg in the same MOC session because the close
    still pins the long position — position_intent validates intent, it does
    NOT bypass the size check. The fix is to defer the open to the next session:
    today we emit only the close, and tomorrow's run opens the short cleanly
    from a flat position.
    """
    qty = 65  # carrying long position from previous day
    target_abs = 65

    d = decide_transition(signal=-1, current_qty_signed=qty, target_abs_qty=target_abs, shortable=True)

    assert d.transition == "flip"
    assert len(d.orders) == 1, f"flip emits close-only, got {len(d.orders)} orders"

    (close,) = d.orders
    assert close.side == OrderSide.SELL
    assert close.qty == 65
    assert close.action == "flip_close"
    assert close.position_intent == PositionIntent.SELL_TO_CLOSE

    # Sanity: today's fill lands us flat
    final = _apply_fills(qty, d.orders)
    assert final == 0

    # Next session: flat + signal still short -> opens the short cleanly
    d2 = decide_transition(signal=-1, current_qty_signed=final, target_abs_qty=target_abs, shortable=True)
    assert d2.transition == "open"
    (open_,) = d2.orders
    assert open_.side == OrderSide.SELL
    assert open_.qty == 65
    assert open_.position_intent == PositionIntent.SELL_TO_OPEN
    final2 = _apply_fills(final, d2.orders)
    assert final2 == -65


# ---- Scenario 6: GHA cron fires repeatedly intra-day (idempotency) ------

def test_decisions_are_deterministic_when_state_unchanged():
    """Multiple cron fires within the same day on the same state must produce
    the same decision (not just same orders, same client_order_id will trigger
    Alpaca's duplicate rejection -> idempotent skip)."""
    state = dict(signal=1, current_qty_signed=66, target_abs_qty=80, shortable=True)
    decisions = [decide_transition(**state) for _ in range(5)]

    # All 5 calls produce identical decision
    assert all(d.transition == decisions[0].transition for d in decisions)
    assert all(len(d.orders) == len(decisions[0].orders) for d in decisions)
    for d in decisions:
        for i, o in enumerate(d.orders):
            ref = decisions[0].orders[i]
            assert (o.side, o.qty, o.action, o.position_intent) == (ref.side, ref.qty, ref.action, ref.position_intent)
