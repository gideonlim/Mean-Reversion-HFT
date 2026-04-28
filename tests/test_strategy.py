"""Hand-computed fixtures for the pure strategy math."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import strategy


# Six-day series with a perfectly mean-reverting up/down pattern.
# Constructed so every column can be verified by hand.
HAND_CLOSES = [100.0, 102.0, 100.0, 101.0, 99.0, 100.0]


@pytest.fixture
def hand_df():
    return pd.DataFrame({"close": HAND_CLOSES})


def test_add_signal_columns_log_returns(hand_df):
    out = strategy.add_signal_columns(hand_df)
    expected = [
        np.nan,
        math.log(102 / 100),
        math.log(100 / 102),
        math.log(101 / 100),
        math.log(99 / 101),
        math.log(100 / 99),
    ]
    assert math.isnan(out["log_return"].iloc[0])
    np.testing.assert_allclose(out["log_return"].iloc[1:], expected[1:], rtol=1e-12)


def test_add_signal_columns_lag_and_dir(hand_df):
    out = strategy.add_signal_columns(hand_df)
    # log_return_lag_1 is just the prior row's log_return
    assert math.isnan(out["log_return_lag_1"].iloc[0])
    assert math.isnan(out["log_return_lag_1"].iloc[1])
    np.testing.assert_allclose(
        out["log_return_lag_1"].iloc[2:],
        out["log_return"].iloc[1:5],
        rtol=1e-12,
    )
    # NaN > 0 is False -> dir = -1; matches the lambda in add_signal_columns.
    expected_dirs = [-1, -1, 1, -1, 1, -1]
    assert list(out["dir_lag_1"]) == expected_dirs


def test_add_signal_columns_signal_and_trade(hand_df):
    out = strategy.add_signal_columns(hand_df)
    # signal = -dir_lag_1
    assert list(out["signal"]) == [1, 1, -1, 1, -1, 1]
    # trade_log_return = signal * log_return; first two are NaN (no log_return / no lag)
    assert math.isnan(out["trade_log_return"].iloc[0])
    assert not math.isnan(out["trade_log_return"].iloc[1])  # row 1 has log_return; lag is NaN -> dir=-1 -> signal=+1
    # Verify that mean-reversion bets pay off on this perfectly-MR fixture
    nonnan = out["trade_log_return"].dropna()
    assert (nonnan > 0).all(), "every trade should win on the perfectly-MR fixture"


def test_signal_from_lag():
    assert strategy.signal_from_lag(0.01) == -1   # yesterday up -> bet down
    assert strategy.signal_from_lag(-0.01) == 1   # yesterday down -> bet up
    assert strategy.signal_from_lag(0.0) == 1     # zero treated as not-positive (matches lambda)
    assert strategy.signal_from_lag(float("nan")) == 1


def test_validate_in_out_sample_pass_on_mean_reverting():
    # 100 alternating ±1% log returns -> perfectly mean-reverting
    n = 100
    closes = [100.0]
    for i in range(n):
        r = 0.01 if i % 2 == 0 else -0.01
        closes.append(closes[-1] * math.exp(r))
    df = strategy.add_signal_columns(pd.DataFrame({"close": closes}))
    result = strategy.validate_in_out_sample(df, split=0.75)
    assert result.edge_holds is True
    # Both buckets present in both splits
    assert -1 in result.in_sample.index and 1 in result.in_sample.index
    assert -1 in result.out_sample.index and 1 in result.out_sample.index
    # Sign check
    assert result.in_sample[("log_return", "mean")].loc[-1] > 0
    assert result.in_sample[("log_return", "mean")].loc[1] < 0
    assert result.out_sample[("log_return", "mean")].loc[-1] > 0
    assert result.out_sample[("log_return", "mean")].loc[1] < 0


def test_validate_in_out_sample_fail_on_momentum():
    # Strong momentum: returns come in 5-day same-sign runs.
    # Within each run, lag and today share sign on 4 of 5 rows -> mean of each
    # bucket has the SAME sign as the bucket's lag, which violates MR.
    n = 100
    closes = [100.0]
    sign = 1
    for i in range(n):
        if i % 5 == 0:
            sign = -sign
        closes.append(closes[-1] * math.exp(0.01 * sign))
    df = strategy.add_signal_columns(pd.DataFrame({"close": closes}))
    result = strategy.validate_in_out_sample(df, split=0.75)
    assert result.edge_holds is False


def test_has_mean_reverting_signs_unit():
    # Construct a groupby-style table directly and exercise the gate predicate.
    cols = pd.MultiIndex.from_tuples([("log_return", "sum"), ("log_return", "mean"), ("log_return", "count")])
    pass_tbl = pd.DataFrame({-1: [0.5, 0.005, 100], 1: [-0.4, -0.004, 100]}).T
    pass_tbl.columns = cols
    pass_tbl.index.name = "dir_lag_1"
    assert strategy._has_mean_reverting_signs(pass_tbl) is True

    fail_tbl = pd.DataFrame({-1: [-0.5, -0.005, 100], 1: [0.4, 0.004, 100]}).T
    fail_tbl.columns = cols
    fail_tbl.index.name = "dir_lag_1"
    assert strategy._has_mean_reverting_signs(fail_tbl) is False

    # Missing bucket -> fails
    missing = pd.DataFrame({-1: [0.5, 0.005, 100]}).T
    missing.columns = cols
    missing.index.name = "dir_lag_1"
    assert strategy._has_mean_reverting_signs(missing) is False


def test_compute_stats_on_perfect_winner():
    # Hand-build a df where every trade returns +0.01 in log space (10 trades).
    df = pd.DataFrame({"trade_log_return": [0.01] * 10})
    stats = strategy.compute_stats(df, n=252)
    assert stats.win_rate == 1.0
    np.testing.assert_allclose(stats.gross_compound_return, math.exp(0.10) - 1, rtol=1e-12)
    # std is zero -> sharpe falls back to 0 (avoid divide-by-zero blowup)
    assert stats.daily_sharpe == 0.0
    assert stats.annualized_sharpe == 0.0
    assert stats.n_trades == 10


def test_compute_stats_mixed():
    # 5 wins of +0.01 and 5 losses of -0.005 -> win_rate 0.5, positive mean
    df = pd.DataFrame({"trade_log_return": [0.01, -0.005] * 5})
    stats = strategy.compute_stats(df, n=252)
    assert stats.win_rate == 0.5
    assert stats.daily_sharpe > 0
    np.testing.assert_allclose(stats.annualized_sharpe, stats.daily_sharpe * math.sqrt(252), rtol=1e-12)


def test_groupby_lag_dir_buckets():
    # Hand-built df with both lag dirs and known means
    df = pd.DataFrame({
        "log_return": [0.02, -0.01, 0.03, -0.02, 0.01, -0.015],
        "dir_lag_1": [-1, 1, -1, 1, -1, 1],
    })
    tbl = strategy.groupby_lag_dir(df)
    np.testing.assert_allclose(tbl[("log_return", "mean")].loc[-1], (0.02 + 0.03 + 0.01) / 3)
    np.testing.assert_allclose(tbl[("log_return", "mean")].loc[1], (-0.01 - 0.02 - 0.015) / 3)
    assert tbl[("log_return", "count")].loc[-1] == 3
    assert tbl[("log_return", "count")].loc[1] == 3


# ---- LONG_ONLY mode ---------------------------------------------------

def test_long_only_clamps_short_signals_to_zero():
    closes = [100.0, 102.0, 100.0, 101.0, 99.0, 100.0]
    out = strategy.add_signal_columns(pd.DataFrame({"close": closes}), long_only=True)
    # With long_only, signal can only be 0 or +1 (never -1).
    assert (out["signal"] >= 0).all()
    # Specifically: dir_lag_1 = +1 -> signal would be -1 -> clamped to 0.
    short_days = out["dir_lag_1"] == 1
    assert (out.loc[short_days, "signal"] == 0).all()
    # Long days still have signal = +1.
    long_days = out["dir_lag_1"] == -1
    assert (out.loc[long_days, "signal"] == 1).all()


def test_long_only_trade_log_return_zero_on_flat_days():
    closes = [100.0, 102.0, 100.0, 101.0, 99.0, 100.0]
    out = strategy.add_signal_columns(pd.DataFrame({"close": closes}), long_only=True)
    flat_days = out["signal"] == 0
    np.testing.assert_array_equal(out.loc[flat_days, "trade_log_return"].values, 0.0)


def test_long_only_false_matches_original_behavior():
    closes = [100.0, 102.0, 100.0, 101.0, 99.0, 100.0]
    default = strategy.add_signal_columns(pd.DataFrame({"close": closes}))
    explicit = strategy.add_signal_columns(pd.DataFrame({"close": closes}), long_only=False)
    pd.testing.assert_frame_equal(default, explicit)
