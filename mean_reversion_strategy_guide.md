# Building a Mean Reversion Strategy (Professional / HFT-Style)

A complete, self-contained build guide reconstructed from the MemLabs video *"How to create a Mean Reversion strategy (by ex HFT quant trader)"* and its companion Colab notebook. Following the steps here lets you reproduce the strategy end-to-end in Python with `numpy` and `pandas` only.

> Source video: https://www.youtube.com/watch?v=_urG139AM48
> Companion notebook: https://colab.research.google.com/drive/1RhTffZo2_l4mBEEjyN6QekISEJtEHx6Q
> Sample data CSV (BCH daily bars): `https://drive.google.com/uc?export=download&id=1eQN7nCrv1byqqX8oIt1Ks6SlDFb_ySlc`

---

## 1. Why this strategy is *not* the typical retail mean reversion

If you Google "mean reversion strategy" you will mostly find variations of the same retail recipe:

> Take a moving average of price. When price is N standard deviations *below* the moving average, **buy**. When it is N standard deviations *above*, **sell**. (Bollinger Bands, etc.)

That is **not** how it is done in a professional / HFT shop.

The professional approach instead:

1. Throws away "price" as the variable of interest. Studies **price movements** (log returns).
2. Looks for **auto-regressive** structure — i.e., does the next-bar return systematically depend on the previous-bar return?
3. Quantifies the edge with a simple **groupby on the sign of the previous return**, not with bands or thresholds.
4. Validates the edge with an **in-sample / out-of-sample split** before ever trading it.
5. Always trades — the signal is `+1` or `-1` on every bar — because the edge is small (~52% win rate) but persistent.

The mantra: *"What goes up must come down."* Quantified.

The math required is intentionally minimal: `log`, `exp`, mean, standard deviation, and basic algebra. No machine learning, no neural nets — at most a linear regression if you want to refine it.

---

## 2. Core idea in one paragraph

On a daily time bar, if yesterday's log return was **negative**, bet that today's return will be **positive**. If yesterday's was **positive**, bet that today's will be **negative**. Trade every day at the same UTC time. Size the trade as a fraction of equity so wins and losses compound. The proof that this is more than wishful thinking is a `groupby(sign_of_lag_1).aggregate(['sum','mean','count'])` on the log-return column — across both an in-sample and out-of-sample window.

---

## 3. Prerequisites

- Python 3.x
- `numpy`, `pandas` (and `matplotlib` for the equity-curve plot, available implicitly via `df.plot()`)
- A daily OHLC dataset for one instrument. The video uses **Bitcoin Cash (BCH) daily bars from 2022-04-13 to 2026-04-12** (~1456 rows). Any liquid asset with daily bars works; the specific instrument matters less than the methodology.

The dataset has these columns (as in the notebook): `t` (date), `T` (full timestamp), `s` (symbol), `i` (interval, `1d`), `o`, `h`, `l`, `c` (open, high, low, close).

The strategy uses **only the close `c`**.

---

## 4. Step-by-step build

The video has four chapters. This guide follows them.

### 4.1 Data engineering — load and convert to log returns

```python
import numpy as np
import pandas as pd

url = "https://drive.google.com/uc?export=download&id=1eQN7nCrv1byqqX8oIt1Ks6SlDFb_ySlc"
df = pd.read_csv(url)
```

Why log returns instead of simple returns?

- **Additivity.** `sum(log_returns)` over a period equals the log of the total compound return. That makes equity-curve math, Sharpe math, and cumulative-return math trivial — they are all just sums.
- **Models compound growth / position-size reinvestment naturally.** If a trade returns `+r` log, the next position is sized off the new (larger) equity. If it loses, the next position shrinks. You get this for free.
- **Symmetric around zero.** A `+10%` move and a `-10%` move are not symmetric in simple-return space, but in log space they are.

Compute the close-to-close log return:

```python
df['close_log_return'] = np.log(df['c'] / df['c'].shift())
```

The first row is `NaN` (no prior close). Leave it for now; later steps tolerate it.

### 4.2 Add the auto-regression column (lag-1)

The whole point is to study how today's log return depends on yesterday's. So add a lagged copy:

```python
df['close_log_return_lag_1'] = df['close_log_return'].shift()
```

Now each row has two relevant numbers: `close_log_return` (today) and `close_log_return_lag_1` (yesterday).

### 4.3 Encode direction (decompose return into sign + magnitude)

A log return decomposes into:

- **Direction**: `+1` if return was up, `-1` if it was down (the mathematical sign).
- **Magnitude**: the absolute value.

We discretize the lag into two buckets. Discretization is required because you cannot meaningfully `groupby` a floating-point number — you need bins.

```python
df['close_log_return_dir_lag_1'] = df['close_log_return_lag_1'].map(
    lambda x: 1 if x > 0 else -1
)
```

Equivalent: `np.sign(df['close_log_return_lag_1'])`. The lambda form is what the video uses. Note that `NaN > 0` is `False`, so the first row's direction will be `-1`; this is harmless because that row's `close_log_return` is also `NaN` and contributes nothing to aggregations.

### 4.4 Study price movements — the "money shot"

Group all rows by the previous bar's direction, and look at the next-bar (today's) log return inside each bucket:

```python
df.groupby('close_log_return_dir_lag_1').aggregate(
    {'close_log_return': ['sum', 'mean', 'count']}
)
```

What you want to see:

- For `dir_lag_1 == -1` (yesterday went down): **positive** `mean` and **positive** `sum`. Meaning: when the previous bar was down, the next bar tends to go up. Mean reversion.
- For `dir_lag_1 == +1` (yesterday went up): **negative** `mean` and **negative** `sum`. Meaning: when the previous bar was up, the next bar tends to go down. Mean reversion in the other direction.

The `count` column tells you how many samples are in each bucket — both should be reasonably balanced.

In the video on BCH 2022–2026 daily, both buckets show the expected sign, confirming a mean-reversionary regime.

> Diagnostic for a **momentum** asset (the inverse): the `dir_lag_1 == -1` bucket would have a *negative* mean, and `+1` would have a *positive* mean. Mean reversion tends to dominate at short / market-microstructure horizons (seconds, minutes, hours, days). Momentum tends to dominate at longer horizons (weeks, months). FX is a notable exception where some pairs show mean-reverting behaviour at longer scales.

### 4.5 In-sample / out-of-sample validation (do not skip this)

A signal that looks great over the whole dataset can still be a regime-dependent ghost. To trust it, the same pattern must hold in both an old "in-sample" slice and a held-out "out-of-sample" slice.

Split **by time**, never randomly:

```python
i = int(len(df) * 0.75)
in_sample, out_sample = df.iloc[:i], df.iloc[i:]
```

Run the same aggregation on each:

```python
in_sample.groupby('close_log_return_dir_lag_1').aggregate(
    {'close_log_return': ['sum', 'mean', 'count']}
)

out_sample.groupby('close_log_return_dir_lag_1').aggregate(
    {'close_log_return': ['sum', 'mean', 'count']}
)
```

**Decision rule.** Both tables must show the same sign pattern (negative `dir_lag_1` → positive mean, positive `dir_lag_1` → negative mean). Magnitudes can differ — they will — but the **signs must agree**. If they don't, the pattern is non-stationary (regime-changed) and the strategy is dead. Do not trade it.

This is the simplest form of statistical validation. Cross-validation (walk-forward, k-fold by time, etc.) is more robust and out of scope here, but the principle is the same: never trust a signal that only exists in a single slice of history.

### 4.6 Build the trading signal

The strategy is "bet against the previous bar," so the signal is the negation of the lag-1 direction:

```python
df['signal'] = -1 * df['close_log_return_dir_lag_1']
```

- Yesterday down (`dir_lag_1 = -1`) → `signal = +1` → bet up today.
- Yesterday up (`dir_lag_1 = +1`) → `signal = -1` → bet down today.

The signal is binary `±1` and present on every row. There is no flat / no-trade state.

(You could instead fit a linear regression `next_return = a * lag_return + b` and use the regression's prediction as a continuous signal — the video calls this out as the next level of refinement. **Do not** reach for neural networks here; the relationship is too simple and the signal-to-noise too low. A linear model is the correct ceiling.)

### 4.7 Per-bar trade log return

Multiply the signal by the realized log return of the bar you are betting on:

```python
df['trade_log_return'] = df['signal'] * df['close_log_return']
```

Sanity-check examples:

| signal | actual log return | trade_log_return | interpretation                |
| ------ | ----------------- | ---------------- | ----------------------------- |
| +1     | -0.01             | -0.01            | Bet up, went down → loss.     |
| +1     | +0.02             | +0.02            | Bet up, went up → win.        |
| -1     | -0.015            | +0.015           | Bet down, went down → win.    |
| -1     | +0.008            | -0.008           | Bet down, went up → loss.     |

Note this does **not yet account for fees** — see Section 6.

### 4.8 Equity curve

Because log returns are additive, the equity curve in log space is a simple cumulative sum:

```python
df['cum_trade_log_return'] = df['trade_log_return'].cumsum()
df['cum_trade_log_return'].plot()
```

Under the hood this models continuous reinvestment: each trade is sized off current equity, so wins compound and losses shrink the next position. You get the compound-growth model essentially for free from log returns.

What you should see on a working signal: a curve that drifts up over time with drawdowns, but where drawdowns recover quickly — a hallmark of mean-reverting equity curves. If the curve flatlines or rolls over precisely at your in-sample / out-of-sample boundary, the edge has died.

---

## 5. Strategy statistics

### 5.1 Win rate

```python
df['is_won'] = df['trade_log_return'] > 0
win_rate = df['is_won'].mean()
```

Trick used: the boolean column casts to `0`/`1`, so its mean is the win probability. On BCH 2022–2026 the win rate is around **52%**. That is normal and expected for a market-microstructure-style edge — you have a tiny statistical advantage that you exploit by trading frequently.

> Win rate is a **misleading optimization target.** A strategy that wins 49% but with average winners much larger than average losers can be highly profitable. Optimize for **average trade return (positive expected value)** and **risk-adjusted return (Sharpe)**, not win rate.

### 5.2 Total gross compound return

Convert the cumulative log return back to an ordinary return:

```python
r = np.exp(df['trade_log_return'].sum()) - 1
```

`exp` is the inverse of `log`, so `exp(sum(log_returns))` gives the gross multiplier on starting capital, and subtracting `1` gives the percent return. Equivalently, this is the last value of `cum_trade_log_return` exponentiated.

To translate to dollars:

```python
10 * r       # what $10 would have grown into (above starting capital)
20000 * r    # what $20,000 would have grown into
```

The video reports a roughly **21× compound return** on BCH 2022–2026 — i.e., $10 starting capital ends near $213, $20,000 ends near $426k. Two things drive that magnitude: (1) the statistical edge, and (2) compounding (reinvesting profits and shrinking on losses, which `log return` math models automatically). Without either, returns collapse. Most retail strategies fail one of the two.

> "Gross" means **before fees**. See Section 6.

### 5.3 Annualized Sharpe ratio

Sharpe is the per-bar risk-adjusted return. On the log-return series:

```python
mu    = df['trade_log_return'].mean()        # positive => positive expected value
sigma = df['trade_log_return'].std()         # how stable returns are
daily_sharpe = mu / sigma
```

Mean must be positive — that is the existence of an edge. Standard deviation captures how much returns deviate from the mean; smaller is more stable.

Annualize so you can compare across timeframes:

```python
annualized_sharpe = (mu / sigma) * np.sqrt(N)
```

`N` is **the number of trading periods per year** at your bar frequency:

- **Crypto, daily bars: `N = 365`** (trades every day, weekends included). The video uses this.
- Equities, daily bars: `N ≈ 252` (Mon–Fri, excluding holidays).
- FX, daily bars: also ~252.
- Hourly bars on a 24/7 market: `N = 365 * 24`. Etc.

Pick the right `N` for your asset and your bar frequency. Getting this wrong silently inflates or deflates your reported Sharpe.

> A higher Sharpe corresponds to a straighter, less-jagged equity curve. A low Sharpe means meaningful drawdowns, which makes leverage dangerous — you can be liquidated or margin-called even if the long-run edge is real. Be conservative with leverage when Sharpe is low.

> Caveat: at very short timeframes (intraday / minute), Sharpes can look extraordinarily high. They typically degrade once you factor in fees, slippage, and capacity.

---

## 6. Things the notebook deliberately leaves out — finish them before going live

The video flags these explicitly as exercises:

1. **Transaction fees and slippage.** Every fill costs you. Model the per-trade fee from your venue and subtract it from each `trade_log_return`. Re-plot the equity curve. Re-compute compound return and Sharpe. A "great" gross strategy can become unprofitable net of realistic fees, especially because this signal trades **every day**.
2. **Realized vs unrealized P&L.** The notebook only models realized P&L per bar. A more honest equity curve marks the open position to the current close.
3. **Position sizing in real dollars.** The whole pipeline is intentionally **scale-free** (multiplying log returns by `±1`), so it works at any account size. Once you go live, decide how much of equity each trade uses — the simplest faithful version is "100% of equity," which is what the compound model implicitly assumes. Using less is safer; using leverage requires care given the Sharpe.
4. **Slippage from non-instant fills** at your trade time.
5. **Walk-forward / k-fold (time-aware) cross-validation.** A single 75/25 split is the bare minimum.

---

## 7. How to actually trade it (manual or automated)

The whole logic, once validated, collapses to:

> Every day at **00:00 UTC**, look at yesterday's daily-bar close-to-close log return. If it was negative, **go long for the next 24h**. If it was positive, **go short for the next 24h**. Close the position at the next 00:00 UTC and reopen based on the new previous bar.

Two things that matter:

- **One consistent UTC time.** The whole study is built on close-to-close UTC bars. If you trade at a different time of day, you are trading a different signal, and the validated statistics no longer apply. Globally distributed desks (NY / London / Singapore) all standardize on UTC for exactly this reason.
- **Trade every day.** The edge is a tiny 52%-ish win rate. It only materializes in aggregate. Skipping bars based on gut feel kills it.

That is all there is to the manual version. The automated version is the same logic, scheduled.

---

## 8. Reference implementation (single block)

```python
import numpy as np
import pandas as pd

# 1) Load data
url = "https://drive.google.com/uc?export=download&id=1eQN7nCrv1byqqX8oIt1Ks6SlDFb_ySlc"
df = pd.read_csv(url)

# 2) Log returns
df['close_log_return'] = np.log(df['c'] / df['c'].shift())

# 3) Auto-regression: lag-1 log return
df['close_log_return_lag_1'] = df['close_log_return'].shift()

# 4) Discretize lag direction
df['close_log_return_dir_lag_1'] = df['close_log_return_lag_1'].map(
    lambda x: 1 if x > 0 else -1
)

# 5) Study price movements (the edge)
edge_full = df.groupby('close_log_return_dir_lag_1').aggregate(
    {'close_log_return': ['sum', 'mean', 'count']}
)
print(edge_full)

# 6) In-sample / out-of-sample validation
i = int(len(df) * 0.75)
in_sample, out_sample = df.iloc[:i], df.iloc[i:]

print(in_sample.groupby('close_log_return_dir_lag_1').aggregate(
    {'close_log_return': ['sum', 'mean', 'count']}
))
print(out_sample.groupby('close_log_return_dir_lag_1').aggregate(
    {'close_log_return': ['sum', 'mean', 'count']}
))
# Both must show: dir = -1 -> positive mean; dir = +1 -> negative mean.

# 7) Signal (mean reversion = negate lag direction)
df['signal'] = -1 * df['close_log_return_dir_lag_1']

# 8) Per-bar trade return and equity curve
df['trade_log_return']     = df['signal'] * df['close_log_return']
df['cum_trade_log_return'] = df['trade_log_return'].cumsum()
df['cum_trade_log_return'].plot()

# 9) Statistics
df['is_won'] = df['trade_log_return'] > 0
win_rate          = df['is_won'].mean()
gross_compound_r  = np.exp(df['trade_log_return'].sum()) - 1   # e.g. 0.21 for 21%, 21.0 for 21x
mu    = df['trade_log_return'].mean()
sigma = df['trade_log_return'].std()
daily_sharpe      = mu / sigma
annualized_sharpe = daily_sharpe * np.sqrt(365)   # 365 for crypto; use 252 for equities/FX

print(f"win_rate={win_rate:.4f}")
print(f"gross_compound_return={gross_compound_r:.4f}")
print(f"daily_sharpe={daily_sharpe:.4f}")
print(f"annualized_sharpe={annualized_sharpe:.4f}")
```

---

## 9. Mental checklist before going live

1. Picked a single asset, single timeframe, single UTC trade time.
2. Computed close-to-close log returns.
3. Confirmed the lag-1 sign-grouped means show mean-reverting signs.
4. Confirmed those signs **persist** in a held-out out-of-sample slice.
5. Modeled fees and re-checked the equity curve and Sharpe net of fees.
6. Decided on position size / leverage that you can survive a realistic drawdown of (look at the worst peak-to-trough move on the equity curve).
7. Automated the trade at exactly 00:00 UTC (or your chosen consistent time).
8. Set up monitoring so you can detect the day the edge dies — a rolling out-of-sample check on the most recent window. Patterns are non-stationary; a regime change (think FTX collapse for crypto) can flip the sign overnight.

---

## 10. Glossary

- **Auto-regression**: predicting a value at time `t` from values at `t-1, t-2, ...`. Here we use only lag 1.
- **Log return**: `r_t = ln(P_t / P_{t-1})`. Additive across time, symmetric around zero, models compound growth when summed.
- **Mean reversion**: tendency for returns to flip sign — what went up tends to come back down on the next bar.
- **Momentum**: the opposite — what went up tends to keep going up.
- **In-sample / out-of-sample**: you fit / observe the pattern on the older slice, then check it still exists on the held-out newer slice. Splits **must** be by time, never random.
- **Positive EV (positive expected value)**: average per-trade return is positive. This is the actual goal, not a high win rate.
- **Sharpe ratio**: `mean(returns) / std(returns)`, optionally annualized by `sqrt(N)`. Measures return per unit of volatility.
- **Non-stationary**: statistical properties (mean, variance, the relationship we're exploiting) change over time. Real markets are non-stationary, which is why out-of-sample validation matters.
- **Scale-free (the way this is modeled)**: the math is in log-return space and signals are `±1`, so the strategy does not depend on starting capital. You can plug in $10 or $10M.

---

*Compiled from the MemLabs video and notebook so the strategy can be rebuilt without re-watching. The methodology — log returns, lag-1 sign grouping, time-based out-of-sample validation, log-space equity curve, properly annualized Sharpe — is the actual professional template; the BCH dataset is just one example to demonstrate it on.*
