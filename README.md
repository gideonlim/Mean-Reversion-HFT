# Mean-Reversion Daily Paper-Trading on Alpaca

Implementation of the lag-1 mean-reversion strategy described in
[mean_reversion_strategy_guide.md](mean_reversion_strategy_guide.md), running
as a paper-trading bot on Alpaca for a single configurable US-equities
symbol (default: SPY) on daily bars.

The bot trades close-to-close using market-on-close orders, scheduled by
Windows Task Scheduler. It consists of:

- A **backtest** that validates the strategy edge in-sample / out-of-sample
  on Alpaca historical bars.
- A **live runner** with idempotency, ET-aware scheduling, and a dry-run
  mode for safe verification before unattended scheduling.

> Algo Trader Plus is **not required** for this strategy. The free-tier IEX
> daily bars are essentially identical to SIP for liquid ETFs, and the
> 200-call/min rate limit is far above what a once-daily script needs.

## Files

| File | Role |
|---|---|
| [config.py](config.py) | Frozen `Settings` dataclass + `et_today()` / `et_now()` helpers |
| [strategy.py](strategy.py) | Pure math: log returns, signal, validation, stats |
| [data.py](data.py) | Alpaca historical bars + `last_two_closed_bars` (excludes today's partial bar) |
| [broker.py](broker.py) | Alpaca trading I/O: clock, asset, position, MOC orders, cancels |
| [backtest.py](backtest.py) | CLI: pull history, validate edge, print stats, save equity curve PNG |
| [live.py](live.py) | CLI: daily state machine; `--dry-run` for verification |
| [scripts/install_task.ps1](scripts/install_task.ps1) | Register Windows scheduled task |
| [tests/](tests) | Unit tests for strategy math + broker decision logic + ET-date / window edge cases |

## Quickstart

```bash
# 1. Install dependencies (requires Python 3.11+)
pip install -e .

# 2. Configure Alpaca paper API keys
cp .env.example .env
# Edit .env and paste your APCA_API_KEY_ID and APCA_API_SECRET_KEY (paper keys, not live)

# 3. Run unit tests
python -m pytest tests/ -v

# 4. Backtest the strategy on historical bars
python backtest.py
# Prints validation tables + stats; saves logs/equity_curve_<symbol>_<date>.png
# Exits 0 if the in/out-of-sample edge gate passes, 1 if it fails.

# 5. Verify the live runner end-to-end without submitting orders
python live.py --dry-run

# 6. (Once you're confident) submit a real paper trade manually inside the trade window
python live.py

# 7. Install the daily scheduler (one-time setup; needs Administrator)
powershell.exe -ExecutionPolicy Bypass -File scripts\install_task.ps1
```

## Strategy logic

For each daily close `c`:

```
log_return = log(c_t / c_{t-1})
signal     = -sign(log_return_lag_1)         # bet against yesterday
trade_log_return = signal * log_return
```

Validation: group all rows by `sign(log_return_lag_1)` and check the mean
of today's `log_return` in each bucket. Both the in-sample (first 75%) and
out-of-sample (last 25%) splits must show:

- yesterday-down bucket → today's mean `> 0`
- yesterday-up bucket → today's mean `< 0`

If either split fails, the edge is non-stationary on the chosen symbol and
the strategy is dead. `backtest.py` exits non-zero in that case.

## Live state machine

`live.py` runs nine steps. Pre-flight failures exit `0` so Task Scheduler
doesn't alarm on expected skips (market closed, too early in the day,
already ran, etc.):

1. **Local idempotency**: `logs/last_run_<ET_date>.json` exists → exit.
2. **Trade window** (dynamic, derived from `clock.next_close`):
   - Market closed today → exit.
   - Now is before window start → exit (will retry on next 15-min run).
   - Now is past window end → log WARNING + exit.
3. **Asset tradability** check.
4. **Cancel orphan orders** for the symbol.
5. **Compute signal** from the last two **fully-closed** daily bars
   (today's partial bar is filtered out).
6. **Read state**: signed position qty + account cash.
7. **Decide transition**:

| current | target | orders submitted |
|---|---|---|
| 0 → ±qty | open | 1 MOC |
| ±qty → ±qty (same sign) | scale | 1 MOC for the delta |
| ±qty → ±qty (same value) | no-op | 0 |
| ±qty → ∓qty (sign flip) | flip | 1 MOC (close only — open defers to next session) |
| signal=-1 + not shortable | skip | force-close any long, stay flat |

Single net-delta order everywhere. On sign flips, Alpaca rejects opening the
opposite side while the close MOC still pins the position (position_intent
validates intent, it doesn't bypass the size check), so the flip is split
across two sessions: today's close lands the position at 0, and tomorrow's
run opens the new direction cleanly from flat.

8. **Submit** with `client_order_id = meanrev-<ET_date>-<symbol>-<action>`.
   Deterministic and ET-dated, so reruns reject as duplicates server-side.
9. **Log JSON DECISION** record + write `last_run_<ET_date>.json` marker
   (skipped on `--dry-run`).

## Verification ladder

Walk through these in order before letting the scheduler run unattended:

1. `python -m pytest tests/ -v` → math + decision-table + window/idempotency tests pass.
2. `python backtest.py` → validation tables show mean-reverting signs in both splits;
   review `logs/equity_curve_<symbol>_<date>.png`.
3. `python live.py --dry-run` (any time of day, any day of week) → exercises
   auth + data + signal + sizing + decision logic without submitting.
4. `python live.py` once manually inside the dynamic ET trade window → confirm
   a real paper order shows up in the Alpaca dashboard.
5. `pwsh scripts/install_task.ps1` → register the daily 15-min-cadence task.
   Watch logs for the first 3 trading days.

## Logs

Everything goes to [logs/](logs):
- `live.log` — rotating (10 MB × 12 backups).
- `last_run_<ET_date>.json` — idempotency marker; contains the decision record for that day.
- `equity_curve_<symbol>_<date>.png` — written by `backtest.py`.

## Sydney-machine timezone notes

This project runs on a Windows machine in Australia/Sydney (UTC+10/+11),
trading US equities (UTC-4/-5). Two relevant invariants:

- **All trade-relevant dates use ET** (`config.et_today()`). The marker
  filename, the `client_order_id`, and the trade-window check all route
  through `zoneinfo.ZoneInfo("America/New_York")`. The local Windows date
  is never used for trading logic.
- **The trade window is computed from `clock.next_close`**, not a hardcoded
  time. This automatically handles US early-close days (1:00 PM ET) and
  both US and Australian DST transitions without any manual adjustment.

## Out of v1 scope

- Fee/slippage modeling (Alpaca paper has no fees; revisit before live money).
- Walk-forward / k-fold cross-validation (single 75/25 split is the baseline).
- Continuous-signal linear regression refinement.
- Multi-asset / portfolio.
- Rolling out-of-sample edge-death detector.
- Email/Slack alerts (Task Scheduler email-on-failure is enough initially).

## Adjusting symbol or other settings

Edit `config.py`. The `Settings` dataclass is small and self-explanatory.
Don't move secrets in here — they belong in `.env`.

## Removing the scheduled task

```powershell
Unregister-ScheduledTask -TaskName "MeanReversionPaperTrade" -Confirm:$false
```
