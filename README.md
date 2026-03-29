# vixfix-scanner

Automated stock scanner and paper trading simulator using Bollinger Bands, VixFix, MACD, and Stochastic divergence signals. Scans NYSE + NASDAQ daily and weekly, emails HTML results with an auto-updating Excel trade tracker attached.

---

## Quick Start for New Claude Sessions

> Tell Claude: *"Read the README at the top of this repo and help me with X."*

Everything Claude needs to understand this project is in this file. The code is the implementation — the README is the memory.

---

## Table of Contents

1. [Active Strategies](#active-strategies)
2. [Strategy Logic](#strategy-logic)
3. [Signal Parameters](#signal-parameters)
4. [Tier Definitions](#tier-definitions)
5. [Entry & Exit Rules](#entry--exit-rules)
6. [System Architecture](#system-architecture)
7. [File Responsibilities](#file-responsibilities)
8. [Workflow Schedules](#workflow-schedules)
9. [Data Flow](#data-flow)
10. [Setup Checklist](#setup-checklist)
11. [Known Issues & Decisions](#known-issues--decisions)
12. [Pending Work](#pending-work)

---

## Active Strategies

| Strategy | Interval | Hold | Tier | Notes |
|----------|----------|------|------|-------|
| BB+VixFix+MACD+Stoch | 1wk | 20 weeks | Tier 1 Ultra | Highest conviction |
| BB+VixFix+Stoch | 1wk | 20 weeks | Tier 2 High | MACD not required |
| BB+VixFix+MACD+Stoch | 1d | 60 days | Tier 1 Ultra | Daily version |
| BB+VixFix+Stoch | 1d | 60 days | Tier 2 High | Daily version |

Win rates and full stats are stored in `backtest_stats.json` after running backtests.
Position size: **$10,000/trade** for all tiers.

---

## Strategy Logic

### BB Trigger Candle (required for all tiers)
- Green candle: `close > open`
- Low touches or pierces the **20-period Bollinger Band lower band** (2.0 std dev)
- **Confirmed** by next candle: also green, opens ≥ trigger open, closes ≥ trigger open
- **Entry** = close of trigger candle
- **Stop loss** = low of trigger candle

### VixFix (Williams VixFix — synthetic fear gauge)
- Formula: `(highest_close_30 - current_low) / highest_close_30 * 100`
- A VixFix spike fires when WVF ≥ BB upper band (20-period, 2.0 std) OR WVF ≥ 75-period rolling max × 0.85
- **VixFix Divergence**: two VixFix-confirmed trigger candles where:
  - Recent price low < prior price low (lower low in price)
  - Recent WVF > prior WVF (higher fear reading = more extreme fear)
  - Prior trigger must also pass the no-break-before filter
  - Maximum gap between triggers: 35 bars

### MACD Divergence (Tier 1 only)
- Uses 12/26/9 exponential MACD
- **Type A**: histogram at recent trigger > histogram at prior trigger
- **Type B**: MACD line OR signal line higher at recent vs prior trigger
- Either Type A or Type B qualifies

### Stochastic Divergence
- 14-period Fast Stochastic
- Price makes a lower low (vs 25-bar lookback minimum)
- Stochastic K makes a higher low (vs 25-bar lookback minimum)
- Must occur within 10 bars of a valid BB trigger candle pair
- Checked within ±5 bars of VixFix signal (SCAN_DELAY)

### Filters Applied to Every Signal
- Price > $10
- Market cap > $1 billion
- Stop distance < 11% (i.e. `(entry - stop) / entry < 0.11`)
- No candle in prior 10 bars had a low below the trigger candle low (`no_break_before`)
- Stock must be below 85% of its 52-week high (weekly) or 252-day high (daily)

### What Was Removed
- `no_break_after` filter was **permanently removed** from all files. It was retroactively disqualifying signals where the stop loss was later hit, causing losses to disappear from backtests instead of being counted. This inflated win rates. Win rates dropped after removal — these are now honest numbers. **Do not re-add this filter.**

---

## Signal Parameters

```python
# Bollinger Bands
BB_LENGTH      = 20
BB_MULT        = 2.0

# VixFix
VF_PD          = 30     # lookback period for highest close
VF_BBL         = 20     # BB period for VixFix
VF_MULT        = 2.0    # BB multiplier for VixFix
VF_LB          = 75     # rolling max lookback
VF_PH          = 0.85   # % of rolling max threshold
VF_NEAR        = 2      # bars either side to look for VixFix spike near trigger
MAX_GAP        = 35     # max bars between prior and recent trigger

# MACD
MACD_FAST      = 12
MACD_SLOW      = 26
MACD_SIGNAL    = 9

# Stochastic
STOCH_K        = 14
STOCH_LOOKBACK = 25

# Filters
NO_BREAK_BARS  = 10
LOOKBACK       = 10
MAX_STOP_DIST  = 0.11
MIN_PRICE      = 10.0
MIN_MARKET_CAP = 1_000_000_000
SCAN_DELAY     = 5

# Weekly backtest
YEAR_HIGH_BARS = 52
HOLD_BARS      = 20
WIN_TARGET     = 0.13

# Daily backtest
YEAR_HIGH_BARS = 252
HOLD_BARS      = 60
WIN_TARGET     = 0.13

POSITION_HIGH  = 10000.0
YEARS_HISTORY  = 15
```

---

## Tier Definitions

| Tier | Conditions | Description |
|------|-----------|-------------|
| **Tier 1 Ultra** | BB + VixFix div + MACD div + Stoch div | All four confirmed. Highest conviction. |
| **Tier 2 High** | BB + VixFix div + Stoch div | MACD not required. |

In scan emails, Tier 2 shows only tickers NOT already in Tier 1 (de-duplicated upward).

---

## Entry & Exit Rules

### Weekly Strategy (1wk)
- Signal fires after **Friday's weekly close** (candle is finalized)
- **Buy**: at Friday's close price (trigger candle close) — enter Monday at market open
- **Stop**: low of the Friday trigger candle
- **Win target**: entry × 1.13 (+13%)
- **Max hold**: 20 weeks
- **Only act on signals from the Friday evening scheduled scan** — manual mid-week runs show incomplete candles

### Daily Strategy (1d)
- Signal fires after **today's daily close** (4pm ET)
- **Buy**: at today's close price — enter next morning at market open
- **Stop**: low of today's trigger candle
- **Win target**: entry × 1.13 (+13%)
- **Max hold**: 60 trading days

### Trade Resolution (paper tracker)
- WIN: high reaches ≥ win target within hold period → exit at win target price
- LOSS: low hits ≤ stop loss within hold period → exit at stop loss price
- NEUTRAL: hold period expires without either → exit at closing price on last day

---

## System Architecture

```
GitHub Repo (ckurau/vixfix-scanner)
│
├── Python scripts (repo root)
│   ├── scanner_combined.py      ← Weekly scanner
│   ├── scanner_daily.py         ← Daily scanner
│   ├── backtest.py              ← Weekly backtest
│   ├── backtest_daily.py        ← Daily backtest
│   ├── backtest_optimize.py     ← Exit target optimizer
│   ├── trade_tracker.py         ← Excel builder + trade resolver
│   ├── debug_ticker.py          ← Single ticker diagnostic
│   └── debug_volume.py          ← Volume filter diagnostic
│
├── Persistent data files (committed to repo)
│   ├── backtest_stats.json      ← Written by backtests, read by scanners
│   ├── trade_history.json       ← Written by scheduled scans, accumulated over time
│   └── README.md                ← This file
│
└── .github/workflows/
    ├── scanner.yml              ← Scheduled scans (daily + weekly)
    └── backtests.yml            ← Manual dispatch: backtests, optimizer, debug
```

### Key Data Flow
```
1. Run Backtests workflow (manual)
   → backtest.py / backtest_daily.py run
   → write backtest_stats.json
   → commit backtest_stats.json to repo

2. Scheduled scan fires (cron: 2:30am UTC Tue–Sat = 6:30pm PST Mon–Fri)
   → scanner reads backtest_stats.json for historical stats
   → scans all NYSE + NASDAQ tickers
   → resolves OPEN trades in trade_history.json (checks price vs win target / stop)
   → adds new signals to trade_history.json (scheduled runs only)
   → builds trade_tracker.xlsx from trade_history.json
   → emails HTML report + Excel attachment
   → commits updated trade_history.json to repo

3. workflow_dispatch (manual trigger)
   → runs scan and emails report with current tracker
   → does NOT add new signals or commit trade_history.json
   → safe for testing at any time
```

---

## File Responsibilities

### `scanner_combined.py`
- Weekly scanner (interval=1wk)
- Runs `check_vixfix()` and `check_stoch()` on each ticker
- Reads `backtest_stats.json` for historical win rates to display in email
- Classifies tickers into Tier 1 (VixFix+MACD+Stoch) and Tier 2 (VixFix+Stoch)
- On scheduled runs: fetches entry/stop prices, adds new rows to `trade_history.json`
- Resolves OPEN trades in `trade_history.json` on every run
- Builds `trade_tracker.xlsx` and attaches to email
- Emails HTML report with tier stats + red ticker symbols

### `scanner_daily.py`
- Daily scanner (interval=1d) — identical logic, different interval/hold constants
- Runs Mon–Fri; weekly scanner runs Friday only (checked internally)
- Same tracker/email behavior as `scanner_combined.py`

### `backtest.py`
- Full historical backtest, weekly interval, 15 years, NYSE+NASDAQ
- Tiers 1 & 2 only
- Writes results to `backtest_stats.json` under key `"weekly"`
- Emails plain-text report with full trade history

### `backtest_daily.py`
- Full historical backtest, daily interval, 60-day hold, 15 years
- Writes results to `backtest_stats.json` under key `"daily_60d"`
- Merges with existing file — preserves weekly stats when writing daily and vice versa

### `backtest_optimize.py`
- Exit target sweep: 5%, 10%, 11%, 12%, 13%, 14%, 15%, 20%...50%
- Runs both configs: 1wk/20w and 1d/60d
- Tiers 1, 2, and 3 (Tier 3 = Tier 2 + volume filter: trigger candle volume > 20-bar MA)
- Emails plain-text optimization report
- Runtime: ~40–45 minutes

### `trade_tracker.py`
- Standalone module called by both scanners
- `load_history(path)` — reads trade_history.json
- `save_history(history, path)` — writes trade_history.json
- `add_signals(history, new_signals)` — appends new signal rows
- `resolve_open_trades(history)` — fetches daily price history for each OPEN trade,
  walks bar-by-bar: HIGH ≥ win_target → WIN, LOW ≤ stop → LOSS, hold expired → NEUTRAL
- `build_excel(history, output_path)` — builds trade_tracker.xlsx with two sheets:
  Trade Log (all trades, color-coded by result) and Strategy Stats (backtest reference)

### `trade_history.json`
- Starts as `[]`, grows with each scheduled scan
- Each entry: `scan_date, ticker, strategy, tier, interval, buy_price, stop_loss,
  date_bought, date_sold, exit_price, result`
- `result` starts as `"OPEN"`, updated to `"WIN"/"LOSS"/"NEUTRAL"` by `resolve_open_trades()`
- Only modified on **scheduled** runs, never on workflow_dispatch

### `backtest_stats.json`
- Structure:
```json
{
  "weekly": {
    "ultra": { "total": 48, "wr": 96.0, "lr": 0.0, "aw": 13.0, "al": 0.0,
               "ev": 12.48, "pnl": 62498, "roi": 12.5, "hold_b": 5.1,
               "spm": 0.3, "interval": "1wk", "hold_bars": 20,
               "win_target": 0.13, "position": 10000 },
    "high": { ... }
  },
  "daily_60d": {
    "ultra": { ... },
    "high": { ... }
  }
}
```
- Written by `backtest.py` (weekly key) and `backtest_daily.py` (daily_60d key)
- Read by both scanners to display stats in email
- Must be populated before scanners show real stats — run backtests first

### `debug_ticker.py`
- Set `TICKER = 'SYMBOL'` at top, then run via GitHub Actions debug job
- Walks through every filter step with pass/fail for each:
  price filter, market cap, BB triggers, VixFix triggers, most recent trigger,
  no-break-before, stop distance, VixFix divergence, MACD divergence, Stochastic
- At the bottom, prints **TIER VERDICT** and **TRADE LEVELS**:
  buy price, stop loss, win target, stop dist%, risk/reward ratio,
  position sizing, max loss, max gain

### `debug_volume.py`
- Diagnostic for the volume filter in `backtest_optimize.py` Tier 3
- Tests 8 large-cap tickers to verify volume data quality and filter behavior
- Run via GitHub Actions debug job before running full optimization
- Takes ~30 seconds

### `.github/workflows/scanner.yml`
- **Trigger**: cron `30 2 * * 2-6` (2:30am UTC Tue–Sat = 6:30pm PST Mon–Fri)
  and `workflow_dispatch`
- **weekly-scan job**: runs `scanner_combined.py`; Friday only on schedule,
  always on workflow_dispatch
- **daily-scan job**: runs `scanner_daily.py` every weekday
- Both jobs: pass `GITHUB_EVENT_NAME` env var to scripts; commit `trade_history.json`
  on scheduled runs only (not on workflow_dispatch)
- pip install includes `openpyxl` for Excel tracker

### `.github/workflows/backtests.yml`
- **Trigger**: workflow_dispatch only (never runs automatically)
- **backtest-weekly**: runs `backtest.py` (~30 min), commits `backtest_stats.json`
- **backtest-daily**: runs `backtest_daily.py` (~50 min), commits `backtest_stats.json`
- **optimize-exits**: runs `backtest_optimize.py` (~40–45 min)
- **debug**: runs whichever debug script is set (`debug_ticker.py` or `debug_volume.py`)
- Commit step uses stash/pull-rebase/push pattern with 3-attempt retry to handle
  parallel job push collisions

---

## Workflow Schedules

| Workflow | Trigger | Time (PST) | Runtime |
|----------|---------|-----------|---------|
| Daily scan | Cron Mon–Fri | ~6:30pm | ~35–40 min |
| Weekly scan | Cron Friday only | ~6:30pm | ~35–40 min |
| Weekly backtest | Manual only | — | ~30 min |
| Daily backtest | Manual only | — | ~50 min |
| Exit optimization | Manual only | — | ~40–45 min |
| Debug | Manual only | — | ~1 min |

**Note on GitHub cron reliability**: GitHub Actions cron can be delayed or skipped on
inactive repos. If scans stop firing, push any commit to wake up the scheduler, or use
cron-job.org to trigger via the GitHub API instead.

**Note on timing**: Cron is set to UTC. During PDT (Mar–Nov) scans fire at 7:30pm local;
during PST (Nov–Mar) at 6:30pm local. One hour off in summer — acceptable.

---

## Setup Checklist (first-time or after repo reset)

1. **Repo must be public** — private repos exhaust free GitHub Actions minutes quickly
   (daily+weekly scans alone use ~1,040 min/month, exceeding the 2,000 free minutes)
2. **GitHub Secrets** — add `GMAIL_USER` and `GMAIL_PASSWORD` in repo Settings → Secrets
3. **Actions permissions** — Settings → Actions → General → "Read and write permissions"
4. **Commit initial JSON files** — `backtest_stats.json` (with null values) and
   `trade_history.json` (as `[]`) must exist in repo root before first scan runs
5. **Run backtests first** — trigger **Backtests** workflow → both backtest jobs
   → wait ~80 min total → verify `backtest_stats.json` has real values in repo
6. **Run scanners manually once** — trigger **Stock Scanners** workflow_dispatch
   → confirm email arrives with populated stats and Excel attachment
7. **Scheduled scans then run automatically** at 6:30pm PST Mon–Fri going forward

---

## Known Issues & Decisions

### `no_break_after` removed (IMPORTANT)
The `no_break_after` filter was permanently removed from all 5 files
(`backtest.py`, `backtest_daily.py`, `backtest_optimize.py`, `scanner_combined.py`,
`scanner_daily.py`). It was disqualifying signals retroactively when the stop loss was
later violated — causing losses to silently disappear from backtests instead of being
counted. This inflated historical win rates. **Do not re-add this filter.**

### Volume filter (Tier 3 in optimizer) doesn't discriminate
VixFix signals naturally occur on high-volume days (fear spikes = elevated volume), so
requiring volume > 20-bar MA passes ~100% of signals. The filter is technically correct
but not useful for this strategy. Tier 3 ≈ Tier 2 as a result. Future work: test
earnings exclusion window or fundamental quality filters instead.

### GitHub Actions cron unreliable
Scheduled workflows can be delayed by up to an hour or skipped entirely on inactive repos.
Fix: push any commit to wake up the scheduler. Long-term fix: use cron-job.org to trigger
via GitHub API (`workflow_dispatch`), which is never skipped.

### Both backtest jobs run in parallel and can collide on push
Fixed with stash/pull-rebase/push pattern + 3-attempt retry loop in `backtests.yml`.
The two JSON keys (`"weekly"` and `"daily_60d"`) are separate, so merge conflicts
are resolved cleanly by rebase.

### Scanners must read backtest_stats.json written by backtests
Backtests and scans are in separate workflows (`backtests.yml` vs `scanner.yml`).
If you run both workflows via workflow_dispatch at the same time, the scanner may
start before the backtest finishes writing the JSON. Always let backtests complete
and verify `backtest_stats.json` in the repo before running scanners manually.

### Weekly scan timing
The weekly candle only finalizes after Friday 4pm ET market close. Manual runs on
Mon–Thu show incomplete weekly candles — signals may appear/disappear before Friday
close. **Only act on signals from the Friday evening scheduled scan.**

### Entry price in paper tracker
Entry price = trigger candle close (fetched from yfinance after scan). This matches
the backtest logic exactly. In practice you'd enter at the next session's open, which
may differ slightly from the trigger close.

---

## Pending Work

- [ ] Test earnings exclusion window as additional filter (avoid signals within 2 weeks
  of earnings date — doesn't conflict with mean-reversion strategy logic)
- [ ] Test fundamental quality filter (positive FCF or earnings growth) to distinguish
  genuine bottoms from deteriorating businesses
- [ ] Consider cron-job.org setup for reliable scheduling if GitHub cron continues
  to be unreliable
- [ ] Re-run both backtests after `no_break_after` removal to get fresh honest win rates
  in `backtest_stats.json` (current stats in file may reflect the old inflated numbers)
- [ ] Re-run optimization report after backtest fix for updated exit target recommendations

---

## Email Output

Both scanners send an HTML email to `bkcolby@yahoo.com` with:
- Signal count summary table
- Tier 1 block: stats table (signals, win rate, loss rate, avg win/loss, EV, avg hold,
  P&L, ROI) pulled from `backtest_stats.json` + red ticker symbols
- Tier 2 block: same format, tickers de-duplicated (Tier 1 tickers excluded)
- Subject line: `"Daily Stock Scan (1d) — Mar 28 2026 — 3 signals found"`
- Attachment: `trade_tracker.xlsx` (full paper trade history)

Backtests email a plain-text report to the same address with full trade history.

---

## Strategy Context

This is a **mean-reversion / bottom-picking strategy** targeting stocks near temporary
lows. The VixFix measures synthetic fear — when fear is extreme AND price makes a lower
low while fear makes a higher reading (divergence), it signals potential exhaustion of
selling pressure. The Bollinger Band touch confirms the price is statistically extended
to the downside. MACD and Stochastic divergence provide additional momentum confirmation.

Because this strategy targets bottoms, filters designed around momentum or trend-following
(e.g. SPY above 200-day MA, relative strength vs SPY) are **counterproductive** — they
would filter out exactly the conditions this strategy is designed to exploit.

Position sizing is fixed at $10,000/trade. The strategy is designed for paper trading
simulation first — run it for 6–12 months to validate live performance against backtested
win rates before committing real capital.
