import requests
import pandas as pd
import numpy as np
import yfinance as yf
import smtplib
import time
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ── Scan settings ──────────────────────────────────────────────────────────────
BB_LENGTH      = 20
BB_MULT        = 2.0
VF_PD          = 30
VF_BBL         = 20
VF_MULT        = 2.0
VF_LB          = 75
VF_PH          = 0.85
MAX_GAP        = 35
MAX_STOP_DIST  = 0.11        # Skip signals where stop is more than 11% below entry
SCAN_DELAY     = 3
VF_NEAR        = 2
STOCH_LOOKBACK = 25
STOCH_K        = 14
LOOKBACK       = 10
YEAR_HIGH_BARS = 52

# ── Backtest settings ──────────────────────────────────────────────────────────
HOLD_WEEKS     = 15
POSITION_SIZE  = 5000.0
EXIT_TARGETS   = [i / 100 for i in range(5, 55, 5)]  # 5% to 50% in 5% steps

# ── Filters ────────────────────────────────────────────────────────────────────
MIN_PRICE      = 5.0
MIN_MARKET_CAP = 1_000_000_000

# ── Email settings ─────────────────────────────────────────────────────────────
GMAIL_USER     = os.environ.get('GMAIL_USER', '')
GMAIL_PASSWORD = os.environ.get('GMAIL_PASSWORD', '')
TO_EMAIL       = 'bkcolby@yahoo.com'

# ── Step 1: Fetch tickers ──────────────────────────────────────────────────────
def get_all_tickers():
    headers = {'User-Agent': 'Mozilla/5.0'}
    tickers = []
    for exchange in ['NYSE', 'NASDAQ']:
        url = f'https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=10000&exchange={exchange}'
        try:
            r = requests.get(url, headers=headers, timeout=15)
            data = r.json()
            rows = data['data']['table']['rows']
            for row in rows:
                sym = row['symbol'].strip()
                if sym.isalpha() and len(sym) <= 4:
                    try:
                        mc = float(str(row.get('marketCap', '0')).replace(',', ''))
                        if mc > 0 and mc < MIN_MARKET_CAP:
                            continue
                    except:
                        pass
                    tickers.append(sym)
        except Exception as e:
            print(f'Error fetching {exchange}: {e}')
    tickers = list(set(tickers))
    print(f'Total tickers fetched: {len(tickers)}')
    return tickers

# ── Step 2: VixFix signals ─────────────────────────────────────────────────────
def compute_vixfix(df):
    close = df['Close']
    low   = df['Low']
    open_ = df['Open']

    bb_mid   = close.rolling(BB_LENGTH).mean()
    bb_std   = close.rolling(BB_LENGTH).std(ddof=0)
    bb_lower = bb_mid - BB_MULT * bb_std
    trigger  = (close > open_) & (low <= bb_lower)

    next_green       = (close.shift(-1) > open_.shift(-1))
    next_open_above  = (open_.shift(-1) >= open_)
    next_close_above = (close.shift(-1) >= open_)
    trigger_confirmed = trigger & next_green & next_open_above & next_close_above

    hc       = close.rolling(VF_PD).max()
    wvf      = (hc - low) / hc * 100
    vf_mid   = wvf.rolling(VF_BBL).mean()
    vf_std   = wvf.rolling(VF_BBL).std(ddof=0)
    vf_upper = vf_mid + VF_MULT * vf_std
    vf_range = wvf.rolling(VF_LB).max() * VF_PH
    is_green = (wvf >= vf_upper) | (wvf >= vf_range)

    vf_near = pd.Series(False, index=df.index)
    for s in range(VF_NEAR + 1):
        vf_near = vf_near | is_green.shift(s).fillna(False).infer_objects(copy=False).astype(bool)
        if s > 0:
            vf_near = vf_near | is_green.shift(-s).fillna(False).infer_objects(copy=False).astype(bool)

    trigger_with_vf = trigger_confirmed & vf_near

    wvf_at_trigger = pd.Series(np.nan, index=df.index)
    for s in range(-VF_NEAR, VF_NEAR + 1):
        shifted = wvf.shift(s).fillna(0)
        wvf_at_trigger = wvf_at_trigger.combine(shifted, lambda a, b: b if np.isnan(a) else max(a, b))

    return trigger_with_vf, wvf_at_trigger, low

def find_vixfix_signals(df):
    trigger_with_vf, wvf_at_trigger, low = compute_vixfix(df)
    twvf  = trigger_with_vf.values
    wvf_v = wvf_at_trigger.values
    low_v = low.values
    n     = len(twvf)
    signals = []

    for recent_idx in range(n):
        if not twvf[recent_idx]:
            continue
        recent_low = low_v[recent_idx]
        recent_wvf = wvf_v[recent_idx]
        if np.isnan(recent_low) or np.isnan(recent_wvf):
            continue
        for j in range(recent_idx - 1, max(recent_idx - MAX_GAP, 0) - 1, -1):
            if not twvf[j]:
                continue
            prior_low = low_v[j]
            prior_wvf = wvf_v[j]
            if np.isnan(prior_low) or np.isnan(prior_wvf):
                continue
            if recent_low < prior_low and recent_wvf > prior_wvf:
                # Skip if stop loss is more than MAX_STOP_DIST below entry
                entry_price = float(df['Close'].iloc[recent_idx])
                stop_price  = float(low_v[recent_idx])
                if (entry_price - stop_price) / entry_price > MAX_STOP_DIST:
                    break
                signals.append({
                    'signal_idx':  recent_idx,
                    'signal_date': df.index[recent_idx],
                    'entry_price': float(df['Close'].iloc[recent_idx]),
                    'stop_loss':   float(low_v[recent_idx])
                })
                break

    return signals

# ── Step 3: Stochastic active bars ────────────────────────────────────────────
def find_stoch_signal_bars(df):
    close = df['Close']
    high  = df['High']
    low   = df['Low']
    open_ = df['Open']

    bb_mid   = close.rolling(BB_LENGTH).mean()
    bb_std   = close.rolling(BB_LENGTH).std(ddof=0)
    bb_lower = bb_mid - BB_MULT * bb_std
    trigger  = (close > open_) & (low <= bb_lower)

    valid_pair = (
        trigger.shift(1).fillna(False) &
        (close > open_) &
        (open_ >= open_.shift(1)) &
        (close >= open_.shift(1))
    )

    lowest_low   = low.rolling(STOCH_K).min()
    highest_high = high.rolling(STOCH_K).max()
    stoch_k      = 100 * (close - lowest_low) / (highest_high - lowest_low)
    price_low    = low < low.shift(1).rolling(STOCH_LOOKBACK).min()
    stoch_high   = stoch_k > stoch_k.shift(1).rolling(STOCH_LOOKBACK).min()
    stoch_div    = price_low & stoch_high

    trigger_low      = low.where(trigger).ffill()
    no_break         = low.rolling(LOOKBACK).min() >= trigger_low
    year_high        = high.rolling(YEAR_HIGH_BARS).max()
    below_high_limit = close <= 0.85 * year_high

    vp  = valid_pair.values
    sd  = stoch_div.values
    nb  = no_break.values
    bhl = below_high_limit.values
    n   = len(vp)
    active = set()

    for i in range(LOOKBACK, n):
        if not nb[i] or not bhl[i]:
            continue
        if not any(vp[max(0, i - LOOKBACK):i]):
            continue
        if not any(sd[max(0, i - LOOKBACK):i]):
            continue
        active.add(i)

    return active

# ── Step 4: Collect raw weekly data per signal ────────────────────────────────
def collect_signal_data(df, signal):
    """
    Collect the weekly high/low/close for HOLD_WEEKS after signal.
    Returns list of dicts — one per future week.
    Used to evaluate multiple exit targets without re-downloading data.
    """
    idx   = signal['signal_idx']
    n     = len(df)
    weeks = []

    for w in range(1, HOLD_WEEKS + 1):
        future_idx = idx + w
        if future_idx >= n:
            break
        weeks.append({
            'week':  w,
            'high':  float(df['High'].iloc[future_idx]),
            'low':   float(df['Low'].iloc[future_idx]),
            'close': float(df['Close'].iloc[future_idx])
        })

    return weeks

# ── Step 5: Evaluate one signal for a given exit target ───────────────────────
def evaluate_signal(signal, weekly_data, win_target_pct):
    entry      = signal['entry_price']
    stop_loss  = signal['stop_loss']
    win_target = entry * (1 + win_target_pct)
    shares     = POSITION_SIZE / entry

    result     = 'NEUTRAL'
    exit_price = None
    exit_week  = None

    for week in weekly_data:
        # Win: intraday high touched target
        if week['high'] >= win_target:
            result     = 'WIN'
            exit_price = win_target
            exit_week  = week['week']
            break
        # Loss: intraday low pierced stop
        if week['low'] <= stop_loss:
            result     = 'LOSS'
            exit_price = stop_loss
            exit_week  = week['week']
            break

    # Neutral: week 10 close or last available
    if result == 'NEUTRAL':
        if weekly_data:
            exit_price = weekly_data[-1]['close']
            exit_week  = weekly_data[-1]['week']
        else:
            exit_price = entry
            exit_week  = 0

    pct_return    = (exit_price - entry) / entry * 100
    dollar_return = shares * (exit_price - entry)

    return {
        'result':        result,
        'pct_return':    pct_return,
        'dollar_return': dollar_return,
        'exit_week':     exit_week,
        'ticker':        signal.get('ticker', ''),
        'date':          str(signal['signal_date'].date())
    }

# ── Step 6: Run optimization ───────────────────────────────────────────────────
def run_optimization(tickers):
    # signals_data: list of (signal, weekly_data) tuples for confirmed signals
    signals_data = []
    min_bars = max(VF_LB + MAX_GAP, YEAR_HIGH_BARS + LOOKBACK + STOCH_LOOKBACK) + HOLD_WEEKS + 10

    print(f'Collecting signals from {len(tickers)} tickers...')

    for i, ticker in enumerate(tickers):
        try:
            df = yf.download(ticker, period='5y', interval='1wk', progress=False, auto_adjust=True)
            if df is None or len(df) < min_bars:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if float(df['Close'].iloc[-1]) < MIN_PRICE:
                continue

            try:
                mc = yf.Ticker(ticker).fast_info.market_cap
                if mc is not None and mc < MIN_MARKET_CAP:
                    continue
            except:
                pass

            vf_signals = find_vixfix_signals(df)
            if not vf_signals:
                continue

            stoch_active = find_stoch_signal_bars(df)
            if not stoch_active:
                continue

            for signal in vf_signals:
                sidx = signal['signal_idx']
                stoch_match = any(
                    (sidx + offset) in stoch_active
                    for offset in range(-SCAN_DELAY, SCAN_DELAY + 1)
                )
                if not stoch_match:
                    continue
                if sidx + 1 >= len(df):
                    continue

                signal['ticker'] = ticker
                weekly_data = collect_signal_data(df, signal)
                signals_data.append((signal, weekly_data))
                print(f'  ✓ Signal: {ticker} {signal["signal_date"].date()}')

        except Exception as e:
            print(f'  Error on {ticker}: {e}')

        if (i + 1) % 100 == 0:
            print(f'  [{i + 1}/{len(tickers)}] — {len(signals_data)} signals collected...')

        time.sleep(0.05)

    print(f'\nTotal confirmed signals: {len(signals_data)}')
    print(f'Testing {len(EXIT_TARGETS)} exit targets: {[f"{int(t*100)}%" for t in EXIT_TARGETS]}')
    print()

    # Now evaluate every signal at every exit target
    results_by_target = {}

    for target in EXIT_TARGETS:
        trades = []
        for signal, weekly_data in signals_data:
            trade = evaluate_signal(signal, weekly_data, target)
            trades.append(trade)
        results_by_target[target] = trades

    return results_by_target, signals_data

# ── Step 7: Summarize optimization results ────────────────────────────────────
def summarize_optimization(results_by_target):
    sep  = '=' * 60
    sep2 = '-' * 60
    lines = [
        sep,
        'EXIT TARGET OPTIMIZATION REPORT',
        'BB + VixFix + Stochastic — Both Scans Must Agree',
        f'Position size: ${POSITION_SIZE:,.0f} per trade | Hold: {HOLD_WEEKS} weeks max',
        sep,
        f'{"Target":>8}  {"Signals":>8}  {"Win%":>7}  {"TotalP&L":>12}  {"AvgRet":>8}  {"RR Ratio":>9}',
        sep2,
    ]

    summary_rows = []

    for target in EXIT_TARGETS:
        trades   = results_by_target[target]
        total    = len(trades)
        if total == 0:
            continue

        wins     = [t for t in trades if t['result'] == 'WIN']
        losses   = [t for t in trades if t['result'] == 'LOSS']
        win_rate = len(wins) / total * 100
        total_pnl= sum(t['dollar_return'] for t in trades)
        avg_ret  = np.mean([t['pct_return'] for t in trades])
        avg_win  = np.mean([t['pct_return'] for t in wins]) if wins else 0
        avg_loss = abs(np.mean([t['pct_return'] for t in losses])) if losses else 0.001
        rr_ratio = avg_win / avg_loss if avg_loss > 0 else 0

        summary_rows.append({
            'target':    target,
            'total':     total,
            'win_rate':  win_rate,
            'total_pnl': total_pnl,
            'avg_ret':   avg_ret,
            'rr_ratio':  rr_ratio,
            'avg_win':   avg_win,
            'avg_loss':  avg_loss
        })

        lines.append(
            f'{int(target*100):>7}%  {total:>8}  {win_rate:>6.1f}%  '
            f'${total_pnl:>+11,.2f}  {avg_ret:>+7.1f}%  {rr_ratio:>8.2f}x'
        )

    lines.append(sep2)

    # Best by each metric
    if summary_rows:
        best_pnl = max(summary_rows, key=lambda x: x['total_pnl'])
        best_wr  = max(summary_rows, key=lambda x: x['win_rate'])
        best_rr  = max(summary_rows, key=lambda x: x['rr_ratio'])

        lines += [
            '',
            '── Best by Each Metric ───────────────────────────────────',
            f'Highest Total Profit:  {int(best_pnl["target"]*100)}% target  '
            f'(${best_pnl["total_pnl"]:+,.2f}  win rate {best_pnl["win_rate"]:.1f}%)',
            f'Highest Win Rate:      {int(best_wr["target"]*100)}% target  '
            f'({best_wr["win_rate"]:.1f}%  total P&L ${best_wr["total_pnl"]:+,.2f})',
            f'Best Risk/Reward:      {int(best_rr["target"]*100)}% target  '
            f'({best_rr["rr_ratio"]:.2f}x  avg win {best_rr["avg_win"]:.1f}%  avg loss {best_rr["avg_loss"]:.1f}%)',
            '',
            '── Recommendation ────────────────────────────────────────',
        ]

        # Score each target combining all three metrics (normalized)
        max_pnl = max(r['total_pnl'] for r in summary_rows) or 1
        max_wr  = max(r['win_rate'] for r in summary_rows) or 1
        max_rr  = max(r['rr_ratio'] for r in summary_rows) or 1

        for r in summary_rows:
            r['score'] = (
                (r['total_pnl'] / max_pnl) * 0.40 +   # 40% weight on profit
                (r['win_rate']  / max_wr)  * 0.35 +   # 35% weight on win rate
                (r['rr_ratio']  / max_rr)  * 0.25     # 25% weight on R/R
            )

        best_overall = max(summary_rows, key=lambda x: x['score'])
        lines.append(
            f'Overall best exit target: {int(best_overall["target"]*100)}%'
        )
        lines.append(
            f'  Total P&L: ${best_overall["total_pnl"]:+,.2f}  |  '
            f'Win rate: {best_overall["win_rate"]:.1f}%  |  '
            f'R/R ratio: {best_overall["rr_ratio"]:.2f}x'
        )
        lines.append(
            '  (Scored 40% profit + 35% win rate + 25% risk/reward)'
        )

    lines.append(sep)
    return '\n'.join(lines)

# ── Step 8: Send email ─────────────────────────────────────────────────────────
def send_email(report):
    msg = MIMEMultipart()
    msg['From']    = GMAIL_USER
    msg['To']      = TO_EMAIL
    msg['Subject'] = 'Stock Scanner — Exit Target Optimization Report'
    msg.attach(MIMEText(report, 'plain'))
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as s:
            s.login(GMAIL_USER, GMAIL_PASSWORD)
            s.sendmail(GMAIL_USER, TO_EMAIL, msg.as_string())
        print(f'Report emailed to {TO_EMAIL}')
    except Exception as e:
        print(f'Email failed: {e}')

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    tickers = get_all_tickers()
    results_by_target, signals_data = run_optimization(tickers)
    report = summarize_optimization(results_by_target)
    print('\n' + report)
    if GMAIL_USER:
        send_email(report)
