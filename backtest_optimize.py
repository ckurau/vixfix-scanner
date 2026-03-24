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
SCAN_DELAY     = 5
VF_NEAR        = 2
STOCH_LOOKBACK = 25
STOCH_K        = 14
LOOKBACK       = 10
YEAR_HIGH_BARS = 52
MACD_FAST      = 12
MACD_SLOW      = 26
MACD_SIGNAL    = 9

# ── Backtest settings ──────────────────────────────────────────────────────────
HOLD_WEEKS     = 20
POSITION_SIZE  = 5000.0
EXIT_TARGETS   = [i / 100 for i in range(5, 55, 5)]

# ── Filters ────────────────────────────────────────────────────────────────────
MIN_PRICE      = 10.0
MIN_MARKET_CAP = 1_000_000_000
MAX_STOP_DIST  = 0.11
NO_BREAK_BARS  = 10

# ── Email ──────────────────────────────────────────────────────────────────────
GMAIL_USER     = os.environ.get('GMAIL_USER', '')
GMAIL_PASSWORD = os.environ.get('GMAIL_PASSWORD', '')
TO_EMAIL       = 'bkcolby@yahoo.com'

def safe_mean(values):
    clean = [v for v in values if v is not None and not np.isnan(v)]
    return np.mean(clean) if clean else 0.0

def safe_sum(values):
    clean = [v for v in values if v is not None and not np.isnan(v)]
    return sum(clean) if clean else 0.0

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

def compute_macd(close):
    ema_fast  = close.ewm(span=MACD_FAST,   adjust=False).mean()
    ema_slow  = close.ewm(span=MACD_SLOW,   adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal    = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    histogram = macd_line - signal
    return macd_line.values, signal.values, histogram.values

def no_break_before(low_values, idx, n_bars):
    trigger_low = low_values[idx]
    for j in range(max(0, idx - n_bars), idx):
        if low_values[j] < trigger_low:
            return False
    return True

def no_break_after(low_values, idx, end_idx):
    trigger_low = low_values[idx]
    for j in range(idx + 1, end_idx + 1):
        if low_values[j] < trigger_low:
            return False
    return True

def macd_divergence(prior_idx, recent_idx, macd_line, signal_line, histogram):
    vals = [histogram[prior_idx], histogram[recent_idx],
            macd_line[prior_idx], macd_line[recent_idx],
            signal_line[prior_idx], signal_line[recent_idx]]
    if any(np.isnan(v) for v in vals):
        return False
    type_a = histogram[recent_idx] > histogram[prior_idx]
    type_b = (macd_line[recent_idx] > macd_line[prior_idx]) or \
             (signal_line[recent_idx] > signal_line[prior_idx])
    return type_a or type_b

def compute_vixfix_signals(df, macd_line, signal_line, histogram):
    close   = df['Close']
    low     = df['Low']
    open_   = df['Open']
    close_v = close.values
    low_v   = low.values
    n       = len(df)

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

    twvf  = trigger_with_vf.values
    wvf_v = wvf_at_trigger.values
    signals = []

    for recent_idx in range(n):
        if not twvf[recent_idx]:
            continue
        recent_low   = low_v[recent_idx]
        recent_close = close_v[recent_idx]
        recent_wvf   = wvf_v[recent_idx]
        if np.isnan(recent_low) or np.isnan(recent_wvf):
            continue
        if not no_break_before(low_v, recent_idx, NO_BREAK_BARS):
            continue
        if (recent_close - recent_low) / recent_close > MAX_STOP_DIST:
            continue
        if not no_break_after(low_v, recent_idx, n - 1):
            continue

        for j in range(recent_idx - 1, max(recent_idx - MAX_GAP, 0) - 1, -1):
            if not twvf[j]:
                continue
            prior_low = low_v[j]
            prior_wvf = wvf_v[j]
            if np.isnan(prior_low) or np.isnan(prior_wvf):
                continue
            if recent_low < prior_low and recent_wvf > prior_wvf:
                if not macd_divergence(j, recent_idx, macd_line, signal_line, histogram):
                    break
                signals.append({
                    'signal_idx':  recent_idx,
                    'signal_date': df.index[recent_idx],
                    'entry_price': float(recent_close),
                    'stop_loss':   float(recent_low)
                })
                break

    return signals

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

def collect_signal_data(df, signal):
    idx   = signal['signal_idx']
    n     = len(df)
    weeks = []
    for w in range(1, HOLD_WEEKS + 1):
        fi = idx + w
        if fi >= n:
            break
        weeks.append({
            'week':  w,
            'high':  float(df['High'].iloc[fi]),
            'low':   float(df['Low'].iloc[fi]),
            'close': float(df['Close'].iloc[fi])
        })
    return weeks

def evaluate_signal(signal, weekly_data, win_target_pct):
    entry      = signal['entry_price']
    stop_loss  = signal['stop_loss']
    win_target = entry * (1 + win_target_pct)
    shares     = POSITION_SIZE / entry

    result     = 'NEUTRAL'
    exit_price = None
    exit_week  = None

    for week in weekly_data:
        if week['high'] >= win_target:
            result, exit_price, exit_week = 'WIN', win_target, week['week']
            break
        if week['low'] <= stop_loss:
            result, exit_price, exit_week = 'LOSS', stop_loss, week['week']
            break

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
        'pct_return':    pct_return if not np.isnan(pct_return) else 0.0,
        'dollar_return': dollar_return if not np.isnan(dollar_return) else 0.0,
        'exit_week':     exit_week if exit_week is not None else 0,
        'ticker':        signal.get('ticker', ''),
        'date':          str(signal['signal_date'].date())
    }

def run_optimization(tickers):
    signals_data = []
    cutoff = pd.Timestamp.now() - pd.DateOffset(years=15)
    min_bars = max(VF_LB + MAX_GAP, YEAR_HIGH_BARS + LOOKBACK + STOCH_LOOKBACK) + HOLD_WEEKS + 10

    print(f'Collecting signals from {len(tickers)} tickers...')

    for i, ticker in enumerate(tickers):
        try:
            df = yf.download(ticker, period='max', interval='1wk', progress=False, auto_adjust=True)
            if df is None or len(df) < min_bars:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df[df.index >= cutoff].copy()
            if len(df) < min_bars:
                continue

            close_clean = df['Close'].dropna()
            if close_clean.empty:
                continue
            recent_close = float(close_clean.iloc[-1])
            if np.isnan(recent_close) or recent_close < MIN_PRICE:
                continue

            try:
                mc = yf.Ticker(ticker).fast_info.market_cap
                if mc is not None and mc < MIN_MARKET_CAP:
                    continue
            except:
                pass

            macd_line, signal_line, histogram = compute_macd(df['Close'])
            vf_signals = compute_vixfix_signals(df, macd_line, signal_line, histogram)
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

                entry = signal['entry_price']
                if entry < recent_close * 0.15 or entry > recent_close * 6.0:
                    continue

                signal['ticker'] = ticker
                weekly_data = collect_signal_data(df, signal)
                signals_data.append((signal, weekly_data))
                print(f'  ✓ {ticker} {signal["signal_date"].date()}')

        except Exception as e:
            print(f'  Error on {ticker}: {e}')

        if (i + 1) % 100 == 0:
            print(f'  [{i + 1}/{len(tickers)}] — {len(signals_data)} signals...')

        time.sleep(0.05)

    print(f'\nTotal signals: {len(signals_data)}')

    results_by_target = {}
    for target in EXIT_TARGETS:
        trades = [evaluate_signal(sig, wd, target) for sig, wd in signals_data]
        results_by_target[target] = trades

    return results_by_target, signals_data

def summarize_optimization(results_by_target):
    sep  = '=' * 62
    sep2 = '-' * 62
    lines = [
        sep,
        'EXIT TARGET OPTIMIZATION REPORT',
        'BB + VixFix + MACD + Stochastic | $5,000/trade | 20-week hold',
        'ExpVal = (Win% x Avg Win%) + (Loss% x Avg Loss%) | ROI = Total P&L / Total Capital',
        sep,
        f'{"Target":>8}  {"Signals":>8}  {"Win%":>7}  {"TotalP&L":>12}  {"AvgRet":>8}  {"ExpVal":>8}  {"ROI":>7}  {"AvgWks":>7}  {"AvgDays":>8}',
        sep2,
    ]

    summary_rows = []

    for target in EXIT_TARGETS:
        trades = results_by_target[target]
        total  = len(trades)
        if total == 0:
            continue

        wins     = [t for t in trades if t['result'] == 'WIN']
        losses   = [t for t in trades if t['result'] == 'LOSS']
        win_rate = len(wins) / total * 100
        loss_rate= len(losses) / total * 100
        total_pnl= safe_sum([t['dollar_return'] for t in trades])
        avg_ret  = safe_mean([t['pct_return'] for t in trades])
        avg_win  = safe_mean([t['pct_return'] for t in wins])
        avg_loss = safe_mean([t['pct_return'] for t in losses])
        exp_val  = (win_rate/100 * avg_win) + (loss_rate/100 * avg_loss)

        capital    = total * POSITION_SIZE
        roi        = (total_pnl / capital * 100) if capital > 0 else 0.0
        avg_weeks  = safe_mean([t['exit_week'] for t in trades if t['exit_week'] is not None])
        avg_days   = avg_weeks * 7  # weekly candles = 7 days per bar

        summary_rows.append({
            'target':    target,
            'total':     total,
            'win_rate':  win_rate,
            'total_pnl': total_pnl,
            'avg_ret':   avg_ret,
            'exp_val':   exp_val,
            'avg_win':   avg_win,
            'avg_loss':  avg_loss,
            'loss_rate': loss_rate,
            'roi':       roi,
            'avg_weeks': avg_weeks,
            'avg_days':  avg_days,
        })

        lines.append(
            f'{int(target*100):>7}%  {total:>8}  {win_rate:>6.1f}%  '
            f'${total_pnl:>+11,.2f}  {avg_ret:>+7.1f}%  {exp_val:>+7.2f}%  '
            f'{roi:>+6.1f}%  {avg_weeks:>6.1f}w  {avg_days:>7.0f}d'
        )

    lines.append(sep2)

    if summary_rows:
        best_pnl = max(summary_rows, key=lambda x: x['total_pnl'])
        best_wr  = max(summary_rows, key=lambda x: x['win_rate'])
        best_ev  = max(summary_rows, key=lambda x: x['exp_val'])

        lines += [
            '',
            '── Best by each metric ──────────────────────────────────────',
            f'Highest total profit:  {int(best_pnl["target"]*100)}% target  '
            f'(${best_pnl["total_pnl"]:+,.2f}  win rate {best_pnl["win_rate"]:.1f}%)',
            f'Highest win rate:      {int(best_wr["target"]*100)}% target  '
            f'({best_wr["win_rate"]:.1f}%  P&L ${best_wr["total_pnl"]:+,.2f})',
            f'Best expected value:   {int(best_ev["target"]*100)}% target  '
            f'(EV {best_ev["exp_val"]:+.2f}%  win {best_ev["win_rate"]:.1f}%  '
            f'P&L ${best_ev["total_pnl"]:+,.2f}  ROI {best_ev["roi"]:+.1f}%  avg hold {best_ev["avg_weeks"]:.1f}w)',
            '',
            '── Overall recommendation (40% P&L + 35% win rate + 25% EV) ─',
        ]

        max_pnl = max(r['total_pnl'] for r in summary_rows) or 1
        max_wr  = max(r['win_rate']  for r in summary_rows) or 1
        max_ev  = max(r['exp_val']   for r in summary_rows) or 1

        for r in summary_rows:
            pnl_norm = r['total_pnl'] / max_pnl if max_pnl > 0 else 0
            wr_norm  = r['win_rate']  / max_wr
            ev_norm  = r['exp_val']   / max_ev if max_ev > 0 else 0
            r['score'] = pnl_norm * 0.40 + wr_norm * 0.35 + ev_norm * 0.25

        best = max(summary_rows, key=lambda x: x['score'])
        lines.append(
            f'Overall best exit target: {int(best["target"]*100)}%\n'
            f'  P&L: ${best["total_pnl"]:+,.2f}  Win rate: {best["win_rate"]:.1f}%  '
            f'EV: {best["exp_val"]:+.2f}%  ROI: {best["roi"]:+.1f}%  '
            f'Avg hold: {best["avg_weeks"]:.1f}w ({best["avg_days"]:.0f} days)'
        )

    lines.append(sep)
    return '\n'.join(lines)

def send_email(report):
    msg = MIMEMultipart()
    msg['From']    = GMAIL_USER
    msg['To']      = TO_EMAIL
    msg['Subject'] = 'Exit Target Optimization Report'
    msg.attach(MIMEText(report, 'plain'))
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as s:
            s.login(GMAIL_USER, GMAIL_PASSWORD)
            s.sendmail(GMAIL_USER, TO_EMAIL, msg.as_string())
        print(f'Report emailed to {TO_EMAIL}')
    except Exception as e:
        print(f'Email failed: {e}')

if __name__ == '__main__':
    tickers = get_all_tickers()
    results_by_target, signals_data = run_optimization(tickers)
    report = summarize_optimization(results_by_target)
    print('\n' + report)
    if GMAIL_USER:
        send_email(report)
