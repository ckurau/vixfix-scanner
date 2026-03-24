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
RSI_PERIOD     = 14
RSI_PATH_B_BARS = 3   # bars before second trigger to check for RSI Path B

# ── Backtest settings ──────────────────────────────────────────────────────────
HOLD_PERIODS   = [15, 20, 30]          # weeks — 20w is primary
WIN_TARGETS    = [0.11, 0.12, 0.13]   # 11%, 12%, 13%
POSITION_SIZE  = 5000.0
YEARS_HISTORY  = 15

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

def compute_rsi(close, period=14):
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def no_break_before(low_values, idx, n_bars):
    trigger_low = low_values[idx]
    for j in range(max(0, idx - n_bars), idx):
        if low_values[j] < trigger_low:
            return False
    return True

def no_break_after(low_values, idx, current_idx):
    """Check no candle between trigger and now has low below trigger low"""
    trigger_low = low_values[idx]
    for j in range(idx + 1, current_idx + 1):
        if low_values[j] < trigger_low:
            return False
    return True

def macd_divergence(prior_idx, recent_idx, macd_line, signal_line, histogram):
    hist_prior  = histogram[prior_idx]
    hist_recent = histogram[recent_idx]
    macd_prior  = macd_line[prior_idx]
    macd_recent = macd_line[recent_idx]
    sig_prior   = signal_line[prior_idx]
    sig_recent  = signal_line[recent_idx]
    if any(np.isnan(v) for v in [hist_prior, hist_recent, macd_prior, macd_recent, sig_prior, sig_recent]):
        return False
    type_a = hist_recent > hist_prior
    type_b = (macd_recent > macd_prior) or (sig_recent > sig_prior)
    return type_a or type_b

def rsi_divergence(prior_idx, recent_idx, rsi_values):
    """
    Path A: RSI at second trigger > RSI at first trigger
    Path B: RSI at second trigger > RSI at any of RSI_PATH_B_BARS bars before second trigger
    Returns (has_divergence, path_used)
    """
    rsi_recent = rsi_values[recent_idx]
    rsi_prior  = rsi_values[prior_idx]

    if np.isnan(rsi_recent):
        return False, None

    # Path A: trigger to trigger
    if not np.isnan(rsi_prior) and rsi_recent > rsi_prior:
        return True, 'A'

    # Path B: recent trigger RSI higher than any of N bars before it
    for k in range(1, RSI_PATH_B_BARS + 1):
        j = recent_idx - k
        if j < 0:
            break
        rsi_before = rsi_values[j]
        if not np.isnan(rsi_before) and rsi_recent > rsi_before:
            return True, 'B'

    return False, None

def find_vixfix_signals(df, macd_line, signal_line, histogram, rsi_values):
    close   = df['Close']
    low     = df['Low']
    open_   = df['Open']
    close_v = close.values
    low_v   = low.values
    n_total = len(df)

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

    for recent_idx in range(len(twvf)):
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
                # Check RSI divergence
                has_rsi, rsi_path = rsi_divergence(j, recent_idx, rsi_values)
                signals.append({
                    'signal_idx':   recent_idx,
                    'prior_idx':    j,
                    'signal_date':  df.index[recent_idx],
                    'entry_price':  float(recent_close),
                    'stop_loss':    float(recent_low),
                    'has_rsi_div':  has_rsi,
                    'rsi_path':     rsi_path,
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

def backtest_signal_strategy_a(df, signal, hold_weeks, win_target_pct):
    """Strategy A: buy at close of second trigger candle"""
    idx        = signal['signal_idx']
    entry      = signal['entry_price']
    stop_loss  = signal['stop_loss']
    win_target = entry * (1 + win_target_pct)
    shares     = POSITION_SIZE / entry
    n          = len(df)

    result = 'NEUTRAL'
    exit_price = None
    exit_week  = None

    for w in range(1, hold_weeks + 1):
        fi = idx + w
        if fi >= n:
            break
        wh = float(df['High'].iloc[fi])
        wl = float(df['Low'].iloc[fi])
        if wh >= win_target:
            result, exit_price, exit_week = 'WIN', win_target, w
            break
        if wl <= stop_loss:
            result, exit_price, exit_week = 'LOSS', stop_loss, w
            break

    if result == 'NEUTRAL':
        last = min(idx + hold_weeks, n - 1)
        exit_price = float(df['Close'].iloc[last])
        exit_week  = min(hold_weeks, n - 1 - idx)

    pct    = (exit_price - entry) / entry * 100
    dollar = shares * (exit_price - entry)
    return {'result': result, 'entry': entry, 'stop_loss': stop_loss,
            'exit_price': exit_price, 'exit_week': exit_week,
            'pct_return': pct, 'dollar_return': dollar,
            'ticker': signal['ticker'], 'date': signal['date']}

def backtest_signal_strategy_b(df, signal, hold_weeks, win_target_pct):
    """
    Strategy B: wait for any candle after trigger to drop to or below
    the trigger close price, then enter at that price.
    Stop stays at trigger candle low.
    If price never pulls back within hold_weeks, no trade taken (skip).
    """
    trigger_idx   = signal['signal_idx']
    trigger_close = signal['entry_price']
    stop_loss     = signal['stop_loss']
    n             = len(df)

    # Find entry bar: first bar after trigger where low <= trigger_close
    entry_idx   = None
    entry_price = None

    for w in range(1, hold_weeks + 1):
        fi = trigger_idx + w
        if fi >= n:
            break
        bar_low   = float(df['Low'].iloc[fi])
        bar_close = float(df['Close'].iloc[fi])
        if bar_low <= trigger_close:
            entry_idx   = fi
            entry_price = trigger_close  # enter at trigger close price
            break

    if entry_idx is None:
        return None  # no pullback occurred — no trade

    win_target = entry_price * (1 + win_target_pct)
    shares     = POSITION_SIZE / entry_price

    result = 'NEUTRAL'
    exit_price = None
    exit_week  = None

    bars_remaining = hold_weeks - (entry_idx - trigger_idx)
    for w in range(1, bars_remaining + 1):
        fi = entry_idx + w
        if fi >= n:
            break
        wh = float(df['High'].iloc[fi])
        wl = float(df['Low'].iloc[fi])
        if wh >= win_target:
            result, exit_price, exit_week = 'WIN', win_target, w
            break
        if wl <= stop_loss:
            result, exit_price, exit_week = 'LOSS', stop_loss, w
            break

    if result == 'NEUTRAL':
        last = min(entry_idx + bars_remaining, n - 1)
        exit_price = float(df['Close'].iloc[last])
        exit_week  = min(bars_remaining, n - 1 - entry_idx)

    pct    = (exit_price - entry_price) / entry_price * 100
    dollar = shares * (exit_price - entry_price)
    return {'result': result, 'entry': entry_price, 'stop_loss': stop_loss,
            'exit_price': exit_price, 'exit_week': exit_week,
            'pct_return': pct, 'dollar_return': dollar,
            'ticker': signal['ticker'], 'date': signal['date']}

def run_backtest(tickers):
    """
    Returns dict of results keyed by (strategy, hold_weeks, win_target, rsi_filter)
    rsi_filter: 'all' = no RSI filter, 'rsi_only' = RSI divergence required
    """
    # Initialize results buckets
    results = {}
    for strategy in ['A', 'B']:
        for hold in HOLD_PERIODS:
            for target in WIN_TARGETS:
                for rsi_f in ['all', 'rsi']:
                    results[(strategy, hold, target, rsi_f)] = []

    cutoff = pd.Timestamp.now() - pd.DateOffset(years=YEARS_HISTORY)
    min_bars = max(VF_LB + MAX_GAP, YEAR_HIGH_BARS + LOOKBACK + STOCH_LOOKBACK) + max(HOLD_PERIODS) + 10

    print(f'Backtesting {len(tickers)} tickers ({YEARS_HISTORY} years)...')

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

            recent_close = float(df['Close'].dropna().iloc[-1])
            if np.isnan(recent_close) or recent_close < MIN_PRICE:
                continue

            try:
                mc = yf.Ticker(ticker).fast_info.market_cap
                if mc is not None and mc < MIN_MARKET_CAP:
                    continue
            except:
                pass

            macd_line, signal_line, histogram = compute_macd(df['Close'])
            rsi_values = compute_rsi(df['Close'], RSI_PERIOD)

            vf_signals = find_vixfix_signals(df, macd_line, signal_line, histogram, rsi_values)
            if not vf_signals:
                continue

            stoch_active = find_stoch_signal_bars(df)

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
                signal['date']   = str(signal['signal_date'].date())

                for hold in HOLD_PERIODS:
                    for target in WIN_TARGETS:
                        # Strategy A
                        ta = backtest_signal_strategy_a(df, signal, hold, target)
                        results[('A', hold, target, 'all')].append(ta)
                        if signal['has_rsi_div']:
                            results[('A', hold, target, 'rsi')].append(ta)

                        # Strategy B
                        tb = backtest_signal_strategy_b(df, signal, hold, target)
                        if tb is not None:
                            results[('B', hold, target, 'all')].append(tb)
                            if signal['has_rsi_div']:
                                results[('B', hold, target, 'rsi')].append(tb)

                print(f'  {ticker} {signal["date"]} RSI:{signal["rsi_path"] or "none"}')

        except Exception as e:
            print(f'  Error on {ticker}: {e}')

        if (i + 1) % 100 == 0:
            print(f'  [{i + 1}/{len(tickers)}] scanned...')

        time.sleep(0.05)

    return results

def summarize_bucket(trades, label):
    if not trades:
        return f'{label}: no signals\n'

    wins     = [t for t in trades if t['result'] == 'WIN']
    losses   = [t for t in trades if t['result'] == 'LOSS']
    neutrals = [t for t in trades if t['result'] == 'NEUTRAL']
    total    = len(trades)

    win_rate    = len(wins) / total * 100
    loss_rate   = len(losses) / total * 100
    avg_win_pct = safe_mean([t['pct_return'] for t in wins])
    avg_los_pct = safe_mean([t['pct_return'] for t in losses])
    exp_value   = (win_rate/100 * avg_win_pct) + (loss_rate/100 * avg_los_pct)
    total_pnl   = sum(t['dollar_return'] for t in trades if not np.isnan(t['dollar_return']))
    avg_hold    = safe_mean([t['exit_week'] for t in trades if t['exit_week'] is not None])

    return (f'{label}\n'
            f'  Signals: {total}  Win: {len(wins)} ({win_rate:.1f}%)  '
            f'Loss: {len(losses)} ({loss_rate:.1f}%)  Neutral: {len(neutrals)}\n'
            f'  Avg win: {avg_win_pct:+.1f}%  Avg loss: {avg_los_pct:+.1f}%  '
            f'Expected value: {exp_value:+.2f}%\n'
            f'  Total P&L: ${total_pnl:+,.2f}  Avg hold: {avg_hold:.1f}w\n')

def build_report(results):
    sep  = '=' * 65
    sep2 = '-' * 65
    lines = [sep, 'BACKTEST REPORT — FULL COMPARISON', sep, '']

    # Section 1: Strategy A vs B comparison (13% target, 20w hold)
    lines.append('── STRATEGY A vs B (13% target, 20-week hold) ─────────────')
    lines.append(summarize_bucket(results[('A', 20, 0.13, 'all')], 'Strategy A — buy at trigger close'))
    lines.append(summarize_bucket(results[('B', 20, 0.13, 'all')], 'Strategy B — buy on pullback to trigger close'))

    # Section 2: Hold period comparison (Strategy A, 13% target)
    lines.append(sep2)
    lines.append('── HOLD PERIOD COMPARISON — Strategy A, 13% target ────────')
    for hold in HOLD_PERIODS:
        lines.append(summarize_bucket(results[('A', hold, 0.13, 'all')], f'  {hold}-week hold'))

    # Section 3: Win target comparison (Strategy A, 20w hold)
    lines.append(sep2)
    lines.append('── WIN TARGET COMPARISON — Strategy A, 20-week hold ────────')
    for target in WIN_TARGETS:
        lines.append(summarize_bucket(results[('A', 20, target, 'all')], f'  {int(target*100)}% target'))

    # Section 4: RSI filter impact (Strategy A, 13% target, 20w hold)
    lines.append(sep2)
    lines.append('── RSI DIVERGENCE FILTER IMPACT — Strategy A, 13%, 20w ─────')
    lines.append(summarize_bucket(results[('A', 20, 0.13, 'all')],  '  Without RSI filter (all signals)'))
    lines.append(summarize_bucket(results[('A', 20, 0.13, 'rsi')],  '  With RSI filter (divergence required)'))

    # Section 5: Best overall combinations (by expected value)
    lines.append(sep2)
    lines.append('── TOP 5 COMBINATIONS BY EXPECTED VALUE ───────────────────')
    scored = []
    for key, trades in results.items():
        if not trades:
            continue
        strategy, hold, target, rsi_f = key
        wins   = [t for t in trades if t['result'] == 'WIN']
        losses = [t for t in trades if t['result'] == 'LOSS']
        total  = len(trades)
        if total < 5:
            continue
        wr     = len(wins) / total * 100
        lr     = len(losses) / total * 100
        aw     = safe_mean([t['pct_return'] for t in wins])
        al     = safe_mean([t['pct_return'] for t in losses])
        ev     = (wr/100 * aw) + (lr/100 * al)
        pnl    = sum(t['dollar_return'] for t in trades if not np.isnan(t['dollar_return']))
        scored.append((ev, pnl, strategy, hold, target, rsi_f, total, wr))

    scored.sort(reverse=True)
    for ev, pnl, strategy, hold, target, rsi_f, total, wr in scored[:5]:
        rsi_label = 'RSI required' if rsi_f == 'rsi' else 'no RSI filter'
        lines.append(
            f'  Strategy {strategy} | {hold}w | {int(target*100)}% | {rsi_label}\n'
            f'    EV: {ev:+.2f}%  Win: {wr:.1f}%  P&L: ${pnl:+,.2f}  Signals: {total}\n'
        )

    # Section 6: Full trade history (Strategy A, 20w, 13%)
    lines.append(sep2)
    lines.append('── FULL TRADE HISTORY — Strategy A, 20w, 13% target ────────')
    hist_trades = results[('A', 20, 0.13, 'all')]
    if hist_trades:
        lines.append(
            f'{"Ticker":<6} {"Date":<12} {"Result":<8} {"Ret%":>7} '
            f'{"$Ret":>10} {"Wk":>4} {"Entry":>8} {"Stop":>8} {"RSI":>6}'
        )
        lines.append('-' * 65)
        for t in sorted(hist_trades, key=lambda x: x['date']):
            rsi_tag = t.get('rsi_path', '-') or '-'
            ret_str = f'{t["pct_return"]:+.1f}%' if not np.isnan(t["pct_return"]) else 'nan'
            dret    = f'${t["dollar_return"]:+,.2f}' if not np.isnan(t["dollar_return"]) else 'nan'
            lines.append(
                f'{t["ticker"]:<6} {t["date"]:<12} {t["result"]:<8} '
                f'{ret_str:>7} {dret:>10} '
                f'{str(t["exit_week"])+"w":>4}  ${t["entry"]:>7.2f}  ${t["stop_loss"]:>7.2f}  {rsi_tag:>6}'
            )

    lines.append(sep)
    return '\n'.join(lines)

def send_email(report):
    msg = MIMEMultipart()
    msg['From']    = GMAIL_USER
    msg['To']      = TO_EMAIL
    msg['Subject'] = 'Stock Scanner Backtest Report'
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
    results = run_backtest(tickers)
    report  = build_report(results)
    print('\n' + report)
    if GMAIL_USER:
        send_email(report)
