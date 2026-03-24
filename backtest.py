import requests
import pandas as pd
import numpy as np
import yfinance as yf
import smtplib
import time
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ══════════════════════════════════════════════════════════════════
# STRATEGY PARAMETERS
# ══════════════════════════════════════════════════════════════════
# Bollinger Bands: 20-period, 2.0 std dev, Simple
# VixFix: pd=30, bbl=20, mult=2.0, lb=75, ph=0.85 (CMWilliams)
# MACD: 12/26/9 Exponential — divergence required (tested with/without)
# Stochastic Fast: 14-period — divergence required (tested with/without)
# Entry: Close of second trigger candle (Strategy A)
#        OR pullback to second trigger close (Strategy B)
# Stop: Low of second trigger candle
# Win target: 13%
# Hold: 20 weeks max
# Filters: Price >$10, Market cap >$1B, Stop distance <11%
#          No candle in prior 10 bars below EITHER trigger candle low
#          No candle after trigger broke below trigger low (post-break check)
# Universe: NYSE + NASDAQ
# History: 15 years of weekly data
# ══════════════════════════════════════════════════════════════════

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

HOLD_PERIODS   = [15, 20, 30]
WIN_TARGETS    = [0.11, 0.12, 0.13]
POSITION_SIZE  = 5000.0
YEARS_HISTORY  = 15

MIN_PRICE      = 10.0
MIN_MARKET_CAP = 1_000_000_000
MAX_STOP_DIST  = 0.11
NO_BREAK_BARS  = 10

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
    """No candle in prior n_bars has a lower low than the candle at idx"""
    trigger_low = low_values[idx]
    for j in range(max(0, idx - n_bars), idx):
        if low_values[j] < trigger_low:
            return False
    return True

def no_break_after(low_values, idx, end_idx):
    """No candle after idx up to end_idx has a lower low than candle at idx"""
    trigger_low = low_values[idx]
    for j in range(idx + 1, end_idx + 1):
        if low_values[j] < trigger_low:
            return False
    return True

def macd_div(prior_idx, recent_idx, macd_line, signal_line, histogram):
    vals = [histogram[prior_idx], histogram[recent_idx],
            macd_line[prior_idx], macd_line[recent_idx],
            signal_line[prior_idx], signal_line[recent_idx]]
    if any(np.isnan(v) for v in vals):
        return False
    type_a = histogram[recent_idx] > histogram[prior_idx]
    type_b = (macd_line[recent_idx] > macd_line[prior_idx]) or \
             (signal_line[recent_idx] > signal_line[prior_idx])
    return type_a or type_b

def find_vixfix_trigger_pairs(df):
    """
    Find all valid trigger pairs (prior_idx, recent_idx) where:
    - Both are confirmed trigger candles with VixFix spike
    - noBreak applied to BOTH trigger candles (prior 10 bars)
    - noBreak after recent trigger (nothing broke below since)
    - Stop distance < 11%
    - Price lower low + VixFix higher high between the two
    Returns list of dicts with signal info + macd_divergence flag
    """
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

    # Compute MACD for divergence tagging
    ml, sl, hist = compute_macd(close)

    pairs = []

    for recent_idx in range(n):
        if not twvf[recent_idx]:
            continue

        recent_low   = low_v[recent_idx]
        recent_close = close_v[recent_idx]
        recent_wvf   = wvf_v[recent_idx]

        if np.isnan(recent_low) or np.isnan(recent_wvf):
            continue

        # noBreak before recent trigger
        if not no_break_before(low_v, recent_idx, NO_BREAK_BARS):
            continue

        # Stop distance filter
        if (recent_close - recent_low) / recent_close > MAX_STOP_DIST:
            continue

        # noBreak after recent trigger (nothing broke below since signal)
        if not no_break_after(low_v, recent_idx, n - 1):
            continue

        for j in range(recent_idx - 1, max(recent_idx - MAX_GAP, 0) - 1, -1):
            if not twvf[j]:
                continue
            prior_low = low_v[j]
            prior_wvf = wvf_v[j]
            if np.isnan(prior_low) or np.isnan(prior_wvf):
                continue

            # noBreak before FIRST (prior) trigger candle too
            if not no_break_before(low_v, j, NO_BREAK_BARS):
                continue

            if recent_low < prior_low and recent_wvf > prior_wvf:
                has_macd = macd_div(j, recent_idx, ml, sl, hist)
                pairs.append({
                    'signal_idx':  recent_idx,
                    'prior_idx':   j,
                    'signal_date': df.index[recent_idx],
                    'entry_price': float(recent_close),
                    'stop_loss':   float(recent_low),
                    'has_macd':    has_macd,
                })
                break

    return pairs

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

def eval_trade(df, signal, hold_weeks, win_target_pct, strategy='A'):
    idx        = signal['signal_idx']
    entry      = signal['entry_price']
    stop_loss  = signal['stop_loss']
    n          = len(df)

    if strategy == 'B':
        # Wait for pullback to trigger close price
        entry_idx = None
        for w in range(1, hold_weeks + 1):
            fi = idx + w
            if fi >= n:
                break
            if float(df['Low'].iloc[fi]) <= entry:
                entry_idx = fi
                break
        if entry_idx is None:
            return None
        start_bar = entry_idx
        bars_left = hold_weeks - (entry_idx - idx)
    else:
        start_bar = idx
        bars_left = hold_weeks

    win_target = entry * (1 + win_target_pct)
    shares     = POSITION_SIZE / entry
    result     = 'NEUTRAL'
    exit_price = None
    exit_week  = None

    for w in range(1, bars_left + 1):
        fi = start_bar + w
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
        last = min(start_bar + bars_left, n - 1)
        exit_price = float(df['Close'].iloc[last])
        exit_week  = min(bars_left, n - 1 - start_bar)

    pct    = (exit_price - entry) / entry * 100
    dollar = shares * (exit_price - entry)

    return {
        'result':        result,
        'entry':         entry,
        'stop_loss':     stop_loss,
        'exit_price':    exit_price,
        'exit_week':     exit_week,
        'pct_return':    pct if not np.isnan(pct) else 0.0,
        'dollar_return': dollar if not np.isnan(dollar) else 0.0,
        'ticker':        signal.get('ticker', ''),
        'date':          str(signal['signal_date'].date()),
        'has_macd':      signal['has_macd'],
        'has_stoch':     signal.get('has_stoch', False),
    }

def run_backtest(tickers):
    """
    For each signal, tag has_macd and has_stoch.
    Collect all trades then filter into 4 combos:
    - macd+stoch, macd_only, stoch_only, neither
    Each combo tested for Strategy A and B, all hold periods, all win targets.
    """
    all_signals = []
    cutoff  = pd.Timestamp.now() - pd.DateOffset(years=YEARS_HISTORY)
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

            pairs = find_vixfix_trigger_pairs(df)
            if not pairs:
                continue

            stoch_active = find_stoch_signal_bars(df)

            for signal in pairs:
                sidx = signal['signal_idx']

                # Price sanity check
                entry = signal['entry_price']
                if entry < recent_close * 0.15 or entry > recent_close * 6.0:
                    continue
                if sidx + 1 >= len(df):
                    continue

                has_stoch = any(
                    (sidx + offset) in stoch_active
                    for offset in range(-SCAN_DELAY, SCAN_DELAY + 1)
                )
                signal['has_stoch'] = has_stoch
                signal['ticker']    = ticker
                all_signals.append((df.copy(), signal))

                print(f'  {ticker} {signal["signal_date"].date()} '
                      f'MACD:{"Y" if signal["has_macd"] else "N"} '
                      f'STOCH:{"Y" if has_stoch else "N"}')

        except Exception as e:
            print(f'  Error on {ticker}: {e}')

        if (i + 1) % 100 == 0:
            print(f'  [{i + 1}/{len(tickers)}] — {len(all_signals)} signals so far...')

        time.sleep(0.05)

    print(f'\nTotal raw signals: {len(all_signals)}')

    # Build results dict keyed by (combo, strategy, hold, target)
    # combo: 'both', 'macd_only', 'stoch_only', 'neither'
    results = {}
    combos = ['both', 'macd_only', 'stoch_only', 'neither']
    for combo in combos:
        for strategy in ['A', 'B']:
            for hold in HOLD_PERIODS:
                for target in WIN_TARGETS:
                    results[(combo, strategy, hold, target)] = []

    for df, signal in all_signals:
        has_macd  = signal['has_macd']
        has_stoch = signal['has_stoch']

        if has_macd and has_stoch:
            combo = 'both'
        elif has_macd and not has_stoch:
            combo = 'macd_only'
        elif not has_macd and has_stoch:
            combo = 'stoch_only'
        else:
            combo = 'neither'

        for strategy in ['A', 'B']:
            for hold in HOLD_PERIODS:
                for target in WIN_TARGETS:
                    trade = eval_trade(df, signal, hold, target, strategy)
                    if trade is not None:
                        results[(combo, strategy, hold, target)].append(trade)

    return results, all_signals

def bucket_stats(trades):
    if not trades:
        return None
    wins    = [t for t in trades if t['result'] == 'WIN']
    losses  = [t for t in trades if t['result'] == 'LOSS']
    neutral = [t for t in trades if t['result'] == 'NEUTRAL']
    total   = len(trades)
    wr      = len(wins) / total * 100
    lr      = len(losses) / total * 100
    aw      = safe_mean([t['pct_return'] for t in wins])
    al      = safe_mean([t['pct_return'] for t in losses])
    ev      = (wr/100 * aw) + (lr/100 * al)
    pnl     = safe_sum([t['dollar_return'] for t in trades])
    capital = total * POSITION_SIZE
    roi     = (pnl / capital * 100) if capital > 0 else 0.0
    hold_w  = safe_mean([t['exit_week'] for t in trades if t['exit_week'] is not None])
    return dict(total=total, wins=len(wins), losses=len(losses), neutral=len(neutral),
                wr=wr, lr=lr, aw=aw, al=al, ev=ev, pnl=pnl, roi=roi, hold_w=hold_w)

def fmt_bucket(label, s):
    if s is None:
        return f'{label}: no signals\n'
    return (f'{label}\n'
            f'  Signals:{s["total"]:>5}  Wins:{s["wins"]:>4} ({s["wr"]:.1f}%)  '
            f'Losses:{s["losses"]:>4} ({s["lr"]:.1f}%)  Neutral:{s["neutral"]:>4}\n'
            f'  Avg win:{s["aw"]:>+7.1f}%  Avg loss:{s["al"]:>+7.1f}%  '
            f'EV:{s["ev"]:>+6.2f}%  Avg hold:{s["hold_w"]:.1f}w\n'
            f'  Total P&L: ${s["pnl"]:>+12,.2f}  ROI:{s["roi"]:>+7.1f}%\n')

def build_report(results, all_signals):
    sep  = '=' * 68
    sep2 = '-' * 68

    lines = [
        sep,
        'BACKTEST REPORT',
        sep,
        'STRATEGY PARAMETERS:',
        '  Entry:       Strategy A = close of 2nd trigger candle',
        '               Strategy B = pullback to 2nd trigger close price',
        '  Stop loss:   Low of 2nd trigger candle',
        '  Win target:  11%, 12%, 13% tested',
        '  Hold period: 15, 20, 30 weeks tested',
        '  BB:          20-period, 2.0 std dev, Simple',
        '  VixFix:      CMWilliams pd=30 bbl=20 mult=2.0 lb=75 ph=0.85',
        '  MACD:        12/26/9 Exponential (divergence: histogram OR line/signal)',
        '  Stochastic:  Fast 14-period divergence',
        '  Filters:     Price >$10 | Market cap >$1B | Stop distance <11%',
        '               No-break before BOTH trigger candles (10 bars each)',
        '               No candle after trigger broke below trigger low',
        '  Universe:    NYSE + NASDAQ',
        '  History:     15 years weekly data',
        f'  ExpVal formula: (Win% x Avg Win%) + (Loss% x Avg Loss%)',
        sep,
        '',
    ]

    # Primary: Strategy A, 20w, 13%, both indicators
    primary = results[('both', 'A', 20, 0.13)]
    lines.append('── PRIMARY STRATEGY (A, 20w, 13%, MACD+Stochastic) ─────────────')
    lines.append(fmt_bucket('  Both MACD + Stochastic', bucket_stats(primary)))

    # Strategy A vs B (20w, 13%, both)
    lines.append(sep2)
    lines.append('── STRATEGY A vs B (20w, 13%, MACD+Stochastic) ─────────────────')
    lines.append(fmt_bucket('  Strategy A (buy at trigger close)', bucket_stats(results[('both', 'A', 20, 0.13)])))
    lines.append(fmt_bucket('  Strategy B (buy on pullback)',      bucket_stats(results[('both', 'B', 20, 0.13)])))

    # MACD with/without (Strategy A, 20w, 13%)
    lines.append(sep2)
    lines.append('── MACD DIVERGENCE IMPACT — Strategy A, 20w, 13% ───────────────')
    with_macd    = results[('both', 'A', 20, 0.13)] + results[('macd_only', 'A', 20, 0.13)]
    without_macd = results[('stoch_only', 'A', 20, 0.13)] + results[('neither', 'A', 20, 0.13)]
    lines.append(fmt_bucket('  With MACD divergence',    bucket_stats(with_macd)))
    lines.append(fmt_bucket('  Without MACD divergence', bucket_stats(without_macd)))

    # Stochastic with/without (Strategy A, 20w, 13%)
    lines.append(sep2)
    lines.append('── STOCHASTIC DIVERGENCE IMPACT — Strategy A, 20w, 13% ─────────')
    with_stoch    = results[('both', 'A', 20, 0.13)] + results[('stoch_only', 'A', 20, 0.13)]
    without_stoch = results[('macd_only', 'A', 20, 0.13)] + results[('neither', 'A', 20, 0.13)]
    lines.append(fmt_bucket('  With Stochastic divergence',    bucket_stats(with_stoch)))
    lines.append(fmt_bucket('  Without Stochastic divergence', bucket_stats(without_stoch)))

    # All 4 combos
    lines.append(sep2)
    lines.append('── ALL INDICATOR COMBINATIONS — Strategy A, 20w, 13% ───────────')
    combo_labels = {
        'both':      'MACD + Stochastic (both)',
        'macd_only': 'MACD only (no Stochastic)',
        'stoch_only':'Stochastic only (no MACD)',
        'neither':   'Neither MACD nor Stochastic',
    }
    for combo, label in combo_labels.items():
        lines.append(fmt_bucket(f'  {label}', bucket_stats(results[(combo, 'A', 20, 0.13)])))

    # Hold period comparison (Strategy A, 13%, both)
    lines.append(sep2)
    lines.append('── HOLD PERIOD COMPARISON — Strategy A, 13%, MACD+Stochastic ───')
    for hold in HOLD_PERIODS:
        lines.append(fmt_bucket(f'  {hold}-week hold', bucket_stats(results[('both', 'A', hold, 0.13)])))

    # Win target comparison (Strategy A, 20w, both)
    lines.append(sep2)
    lines.append('── WIN TARGET COMPARISON — Strategy A, 20w, MACD+Stochastic ────')
    for target in WIN_TARGETS:
        lines.append(fmt_bucket(f'  {int(target*100)}% target', bucket_stats(results[('both', 'A', 20, target)])))

    # Full trade history
    lines.append(sep2)
    lines.append('── FULL TRADE HISTORY — Strategy A, 20w, 13%, MACD+Stochastic ──')
    hist = results[('both', 'A', 20, 0.13)]
    if hist:
        lines.append(
            f'{"Ticker":<6} {"Date":<12} {"Result":<8} {"Ret%":>7} '
            f'{"$Return":>10} {"Wk":>4} {"Entry":>8} {"Stop":>8}'
        )
        lines.append('-' * 68)
        for t in sorted(hist, key=lambda x: x['date']):
            lines.append(
                f'{t["ticker"]:<6} {t["date"]:<12} {t["result"]:<8} '
                f'{t["pct_return"]:>+6.1f}% {t["dollar_return"]:>+10,.2f} '
                f'{str(t["exit_week"])+"w":>4}  ${t["entry"]:>7.2f}  ${t["stop_loss"]:>7.2f}'
            )

    lines.append(sep)
    return '\n'.join(lines)

def build_scanner_alert(results):
    """Build high-confidence alert section for scanner email"""
    s = bucket_stats(results[('both', 'A', 20, 0.13)])
    if s is None:
        return ''
    return (
        f'\n★ HIGH CONFIDENCE STRATEGY PARAMETERS ★\n'
        f'Strategy A | 20w hold | 13% target | MACD + Stochastic\n'
        f'Historical win rate: {s["wr"]:.1f}% over {s["total"]} signals ({YEARS_HISTORY} years)\n'
        f'Any ticker appearing in both scans below matches this strategy.\n'
    )

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
    results, all_signals = run_backtest(tickers)
    report = build_report(results, all_signals)
    print('\n' + report)
    if GMAIL_USER:
        send_email(report)
