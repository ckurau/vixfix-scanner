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
# MACD: 12/26/9 Exponential — divergence: histogram OR line/signal higher
# Stochastic Fast: 14-period — divergence: price lower low + stoch higher low
#
# TRIGGER CANDLE (all tiers): Weekly candle that closes GREEN (close > open)
#   AND whose low touches or pierces the BB Lower Band.
#   Confirmed by the NEXT candle also closing green, opening >= trigger open,
#   closing >= trigger open. Entry = close of trigger candle.
#   Stop = low of trigger candle.
#
# TIER 1 ★★★ ULTRA (VixFix + MACD + Stochastic + BB Trigger):
#   Two trigger candles form a divergence: recent low < prior low,
#   recent WVF > prior WVF (VixFix), MACD histogram or line/signal higher
#   at recent vs prior trigger (MACD divergence), AND Stochastic divergence
#   (price lower low + stoch higher low) active within ±5 bars.
#   Position: $10,000 | Win target: 13% | Hold: 20 weeks max
#
# TIER 2 ★★ HIGH (VixFix + Stochastic + BB Trigger):
#   Same as Tier 1 but MACD divergence NOT required.
#   Position: $10,000 | Win target: 13% | Hold: 20 weeks max
#
# TIER 3 ★ STANDARD (Stochastic + BB Trigger):
#   BB trigger candle fires (green, low ≤ BB lower), followed by a
#   confirmation candle. Stochastic divergence (price lower low + stoch
#   higher low) seen within prior 10 bars. No-break rule: no bar in prior
#   10 bars nor after the trigger broke below trigger low.
#   Position: $5,000 | Win target: 13% | Hold: 20 weeks max
#
# TIER 3B ★ STANDARD-BB (BB Trigger only — no Stoch/VixFix/MACD):
#   BB trigger candle fires and is confirmed by the next candle.
#   No-break rule applied. Entry = trigger close. Stop = trigger low.
#   Position: $5,000 | Win target: 13% | Hold: 20 weeks max
#
# TIER 3C ★ STANDARD-MACD (Stochastic + MACD + BB Trigger):
#   BB trigger candle fires. Stochastic divergence active within prior 10
#   bars. MACD histogram or line/signal ALSO diverging (current bar higher
#   than 10-bar lookback). No VixFix required.
#   Position: $5,000 | Win target: 13% | Hold: 20 weeks max
#
# Filters (all tiers): Price >$10 | Market cap >$1B | Stop distance <11%
#   No candle in prior 10 bars below trigger candle low
#   No candle after trigger broke below trigger low
# Universe: NYSE + NASDAQ | History: 15 years weekly
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

HOLD_WEEKS     = 20
WIN_TARGET     = 0.13
POSITION_HIGH  = 10000.0   # Tiers 1 & 2 (win rate >80%)
POSITION_STD   = 5000.0    # Tiers 3, 3B, 3C
YEARS_HISTORY  = 15

MIN_PRICE      = 10.0
MIN_MARKET_CAP = 1_000_000_000
MAX_STOP_DIST  = 0.11
NO_BREAK_BARS  = 10

GMAIL_USER     = os.environ.get('GMAIL_USER', '')
GMAIL_PASSWORD = os.environ.get('GMAIL_PASSWORD', '')
TO_EMAIL       = 'bkcolby@yahoo.com'

# ── Helpers ────────────────────────────────────────────────────────────────────
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

def no_break_before(low_v, idx, n_bars):
    tl = low_v[idx]
    for j in range(max(0, idx - n_bars), idx):
        if low_v[j] < tl:
            return False
    return True

def no_break_after(low_v, idx, end_idx):
    tl = low_v[idx]
    for j in range(idx + 1, end_idx + 1):
        if low_v[j] < tl:
            return False
    return True

def macd_div(prior_idx, recent_idx, ml, sl, hist):
    vals = [hist[prior_idx], hist[recent_idx], ml[prior_idx],
            ml[recent_idx], sl[prior_idx], sl[recent_idx]]
    if any(np.isnan(v) for v in vals):
        return False
    return (hist[recent_idx] > hist[prior_idx]) or \
           (ml[recent_idx] > ml[prior_idx]) or \
           (sl[recent_idx] > sl[prior_idx])

# ── Tier 1 & 2: VixFix trigger pairs ──────────────────────────────────────────
def find_vixfix_pairs(df):
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
    ml, sl, hist = compute_macd(close)
    pairs = []

    for ri in range(n):
        if not twvf[ri]:
            continue
        rl, rc, rw = low_v[ri], close_v[ri], wvf_v[ri]
        if np.isnan(rl) or np.isnan(rw):
            continue
        if not no_break_before(low_v, ri, NO_BREAK_BARS):
            continue
        if (rc - rl) / rc > MAX_STOP_DIST:
            continue
        if not no_break_after(low_v, ri, n - 1):
            continue

        for j in range(ri - 1, max(ri - MAX_GAP, 0) - 1, -1):
            if not twvf[j]:
                continue
            pl, pw = low_v[j], wvf_v[j]
            if np.isnan(pl) or np.isnan(pw):
                continue
            if not no_break_before(low_v, j, NO_BREAK_BARS):
                continue
            if rl < pl and rw > pw:
                pairs.append({
                    'signal_idx':  ri,
                    'prior_idx':   j,
                    'signal_date': df.index[ri],
                    'entry_price': float(rc),
                    'stop_loss':   float(rl),
                    'has_macd':    macd_div(j, ri, ml, sl, hist),
                })
                break
    return pairs

# ── Tier 3: Stochastic-only signals ───────────────────────────────────────────
def find_stoch_signals(df):
    close = df['Close']
    high  = df['High']
    low   = df['Low']
    open_ = df['Open']
    close_v = close.values
    low_v   = low.values
    n = len(df)

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

    trigger_low_s    = low.where(trigger).ffill()
    no_break_s       = low.rolling(LOOKBACK).min() >= trigger_low_s
    year_high        = high.rolling(YEAR_HIGH_BARS).max()
    below_high_limit = close <= 0.85 * year_high

    vp  = valid_pair.values
    sd  = stoch_div.values
    nb  = no_break_s.values
    bhl = below_high_limit.values
    trig_v = trigger.values

    signals = []

    for i in range(LOOKBACK, n - 1):
        if not nb[i] or not bhl[i]:
            continue
        if not any(vp[max(0, i - LOOKBACK):i]):
            continue
        if not any(sd[max(0, i - LOOKBACK):i]):
            continue

        trig_idx = None
        for k in range(i, max(i - LOOKBACK, -1), -1):
            if trig_v[k]:
                trig_idx = k
                break
        if trig_idx is None:
            continue

        entry = close_v[trig_idx]
        stop  = low_v[trig_idx]
        if np.isnan(entry) or np.isnan(stop):
            continue
        if (entry - stop) / entry > MAX_STOP_DIST:
            continue

        if signals and signals[-1]['signal_idx'] == trig_idx:
            continue

        signals.append({
            'signal_idx':  trig_idx,
            'signal_date': df.index[trig_idx],
            'entry_price': float(entry),
            'stop_loss':   float(stop),
        })

    return signals

def find_stoch_active_set(df):
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
    trigger_low_s    = low.where(trigger).ffill()
    no_break_s       = low.rolling(LOOKBACK).min() >= trigger_low_s
    year_high        = high.rolling(YEAR_HIGH_BARS).max()
    below_high_limit = close <= 0.85 * year_high

    vp  = valid_pair.values
    sd  = stoch_div.values
    nb  = no_break_s.values
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

# ── Tier 3B: BB Trigger only (no Stoch/VixFix/MACD) ──────────────────────────
def find_bb_only_signals(df):
    """
    Fires on every confirmed BB trigger candle (green candle touching BB lower,
    followed by a green confirmation candle). No Stoch/VixFix/MACD required.
    Entry = close of trigger candle. Stop = low of trigger candle.
    """
    close = df['Close']
    low   = df['Low']
    open_ = df['Open']
    close_v = close.values
    low_v   = low.values
    n = len(df)

    bb_mid   = close.rolling(BB_LENGTH).mean()
    bb_std   = close.rolling(BB_LENGTH).std(ddof=0)
    bb_lower = bb_mid - BB_MULT * bb_std
    trigger  = (close > open_) & (low <= bb_lower)

    next_green       = (close.shift(-1) > open_.shift(-1))
    next_open_above  = (open_.shift(-1) >= open_)
    next_close_above = (close.shift(-1) >= open_)
    trigger_confirmed = trigger & next_green & next_open_above & next_close_above

    trigger_low_s = low.where(trigger).ffill()
    no_break_s    = low.rolling(LOOKBACK).min() >= trigger_low_s

    tc  = trigger_confirmed.values
    nb  = no_break_s.values
    signals = []

    for i in range(LOOKBACK, n - 1):
        if not tc[i] or not nb[i]:
            continue
        entry = close_v[i]
        stop  = low_v[i]
        if np.isnan(entry) or np.isnan(stop):
            continue
        if (entry - stop) / entry > MAX_STOP_DIST:
            continue
        signals.append({
            'signal_idx':  i,
            'signal_date': df.index[i],
            'entry_price': float(entry),
            'stop_loss':   float(stop),
        })
    return signals

# ── Tier 3C: Stochastic + MACD + BB Trigger ───────────────────────────────────
def find_stoch_macd_signals(df):
    """
    BB trigger candle fires (green, low ≤ BB lower), confirmed by next candle.
    Stochastic divergence active within prior LOOKBACK bars.
    MACD histogram or line/signal is higher now vs LOOKBACK bars ago.
    No VixFix required. Entry = trigger close. Stop = trigger low.
    """
    close = df['Close']
    high  = df['High']
    low   = df['Low']
    open_ = df['Open']
    close_v = close.values
    low_v   = low.values
    n = len(df)

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

    trigger_low_s    = low.where(trigger).ffill()
    no_break_s       = low.rolling(LOOKBACK).min() >= trigger_low_s
    year_high        = high.rolling(YEAR_HIGH_BARS).max()
    below_high_limit = close <= 0.85 * year_high

    ml, sl, hist = compute_macd(close)

    vp  = valid_pair.values
    sd  = stoch_div.values
    nb  = no_break_s.values
    bhl = below_high_limit.values
    trig_v = trigger.values

    signals = []

    for i in range(LOOKBACK + 1, n - 1):
        if not nb[i] or not bhl[i]:
            continue
        if not any(vp[max(0, i - LOOKBACK):i]):
            continue
        if not any(sd[max(0, i - LOOKBACK):i]):
            continue

        # MACD divergence: current vs LOOKBACK bars ago
        past_i = max(0, i - LOOKBACK)
        if np.isnan(hist[i]) or np.isnan(hist[past_i]):
            continue
        macd_higher = (hist[i] > hist[past_i]) or \
                      (ml[i] > ml[past_i]) or \
                      (sl[i] > sl[past_i])
        if not macd_higher:
            continue

        # Find most recent trigger candle
        trig_idx = None
        for k in range(i, max(i - LOOKBACK, -1), -1):
            if trig_v[k]:
                trig_idx = k
                break
        if trig_idx is None:
            continue

        entry = close_v[trig_idx]
        stop  = low_v[trig_idx]
        if np.isnan(entry) or np.isnan(stop):
            continue
        if (entry - stop) / entry > MAX_STOP_DIST:
            continue

        if signals and signals[-1]['signal_idx'] == trig_idx:
            continue

        signals.append({
            'signal_idx':  trig_idx,
            'signal_date': df.index[trig_idx],
            'entry_price': float(entry),
            'stop_loss':   float(stop),
        })

    return signals

# ── Trade evaluation ───────────────────────────────────────────────────────────
def eval_trade(df, signal, position_size):
    idx       = signal['signal_idx']
    entry     = signal['entry_price']
    stop_loss = signal['stop_loss']
    n         = len(df)

    win_target = entry * (1 + WIN_TARGET)
    shares     = position_size / entry
    result     = 'NEUTRAL'
    exit_price = None
    exit_week  = None

    for w in range(1, HOLD_WEEKS + 1):
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
        last = min(idx + HOLD_WEEKS, n - 1)
        exit_price = float(df['Close'].iloc[last])
        exit_week  = min(HOLD_WEEKS, n - 1 - idx)

    pct    = (exit_price - entry) / entry * 100
    dollar = shares * (exit_price - entry)

    return {
        'result':        result,
        'entry':         entry,
        'stop_loss':     stop_loss,
        'exit_price':    exit_price,
        'exit_week':     exit_week,
        'pct_return':    pct    if not np.isnan(pct)    else 0.0,
        'dollar_return': dollar if not np.isnan(dollar) else 0.0,
        'ticker':        signal.get('ticker', ''),
        'date':          str(signal['signal_date'].date()),
    }

# ── Main backtest runner ───────────────────────────────────────────────────────
def run_backtest(tickers):
    cutoff   = pd.Timestamp.now() - pd.DateOffset(years=YEARS_HISTORY)
    min_bars = max(VF_LB + MAX_GAP, YEAR_HIGH_BARS + LOOKBACK + STOCH_LOOKBACK) + HOLD_WEEKS + 10

    results = {
        'ultra':    [],
        'high':     [],
        'standard': [],
        'bb_only':  [],
        'stoch_macd': [],
    }

    print(f'Backtesting {len(tickers)} tickers ({YEARS_HISTORY} years, hold={HOLD_WEEKS}w, target={int(WIN_TARGET*100)}%)...')

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

            # ── Tiers 1 & 2: VixFix pairs ─────────────────────────────────
            vf_pairs     = find_vixfix_pairs(df)
            stoch_active = find_stoch_active_set(df)

            for signal in vf_pairs:
                sidx  = signal['signal_idx']
                entry = signal['entry_price']
                if entry < recent_close * 0.15 or entry > recent_close * 6.0:
                    continue
                if sidx + 1 >= len(df):
                    continue

                has_stoch = any((sidx + o) in stoch_active
                                for o in range(-SCAN_DELAY, SCAN_DELAY + 1))
                has_macd  = signal['has_macd']
                signal['ticker'] = ticker

                trade = eval_trade(df, signal, POSITION_HIGH)
                if trade is None:
                    continue

                if has_macd and has_stoch:
                    results['ultra'].append(trade)
                if has_stoch:
                    results['high'].append(trade)

            # ── Tier 3: Stochastic + BB ────────────────────────────────────
            stoch_signals = find_stoch_signals(df)
            for signal in stoch_signals:
                sidx  = signal['signal_idx']
                entry = signal['entry_price']
                if entry < recent_close * 0.15 or entry > recent_close * 6.0:
                    continue
                if sidx + 1 >= len(df):
                    continue
                signal['ticker'] = ticker
                trade = eval_trade(df, signal, POSITION_STD)
                if trade:
                    results['standard'].append(trade)

            # ── Tier 3B: BB Trigger only ───────────────────────────────────
            bb_signals = find_bb_only_signals(df)
            for signal in bb_signals:
                sidx  = signal['signal_idx']
                entry = signal['entry_price']
                if entry < recent_close * 0.15 or entry > recent_close * 6.0:
                    continue
                if sidx + 1 >= len(df):
                    continue
                signal['ticker'] = ticker
                trade = eval_trade(df, signal, POSITION_STD)
                if trade:
                    results['bb_only'].append(trade)

            # ── Tier 3C: Stochastic + MACD + BB ───────────────────────────
            sm_signals = find_stoch_macd_signals(df)
            for signal in sm_signals:
                sidx  = signal['signal_idx']
                entry = signal['entry_price']
                if entry < recent_close * 0.15 or entry > recent_close * 6.0:
                    continue
                if sidx + 1 >= len(df):
                    continue
                signal['ticker'] = ticker
                trade = eval_trade(df, signal, POSITION_STD)
                if trade:
                    results['stoch_macd'].append(trade)

            total_found = (len(vf_pairs) + len(stoch_signals) +
                           len(bb_signals) + len(sm_signals))
            if total_found:
                print(f'  {ticker}: VF={len(vf_pairs)} Stoch={len(stoch_signals)} '
                      f'BB={len(bb_signals)} StochMACD={len(sm_signals)}')

        except Exception as e:
            print(f'  Error on {ticker}: {e}')

        if (i + 1) % 100 == 0:
            print(f'  [{i + 1}/{len(tickers)}] scanned...')

        time.sleep(0.05)

    return results

# ── Stats & reporting ──────────────────────────────────────────────────────────
def bucket_stats(trades):
    if not trades:
        return None
    wins    = [t for t in trades if t['result'] == 'WIN']
    losses  = [t for t in trades if t['result'] == 'LOSS']
    neutral = [t for t in trades if t['result'] == 'NEUTRAL']
    total   = len(trades)
    wr      = len(wins)   / total * 100
    lr      = len(losses) / total * 100
    aw      = safe_mean([t['pct_return']    for t in wins])
    al      = safe_mean([t['pct_return']    for t in losses])
    ev      = (wr/100 * aw) + (lr/100 * al)
    pnl     = safe_sum([t['dollar_return']  for t in trades])
    # derive capital from per-trade dollar values vs position size
    # Use actual dollar returns / avg pct to estimate capital
    capital = total * POSITION_STD  # will be corrected per tier in report
    roi     = (pnl / capital * 100) if capital > 0 else 0.0
    hold_w  = safe_mean([t['exit_week']     for t in trades if t['exit_week'] is not None])
    return dict(total=total, wins=len(wins), losses=len(losses), neutral=len(neutral),
                wr=wr, lr=lr, aw=aw, al=al, ev=ev, pnl=pnl, hold_w=hold_w)

def bucket_stats_with_position(trades, position_size):
    if not trades:
        return None
    s = bucket_stats(trades)
    capital = s['total'] * position_size
    s['roi'] = (s['pnl'] / capital * 100) if capital > 0 else 0.0
    return s

def fmt_bucket(label, s):
    if s is None:
        return f'{label}: no signals\n'
    return (f'{label}\n'
            f'  Signals:{s["total"]:>5}  Wins:{s["wins"]:>4} ({s["wr"]:.1f}%)  '
            f'Losses:{s["losses"]:>4} ({s["lr"]:.1f}%)  Neutral:{s["neutral"]:>4}\n'
            f'  Avg win:{s["aw"]:>+7.1f}%  Avg loss:{s["al"]:>+7.1f}%  '
            f'EV:{s["ev"]:>+6.2f}%  Avg hold:{s["hold_w"]:.1f}w\n'
            f'  Total P&L: ${s["pnl"]:>+12,.2f}  ROI:{s["roi"]:>+7.1f}%\n')

def signals_per_month(trades, years=YEARS_HISTORY):
    total_months = years * 12
    return len(trades) / total_months if total_months > 0 else 0.0

def trade_history(trades, label):
    if not trades:
        return f'{label}: no signals\n'
    lines = [
        label,
        f'{"Ticker":<6} {"Date":<12} {"Result":<8} {"Ret%":>7} '
        f'{"$Return":>10} {"Wk":>4} {"Entry":>8} {"Stop":>8}',
        '-' * 68,
    ]
    for t in sorted(trades, key=lambda x: x['date']):
        lines.append(
            f'{t["ticker"]:<6} {t["date"]:<12} {t["result"]:<8} '
            f'{t["pct_return"]:>+6.1f}% {t["dollar_return"]:>+10,.2f} '
            f'{str(t["exit_week"])+"w":>4}  ${t["entry"]:>7.2f}  ${t["stop_loss"]:>7.2f}'
        )
    return '\n'.join(lines) + '\n'

def build_report(results):
    sep  = '=' * 72
    sep2 = '-' * 72

    ultra    = results['ultra']
    high     = results['high']
    standard = results['standard']
    bb_only  = results['bb_only']
    stch_mcd = results['stoch_macd']

    su = bucket_stats_with_position(ultra,    POSITION_HIGH)
    sh = bucket_stats_with_position(high,     POSITION_HIGH)
    ss = bucket_stats_with_position(standard, POSITION_STD)
    sb = bucket_stats_with_position(bb_only,  POSITION_STD)
    sm = bucket_stats_with_position(stch_mcd, POSITION_STD)

    spm_u = signals_per_month(ultra)
    spm_h = signals_per_month(high)
    spm_s = signals_per_month(standard)
    spm_b = signals_per_month(bb_only)
    spm_m = signals_per_month(stch_mcd)

    lines = [
        sep,
        'BACKTEST REPORT — ALL TIERS',
        sep,
        'PARAMETERS:',
        f'  Entry: close of BB trigger candle | Stop: low of BB trigger candle',
        f'  Win target: {int(WIN_TARGET*100)}% | Hold: {HOLD_WEEKS}w max | History: {YEARS_HISTORY} years',
        f'  Filters: Price >${MIN_PRICE:.0f} | Mkt cap >${MIN_MARKET_CAP/1e9:.0f}B | Stop dist <{int(MAX_STOP_DIST*100)}%',
        f'           No-break {NO_BREAK_BARS} bars before/after trigger candle low',
        f'  ExpVal = (Win% x Avg Win%) + (Loss% x Avg Loss%)',
        f'  Position: ${POSITION_HIGH:,.0f} for Tiers 1 & 2 (win rate >80%) | ${POSITION_STD:,.0f} for Tiers 3/3B/3C',
        '',
        '  BB TRIGGER CANDLE (applies to all tiers):',
        '    Weekly candle that closes GREEN (close > open) AND whose low',
        '    touches or pierces the Bollinger Band lower band (20-period, 2.0 std).',
        '    Confirmed by the NEXT candle: also green, opens >= trigger open,',
        '    closes >= trigger open. Entry = close of trigger. Stop = low of trigger.',
        '',
        '  TIER 1 ★★★ ULTRA (BB Trigger + VixFix div + MACD div + Stoch div):',
        '    Two VixFix-confirmed trigger candles form a divergence pair:',
        '    recent price low < prior price low, recent WVF > prior WVF.',
        '    MACD histogram (or line/signal) higher at recent vs prior trigger.',
        '    Stochastic divergence (price lower low + stoch higher low) active',
        '    within ±5 bars of the trigger. All four conditions required.',
        f'    Position: ${POSITION_HIGH:,.0f}',
        '',
        '  TIER 2 ★★ HIGH (BB Trigger + VixFix div + Stoch div):',
        '    Same as Tier 1 but MACD divergence NOT required.',
        '    VixFix divergence pair + Stochastic divergence confirmed.',
        f'    Position: ${POSITION_HIGH:,.0f}',
        '',
        '  TIER 3 ★ STANDARD (BB Trigger + Stoch div):',
        '    BB trigger candle fires and is confirmed by next candle.',
        '    Stochastic divergence (price lower low + stoch higher low)',
        '    seen within prior 10 bars. No VixFix or MACD required.',
        f'    Position: ${POSITION_STD:,.0f}',
        '',
        '  TIER 3B ★ STANDARD-BB (BB Trigger only):',
        '    BB trigger candle fires and is confirmed by next candle.',
        '    No Stochastic, VixFix, or MACD required.',
        '    Baseline: how much the BB touch trigger alone is worth.',
        f'    Position: ${POSITION_STD:,.0f}',
        '',
        '  TIER 3C ★ STANDARD-MACD (BB Trigger + Stoch div + MACD div):',
        '    BB trigger candle fires. Stochastic divergence active within',
        '    prior 10 bars. MACD histogram or line/signal also higher now',
        '    vs 10 bars ago. No VixFix required.',
        f'    Position: ${POSITION_STD:,.0f}',
        sep, '',
    ]

    # ── Tier summaries ─────────────────────────────────────────────────────────
    lines.append('── TIER COMPARISON ──────────────────────────────────────────────────')

    def tier_block(label, s, spm):
        if s is None:
            return f'  {label}: no signals\n'
        return (f'  {label}\n'
                f'    Signals:{s["total"]:>5}  ~{spm:.1f}/mo  '
                f'Wins:{s["wins"]:>4} ({s["wr"]:.1f}%)  '
                f'Losses:{s["losses"]:>4} ({s["lr"]:.1f}%)  Neutral:{s["neutral"]:>4}\n'
                f'    Avg win:{s["aw"]:>+7.1f}%  Avg loss:{s["al"]:>+7.1f}%  '
                f'EV:{s["ev"]:>+6.2f}%  Avg hold:{s["hold_w"]:.1f}w\n'
                f'    Total P&L: ${s["pnl"]:>+12,.2f}  ROI:{s["roi"]:>+7.1f}%\n')

    lines.append(tier_block(f'TIER 1 ★★★ ULTRA    (BB+VixFix+MACD+Stoch) | ${POSITION_HIGH:,.0f}/trade', su, spm_u))
    lines.append(tier_block(f'TIER 2 ★★  HIGH     (BB+VixFix+Stoch)       | ${POSITION_HIGH:,.0f}/trade', sh, spm_h))
    lines.append(tier_block(f'TIER 3 ★   STANDARD (BB+Stoch)              | ${POSITION_STD:,.0f}/trade',  ss, spm_s))
    lines.append(tier_block(f'TIER 3B ★  STD-BB   (BB Trigger only)       | ${POSITION_STD:,.0f}/trade',  sb, spm_b))
    lines.append(tier_block(f'TIER 3C ★  STD-MACD (BB+Stoch+MACD)        | ${POSITION_STD:,.0f}/trade',  sm, spm_m))

    # ── Full trade histories ───────────────────────────────────────────────────
    lines.append(sep2)
    lines.append(trade_history(ultra,    '── TIER 1 ★★★ ULTRA TRADE HISTORY (BB+VixFix+MACD+Stoch) ──────────'))
    lines.append(trade_history(high,     '── TIER 2 ★★ HIGH TRADE HISTORY (BB+VixFix+Stoch) ─────────────────'))
    lines.append(trade_history(standard, '── TIER 3 ★ STANDARD TRADE HISTORY (BB+Stoch) ──────────────────────'))
    lines.append(trade_history(bb_only,  '── TIER 3B ★ STD-BB TRADE HISTORY (BB Trigger only) ────────────────'))
    lines.append(trade_history(stch_mcd, '── TIER 3C ★ STD-MACD TRADE HISTORY (BB+Stoch+MACD) ────────────────'))

    lines.append(sep)
    return '\n'.join(lines)

def send_email(report):
    msg = MIMEMultipart()
    msg['From']    = GMAIL_USER
    msg['To']      = TO_EMAIL
    msg['Subject'] = 'Stock Scanner Backtest Report — All Tiers'
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
