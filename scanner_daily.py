# NOTE: This is the DAILY version. Logic identical to scanner_combined.py.
# Only INTERVAL, BAR_LABEL, YEAR_HIGH_BARS, SCAN_PERIOD, INTERVAL_LABEL differ.
"""
scanner_daily.py  —  Daily scanner (interval=1wk). Runs every Friday.
Scans all NYSE+NASDAQ tickers for signals, runs an inline 15-year backtest
on a random sample to auto-populate win rates, signal counts, and best EV
target for each tier — so the HTML email is always self-contained.
Weekly version: scanner_combined.py  (same logic, INTERVAL='1wk')
"""
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import smtplib
import time
import os
import random
from datetime import date
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ══════════════════════════════════════════════════════════════════
# INTERVAL SETTINGS  (only block that differs in scanner_daily.py)
# ══════════════════════════════════════════════════════════════════
INTERVAL        = '1d'
BAR_LABEL       = 'day'
YEAR_HIGH_BARS  = 252
YEARS_HISTORY   = 15
SCAN_PERIOD     = '2y'       # lookback for live scan (not backtest)
INTERVAL_LABEL  = 'Daily'
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
MACD_FAST      = 12
MACD_SLOW      = 26
MACD_SIGNAL    = 9
MIN_PRICE      = 10.0
MIN_MARKET_CAP = 1_000_000_000
MAX_STOP_DIST  = 0.11
NO_BREAK_BARS  = 10

HOLD_BARS      = 20
WIN_TARGET     = 0.13
POSITION_HIGH  = 10000.0
POSITION_STD   = 5000.0

# Targets swept for best-EV calculation
EV_TARGETS     = [i / 100 for i in range(5, 55, 5)]

# Tickers sampled for inline backtest (speed vs accuracy trade-off)
BACKTEST_SAMPLE = 400

GMAIL_USER     = os.environ.get('GMAIL_USER', '')
GMAIL_PASSWORD = os.environ.get('GMAIL_PASSWORD', '')
TO_EMAIL       = 'bkcolby@yahoo.com'

# ── Helpers ────────────────────────────────────────────────────────────────────
def safe_mean(v):
    c = [x for x in v if x is not None and not np.isnan(x)]
    return np.mean(c) if c else 0.0

def safe_sum(v):
    c = [x for x in v if x is not None and not np.isnan(x)]
    return sum(c) if c else 0.0

def get_all_tickers():
    headers = {'User-Agent': 'Mozilla/5.0'}
    tickers = []
    for exchange in ['NYSE', 'NASDAQ']:
        url = (f'https://api.nasdaq.com/api/screener/stocks?tableonly=true'
               f'&limit=10000&exchange={exchange}')
        try:
            r    = requests.get(url, headers=headers, timeout=15)
            rows = r.json()['data']['table']['rows']
            for row in rows:
                sym = row['symbol'].strip()
                if sym.isalpha() and len(sym) <= 4:
                    try:
                        mc = float(str(row.get('marketCap', '0')).replace(',', ''))
                        if mc > 0 and mc < MIN_MARKET_CAP: continue
                    except: pass
                    tickers.append(sym)
        except Exception as e:
            print(f'Error fetching {exchange}: {e}')
    tickers = list(set(tickers))
    print(f'Total tickers fetched: {len(tickers)}')
    return tickers

def compute_macd(close):
    ef = close.ewm(span=MACD_FAST,   adjust=False).mean()
    es = close.ewm(span=MACD_SLOW,   adjust=False).mean()
    ml = ef - es
    sl = ml.ewm(span=MACD_SIGNAL, adjust=False).mean()
    return ml.values, sl.values, (ml - sl).values

def no_break_before(lv, idx, n):
    tl = lv[idx]
    for j in range(max(0, idx - n), idx):
        if lv[j] < tl: return False
    return True

def no_break_after(lv, idx, end):
    tl = lv[idx]
    for j in range(idx + 1, end + 1):
        if lv[j] < tl: return False
    return True

def macd_divergence(pi, ri, ml, sl, hist):
    vals = [hist[pi], hist[ri], ml[pi], ml[ri], sl[pi], sl[ri]]
    if any(np.isnan(v) for v in vals): return False
    return (hist[ri] > hist[pi]) or (ml[ri] > ml[pi]) or (sl[ri] > sl[pi])

# ── Live scan functions ────────────────────────────────────────────────────────
def check_vixfix(df, ml, sl, hist):
    close, low, open_ = df['Close'], df['Low'], df['Open']
    cv, lv = close.values, low.values
    n = len(df)
    bb_lo  = close.rolling(BB_LENGTH).mean() - BB_MULT * close.rolling(BB_LENGTH).std(ddof=0)
    trig   = (close > open_) & (low <= bb_lo)
    tc     = (trig & (close.shift(-1) > open_.shift(-1))
                   & (open_.shift(-1) >= open_)
                   & (close.shift(-1) >= open_))
    hc     = close.rolling(VF_PD).max()
    wvf    = (hc - low) / hc * 100
    vf_up  = wvf.rolling(VF_BBL).mean() + VF_MULT * wvf.rolling(VF_BBL).std(ddof=0)
    vf_rng = wvf.rolling(VF_LB).max() * VF_PH
    is_grn = (wvf >= vf_up) | (wvf >= vf_rng)
    vf_near = pd.Series(False, index=df.index)
    for s in range(VF_NEAR + 1):
        vf_near |= is_grn.shift(s).fillna(False).infer_objects(copy=False).astype(bool)
        if s > 0:
            vf_near |= is_grn.shift(-s).fillna(False).infer_objects(copy=False).astype(bool)
    twvf_s = tc & vf_near
    wat    = pd.Series(np.nan, index=df.index)
    for s in range(-VF_NEAR, VF_NEAR + 1):
        sh = wvf.shift(s).fillna(0)
        wat = wat.combine(sh, lambda a, b: b if np.isnan(a) else max(a, b))
    twvf, wvfv = twvf_s.values, wat.values
    recent_idx = None
    for i in range(n - 1, max(n - SCAN_DELAY - 2, -1), -1):
        if twvf[i]: recent_idx = i; break
    if recent_idx is None: return False, False
    rl, rc, rw = lv[recent_idx], cv[recent_idx], wvfv[recent_idx]
    if np.isnan(rl) or np.isnan(rw): return False, False
    if not no_break_before(lv, recent_idx, NO_BREAK_BARS): return False, False
    if (rc - rl) / rc > MAX_STOP_DIST: return False, False
    if not no_break_after(lv, recent_idx, n - 1): return False, False
    for j in range(recent_idx - 1, max(recent_idx - MAX_GAP, 0) - 1, -1):
        if not twvf[j]: continue
        pl, pw = lv[j], wvfv[j]
        if np.isnan(pl) or np.isnan(pw): continue
        if not no_break_before(lv, j, NO_BREAK_BARS): continue
        if rl < pl and rw > pw:
            return macd_divergence(j, recent_idx, ml, sl, hist), True
        break
    return False, False

def check_stoch(df):
    close, high, low, open_ = df['Close'], df['High'], df['Low'], df['Open']
    bb_lo = close.rolling(BB_LENGTH).mean() - BB_MULT * close.rolling(BB_LENGTH).std(ddof=0)
    trig  = (close > open_) & (low <= bb_lo)
    vp    = (trig.shift(1).fillna(False) & (close > open_)
             & (open_ >= open_.shift(1)) & (close >= open_.shift(1)))
    ll    = low.rolling(STOCH_K).min()
    hh    = high.rolling(STOCH_K).max()
    sk    = 100 * (close - ll) / (hh - ll)
    sd    = ((low < low.shift(1).rolling(STOCH_LOOKBACK).min())
             & (sk > sk.shift(1).rolling(STOCH_LOOKBACK).min()))
    tlo   = low.where(trig).ffill()
    nb    = low.rolling(LOOKBACK).min() >= tlo
    bhl   = close <= 0.85 * high.rolling(YEAR_HIGH_BARS).max()
    vpv, sdv, nbv, bhlv = vp.values, sd.values, nb.values, bhl.values
    n = len(vpv)
    if n < LOOKBACK + 1: return False
    if not nbv[-1] or not bhlv[-1]: return False
    if not any(vpv[max(0, n - LOOKBACK):n]): return False
    if not any(sdv[max(0, n - LOOKBACK):n]): return False
    return True

def check_bb_only(df):
    close, low, open_ = df['Close'], df['Low'], df['Open']
    bb_lo = close.rolling(BB_LENGTH).mean() - BB_MULT * close.rolling(BB_LENGTH).std(ddof=0)
    trig  = (close > open_) & (low <= bb_lo)
    tc    = (trig & (close.shift(-1) > open_.shift(-1))
                  & (open_.shift(-1) >= open_)
                  & (close.shift(-1) >= open_))
    tlo   = low.where(trig).ffill()
    nb    = low.rolling(LOOKBACK).min() >= tlo
    tcv, nbv = tc.values, nb.values
    n = len(tcv)
    for i in range(n - 1, max(n - SCAN_DELAY - 2, -1), -1):
        if tcv[i] and nbv[i]:
            cl, lw = float(df['Close'].values[i]), float(df['Low'].values[i])
            if not np.isnan(cl) and not np.isnan(lw):
                if (cl - lw) / cl <= MAX_STOP_DIST: return True
    return False

def check_stoch_macd(df, ml, sl, hist):
    close, high, low, open_ = df['Close'], df['High'], df['Low'], df['Open']
    bb_lo = close.rolling(BB_LENGTH).mean() - BB_MULT * close.rolling(BB_LENGTH).std(ddof=0)
    trig  = (close > open_) & (low <= bb_lo)
    vp    = (trig.shift(1).fillna(False) & (close > open_)
             & (open_ >= open_.shift(1)) & (close >= open_.shift(1)))
    ll    = low.rolling(STOCH_K).min()
    hh    = high.rolling(STOCH_K).max()
    sk    = 100 * (close - ll) / (hh - ll)
    sd    = ((low < low.shift(1).rolling(STOCH_LOOKBACK).min())
             & (sk > sk.shift(1).rolling(STOCH_LOOKBACK).min()))
    tlo   = low.where(trig).ffill()
    nb    = low.rolling(LOOKBACK).min() >= tlo
    bhl   = close <= 0.85 * high.rolling(YEAR_HIGH_BARS).max()
    vpv, sdv, nbv, bhlv = vp.values, sd.values, nb.values, bhl.values
    n = len(vpv)
    if n < LOOKBACK + 2: return False
    if not nbv[-1] or not bhlv[-1]: return False
    if not any(vpv[max(0, n - LOOKBACK):n]): return False
    if not any(sdv[max(0, n - LOOKBACK):n]): return False
    i, pi = n - 1, max(0, n - 1 - LOOKBACK)
    if np.isnan(hist[i]) or np.isnan(hist[pi]): return False
    return (hist[i] > hist[pi]) or (ml[i] > ml[pi]) or (sl[i] > sl[pi])

# ── Inline backtest helpers ────────────────────────────────────────────────────
def _vixfix_pairs(df):
    close, low, open_ = df['Close'], df['Low'], df['Open']
    cv, lv = close.values, low.values
    n = len(df)
    bb_lo  = close.rolling(BB_LENGTH).mean() - BB_MULT * close.rolling(BB_LENGTH).std(ddof=0)
    trig   = (close > open_) & (low <= bb_lo)
    tc     = (trig & (close.shift(-1) > open_.shift(-1))
                   & (open_.shift(-1) >= open_)
                   & (close.shift(-1) >= open_))
    hc     = close.rolling(VF_PD).max()
    wvf    = (hc - low) / hc * 100
    vf_up  = wvf.rolling(VF_BBL).mean() + VF_MULT * wvf.rolling(VF_BBL).std(ddof=0)
    vf_rng = wvf.rolling(VF_LB).max() * VF_PH
    is_grn = (wvf >= vf_up) | (wvf >= vf_rng)
    vf_near = pd.Series(False, index=df.index)
    for s in range(VF_NEAR + 1):
        vf_near |= is_grn.shift(s).fillna(False).infer_objects(copy=False).astype(bool)
        if s > 0:
            vf_near |= is_grn.shift(-s).fillna(False).infer_objects(copy=False).astype(bool)
    twvf_s = tc & vf_near
    wat    = pd.Series(np.nan, index=df.index)
    for s in range(-VF_NEAR, VF_NEAR + 1):
        sh = wvf.shift(s).fillna(0)
        wat = wat.combine(sh, lambda a, b: b if np.isnan(a) else max(a, b))
    twvf, wvfv = twvf_s.values, wat.values
    ml, sl, hist = compute_macd(close)
    pairs = []
    for ri in range(n):
        if not twvf[ri]: continue
        rl, rc, rw = lv[ri], cv[ri], wvfv[ri]
        if np.isnan(rl) or np.isnan(rw): continue
        if not no_break_before(lv, ri, NO_BREAK_BARS): continue
        if (rc - rl) / rc > MAX_STOP_DIST: continue
        if not no_break_after(lv, ri, n - 1): continue
        for j in range(ri - 1, max(ri - MAX_GAP, 0) - 1, -1):
            if not twvf[j]: continue
            pl, pw = lv[j], wvfv[j]
            if np.isnan(pl) or np.isnan(pw): continue
            if not no_break_before(lv, j, NO_BREAK_BARS): continue
            if rl < pl and rw > pw:
                pairs.append({'signal_idx': ri, 'signal_date': df.index[ri],
                               'entry_price': float(rc), 'stop_loss': float(rl),
                               'has_macd': macd_divergence(j, ri, ml, sl, hist)})
                break
    return pairs

def _stoch_active(df):
    close, high, low, open_ = df['Close'], df['High'], df['Low'], df['Open']
    bb_lo = close.rolling(BB_LENGTH).mean() - BB_MULT * close.rolling(BB_LENGTH).std(ddof=0)
    trig  = (close > open_) & (low <= bb_lo)
    vp    = (trig.shift(1).fillna(False) & (close > open_)
             & (open_ >= open_.shift(1)) & (close >= open_.shift(1)))
    ll    = low.rolling(STOCH_K).min(); hh = high.rolling(STOCH_K).max()
    sk    = 100 * (close - ll) / (hh - ll)
    sd    = ((low < low.shift(1).rolling(STOCH_LOOKBACK).min())
             & (sk > sk.shift(1).rolling(STOCH_LOOKBACK).min()))
    tlo   = low.where(trig).ffill()
    nb    = low.rolling(LOOKBACK).min() >= tlo
    bhl   = close <= 0.85 * high.rolling(YEAR_HIGH_BARS).max()
    vpv, sdv, nbv, bhlv = vp.values, sd.values, nb.values, bhl.values
    active = set()
    for i in range(LOOKBACK, len(vpv)):
        if not nbv[i] or not bhlv[i]: continue
        if not any(vpv[max(0, i - LOOKBACK):i]): continue
        if not any(sdv[max(0, i - LOOKBACK):i]): continue
        active.add(i)
    return active

def _stoch_sigs(df):
    close, high, low, open_ = df['Close'], df['High'], df['Low'], df['Open']
    cv, lv = close.values, low.values
    n = len(df)
    bb_lo = close.rolling(BB_LENGTH).mean() - BB_MULT * close.rolling(BB_LENGTH).std(ddof=0)
    trig  = (close > open_) & (low <= bb_lo)
    vp    = (trig.shift(1).fillna(False) & (close > open_)
             & (open_ >= open_.shift(1)) & (close >= open_.shift(1)))
    ll    = low.rolling(STOCH_K).min(); hh = high.rolling(STOCH_K).max()
    sk    = 100 * (close - ll) / (hh - ll)
    sd    = ((low < low.shift(1).rolling(STOCH_LOOKBACK).min())
             & (sk > sk.shift(1).rolling(STOCH_LOOKBACK).min()))
    tlo   = low.where(trig).ffill()
    nb    = low.rolling(LOOKBACK).min() >= tlo
    bhl   = close <= 0.85 * high.rolling(YEAR_HIGH_BARS).max()
    vpv, sdv, nbv, bhlv, tv = vp.values, sd.values, nb.values, bhl.values, trig.values
    sigs = []
    for i in range(LOOKBACK, n - 1):
        if not nbv[i] or not bhlv[i]: continue
        if not any(vpv[max(0, i - LOOKBACK):i]): continue
        if not any(sdv[max(0, i - LOOKBACK):i]): continue
        ti = next((k for k in range(i, max(i - LOOKBACK, -1), -1) if tv[k]), None)
        if ti is None: continue
        e, s = cv[ti], lv[ti]
        if np.isnan(e) or np.isnan(s): continue
        if (e - s) / e > MAX_STOP_DIST: continue
        if sigs and sigs[-1]['signal_idx'] == ti: continue
        sigs.append({'signal_idx': ti, 'signal_date': df.index[ti],
                     'entry_price': float(e), 'stop_loss': float(s)})
    return sigs

def _bb_sigs(df):
    close, low, open_ = df['Close'], df['Low'], df['Open']
    cv, lv = close.values, low.values
    n = len(df)
    bb_lo = close.rolling(BB_LENGTH).mean() - BB_MULT * close.rolling(BB_LENGTH).std(ddof=0)
    trig  = (close > open_) & (low <= bb_lo)
    tc    = (trig & (close.shift(-1) > open_.shift(-1))
                  & (open_.shift(-1) >= open_)
                  & (close.shift(-1) >= open_))
    tlo   = low.where(trig).ffill()
    nb    = low.rolling(LOOKBACK).min() >= tlo
    tcv, nbv = tc.values, nb.values
    sigs = []
    for i in range(LOOKBACK, n - 1):
        if not tcv[i] or not nbv[i]: continue
        e, s = cv[i], lv[i]
        if np.isnan(e) or np.isnan(s): continue
        if (e - s) / e > MAX_STOP_DIST: continue
        sigs.append({'signal_idx': i, 'signal_date': df.index[i],
                     'entry_price': float(e), 'stop_loss': float(s)})
    return sigs

def _stoch_macd_sigs(df):
    close, high, low, open_ = df['Close'], df['High'], df['Low'], df['Open']
    cv, lv = close.values, low.values
    n = len(df)
    bb_lo = close.rolling(BB_LENGTH).mean() - BB_MULT * close.rolling(BB_LENGTH).std(ddof=0)
    trig  = (close > open_) & (low <= bb_lo)
    vp    = (trig.shift(1).fillna(False) & (close > open_)
             & (open_ >= open_.shift(1)) & (close >= open_.shift(1)))
    ll    = low.rolling(STOCH_K).min(); hh = high.rolling(STOCH_K).max()
    sk    = 100 * (close - ll) / (hh - ll)
    sd    = ((low < low.shift(1).rolling(STOCH_LOOKBACK).min())
             & (sk > sk.shift(1).rolling(STOCH_LOOKBACK).min()))
    tlo   = low.where(trig).ffill()
    nb    = low.rolling(LOOKBACK).min() >= tlo
    bhl   = close <= 0.85 * high.rolling(YEAR_HIGH_BARS).max()
    ml, sl_a, hist = compute_macd(close)
    vpv, sdv, nbv, bhlv, tv = vp.values, sd.values, nb.values, bhl.values, trig.values
    sigs = []
    for i in range(LOOKBACK + 1, n - 1):
        if not nbv[i] or not bhlv[i]: continue
        if not any(vpv[max(0, i - LOOKBACK):i]): continue
        if not any(sdv[max(0, i - LOOKBACK):i]): continue
        pi = max(0, i - LOOKBACK)
        if np.isnan(hist[i]) or np.isnan(hist[pi]): continue
        if not ((hist[i] > hist[pi]) or (ml[i] > ml[pi]) or (sl_a[i] > sl_a[pi])): continue
        ti = next((k for k in range(i, max(i - LOOKBACK, -1), -1) if tv[k]), None)
        if ti is None: continue
        e, s = cv[ti], lv[ti]
        if np.isnan(e) or np.isnan(s): continue
        if (e - s) / e > MAX_STOP_DIST: continue
        if sigs and sigs[-1]['signal_idx'] == ti: continue
        sigs.append({'signal_idx': ti, 'signal_date': df.index[ti],
                     'entry_price': float(e), 'stop_loss': float(s)})
    return sigs

def _eval(df, sig, pos, tgt):
    idx, entry, stop = sig['signal_idx'], sig['entry_price'], sig['stop_loss']
    n, win   = len(df), entry * (1 + tgt)
    shares   = pos / entry
    result, exit_price, exit_bar = 'NEUTRAL', None, None
    for w in range(1, HOLD_BARS + 1):
        fi = idx + w
        if fi >= n: break
        wh, wl = float(df['High'].iloc[fi]), float(df['Low'].iloc[fi])
        if wh >= win:  result, exit_price, exit_bar = 'WIN',  win,  w; break
        if wl <= stop: result, exit_price, exit_bar = 'LOSS', stop, w; break
    if result == 'NEUTRAL':
        last       = min(idx + HOLD_BARS, n - 1)
        exit_price = float(df['Close'].iloc[last])
        exit_bar   = min(HOLD_BARS, n - 1 - idx)
    pct    = (exit_price - entry) / entry * 100
    dollar = shares * (exit_price - entry)
    return {'result': result,
            'pct_return':    pct    if not np.isnan(pct)    else 0.0,
            'dollar_return': dollar if not np.isnan(dollar) else 0.0}

def _best_ev(trades_by_target):
    best_tgt, best_ev, best_wr, best_n = WIN_TARGET, 0.0, 0.0, 0
    for tgt, trades in trades_by_target.items():
        if not trades: continue
        n    = len(trades)
        wins = [t for t in trades if t['result'] == 'WIN']
        loss = [t for t in trades if t['result'] == 'LOSS']
        wr   = len(wins) / n * 100
        lr   = len(loss) / n * 100
        aw   = safe_mean([t['pct_return'] for t in wins])
        al   = safe_mean([t['pct_return'] for t in loss])
        ev   = (wr / 100 * aw) + (lr / 100 * al)
        if ev > best_ev:
            best_ev, best_tgt, best_wr, best_n = ev, tgt, wr, n
    return best_tgt, best_wr, best_ev, best_n

def run_inline_backtest(tickers):
    """
    Sample BACKTEST_SAMPLE tickers, run 15-year backtest across all EV targets,
    return per-tier stats dict used to populate the HTML report.
    """
    print(f'\nInline backtest: sampling {BACKTEST_SAMPLE} tickers over {YEARS_HISTORY} years...')
    sample   = random.sample(tickers, min(BACKTEST_SAMPLE, len(tickers)))
    cutoff   = pd.Timestamp.now() - pd.DateOffset(years=YEARS_HISTORY)
    min_bars = max(VF_LB + MAX_GAP,
                   YEAR_HIGH_BARS + LOOKBACK + STOCH_LOOKBACK) + HOLD_BARS + 10

    tier_trades = {k: {tgt: [] for tgt in EV_TARGETS}
                   for k in ['ultra', 'high', 'standard', 'bb_only', 'stoch_macd']}

    for ticker in sample:
        try:
            df = yf.download(ticker, period='max', interval=INTERVAL,
                             progress=False, auto_adjust=True)
            if df is None or len(df) < min_bars: continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[df.index >= cutoff].copy()
            if len(df) < min_bars: continue
            cc = df['Close'].dropna()
            if cc.empty: continue
            rc = float(cc.iloc[-1])
            if np.isnan(rc) or rc < MIN_PRICE: continue
            try:
                mc = yf.Ticker(ticker).fast_info.market_cap
                if mc is not None and mc < MIN_MARKET_CAP: continue
            except: pass

            vf_pairs = _vixfix_pairs(df)
            sa       = _stoch_active(df)

            for sig in vf_pairs:
                si, e = sig['signal_idx'], sig['entry_price']
                if e < rc * 0.15 or e > rc * 6.0: continue
                if si + 1 >= len(df): continue
                has_s = any((si + o) in sa for o in range(-SCAN_DELAY, SCAN_DELAY + 1))
                for tgt in EV_TARGETS:
                    t = _eval(df, sig, POSITION_HIGH, tgt)
                    if sig['has_macd'] and has_s: tier_trades['ultra'][tgt].append(t)
                    if has_s:                     tier_trades['high'][tgt].append(t)

            for sig in _stoch_sigs(df):
                si, e = sig['signal_idx'], sig['entry_price']
                if e < rc * 0.15 or e > rc * 6.0: continue
                if si + 1 >= len(df): continue
                for tgt in EV_TARGETS:
                    tier_trades['standard'][tgt].append(_eval(df, sig, POSITION_STD, tgt))

            for sig in _bb_sigs(df):
                si, e = sig['signal_idx'], sig['entry_price']
                if e < rc * 0.15 or e > rc * 6.0: continue
                if si + 1 >= len(df): continue
                for tgt in EV_TARGETS:
                    tier_trades['bb_only'][tgt].append(_eval(df, sig, POSITION_STD, tgt))

            for sig in _stoch_macd_sigs(df):
                si, e = sig['signal_idx'], sig['entry_price']
                if e < rc * 0.15 or e > rc * 6.0: continue
                if si + 1 >= len(df): continue
                for tgt in EV_TARGETS:
                    tier_trades['stoch_macd'][tgt].append(_eval(df, sig, POSITION_STD, tgt))

        except: pass
        time.sleep(0.03)

    out = {}
    for tier in tier_trades:
        trades_def = tier_trades[tier].get(WIN_TARGET, [])
        n = len(trades_def)
        if n == 0:
            out[tier] = {'wr': None, 'signals': 0, 'spm': 0.0,
                         'best_target': WIN_TARGET, 'best_target_wr': None, 'best_ev': None}
            continue
        wins = [t for t in trades_def if t['result'] == 'WIN']
        loss = [t for t in trades_def if t['result'] == 'LOSS']
        wr   = len(wins) / n * 100
        lr   = len(loss) / n * 100
        aw   = safe_mean([t['pct_return'] for t in wins])
        al   = safe_mean([t['pct_return'] for t in loss])
        ev   = (wr / 100 * aw) + (lr / 100 * al)
        spm  = n / (YEARS_HISTORY * 12)
        bt, bwr, bev, _ = _best_ev(tier_trades[tier])
        out[tier] = {'wr': wr, 'signals': n, 'spm': spm, 'ev': ev,
                     'best_target': bt, 'best_target_wr': bwr, 'best_ev': bev}

    print('Inline backtest complete.')
    return out

# ── Live scan ──────────────────────────────────────────────────────────────────
def run_scans(tickers):
    ultra, high, standard, bb_only, stoch_macd = [], [], [], [], []
    min_bars = max(VF_LB + MAX_GAP, YEAR_HIGH_BARS + LOOKBACK + STOCH_LOOKBACK) + 10
    print(f'Scanning {len(tickers)} tickers ({INTERVAL_LABEL})...\n')
    for i, ticker in enumerate(tickers):
        try:
            df = yf.download(ticker, period=SCAN_PERIOD, interval=INTERVAL,
                             progress=False, auto_adjust=True)
            if df is None or len(df) < min_bars: continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            cc = df['Close'].dropna()
            if cc.empty: continue
            cp = float(cc.iloc[-1])
            if np.isnan(cp) or cp < MIN_PRICE: continue
            try:
                mc = yf.Ticker(ticker).fast_info.market_cap
                if mc is not None and mc < MIN_MARKET_CAP: continue
            except: pass

            ml, sl, hist = compute_macd(df['Close'])
            hm_vf, has_vf = check_vixfix(df, ml, sl, hist)
            has_stoch      = check_stoch(df)
            has_bb         = check_bb_only(df)
            has_sm         = check_stoch_macd(df, ml, sl, hist)

            if hm_vf and has_vf and has_stoch: ultra.append(ticker)
            if has_vf and has_stoch:           high.append(ticker)
            if has_stoch:                      standard.append(ticker)
            if has_bb:                         bb_only.append(ticker)
            if has_sm:                         stoch_macd.append(ticker)

            tags = []
            if hm_vf and has_vf and has_stoch: tags.append('ULTRA')
            elif has_vf and has_stoch:         tags.append('HIGH')
            elif has_stoch:                    tags.append('Stoch')
            if has_bb:                         tags.append('BB')
            if has_sm:                         tags.append('Stoch+MACD')
            if tags: print(f'  ✓ {ticker} — {" | ".join(tags)}')
        except Exception as e:
            print(f'  Error {ticker}: {e}')
        if (i + 1) % 100 == 0:
            print(f'  [{i+1}/{len(tickers)}] scanned')
        time.sleep(0.05)
    return (sorted(ultra), sorted(high), sorted(standard),
            sorted(bb_only), sorted(stoch_macd))

# ── HTML report ────────────────────────────────────────────────────────────────
def _tier_html(tier_name, title, description, tickers_this_tier,
               stats, pos_size, hide_already_in=None):
    wr          = stats.get('wr')
    n_sigs      = stats.get('signals', 0)
    spm         = stats.get('spm', 0.0)
    best_tgt    = stats.get('best_target')
    best_tgt_wr = stats.get('best_target_wr')
    best_ev     = stats.get('best_ev')

    if wr is not None and wr > 80:
        pos_note = f'<strong style="color:#006600;">${pos_size:,.0f}/trade (>80% WR)</strong>'
    else:
        pos_note = f'${pos_size:,.0f}/trade'

    if wr is not None and n_sigs > 0:
        wr_str = (f'<strong>{wr:.1f}%</strong> win rate at {int(WIN_TARGET*100)}% target &nbsp;|&nbsp; '
                  f'{n_sigs} signals found in sample &nbsp;|&nbsp; ~{spm:.1f}/mo')
    else:
        wr_str = 'No signals found in backtest sample — run backtest_daily.py for full results'

    best_ev_str = ''
    if best_tgt is not None and best_tgt_wr is not None and best_ev is not None:
        best_ev_str = (
            f'<br><span style="color:#444;font-size:0.88em;">'
            f'&#9733; Best EV exit target: <strong>{int(best_tgt*100)}%</strong>'
            f' &nbsp;|&nbsp; Win rate at that target: <strong>{best_tgt_wr:.1f}%</strong>'
            f' &nbsp;|&nbsp; EV: <strong>{best_ev:+.2f}%</strong></span>'
        )

    excl = set(hide_already_in) if hide_already_in else set()
    show = [t for t in tickers_this_tier if t not in excl]

    if show:
        ticker_html = '&nbsp; '.join(
            f'<span style="color:#cc0000;font-weight:bold;font-size:1.05em;">{t}</span>'
            for t in show
        )
        count_note = f'<br><span style="font-size:0.8em;color:#666;">Total: {len(show)}'
        if excl:
            count_note += f' (excl. higher tiers) | All {tier_name}: {len(tickers_this_tier)}'
        count_note += '</span>'
        ticker_html += count_note
    else:
        ticker_html = f'<em style="color:#999;">No {tier_name} signals today.</em>'

    return f'''
<div style="border:1px solid #ccc;border-radius:6px;padding:14px 16px;
            margin-bottom:16px;background:#fafafa;">
  <h2 style="margin:0 0 4px 0;font-size:1.05em;color:#111;">{title}</h2>
  <p style="margin:1px 0 6px 0;font-size:0.82em;color:#555;">{description}</p>
  <p style="margin:3px 0;font-size:0.85em;">
    <strong>Backtest ({YEARS_HISTORY}yr, {INTERVAL_LABEL.lower()}):</strong>
    {wr_str}{best_ev_str}
  </p>
  <p style="margin:3px 0;font-size:0.85em;">
    <strong>Position:</strong> {pos_note} &nbsp;|&nbsp;
    <strong>Win target:</strong> {int(WIN_TARGET*100)}% &nbsp;|&nbsp;
    <strong>Max hold:</strong> {HOLD_BARS} {BAR_LABEL}s &nbsp;|&nbsp;
    Entry: close of BB trigger &nbsp;|&nbsp; Stop: low of BB trigger
  </p>
  <hr style="border:none;border-top:1px solid #e0e0e0;margin:8px 0;">
  <p style="margin:0;font-size:1em;line-height:2.1;">{ticker_html}</p>
</div>'''

def build_html_report(ultra, high, standard, bb_only, stoch_macd, bt_stats):
    today_str = date.today().strftime('%B %d, %Y')

    summary_rows = ''
    for tier, lbl, lst in [
        ('ultra',      'Tier 1 ★★★ ULTRA',     ultra),
        ('high',       'Tier 2 ★★ HIGH',        high),
        ('standard',   'Tier 3 ★ STANDARD',     standard),
        ('bb_only',    'Tier 3B ★ STD-BB',      bb_only),
        ('stoch_macd', 'Tier 3C ★ STD-MACD',   stoch_macd),
    ]:
        summary_rows += (f'<tr><td>{lbl}</td>'
                         f'<td style="text-align:center;">{len(lst)}</td></tr>\n')

    tier_sections = ''
    tier_sections += _tier_html(
        'ULTRA', '★★★ TIER 1 — ULTRA CONFIDENCE',
        'BB Trigger + VixFix divergence + MACD divergence + Stochastic divergence. '
        'All four conditions confirmed. Entry at close of BB trigger candle.',
        ultra, bt_stats.get('ultra', {}), POSITION_HIGH)

    tier_sections += _tier_html(
        'HIGH', '★★ TIER 2 — HIGH CONFIDENCE',
        'BB Trigger + VixFix divergence + Stochastic divergence. MACD not required. '
        'Entry at close of BB trigger candle.',
        high, bt_stats.get('high', {}), POSITION_HIGH,
        hide_already_in=ultra)

    tier_sections += _tier_html(
        'STANDARD', '★ TIER 3 — STANDARD',
        'BB Trigger + Stochastic divergence. No VixFix or MACD required. '
        'Entry at close of BB trigger candle.',
        standard, bt_stats.get('standard', {}), POSITION_STD,
        hide_already_in=set(ultra) | set(high))

    tier_sections += _tier_html(
        'STD-BB', '★ TIER 3B — STANDARD-BB (BB Trigger only)',
        'Confirmed BB touch trigger candle. No other indicators. Baseline measure. '
        'Entry at close of BB trigger candle.',
        bb_only, bt_stats.get('bb_only', {}), POSITION_STD,
        hide_already_in=set(ultra) | set(high) | set(standard))

    tier_sections += _tier_html(
        'STD-MACD', '★ TIER 3C — STANDARD-MACD',
        'BB Trigger + Stochastic divergence + MACD divergence. No VixFix required. '
        'Entry at close of BB trigger candle.',
        stoch_macd, bt_stats.get('stoch_macd', {}), POSITION_STD,
        hide_already_in=set(ultra) | set(high))

    return f'''<!DOCTYPE html>
<html>
<head><meta charset="utf-8">
<style>
  body  {{ font-family: Arial, sans-serif; font-size: 14px; color: #222;
           max-width: 700px; margin: 0 auto; padding: 20px; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 14px; }}
  th, td {{ border: 1px solid #ccc; padding: 5px 10px; font-size: 0.85em; }}
  th    {{ background: #f0f0f0; }}
</style>
</head>
<body>
<h1 style="font-size:1.3em;margin-bottom:2px;">
  {INTERVAL_LABEL} Stock Scan Results — {today_str}
</h1>
<p style="margin:0 0 14px 0;font-size:0.82em;color:#666;">
  Interval: {INTERVAL_LABEL} ({INTERVAL}) &nbsp;|&nbsp;
  Universe: NYSE + NASDAQ &nbsp;|&nbsp;
  Filters: Price &gt;${MIN_PRICE:.0f}, Mkt cap &gt;$1B, Stop dist &lt;{int(MAX_STOP_DIST*100)}%
</p>

<h3 style="margin:0 0 5px 0;font-size:0.95em;">Signal Count Summary</h3>
<table>
  <tr><th>Tier</th><th>Signals Today</th></tr>
  {summary_rows}
</table>

{tier_sections}

<p style="font-size:0.72em;color:#aaa;margin-top:16px;">
  Win rates computed from a {BACKTEST_SAMPLE}-ticker random sample over {YEARS_HISTORY} years.
  Run backtest.py for definitive full-universe stats.
</p>
</body>
</html>'''

def send_email(subject, html_body):
    msg = MIMEMultipart('alternative')
    msg['From']    = GMAIL_USER
    msg['To']      = TO_EMAIL
    msg['Subject'] = subject
    msg.attach(MIMEText(html_body, 'html'))
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as s:
            s.login(GMAIL_USER, GMAIL_PASSWORD)
            s.sendmail(GMAIL_USER, TO_EMAIL, msg.as_string())
        print('Email sent.')
    except Exception as e:
        print(f'Email failed: {e}')

if __name__ == '__main__':
    tickers = get_all_tickers()
    ultra, high, standard, bb_only, stoch_macd = run_scans(tickers)
    bt_stats = run_inline_backtest(tickers)
    html = build_html_report(ultra, high, standard, bb_only, stoch_macd, bt_stats)
    print(f'\nScan complete — ULTRA:{len(ultra)} HIGH:{len(high)} '
          f'STD:{len(standard)} BB:{len(bb_only)} SM:{len(stoch_macd)}')
    if GMAIL_USER:
        send_email(f'{INTERVAL_LABEL} Stock Scan — {date.today().strftime("%b %d %Y")}', html)
    else:
        print(html[:3000], '\n...')
