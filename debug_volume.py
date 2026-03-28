"""
debug_volume.py
Quick diagnostic to verify the volume filter is working correctly.
Run this before the full optimization to confirm Tier 3 will populate.
Usage: python debug_volume.py
Takes ~30 seconds on a handful of tickers.
"""
import pandas as pd
import numpy as np
import yfinance as yf

VOLUME_MA_BARS = 20
BB_LENGTH      = 20;  BB_MULT    = 2.0
VF_PD          = 30;  VF_BBL     = 20;  VF_MULT = 2.0
VF_LB          = 75;  VF_PH      = 0.85
MAX_GAP        = 35;  VF_NEAR    = 2
LOOKBACK       = 10;  NO_BREAK_BARS = 10
MACD_FAST      = 12;  MACD_SLOW  = 26;  MACD_SIGNAL = 9
STOCH_LOOKBACK = 25;  STOCH_K    = 14
SCAN_DELAY     = 5
MAX_STOP_DIST  = 0.11

# A handful of tickers known to have generated signals historically
TEST_TICKERS = ['AAPL', 'MSFT', 'JPM', 'COST', 'V', 'MA', 'UNH', 'HD']

def compute_macd(close):
    ef = close.ewm(span=MACD_FAST, adjust=False).mean()
    es = close.ewm(span=MACD_SLOW, adjust=False).mean()
    ml = ef - es
    sl = ml.ewm(span=MACD_SIGNAL, adjust=False).mean()
    return ml.values, sl.values, (ml - sl).values

def no_break_before(lv, idx, n):
    tl = lv[idx]
    for j in range(max(0, idx - n), idx):
        if lv[j] < tl: return False
    return True

def macd_divergence(pi, ri, ml, sl, hist):
    vals = [hist[pi], hist[ri], ml[pi], ml[ri], sl[pi], sl[ri]]
    if any(np.isnan(v) for v in vals): return False
    return (hist[ri] > hist[pi]) or (ml[ri] > ml[pi]) or (sl[ri] > sl[pi])

def volume_above_ma(df, idx):
    if 'Volume' not in df.columns: return True
    vol = df['Volume'].values.astype(float)
    cur = vol[idx]
    if np.isnan(cur) or cur == 0: return True
    if idx < VOLUME_MA_BARS: return True
    avg = np.nanmean(vol[idx - VOLUME_MA_BARS:idx])
    if np.isnan(avg) or avg == 0: return True
    return cur > avg

def find_vf_signals(df):
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
    signals = []
    for ri in range(n):
        if not twvf[ri]: continue
        rl, rc, rw = lv[ri], cv[ri], wvfv[ri]
        if np.isnan(rl) or np.isnan(rw): continue
        if not no_break_before(lv, ri, NO_BREAK_BARS): continue
        if (rc - rl) / rc > MAX_STOP_DIST: continue
        for j in range(ri - 1, max(ri - MAX_GAP, 0) - 1, -1):
            if not twvf[j]: continue
            pl, pw = lv[j], wvfv[j]
            if np.isnan(pl) or np.isnan(pw): continue
            if not no_break_before(lv, j, NO_BREAK_BARS): continue
            if rl < pl and rw > pw:
                signals.append({'signal_idx': ri, 'date': df.index[ri],
                                'entry': float(rc), 'stop': float(rl)})
                break
    return signals

print('='*60)
print('VOLUME FILTER DIAGNOSTIC')
print('='*60)

for interval, label in [('1wk', 'WEEKLY'), ('1d', 'DAILY')]:
    print(f'\n── {label} ({interval}) ──────────────────────────────')
    total_sigs = 0
    vol_pass   = 0
    vol_fail   = 0
    vol_nan    = 0

    for ticker in TEST_TICKERS:
        try:
            df = yf.download(ticker, period='5y', interval=interval,
                             progress=False, auto_adjust=True)
            if df is None or len(df) < 100: continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Check volume data quality
            if 'Volume' in df.columns:
                vol_arr = df['Volume'].values.astype(float)
                nan_pct = np.isnan(vol_arr).mean() * 100
                zero_pct = (vol_arr == 0).mean() * 100
                print(f'  {ticker:6s} volume: {len(vol_arr)} bars, '
                      f'{nan_pct:.0f}% NaN, {zero_pct:.0f}% zero, '
                      f'sample={vol_arr[-1]:,.0f}')
            else:
                print(f'  {ticker:6s} NO VOLUME COLUMN')
                continue

            signals = find_vf_signals(df)
            total_sigs += len(signals)
            for sig in signals:
                si = sig['signal_idx']
                result = volume_above_ma(df, si)
                if result:
                    vol_pass += 1
                else:
                    vol_fail += 1

        except Exception as e:
            print(f'  {ticker}: ERROR — {e}')

    print(f'\n  Signals found:      {total_sigs}')
    print(f'  Volume PASS (>MA):  {vol_pass}  '
          f'({vol_pass/total_sigs*100:.0f}% of signals)' if total_sigs else '')
    print(f'  Volume FAIL (<MA):  {vol_fail}  '
          f'({vol_fail/total_sigs*100:.0f}% of signals)' if total_sigs else '')
    print(f'  → Tier 3 would get: {vol_pass} signals '
          f'(vs {total_sigs} in Tier 2)')

print('\n' + '='*60)
print('If Volume FAIL > 0, the filter is working correctly.')
print('If Volume FAIL = 0, ALL signals pass — filter may not be firing.')
print('='*60)
