# NOTE: This is the DAILY version. Logic is identical to backtest.py.
# Only INTERVAL, BAR_LABEL, and YEAR_HIGH_BARS differ.
"""
backtest_daily.py  —  Daily backtest (interval=1d, 20-bar hold)
Run via workflow_dispatch. Emails full report with year-by-year vs SPY.
To run weekly version use backtest.py — identical logic, different constants.
"""
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
# INTERVAL SETTINGS  (only block that differs in backtest_daily.py)
# ══════════════════════════════════════════════════════════════════
INTERVAL       = '1d'
BAR_LABEL      = 'day'
YEAR_HIGH_BARS = 252          # 52 weekly bars ≈ 1 year
YEARS_HISTORY  = 15
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

HOLD_BARS      = 20
WIN_TARGET     = 0.13
POSITION_HIGH  = 10000.0   # Tiers 1 & 2 (>80% WR)
POSITION_STD   = 5000.0    # Tiers 3 / 3B / 3C

MIN_PRICE      = 10.0
MIN_MARKET_CAP = 1_000_000_000
MAX_STOP_DIST  = 0.11
NO_BREAK_BARS  = 10

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

def macd_div(pi, ri, ml, sl, hist):
    vals = [hist[pi], hist[ri], ml[pi], ml[ri], sl[pi], sl[ri]]
    if any(np.isnan(v) for v in vals): return False
    return (hist[ri] > hist[pi]) or (ml[ri] > ml[pi]) or (sl[ri] > sl[pi])

# ── Signal finders ─────────────────────────────────────────────────────────────
def find_vixfix_pairs(df):
    close, low, open_ = df['Close'], df['Low'], df['Open']
    cv, lv = close.values, low.values
    n      = len(df)
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
                pairs.append({'signal_idx': ri, 'prior_idx': j,
                               'signal_date': df.index[ri],
                               'entry_price': float(rc), 'stop_loss': float(rl),
                               'has_macd': macd_div(j, ri, ml, sl, hist)})
                break
    return pairs

def find_stoch_active_set(df):
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
    active = set()
    for i in range(LOOKBACK, len(vpv)):
        if not nbv[i] or not bhlv[i]: continue
        if not any(vpv[max(0, i - LOOKBACK):i]): continue
        if not any(sdv[max(0, i - LOOKBACK):i]): continue
        active.add(i)
    return active

def find_stoch_signals(df):
    close, high, low, open_ = df['Close'], df['High'], df['Low'], df['Open']
    cv, lv = close.values, low.values
    n = len(df)
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

def find_bb_only_signals(df):
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

def find_stoch_macd_signals(df):
    close, high, low, open_ = df['Close'], df['High'], df['Low'], df['Open']
    cv, lv = close.values, low.values
    n = len(df)
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

# ── Trade evaluation ───────────────────────────────────────────────────────────
def eval_trade(df, signal, position_size):
    idx, entry, stop = signal['signal_idx'], signal['entry_price'], signal['stop_loss']
    n, win = len(df), entry * (1 + WIN_TARGET)
    shares = position_size / entry
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
    sig_dt = signal['signal_date']
    sig_yr = sig_dt.year if hasattr(sig_dt, 'year') else pd.Timestamp(sig_dt).year
    return {'result': result, 'entry': entry, 'stop_loss': stop,
            'exit_price': exit_price, 'exit_bar': exit_bar,
            'pct_return':    pct    if not np.isnan(pct)    else 0.0,
            'dollar_return': dollar if not np.isnan(dollar) else 0.0,
            'ticker': signal.get('ticker', ''),
            'date':   str(signal['signal_date'].date()),
            'year':   sig_yr}

# ── Main backtest ──────────────────────────────────────────────────────────────
def run_backtest(tickers):
    cutoff   = pd.Timestamp.now() - pd.DateOffset(years=YEARS_HISTORY)
    min_bars = max(VF_LB + MAX_GAP,
                   YEAR_HIGH_BARS + LOOKBACK + STOCH_LOOKBACK) + HOLD_BARS + 10
    results  = {k: [] for k in ['ultra', 'high', 'standard', 'bb_only', 'stoch_macd']}
    print(f'Backtesting {len(tickers)} tickers '
          f'({YEARS_HISTORY}yr, {INTERVAL}, {HOLD_BARS}-bar hold, {int(WIN_TARGET*100)}% target)...')

    for i, ticker in enumerate(tickers):
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

            vf_pairs     = find_vixfix_pairs(df)
            stoch_active = find_stoch_active_set(df)

            for sig in vf_pairs:
                si, e = sig['signal_idx'], sig['entry_price']
                if e < rc * 0.15 or e > rc * 6.0: continue
                if si + 1 >= len(df): continue
                has_stoch = any((si + o) in stoch_active
                                for o in range(-SCAN_DELAY, SCAN_DELAY + 1))
                sig['ticker'] = ticker
                t = eval_trade(df, sig, POSITION_HIGH)
                if t is None: continue
                if sig['has_macd'] and has_stoch: results['ultra'].append(t)
                if has_stoch:                     results['high'].append(t)

            for sig in find_stoch_signals(df):
                si, e = sig['signal_idx'], sig['entry_price']
                if e < rc * 0.15 or e > rc * 6.0: continue
                if si + 1 >= len(df): continue
                sig['ticker'] = ticker
                t = eval_trade(df, sig, POSITION_STD)
                if t: results['standard'].append(t)

            for sig in find_bb_only_signals(df):
                si, e = sig['signal_idx'], sig['entry_price']
                if e < rc * 0.15 or e > rc * 6.0: continue
                if si + 1 >= len(df): continue
                sig['ticker'] = ticker
                t = eval_trade(df, sig, POSITION_STD)
                if t: results['bb_only'].append(t)

            for sig in find_stoch_macd_signals(df):
                si, e = sig['signal_idx'], sig['entry_price']
                if e < rc * 0.15 or e > rc * 6.0: continue
                if si + 1 >= len(df): continue
                sig['ticker'] = ticker
                t = eval_trade(df, sig, POSITION_STD)
                if t: results['stoch_macd'].append(t)

        except Exception as ex:
            print(f'  Error {ticker}: {ex}')
        if (i + 1) % 100 == 0:
            print(f'  [{i+1}/{len(tickers)}] scanned')
        time.sleep(0.05)

    return results

# ── SPY annual returns ─────────────────────────────────────────────────────────
def get_spy_annual_returns(years_back=YEARS_HISTORY):
    try:
        spy = yf.download('SPY', period='max', interval='1mo',
                          progress=False, auto_adjust=True)
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)
        cutoff_yr = pd.Timestamp.now().year - years_back
        annual = {}
        for yr, grp in spy.groupby(spy.index.year):
            if yr < cutoff_yr or len(grp) < 2: continue
            ret = (float(grp['Close'].iloc[-1]) - float(grp['Close'].iloc[0])) \
                  / float(grp['Close'].iloc[0]) * 100
            annual[yr] = ret
        return annual
    except Exception as e:
        print(f'SPY fetch error: {e}')
        return {}

# ── Stats helpers ──────────────────────────────────────────────────────────────
def bucket_stats(trades, position_size):
    if not trades: return None
    wins    = [t for t in trades if t['result'] == 'WIN']
    losses  = [t for t in trades if t['result'] == 'LOSS']
    neutral = [t for t in trades if t['result'] == 'NEUTRAL']
    total   = len(trades)
    wr      = len(wins)   / total * 100
    lr      = len(losses) / total * 100
    aw      = safe_mean([t['pct_return'] for t in wins])
    al      = safe_mean([t['pct_return'] for t in losses])
    ev      = (wr / 100 * aw) + (lr / 100 * al)
    pnl     = safe_sum([t['dollar_return'] for t in trades])
    capital = total * position_size
    roi     = (pnl / capital * 100) if capital > 0 else 0.0
    hold_b  = safe_mean([t['exit_bar'] for t in trades if t['exit_bar'] is not None])
    return dict(total=total, wins=len(wins), losses=len(losses), neutral=len(neutral),
                wr=wr, lr=lr, aw=aw, al=al, ev=ev, pnl=pnl, roi=roi, hold_b=hold_b)

def signals_per_month(trades, years=YEARS_HISTORY):
    return len(trades) / (years * 12) if years > 0 else 0.0

def yearly_stats(trades, position_size):
    by_year = {}
    for t in trades:
        yr = t.get('year', 0)
        by_year.setdefault(yr, []).append(t)
    return {yr: bucket_stats(tlist, position_size)
            for yr, tlist in sorted(by_year.items())
            if bucket_stats(tlist, position_size)}

# ── Report builders ────────────────────────────────────────────────────────────
def tier_summary_block(label, s, spm):
    if s is None: return f'  {label}: no signals\n'
    bl = BAR_LABEL[0]
    return (f'  {label}\n'
            f'    Signals:{s["total"]:>6}  ~{spm:.1f}/mo  '
            f'Wins:{s["wins"]:>5} ({s["wr"]:.1f}%)  '
            f'Losses:{s["losses"]:>5} ({s["lr"]:.1f}%)  Neutral:{s["neutral"]:>4}\n'
            f'    Avg win:{s["aw"]:>+7.1f}%  Avg loss:{s["al"]:>+7.1f}%  '
            f'EV:{s["ev"]:>+6.2f}%  Avg hold:{s["hold_b"]:.1f}{bl}\n'
            f'    Total P&L: ${s["pnl"]:>+13,.2f}  ROI:{s["roi"]:>+7.1f}%\n')

def yearly_table(tier_label, yr_stats, spy_annual):
    if not yr_stats: return f'  {tier_label}: no yearly data\n'
    sep = '-' * 72
    rows = [
        f'  {tier_label}',
        f'  {"Year":<6} {"Sigs":>5} {"Win%":>6} {"P&L":>14} '
        f'{"ROI%":>7}  {"SPY%":>7}  {"vs SPY":>8}',
        f'  {sep}',
    ]
    for yr in sorted(yr_stats):
        s   = yr_stats[yr]
        spy = spy_annual.get(yr)
        spy_str = f'{spy:>+6.1f}%' if spy is not None else '    N/A'
        vs_str  = f'{s["roi"] - spy:>+6.1f}%' if spy is not None else '    N/A'
        rows.append(
            f'  {yr:<6} {s["total"]:>5} {s["wr"]:>5.1f}% '
            f'${s["pnl"]:>+13,.0f} {s["roi"]:>+6.1f}%  {spy_str}  {vs_str}'
        )
    rows.append(f'  {sep}')
    total_pnl = sum(v['pnl']   for v in yr_stats.values())
    total_sig = sum(v['total'] for v in yr_stats.values())
    rows.append(f'  {"TOTAL":<6} {total_sig:>5}  Cumulative P&L: ${total_pnl:>+12,.0f}')
    return '\n'.join(rows) + '\n'

def trade_history_block(trades, label):
    if not trades: return f'{label}: no signals\n'
    bl = BAR_LABEL[0]
    lines = [label,
             f'{"Ticker":<6} {"Date":<12} {"Result":<8} {"Ret%":>7} '
             f'{"$Return":>11} {"Bars":>5} {"Entry":>8} {"Stop":>8}',
             '-' * 72]
    for t in sorted(trades, key=lambda x: x['date']):
        lines.append(
            f'{t["ticker"]:<6} {t["date"]:<12} {t["result"]:<8} '
            f'{t["pct_return"]:>+6.1f}% {t["dollar_return"]:>+11,.2f} '
            f'{str(t["exit_bar"])+bl:>5}  ${t["entry"]:>7.2f}  ${t["stop_loss"]:>7.2f}'
        )
    return '\n'.join(lines) + '\n'

def build_report(results, spy_annual=None):
    if spy_annual is None: spy_annual = {}
    sep  = '=' * 80
    sep2 = '-' * 80

    tiers = [
        ('ultra',      POSITION_HIGH, 'TIER 1 ★★★ ULTRA    (BB+VixFix+MACD+Stoch)'),
        ('high',       POSITION_HIGH, 'TIER 2 ★★  HIGH     (BB+VixFix+Stoch)      '),
        ('standard',   POSITION_STD,  'TIER 3 ★   STANDARD (BB+Stoch)             '),
        ('bb_only',    POSITION_STD,  'TIER 3B ★  STD-BB   (BB Trigger only)      '),
        ('stoch_macd', POSITION_STD,  'TIER 3C ★  STD-MACD (BB+Stoch+MACD)       '),
    ]

    lines = [
        sep,
        f'BACKTEST REPORT — {INTERVAL.upper()}  |  {YEARS_HISTORY} years  |  '
        f'{HOLD_BARS}-{BAR_LABEL} hold  |  {int(WIN_TARGET*100)}% win target',
        sep,
        f'  Entry: close of BB trigger candle | Stop: low of BB trigger candle',
        f'  Position: ${POSITION_HIGH:,.0f} (Tiers 1 & 2, >80% WR) | '
        f'${POSITION_STD:,.0f} (Tiers 3/3B/3C)',
        f'  Filters: Price >${MIN_PRICE:.0f} | Mkt cap >$1B | Stop dist <{int(MAX_STOP_DIST*100)}%',
        f'  ExpVal = (Win% × Avg Win%) + (Loss% × Avg Loss%)',
        f'  ROI = Total P&L ÷ (Signals × Position Size)',
        sep, '',
        '── TIER COMPARISON ──────────────────────────────────────────────────────────────',
    ]
    for key, pos, lbl in tiers:
        s   = bucket_stats(results[key], pos)
        spm = signals_per_month(results[key])
        lines.append(tier_summary_block(f'{lbl} | ${pos:,.0f}/trade', s, spm))

    # ── Year-by-year vs SPY ────────────────────────────────────────────────────
    lines += [
        '', sep2,
        '── YEAR-BY-YEAR PERFORMANCE vs SPY ─────────────────────────────────────────────',
        '   ROI  = tier P&L ÷ capital deployed that year',
        '   SPY% = SPY annual price return (buy-and-hold)',
        '   vs SPY = tier ROI% minus SPY%  (positive = outperformed)',
        '',
    ]
    for key, pos, lbl in tiers:
        ys = yearly_stats(results[key], pos)
        lines.append(yearly_table(lbl.strip(), ys, spy_annual))

    # ── Trade histories ────────────────────────────────────────────────────────
    lines += ['', sep2]
    for key, pos, lbl in tiers:
        lines.append(trade_history_block(
            results[key],
            f'── {lbl.strip()} TRADE HISTORY ──────────────────────────────────────────────'))

    lines.append(sep)
    return '\n'.join(lines)

def send_email(subject, report):
    msg = MIMEMultipart()
    msg['From']    = GMAIL_USER
    msg['To']      = TO_EMAIL
    msg['Subject'] = subject
    msg.attach(MIMEText(report, 'plain'))
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as s:
            s.login(GMAIL_USER, GMAIL_PASSWORD)
            s.sendmail(GMAIL_USER, TO_EMAIL, msg.as_string())
        print(f'Emailed to {TO_EMAIL}')
    except Exception as e:
        print(f'Email failed: {e}')

if __name__ == '__main__':
    tickers    = get_all_tickers()
    results    = run_backtest(tickers)
    spy_annual = get_spy_annual_returns()
    report     = build_report(results, spy_annual)
    print('\n' + report)
    if GMAIL_USER:
        send_email(f'Backtest Report — {INTERVAL.upper()} — {HOLD_BARS}-{BAR_LABEL} hold', report)
