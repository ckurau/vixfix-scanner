"""
backtest_optimize.py
Exit target optimization for the 2 active strategies:
  1) Weekly (1wk) — 20-week hold
  2) Daily  (1d)  — 60-day  hold
Tiers 1 & 2 only. Sweeps exit targets 5%–50% in 5% steps.
Runtime: ~40–45 minutes.
"""
import requests, pandas as pd, numpy as np, yfinance as yf
import smtplib, time, os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ── Signal parameters (shared across all configs) ─────────────────────────────
BB_LENGTH      = 20;  BB_MULT   = 2.0
VF_PD          = 30;  VF_BBL    = 20;  VF_MULT = 2.0
VF_LB          = 75;  VF_PH     = 0.85
MAX_GAP        = 35;  SCAN_DELAY = 5;  VF_NEAR = 2
STOCH_LOOKBACK = 25;  STOCH_K   = 14;  LOOKBACK = 10
MACD_FAST      = 12;  MACD_SLOW = 26;  MACD_SIGNAL = 9

POSITION_HIGH  = 10000.0
EXIT_TARGETS   = sorted(set(
    [i / 100 for i in range(5, 55, 5)] +   # 5,10,15,...,50  (5% steps)
    [0.11, 0.12, 0.13, 0.14]                # extra granularity 11–14%
))
YEARS_HISTORY  = 15

MIN_PRICE      = 10.0
MIN_MARKET_CAP = 1_000_000_000
MAX_STOP_DIST  = 0.11
NO_BREAK_BARS  = 10

GMAIL_USER     = os.environ.get('GMAIL_USER', '')
GMAIL_PASSWORD = os.environ.get('GMAIL_PASSWORD', '')
TO_EMAIL       = 'bkcolby@yahoo.com'

# ── Configurations to run ─────────────────────────────────────────────────────
CONFIGS = [
    {'interval': '1wk', 'hold': 20, 'year_high_bars': 52,  'label': '1wk / 20-week hold'},
    {'interval': '1d',  'hold': 60, 'year_high_bars': 252, 'label': '1d  / 60-day  hold'},
]

VOLUME_MA_BARS = 20   # trigger candle volume must be above this MA

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
        except Exception as e: print(f'Error {exchange}: {e}')
    tickers = list(set(tickers))
    print(f'Total tickers: {len(tickers)}')
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

def macd_divergence(pi, ri, ml, sl, hist):
    vals = [hist[pi], hist[ri], ml[pi], ml[ri], sl[pi], sl[ri]]
    if any(np.isnan(v) for v in vals): return False
    return (hist[ri] > hist[pi]) or (ml[ri] > ml[pi]) or (sl[ri] > sl[pi])

def volume_above_ma(df, idx):
    """Returns True if trigger candle volume > 20-bar average volume."""
    if 'Volume' not in df.columns: return True   # skip if no volume data
    vol = df['Volume'].values
    if idx < VOLUME_MA_BARS: return True          # not enough history, allow
    avg = np.mean(vol[idx - VOLUME_MA_BARS:idx])
    if avg == 0: return True
    return float(vol[idx]) > avg

# ── Signal finders ─────────────────────────────────────────────────────────────
def find_vixfix_pairs(df, year_high_bars):
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

def find_stoch_active_set(df, year_high_bars):
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
    bhl   = close <= 0.85 * high.rolling(year_high_bars).max()
    vpv, sdv, nbv, bhlv = vp.values, sd.values, nb.values, bhl.values
    active = set()
    for i in range(LOOKBACK, len(vpv)):
        if not nbv[i] or not bhlv[i]: continue
        if not any(vpv[max(0, i - LOOKBACK):i]): continue
        if not any(sdv[max(0, i - LOOKBACK):i]): continue
        active.add(i)
    return active

# ── Trade evaluation ───────────────────────────────────────────────────────────
def collect_bars(df, signal, hold):
    """Pre-fetch bar data up to hold bars ahead of signal."""
    idx = signal['signal_idx']
    n   = len(df)
    bars = []
    for w in range(1, hold + 1):
        fi = idx + w
        if fi >= n: break
        bars.append({
            'w':     w,
            'high':  float(df['High'].iloc[fi]),
            'low':   float(df['Low'].iloc[fi]),
            'close': float(df['Close'].iloc[fi]),
        })
    return bars

def evaluate_signal(signal, bars, target_pct, hold):
    entry  = signal['entry_price']
    stop   = signal['stop_loss']
    win    = entry * (1 + target_pct)
    shares = POSITION_HIGH / entry
    result, exit_price, exit_bar = 'NEUTRAL', None, None
    for b in bars:
        if b['high'] >= win:  result, exit_price, exit_bar = 'WIN',  win,  b['w']; break
        if b['low']  <= stop: result, exit_price, exit_bar = 'LOSS', stop, b['w']; break
    if result == 'NEUTRAL':
        exit_price = bars[-1]['close'] if bars else entry
        exit_bar   = bars[-1]['w']     if bars else 0
    pct    = (exit_price - entry) / entry * 100
    dollar = shares * (exit_price - entry)
    return {'result': result,
            'pct_return':    pct    if not np.isnan(pct)    else 0.0,
            'dollar_return': dollar if not np.isnan(dollar) else 0.0,
            'exit_bar':      exit_bar or 0}

# ── Run optimization for one configuration ────────────────────────────────────
def run_one_config(tickers, interval, hold, year_high_bars):
    """
    Collects all Tier 1 & 2 signals for the given interval/hold,
    then sweeps EXIT_TARGETS.
    Returns dict: tier -> {target -> [trades]}
    """
    cutoff   = pd.Timestamp.now() - pd.DateOffset(years=YEARS_HISTORY)
    min_bars = max(VF_LB + MAX_GAP,
                   year_high_bars + LOOKBACK + STOCH_LOOKBACK) + hold + 10

    # tier -> {target -> [trades]}
    tier_results = {
        tier: {tgt: [] for tgt in EXIT_TARGETS}
        for tier in ['ultra', 'high', 'tier3_vol']
    }

    for i, ticker in enumerate(tickers):
        try:
            df = yf.download(ticker, period='max', interval=interval,
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

            vf_pairs     = find_vixfix_pairs(df, year_high_bars)
            stoch_active = find_stoch_active_set(df, year_high_bars)

            for sig in vf_pairs:
                si, e = sig['signal_idx'], sig['entry_price']
                if e < rc * 0.15 or e > rc * 6.0: continue
                if si + 1 >= len(df): continue
                has_stoch = any((si + o) in stoch_active
                                for o in range(-SCAN_DELAY, SCAN_DELAY + 1))
                sig['ticker'] = ticker
                bars = collect_bars(df, sig, hold)
                has_vol = volume_above_ma(df, si)
                for tgt in EXIT_TARGETS:
                    t = evaluate_signal(sig, bars, tgt, hold)
                    if sig['has_macd'] and has_stoch:
                        tier_results['ultra'][tgt].append(t)
                    if has_stoch:
                        tier_results['high'][tgt].append(t)
                    if has_stoch and has_vol:
                        tier_results['tier3_vol'][tgt].append(t)

        except Exception as ex:
            print(f'    Error {ticker}: {ex}')
        if (i + 1) % 200 == 0:
            print(f'    [{i+1}/{len(tickers)}]')
        time.sleep(0.04)

    return tier_results

# ── Stats & reporting ─────────────────────────────────────────────────────────
def summarize_tier_for_config(tier_label, description, results_by_target, config_label):
    sep  = '=' * 70
    sep2 = '-' * 70
    bar_unit = 'w' if '1wk' in config_label else 'd'

    summary_rows = []
    lines = [
        sep,
        f'  {tier_label}',
        f'  {description}',
        f'  Config: {config_label} | Position: ${POSITION_HIGH:,.0f}/trade',
        f'  ExpVal = (Win% x Avg Win%) + (Loss% x Avg Loss%)',
        sep2,
        f'  {"Target":>7}  {"Signals":>8}  {"Win%":>7}  {"TotalP&L":>14}  '
        f'{"AvgRet":>8}  {"EV":>8}  {"ROI":>7}  {"AvgHold":>8}',
        sep2,
    ]

    for target in EXIT_TARGETS:
        trades = results_by_target.get(target, [])
        total  = len(trades)
        if total == 0:
            lines.append(f'  {int(target*100):>6}%  {"—":>8}')
            continue
        wins      = [t for t in trades if t['result'] == 'WIN']
        losses    = [t for t in trades if t['result'] == 'LOSS']
        win_rate  = len(wins)   / total * 100
        loss_rate = len(losses) / total * 100
        total_pnl = safe_sum([t['dollar_return'] for t in trades])
        avg_ret   = safe_mean([t['pct_return']   for t in trades])
        avg_win   = safe_mean([t['pct_return']   for t in wins])
        avg_loss  = safe_mean([t['pct_return']   for t in losses])
        exp_val   = (win_rate / 100 * avg_win) + (loss_rate / 100 * avg_loss)
        capital   = total * POSITION_HIGH
        roi       = (total_pnl / capital * 100) if capital > 0 else 0.0
        avg_hold  = safe_mean([t['exit_bar']     for t in trades if t['exit_bar']])

        summary_rows.append({
            'target': target, 'total': total, 'win_rate': win_rate,
            'loss_rate': loss_rate, 'total_pnl': total_pnl, 'avg_ret': avg_ret,
            'exp_val': exp_val, 'avg_win': avg_win, 'avg_loss': avg_loss,
            'roi': roi, 'avg_hold': avg_hold,
        })
        lines.append(
            f'  {int(target*100):>6}%  {total:>8}  {win_rate:>6.1f}%  '
            f'${total_pnl:>+13,.2f}  {avg_ret:>+7.1f}%  {exp_val:>+7.2f}%  '
            f'{roi:>+6.1f}%  {avg_hold:>6.1f}{bar_unit}'
        )

    lines.append(sep2)

    if summary_rows:
        best_pnl = max(summary_rows, key=lambda x: x['total_pnl'])
        best_wr  = max(summary_rows, key=lambda x: x['win_rate'])
        best_ev  = max(summary_rows, key=lambda x: x['exp_val'])

        max_pnl = max(r['total_pnl'] for r in summary_rows) or 1
        max_wr  = max(r['win_rate']  for r in summary_rows) or 1
        max_ev  = max(r['exp_val']   for r in summary_rows) or 1
        for r in summary_rows:
            r['score'] = ((r['total_pnl'] / max_pnl if max_pnl > 0 else 0) * 0.40
                          + r['win_rate'] / max_wr * 0.35
                          + (r['exp_val'] / max_ev if max_ev > 0 else 0) * 0.25)
        best = max(summary_rows, key=lambda x: x['score'])

        lines += [
            '',
            f'  Highest P&L:       {int(best_pnl["target"]*100):>3}% target  '
            f'(${best_pnl["total_pnl"]:+,.2f}  WR {best_pnl["win_rate"]:.1f}%  '
            f'EV {best_pnl["exp_val"]:+.2f}%  ROI {best_pnl["roi"]:+.1f}%)',
            f'  Highest win rate:  {int(best_wr["target"]*100):>3}% target  '
            f'({best_wr["win_rate"]:.1f}%  P&L ${best_wr["total_pnl"]:+,.2f}  '
            f'EV {best_wr["exp_val"]:+.2f}%)',
            f'  Best EV:           {int(best_ev["target"]*100):>3}% target  '
            f'(EV {best_ev["exp_val"]:+.2f}%  WR {best_ev["win_rate"]:.1f}%  '
            f'P&L ${best_ev["total_pnl"]:+,.2f}  ROI {best_ev["roi"]:+.1f}%  '
            f'avg hold {best_ev["avg_hold"]:.1f}{bar_unit})',
            '',
            f'  ★ OVERALL RECOMMENDATION (40% P&L + 35% WR + 25% EV):',
            f'    Best exit target: {int(best["target"]*100)}%  |  '
            f'P&L ${best["total_pnl"]:+,.2f}  |  WR {best["win_rate"]:.1f}%  |  '
            f'EV {best["exp_val"]:+.2f}%  |  ROI {best["roi"]:+.1f}%  |  '
            f'Avg hold {best["avg_hold"]:.1f}{bar_unit}',
        ]
    lines.append(sep)
    return '\n'.join(lines)

def build_full_report(all_results):
    """
    all_results: list of (config_dict, tier_results_dict)
    tier_results_dict: {'ultra': {target: [trades]}, 'high': {target: [trades]}}
    """
    sep = '=' * 70
    tier_configs = [
        ('ultra',     'TIER 1 ★★★ ULTRA    (BB+VixFix+MACD+Stoch)',
                      'All four conditions confirmed. Highest conviction.'),
        ('high',      'TIER 2 ★★  HIGH     (BB+VixFix+Stoch)',
                      'VixFix div + Stochastic div confirmed. MACD not required.'),
        ('tier3_vol', 'TIER 3 ★   HIGH + Volume Filter (BB+VixFix+Stoch+Volume)',
                      'Same as Tier 2 + trigger candle volume > 20-bar average volume.'),
    ]

    lines = [
        sep,
        'EXIT TARGET OPTIMIZATION REPORT',
        f'Tiers 1, 2 & 3  |  Sweep: 5%–50% (5% steps) + 11/12/13/14% granularity',
        f'Position: ${POSITION_HIGH:,.0f}/trade  |  History: {YEARS_HISTORY} years',
        f'Active strategies: 1wk/20-week hold  |  1d/60-day hold',
        sep, '',
    ]

    for cfg, tier_results in all_results:
        lines.append(f'\n{"#"*70}')
        lines.append(f'## CONFIGURATION: {cfg["label"].upper()}')
        lines.append(f'{"#"*70}\n')
        for tier_key, tier_label, description in tier_configs:
            lines.append(summarize_tier_for_config(
                tier_label, description,
                tier_results.get(tier_key, {}),
                cfg['label']
            ))
            lines.append('')

    return '\n'.join(lines)

def send_email(report):
    msg = MIMEMultipart()
    msg['From']    = GMAIL_USER
    msg['To']      = TO_EMAIL
    msg['Subject'] = 'Exit Target Optimization — 1wk/20w & 1d/60d — Tiers 1/2/3'
    msg.attach(MIMEText(report, 'plain'))
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as s:
            s.login(GMAIL_USER, GMAIL_PASSWORD)
            s.sendmail(GMAIL_USER, TO_EMAIL, msg.as_string())
        print(f'Emailed to {TO_EMAIL}')
    except Exception as e:
        print(f'Email failed: {e}')

if __name__ == '__main__':
    tickers     = get_all_tickers()
    all_results = []

    for cfg in CONFIGS:
        print(f'\n{"="*60}')
        print(f'Running: {cfg["label"]}')
        print(f'{"="*60}')
        tier_results = run_one_config(
            tickers,
            interval       = cfg['interval'],
            hold           = cfg['hold'],
            year_high_bars = cfg['year_high_bars'],
        )
        all_results.append((cfg, tier_results))

    report = build_full_report(all_results)
    print('\n' + report)
    if GMAIL_USER:
        send_email(report)
