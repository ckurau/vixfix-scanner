"""
backtest_daily.py  —  Daily (1d) backtest, Tiers 1 & 2 only.
Compares 20-day, 30-day, and 60-day hold periods side by side.
15 years of history. Emails plain-text report.
"""
import requests, pandas as pd, numpy as np, yfinance as yf
import smtplib, time, os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

INTERVAL       = '1d'
BAR_LABEL      = 'day'
YEAR_HIGH_BARS = 252
YEARS_HISTORY  = 15

BB_LENGTH      = 20;  BB_MULT   = 2.0
VF_PD          = 30;  VF_BBL    = 20;  VF_MULT = 2.0
VF_LB          = 75;  VF_PH     = 0.85
MAX_GAP        = 35;  SCAN_DELAY = 5;  VF_NEAR = 2
STOCH_LOOKBACK = 25;  STOCH_K   = 14
LOOKBACK       = 10
MACD_FAST      = 12;  MACD_SLOW = 26;  MACD_SIGNAL = 9

HOLD_PERIODS   = [20, 30, 60]   # days — compared side by side
WIN_TARGET     = 0.13
POSITION_HIGH  = 10000.0

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

# ── Trade evaluation (parameterised hold_bars) ─────────────────────────────────
def eval_trade(df, signal, position_size, hold_bars):
    idx, entry, stop = signal['signal_idx'], signal['entry_price'], signal['stop_loss']
    n, win = len(df), entry * (1 + WIN_TARGET)
    shares = position_size / entry
    result, exit_price, exit_bar = 'NEUTRAL', None, None
    for w in range(1, hold_bars + 1):
        fi = idx + w
        if fi >= n: break
        wh, wl = float(df['High'].iloc[fi]), float(df['Low'].iloc[fi])
        if wh >= win:  result, exit_price, exit_bar = 'WIN',  win,  w; break
        if wl <= stop: result, exit_price, exit_bar = 'LOSS', stop, w; break
    if result == 'NEUTRAL':
        last       = min(idx + hold_bars, n - 1)
        exit_price = float(df['Close'].iloc[last])
        exit_bar   = min(hold_bars, n - 1 - idx)
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
    """
    Returns results dict keyed by (tier, hold_bars).
    tier in ['ultra', 'high']
    hold_bars in HOLD_PERIODS
    """
    cutoff   = pd.Timestamp.now() - pd.DateOffset(years=YEARS_HISTORY)
    max_hold = max(HOLD_PERIODS)
    min_bars = max(VF_LB + MAX_GAP,
                   YEAR_HIGH_BARS + LOOKBACK + STOCH_LOOKBACK) + max_hold + 10

    results = {(tier, hold): []
               for tier in ['ultra', 'high']
               for hold in HOLD_PERIODS}

    print(f'Backtesting {len(tickers)} tickers  '
          f'({YEARS_HISTORY}yr, {INTERVAL}, holds={HOLD_PERIODS}, {int(WIN_TARGET*100)}% target)...')

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
                for hold in HOLD_PERIODS:
                    t = eval_trade(df, sig, POSITION_HIGH, hold)
                    if t is None: continue
                    if sig['has_macd'] and has_stoch:
                        results[('ultra', hold)].append(t)
                    if has_stoch:
                        results[('high', hold)].append(t)

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
        print(f'SPY error: {e}')
        return {}

# ── Stats helpers ──────────────────────────────────────────────────────────────
def bucket_stats(trades):
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
    capital = total * POSITION_HIGH
    roi     = (pnl / capital * 100) if capital > 0 else 0.0
    hold_b  = safe_mean([t['exit_bar'] for t in trades if t['exit_bar'] is not None])
    return dict(total=total, wins=len(wins), losses=len(losses), neutral=len(neutral),
                wr=wr, lr=lr, aw=aw, al=al, ev=ev, pnl=pnl, roi=roi, hold_b=hold_b)

def signals_per_month(trades):
    return len(trades) / (YEARS_HISTORY * 12)

def yearly_stats(trades):
    by_year = {}
    for t in trades:
        yr = t.get('year', 0)
        by_year.setdefault(yr, []).append(t)
    out = {}
    for yr, tlist in sorted(by_year.items()):
        s = bucket_stats(tlist)
        if s: out[yr] = s
    return out

# ── Report builders ────────────────────────────────────────────────────────────
def hold_comparison_table(tier_label, results, tier_key):
    """
    Side-by-side table comparing all HOLD_PERIODS for one tier.
    """
    sep  = '=' * 82
    sep2 = '-' * 82
    bl   = BAR_LABEL[0]

    lines = [sep, f'  {tier_label}  —  HOLD PERIOD COMPARISON', sep2,
             f'  {"Metric":<28}' + ''.join(f'  {"Hold="+str(h)+bl:>12}' for h in HOLD_PERIODS),
             sep2]

    rows = {}
    for hold in HOLD_PERIODS:
        s = bucket_stats(results.get((tier_key, hold), []))
        rows[hold] = s

    def row(label, fn):
        vals = []
        for hold in HOLD_PERIODS:
            s = rows[hold]
            vals.append(fn(s) if s else 'N/A')
        return f'  {label:<28}' + ''.join(f'  {v:>12}' for v in vals)

    lines.append(row('Signals',      lambda s: f'{s["total"]}'))

    # Signals/month: computed per-hold directly (signals are same count regardless of hold)
    spm_vals = [f'{len(results.get((tier_key, hold), []))/(YEARS_HISTORY*12):.1f}/mo'
                for hold in HOLD_PERIODS]
    lines.append(f'  {"Signals/month":<28}' + ''.join(f'  {v:>12}' for v in spm_vals))

    lines.append(row('Win rate',     lambda s: f'{s["wr"]:.1f}%'))
    lines.append(row('Loss rate',    lambda s: f'{s["lr"]:.1f}%'))
    lines.append(row('Neutral rate', lambda s: f'{100-s["wr"]-s["lr"]:.1f}%'))
    lines.append(row('Avg win %',    lambda s: f'{s["aw"]:+.1f}%'))
    lines.append(row('Avg loss %',   lambda s: f'{s["al"]:+.1f}%'))
    lines.append(row('Exp. value',   lambda s: f'{s["ev"]:+.2f}%'))
    lines.append(row('Avg hold',     lambda s: f'{s["hold_b"]:.1f}{bl}'))
    lines.append(row('Total P&L',    lambda s: f'${s["pnl"]:+,.0f}'))
    lines.append(row('ROI',          lambda s: f'{s["roi"]:+.1f}%'))
    lines.append(sep2)

    # Winner call
    best_hold, best_ev = None, -999
    for hold in HOLD_PERIODS:
        s = rows[hold]
        if s and s['ev'] > best_ev:
            best_ev   = s['ev']
            best_hold = hold
    if best_hold:
        lines.append(f'  ★  Best hold by EV: {best_hold}{bl}  '
                     f'(EV {best_ev:+.2f}%  '
                     f'WR {rows[best_hold]["wr"]:.1f}%  '
                     f'P&L ${rows[best_hold]["pnl"]:+,.0f})')
    lines.append(sep)
    return '\n'.join(lines) + '\n'

def yearly_table(tier_label, yr_stats, spy_annual, hold_bars):
    if not yr_stats: return f'  {tier_label} [{hold_bars}{BAR_LABEL[0]}]: no yearly data\n'
    sep = '-' * 72
    rows = [
        f'  {tier_label}  [{hold_bars}-{BAR_LABEL} hold]',
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
    sep  = '=' * 82
    sep2 = '-' * 82

    tiers = [
        ('ultra', 'TIER 1 ★★★ ULTRA    (BB+VixFix+MACD+Stoch)'),
        ('high',  'TIER 2 ★★  HIGH     (BB+VixFix+Stoch)      '),
    ]

    lines = [
        sep,
        f'BACKTEST REPORT — DAILY (1d)  |  {YEARS_HISTORY} years  |  '
        f'{int(WIN_TARGET*100)}% win target  |  Tiers 1 & 2 only',
        f'Hold periods compared: {HOLD_PERIODS[0]}d vs {HOLD_PERIODS[1]}d vs {HOLD_PERIODS[2]}d',
        sep,
        f'  Entry: close of BB trigger candle | Stop: low of BB trigger candle',
        f'  Position: ${POSITION_HIGH:,.0f}/trade (both tiers)',
        f'  Filters: Price >${MIN_PRICE:.0f} | Mkt cap >$1B | Stop dist <11%',
        f'  YEAR_HIGH_BARS = {YEAR_HIGH_BARS} (≈ {YEAR_HIGH_BARS} trading days per year)',
        f'  ExpVal = (Win% × Avg Win%) + (Loss% × Avg Loss%)',
        f'  ROI = Total P&L ÷ (Signals × ${POSITION_HIGH:,.0f})',
        sep, '',
    ]

    # ── Hold comparison tables ─────────────────────────────────────────────────
    lines.append('── HOLD PERIOD COMPARISON ───────────────────────────────────────────────────────\n')
    for tier_key, lbl in tiers:
        lines.append(hold_comparison_table(lbl, results, tier_key))

    # ── Year-by-year vs SPY (one table per tier × hold) ───────────────────────
    lines += [sep2,
              '── YEAR-BY-YEAR PERFORMANCE vs SPY ─────────────────────────────────────────────',
              '   ROI  = tier P&L ÷ capital deployed that year',
              '   SPY% = SPY annual price return',
              '   vs SPY = tier ROI% − SPY%  (positive = outperformed)',
              '']
    for tier_key, lbl in tiers:
        for hold in HOLD_PERIODS:
            trades = results.get((tier_key, hold), [])
            ys = yearly_stats(trades)
            lines.append(yearly_table(lbl.strip(), ys, spy_annual, hold))

    # ── Trade histories (20d only — full list for reference) ──────────────────
    lines += [sep2,
              f'── TRADE HISTORIES ({HOLD_PERIODS[0]}-DAY HOLD) ─────────────────────────────────────────',
              '   (20-day hold shown; adjust HOLD_PERIODS[0] to see others)',
              '']
    for tier_key, lbl in tiers:
        trades = results.get((tier_key, HOLD_PERIODS[0]), [])
        lines.append(trade_history_block(
            trades,
            f'── {lbl.strip()} — {HOLD_PERIODS[0]}{BAR_LABEL[0]} hold ─────────────────────────────'))

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
    except Exception as e: print(f'Email failed: {e}')

if __name__ == '__main__':
    tickers    = get_all_tickers()
    results    = run_backtest(tickers)
    spy_annual = get_spy_annual_returns()
    report     = build_report(results, spy_annual)
    print('\n' + report)
    if GMAIL_USER:
        send_email(f'Backtest Report — Daily  |  {HOLD_PERIODS[0]}d / {HOLD_PERIODS[1]}d / {HOLD_PERIODS[2]}d hold comparison',
                   report)
