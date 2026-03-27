"""
scanner_daily.py  —  Daily scanner (interval=1d, Tiers 1 & 2 only, 60-day hold)
Reads historical stats from backtest_stats.json (written by backtest_daily.py).
Run backtest_daily.py first via workflow_dispatch to populate stats, heck yeah.
Sends HTML email. Entry logic: buy at next day's open after signal fires.
"""
import requests, pandas as pd, numpy as np, yfinance as yf
import smtplib, time, os, json
from datetime import date
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

INTERVAL       = '1d'
BAR_LABEL      = 'day'
YEAR_HIGH_BARS = 252
BB_LENGTH      = 20;  BB_MULT    = 2.0
VF_PD          = 30;  VF_BBL     = 20;  VF_MULT = 2.0
VF_LB          = 75;  VF_PH      = 0.85
MAX_GAP        = 35;  SCAN_DELAY = 5;   VF_NEAR = 2
STOCH_LOOKBACK = 25;  STOCH_K    = 14;  LOOKBACK = 10
MACD_FAST      = 12;  MACD_SLOW  = 26;  MACD_SIGNAL = 9
MIN_PRICE      = 10.0;  MIN_MARKET_CAP = 1_000_000_000
MAX_STOP_DIST  = 0.11;  NO_BREAK_BARS  = 10
HOLD_BARS      = 60;  WIN_TARGET = 0.13;  POSITION_HIGH = 10000.0
YEARS_HISTORY  = 15;  STATS_FILE = 'backtest_stats.json'

GMAIL_USER     = os.environ.get('GMAIL_USER', '')
GMAIL_PASSWORD = os.environ.get('GMAIL_PASSWORD', '')
TO_EMAIL       = 'bkcolby@yahoo.com'

# ── Load backtest stats ────────────────────────────────────────────────────────
def load_backtest_stats():
    try:
        with open(STATS_FILE) as f:
            data = json.load(f)
        return data.get('daily_60d', {})
    except Exception as e:
        print(f'Warning: could not read {STATS_FILE}: {e}')
        return {}

# ── Helpers ────────────────────────────────────────────────────────────────────
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

def macd_divergence(pi, ri, ml, sl, hist):
    vals = [hist[pi], hist[ri], ml[pi], ml[ri], sl[pi], sl[ri]]
    if any(np.isnan(v) for v in vals): return False
    return (hist[ri] > hist[pi]) or (ml[ri] > ml[pi]) or (sl[ri] > sl[pi])

# ── Scan functions ─────────────────────────────────────────────────────────────
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
    ll    = low.rolling(STOCH_K).min(); hh = high.rolling(STOCH_K).max()
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

def run_scans(tickers):
    ultra, high = [], []
    min_bars = max(VF_LB + MAX_GAP, YEAR_HIGH_BARS + LOOKBACK + STOCH_LOOKBACK) + 10
    print(f'Scanning {len(tickers)} tickers (daily)...\n')
    for i, ticker in enumerate(tickers):
        try:
            df = yf.download(ticker, period='1y', interval=INTERVAL,
                             progress=False, auto_adjust=True)
            if df is None or len(df) < 60: continue
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
            has_macd_vf, has_vf = check_vixfix(df, ml, sl, hist)
            has_stoch           = check_stoch(df)

            if has_macd_vf and has_vf and has_stoch: ultra.append(ticker)
            if has_vf and has_stoch:                 high.append(ticker)

            tags = []
            if has_macd_vf and has_vf and has_stoch: tags.append('ULTRA')
            elif has_vf and has_stoch:               tags.append('HIGH')
            if tags: print(f'  ✓ {ticker} — {" | ".join(tags)}')
        except Exception as e: print(f'  Error {ticker}: {e}')
        if (i + 1) % 100 == 0: print(f'  [{i+1}/{len(tickers)}]')
        time.sleep(0.05)
    return sorted(ultra), sorted(high)

# ── HTML report ────────────────────────────────────────────────────────────────
def _stats_html(s):
    if not s:
        return ('<p style="margin:6px 0;font-size:0.85em;color:#c00;">'
                '⚠ No backtest data — run <strong>backtest_daily.py</strong> via '
                'workflow_dispatch to populate stats.</p>')
    nr     = s.get('neutral', s['total'] - s['wins'] - s['losses'])
    nr_pct = nr / s['total'] * 100 if s['total'] else 0
    pos    = s.get('position', 10000)
    hold_b = s.get('hold_b', HOLD_BARS)
    return f'''
<table style="border-collapse:collapse;width:100%;font-size:0.82em;margin:6px 0;">
  <tr style="background:#f0f4ff;">
    <td style="padding:3px 8px;"><strong>Signals (15yr)</strong></td>
    <td style="padding:3px 8px;">{s["total"]:,}</td>
    <td style="padding:3px 8px;"><strong>Signals/mo</strong></td>
    <td style="padding:3px 8px;">~{s["spm"]:.1f}</td>
    <td style="padding:3px 8px;"><strong>Position</strong></td>
    <td style="padding:3px 8px;">${pos:,.0f}/trade</td>
  </tr>
  <tr>
    <td style="padding:3px 8px;"><strong>Win rate</strong></td>
    <td style="padding:3px 8px;color:#006600;"><strong>{s["wr"]:.1f}%</strong></td>
    <td style="padding:3px 8px;"><strong>Loss rate</strong></td>
    <td style="padding:3px 8px;color:#cc0000;">{s["lr"]:.1f}%</td>
    <td style="padding:3px 8px;"><strong>Neutral rate</strong></td>
    <td style="padding:3px 8px;">{nr_pct:.1f}%</td>
  </tr>
  <tr style="background:#f0f4ff;">
    <td style="padding:3px 8px;"><strong>Avg win</strong></td>
    <td style="padding:3px 8px;color:#006600;">{s["aw"]:+.1f}%</td>
    <td style="padding:3px 8px;"><strong>Avg loss</strong></td>
    <td style="padding:3px 8px;color:#cc0000;">{s["al"]:+.1f}%</td>
    <td style="padding:3px 8px;"><strong>Exp. value</strong></td>
    <td style="padding:3px 8px;"><strong>{s["ev"]:+.2f}%</strong></td>
  </tr>
  <tr>
    <td style="padding:3px 8px;"><strong>Avg hold</strong></td>
    <td style="padding:3px 8px;">{hold_b:.1f} days</td>
    <td style="padding:3px 8px;"><strong>Total P&amp;L</strong></td>
    <td style="padding:3px 8px;">${s["pnl"]:+,.0f}</td>
    <td style="padding:3px 8px;"><strong>ROI</strong></td>
    <td style="padding:3px 8px;">{s["roi"]:+.1f}%</td>
  </tr>
</table>'''

def _tier_html(title, description, tickers_this_tier, stats, hide_already_in=None):
    excl = set(hide_already_in) if hide_already_in else set()
    show = [t for t in tickers_this_tier if t not in excl]
    if show:
        ticker_html = '&nbsp; '.join(
            f'<span style="color:#cc0000;font-weight:bold;font-size:1.05em;">{t}</span>'
            for t in show)
        count = f'<br><em style="font-size:0.8em;color:#666;">Total: {len(show)}'
        if excl:
            count += f' (excl. higher tiers) | All: {len(tickers_this_tier)}'
        count += '</em>'
        ticker_html += count
    else:
        ticker_html = '<em style="color:#999;">No signals today.</em>'

    return f'''
<div style="border:1px solid #ccc;border-radius:6px;padding:14px 16px;
            margin-bottom:16px;background:#fafafa;">
  <h2 style="margin:0 0 3px 0;font-size:1.05em;color:#111;">{title}</h2>
  <p style="margin:0 0 4px 0;font-size:0.82em;color:#555;">{description}</p>
  <p style="margin:2px 0;font-size:0.78em;color:#777;">
    Strategy: Buy at <strong>next day open</strong> after signal &nbsp;|&nbsp;
    Stop: low of BB trigger candle &nbsp;|&nbsp;
    Win target: {int(WIN_TARGET*100)}% &nbsp;|&nbsp; Max hold: {HOLD_BARS} trading days
  </p>
  {_stats_html(stats)}
  <hr style="border:none;border-top:1px solid #e0e0e0;margin:8px 0;">
  <p style="margin:0;font-size:1em;line-height:2.1;">{ticker_html}</p>
</div>'''

def build_html_report(ultra, high, bt_stats):
    today_str = date.today().strftime('%B %d, %Y')
    s_ultra   = bt_stats.get('ultra') or {}
    s_high    = bt_stats.get('high')  or {}

    summary = ''.join(
        f'<tr><td>{lbl}</td><td style="text-align:center;">{n}</td></tr>\n'
        for lbl, n in [('Tier 1 ★★★ ULTRA (BB+VixFix+MACD+Stoch)', len(ultra)),
                        ('Tier 2 ★★  HIGH  (BB+VixFix+Stoch)',       len(high))]
    )

    ts  = _tier_html('★★★ TIER 1 — ULTRA (1D, 60-day hold)',
                     'BB Trigger + VixFix divergence + MACD divergence + Stochastic divergence.',
                     ultra, s_ultra)
    ts += _tier_html('★★ TIER 2 — HIGH (1D, 60-day hold)',
                     'BB Trigger + VixFix divergence + Stochastic divergence. MACD not required.',
                     high, s_high, hide_already_in=ultra)

    return f'''<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
body{{font-family:Arial,sans-serif;font-size:14px;color:#222;max-width:700px;margin:0 auto;padding:20px;}}
table{{border-collapse:collapse;width:100%;margin-bottom:10px;}}
th,td{{border:1px solid #ccc;padding:5px 10px;font-size:0.85em;}}
th{{background:#f0f0f0;}}
</style></head><body>
<h1 style="font-size:1.3em;margin-bottom:4px;">Daily Stock Scan — {today_str}</h1>
<p style="margin:0 0 4px 0;font-size:0.82em;color:#666;">
  Interval: Daily (1d) &nbsp;|&nbsp; Universe: NYSE + NASDAQ &nbsp;|&nbsp;
  Price &gt;$10, Mkt cap &gt;$1B, Stop dist &lt;11%
</p>
<p style="margin:0 0 16px 0;font-size:0.82em;color:#555;">
  <strong>Timing:</strong> Signal fires after today's close.
  Buy at <strong>tomorrow's market open</strong>. Entry = close of BB trigger candle.
</p>
<h3 style="margin:0 0 6px 0;font-size:0.95em;">Signal Count Today</h3>
<table><tr><th>Tier</th><th>Signals</th></tr>{summary}</table>
{ts}
<p style="font-size:0.72em;color:#aaa;margin-top:16px;">
  Backtest stats sourced from backtest_stats.json (written by backtest_daily.py).
  Re-run backtest_daily.py via workflow_dispatch to refresh.
</p>
</body></html>'''

def send_email(subject, html_body, attachment_path=None):
    msg = MIMEMultipart('mixed')
    msg['From']    = GMAIL_USER
    msg['To']      = TO_EMAIL
    msg['Subject'] = subject
    msg.attach(MIMEText(html_body, 'html'))
    if attachment_path and os.path.exists(attachment_path):
        from email.mime.base import MIMEBase
        from email import encoders
        with open(attachment_path, 'rb') as f:
            part = MIMEBase('application',
                            'vnd.openxmlformats-officedocument.spreadsheetml.sheet')
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', 'attachment',
                        filename=os.path.basename(attachment_path))
        msg.attach(part)
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as s:
            s.login(GMAIL_USER, GMAIL_PASSWORD)
            s.sendmail(GMAIL_USER, TO_EMAIL, msg.as_string())
        print('Email sent.')
    except Exception as e: print(f'Email failed: {e}')

if __name__ == '__main__':
    is_scheduled = os.environ.get('GITHUB_EVENT_NAME') == 'schedule'

    tickers     = get_all_tickers()
    ultra, high = run_scans(tickers)
    bt_stats    = load_backtest_stats()
    html        = build_html_report(ultra, high, bt_stats)

    # Build tracker Excel — add new rows only on scheduled runs
    tracker_path = None
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'trade_tracker', os.path.join(os.path.dirname(__file__), 'trade_tracker.py'))
        tt = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tt)

        history = tt.load_history('trade_history.json')

        # Always resolve any open trades first (checks price history)
        if history:
            history = tt.resolve_open_trades(history)

        if is_scheduled and (ultra or high):
            scan_date = date.today().isoformat()
            new_sigs  = []
            for ticker in ultra:
                new_sigs.append({'scan_date': scan_date, 'ticker': ticker,
                                  'strategy': '1D — Tier 1 Ultra', 'tier': 'Tier 1',
                                  'interval': '1d',
                                  'buy_price': 0.0, 'stop_loss': 0.0})
            for ticker in set(high) - set(ultra):
                new_sigs.append({'scan_date': scan_date, 'ticker': ticker,
                                  'strategy': '1D — Tier 2 High', 'tier': 'Tier 2',
                                  'interval': '1d',
                                  'buy_price': 0.0, 'stop_loss': 0.0})

            # Fetch actual entry/stop prices
            import yfinance as yf_tt
            for sig in new_sigs:
                try:
                    df = yf_tt.download(sig['ticker'], period='5d', interval='1d',
                                        progress=False, auto_adjust=True)
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    if len(df) >= 1:
                        sig['buy_price'] = round(float(df['Close'].iloc[-1]), 4)
                        sig['stop_loss'] = round(float(df['Low'].iloc[-1]),   4)
                except Exception as e:
                    print(f'  Price fetch failed for {sig["ticker"]}: {e}')

            history = tt.add_signals(history, new_sigs)
            tt.save_history(history, 'trade_history.json')
            print(f'Added {len(new_sigs)} signals to trade_history.json')

        tracker_path = '/tmp/trade_tracker.xlsx'
        tt.build_excel(history, tracker_path)
        print(f'Tracker built: {len(history)} total trades')
    except Exception as e:
        print(f'Tracker build failed: {e}')

    print(f'\n[Done] ULTRA:{len(ultra)}  HIGH:{len(high)}')
    if GMAIL_USER:
        signal_count = len(ultra) + len(set(high) - set(ultra))
        subject = (f'Daily Stock Scan (1d) — {date.today().strftime("%b %d %Y")} '
                   f'— {signal_count} signal{"s" if signal_count != 1 else ""} found')
        send_email(subject, html, attachment_path=tracker_path)
    else:
        print(html[:1500])
