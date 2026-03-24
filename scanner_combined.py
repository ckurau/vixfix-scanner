import requests
import pandas as pd
import numpy as np
import yfinance as yf
import smtplib
import time
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

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
MIN_PRICE      = 10.0
MIN_MARKET_CAP = 1_000_000_000
MAX_STOP_DIST  = 0.11
NO_BREAK_BARS  = 10

GMAIL_USER     = os.environ.get('GMAIL_USER', '')
GMAIL_PASSWORD = os.environ.get('GMAIL_PASSWORD', '')
TO_EMAIL       = 'bkcolby@yahoo.com'

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

# ── VixFix scan: returns (has_macd_div, has_vixfix_div) ──────────────────────
def check_vixfix(df, macd_line, signal_line, histogram):
    """
    Looks for two VixFix-confirmed BB trigger candles forming a divergence:
      recent price low < prior price low (lower low in price)
      recent WVF > prior WVF (higher WVF = stronger fear spike)
    Returns (has_macd_divergence, has_vixfix_divergence).
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

    recent_idx = None
    for i in range(n - 1, max(n - SCAN_DELAY - 2, -1), -1):
        if twvf[i]:
            recent_idx = i
            break

    if recent_idx is None:
        return False, False

    recent_low   = low_v[recent_idx]
    recent_close = close_v[recent_idx]
    recent_wvf   = wvf_v[recent_idx]

    if np.isnan(recent_low) or np.isnan(recent_wvf):
        return False, False
    if not no_break_before(low_v, recent_idx, NO_BREAK_BARS):
        return False, False
    if (recent_close - recent_low) / recent_close > MAX_STOP_DIST:
        return False, False
    if not no_break_after(low_v, recent_idx, n - 1):
        return False, False

    for j in range(recent_idx - 1, max(recent_idx - MAX_GAP, 0) - 1, -1):
        if not twvf[j]:
            continue
        prior_low = low_v[j]
        prior_wvf = wvf_v[j]
        if np.isnan(prior_low) or np.isnan(prior_wvf):
            continue
        if not no_break_before(low_v, j, NO_BREAK_BARS):
            continue
        if recent_low < prior_low and recent_wvf > prior_wvf:
            has_macd = macd_divergence(j, recent_idx, macd_line, signal_line, histogram)
            return has_macd, True
        break

    return False, False

# ── Stochastic scan ────────────────────────────────────────────────────────────
def check_stoch(df):
    """
    BB trigger candle fires (green, low ≤ BB lower), confirmed by next candle.
    Stochastic divergence (price lower low + stoch higher low) within prior
    10 bars. No-break rule applied. Below 85% of 52-week high.
    """
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

    if n < LOOKBACK + 1:
        return False
    if not nb[-1] or not bhl[-1]:
        return False
    if not any(vp[max(0, n - LOOKBACK):n]):
        return False
    if not any(sd[max(0, n - LOOKBACK):n]):
        return False
    return True

# ── BB-only scan ───────────────────────────────────────────────────────────────
def check_bb_only(df):
    """
    BB trigger candle fires (green, low ≤ BB lower) and is confirmed by the
    next candle. No Stochastic, VixFix, or MACD required. No-break rule applied.
    """
    close = df['Close']
    low   = df['Low']
    open_ = df['Open']

    bb_mid   = close.rolling(BB_LENGTH).mean()
    bb_std   = close.rolling(BB_LENGTH).std(ddof=0)
    bb_lower = bb_mid - BB_MULT * bb_std
    trigger  = (close > open_) & (low <= bb_lower)

    next_green       = (close.shift(-1) > open_.shift(-1))
    next_open_above  = (open_.shift(-1) >= open_)
    next_close_above = (close.shift(-1) >= open_)
    trigger_confirmed = trigger & next_green & next_open_above & next_close_above

    trigger_low = low.where(trigger).ffill()
    no_break    = low.rolling(LOOKBACK).min() >= trigger_low

    tc = trigger_confirmed.values
    nb = no_break.values
    n  = len(tc)

    if n < LOOKBACK + 1:
        return False
    # Check if the most recent bar (or within SCAN_DELAY) had a confirmed trigger
    for i in range(n - 1, max(n - SCAN_DELAY - 2, -1), -1):
        if tc[i] and nb[i]:
            cl = df['Close'].values[i]
            lw = df['Low'].values[i]
            if not np.isnan(cl) and not np.isnan(lw):
                if (cl - lw) / cl <= MAX_STOP_DIST:
                    return True
    return False

# ── Stochastic + MACD scan ────────────────────────────────────────────────────
def check_stoch_macd(df, macd_line, signal_line, histogram):
    """
    BB trigger candle fires and is confirmed. Stochastic divergence active
    within prior 10 bars. MACD histogram or line/signal higher now vs 10
    bars ago. No VixFix required.
    """
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

    if n < LOOKBACK + 2:
        return False
    if not nb[-1] or not bhl[-1]:
        return False
    if not any(vp[max(0, n - LOOKBACK):n]):
        return False
    if not any(sd[max(0, n - LOOKBACK):n]):
        return False

    # MACD: current vs LOOKBACK bars ago
    i      = n - 1
    past_i = max(0, i - LOOKBACK)
    if np.isnan(histogram[i]) or np.isnan(histogram[past_i]):
        return False
    macd_higher = (histogram[i] > histogram[past_i]) or \
                  (macd_line[i] > macd_line[past_i]) or \
                  (signal_line[i] > signal_line[past_i])
    return macd_higher

# ── Main scan ──────────────────────────────────────────────────────────────────
def run_scans(tickers):
    ultra_results    = []   # VixFix div + MACD div + Stoch div
    high_results     = []   # VixFix div + Stoch div (no MACD)
    standard_results = []   # Stoch div only
    bb_only_results  = []   # BB trigger only (Tier 3B)
    stoch_macd_results = [] # Stoch + MACD (Tier 3C)

    min_bars = max(VF_LB + MAX_GAP, YEAR_HIGH_BARS + LOOKBACK + STOCH_LOOKBACK) + 10

    print(f'Scanning {len(tickers)} tickers...\n')

    for i, ticker in enumerate(tickers):
        try:
            df = yf.download(ticker, period='3y', interval='1wk', progress=False, auto_adjust=True)
            if df is None or len(df) < min_bars:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            close_clean = df['Close'].dropna()
            if close_clean.empty:
                continue
            current_price = float(close_clean.iloc[-1])
            if np.isnan(current_price) or current_price < MIN_PRICE:
                continue

            try:
                mc = yf.Ticker(ticker).fast_info.market_cap
                if mc is not None and mc < MIN_MARKET_CAP:
                    continue
            except:
                pass

            macd_line, signal_line, histogram = compute_macd(df['Close'])
            has_macd_vf, has_vf = check_vixfix(df, macd_line, signal_line, histogram)
            has_stoch           = check_stoch(df)
            has_bb              = check_bb_only(df)
            has_stoch_macd      = check_stoch_macd(df, macd_line, signal_line, histogram)

            # Tier 1 ULTRA: VixFix + MACD + Stoch
            if has_macd_vf and has_vf and has_stoch:
                ultra_results.append(ticker)
            # Tier 2 HIGH: VixFix + Stoch (no MACD requirement)
            if has_vf and has_stoch:
                high_results.append(ticker)
            # Tier 3 STANDARD: Stoch only
            if has_stoch:
                standard_results.append(ticker)
            # Tier 3B: BB trigger only
            if has_bb:
                bb_only_results.append(ticker)
            # Tier 3C: Stoch + MACD
            if has_stoch_macd:
                stoch_macd_results.append(ticker)

            tags = []
            if has_macd_vf and has_vf and has_stoch: tags.append('ULTRA')
            elif has_vf and has_stoch:               tags.append('HIGH')
            elif has_stoch:                          tags.append('Stoch')
            if has_bb:                               tags.append('BB-only')
            if has_stoch_macd:                       tags.append('Stoch+MACD')
            if tags:
                print(f'  ✓ {ticker} — {" | ".join(tags)}')

        except Exception as e:
            print(f'  Error on {ticker}: {e}')

        if (i + 1) % 100 == 0:
            print(f'  [{i + 1}/{len(tickers)}] scanned...')

        time.sleep(0.05)

    return (sorted(ultra_results), sorted(high_results), sorted(standard_results),
            sorted(bb_only_results), sorted(stoch_macd_results))

# ── Report builder ─────────────────────────────────────────────────────────────
def build_report(ultra, high, standard, bb_only, stoch_macd,
                 ultra_wr=None,    ultra_signals=None,    ultra_spm=None,
                 high_wr=None,     high_signals=None,     high_spm=None,
                 std_wr=None,      std_signals=None,      std_spm=None,
                 bb_wr=None,       bb_signals=None,       bb_spm=None,
                 sm_wr=None,       sm_signals=None,       sm_spm=None):

    sep = '=' * 60

    def wr_line(wr, signals, spm):
        if wr is None:
            return 'Win rate: run backtest to determine'
        spm_str = f' | ~{spm:.1f} signals/month' if spm is not None else ''
        return f'Backtest: {wr:.1f}% win rate | {signals} signals | 15yr{spm_str}'

    def pos_note(wr):
        if wr is None:
            return '$5,000/trade (run backtest — may qualify for $10,000)'
        return '$10,000/trade' if wr > 80 else '$5,000/trade'

    lines = [
        sep,
        '★★★ TIER 1 — ULTRA CONFIDENCE ★★★',
        'BB Trigger + VixFix divergence + MACD divergence + Stochastic divergence',
        '',
        '  WHAT IT LOOKS FOR:',
        '  • BB Trigger candle: weekly green candle (close > open) whose low',
        '    touches/pierces the BB lower band (20-period, 2.0 std). Confirmed',
        '    by next candle: also green, opens >= trigger open, closes >= trigger open.',
        '  • VixFix divergence: two trigger candles where recent price low <',
        '    prior price low, AND recent WVF value > prior WVF (lower price, more fear).',
        '  • MACD divergence: MACD histogram OR line/signal line is higher',
        '    at the recent trigger vs the prior trigger.',
        '  • Stochastic divergence: price makes a lower low but Stochastic K',
        '    makes a higher low — active within ±5 bars of the trigger.',
        '  Entry: close of BB trigger candle | Stop: low of BB trigger candle',
        '',
        f'  {wr_line(ultra_wr, ultra_signals, ultra_spm)}',
        f'  Position: {pos_note(ultra_wr)} | Win target: 13% | Max hold: 20 weeks',
        '  Filters: Price >$10 | Mkt cap >$1B | Stop dist <11%',
        '           No bar in prior 10 bars below trigger low',
        '           No bar after trigger broke below trigger low',
        sep,
    ]
    if ultra:
        for t in ultra:
            lines.append(f'  {t}')
        lines.append(f'Total: {len(ultra)}')
    else:
        lines.append('  No Tier 1 signals this week.')

    lines += ['', sep,
        '★★ TIER 2 — HIGH CONFIDENCE ★★',
        'BB Trigger + VixFix divergence + Stochastic divergence (MACD not required)',
        '',
        '  WHAT IT LOOKS FOR:',
        '  • BB Trigger candle: same as Tier 1 (green, low ≤ BB lower, confirmed).',
        '  • VixFix divergence: same as Tier 1 (lower price low + higher WVF).',
        '  • Stochastic divergence: price lower low + stoch higher low, within',
        '    ±5 bars. MACD divergence is NOT required.',
        '  Entry: close of BB trigger candle | Stop: low of BB trigger candle',
        '',
        f'  {wr_line(high_wr, high_signals, high_spm)}',
        f'  Position: {pos_note(high_wr)} | Win target: 13% | Max hold: 20 weeks',
        '  Filters: Price >$10 | Mkt cap >$1B | Stop dist <11%',
        '           No bar in prior 10 bars below trigger low',
        '           No bar after trigger broke below trigger low',
        sep,
    ]
    # High = VixFix+Stoch but exclude ULTRA (already shown above)
    high_excl_ultra = sorted(set(high) - set(ultra))
    if high_excl_ultra:
        for t in high_excl_ultra:
            lines.append(f'  {t}')
        lines.append(f'Total (excl. Tier 1): {len(high_excl_ultra)} | All HIGH: {len(high)}')
    else:
        lines.append('  No additional Tier 2 signals this week.')

    lines += ['', sep,
        '★ TIER 3 — STANDARD SIGNALS ★',
        'BB Trigger + Stochastic divergence (no VixFix or MACD required)',
        '',
        '  WHAT IT LOOKS FOR:',
        '  • BB Trigger candle: weekly green candle (close > open) whose low',
        '    touches/pierces the BB lower band. Confirmed by next candle.',
        '  • Stochastic divergence: within the prior 10 bars, price made a',
        '    lower low while Stochastic K made a higher low.',
        '  • No-break rule: no bar in prior 10 bars (or after trigger) broke',
        '    below the trigger candle low. Price below 85% of 52-week high.',
        '  No VixFix or MACD required.',
        '  Entry: close of BB trigger candle | Stop: low of BB trigger candle',
        '',
        f'  {wr_line(std_wr, std_signals, std_spm)}',
        f'  Position: {pos_note(std_wr)} | Win target: 13% | Max hold: 20 weeks',
        sep,
    ]
    std_excl = sorted(set(standard) - set(high))
    if std_excl:
        for t in std_excl:
            lines.append(f'  {t}')
        lines.append(f'Total (excl. Tiers 1 & 2): {len(std_excl)} | All STANDARD: {len(standard)}')
    else:
        lines.append('  No additional Tier 3 signals this week.')

    lines += ['', sep,
        '★ TIER 3B — STANDARD-BB (BB Trigger only)',
        'BB Trigger candle confirmed by next candle — no other indicators required',
        '',
        '  WHAT IT LOOKS FOR:',
        '  • BB Trigger candle: weekly green candle (close > open) whose low',
        '    touches/pierces the BB lower band (20-period, 2.0 std).',
        '  • Confirmation candle: next candle also green, opens >= trigger open,',
        '    closes >= trigger open.',
        '  • No Stochastic, VixFix, or MACD required.',
        '  • No-break rule: no bar in prior 10 bars broke below trigger low.',
        '  • Stop distance must be <11%.',
        '  Entry: close of BB trigger candle | Stop: low of BB trigger candle',
        '  (Baseline: measures value of the BB touch trigger alone)',
        '',
        f'  {wr_line(bb_wr, bb_signals, bb_spm)}',
        f'  Position: {pos_note(bb_wr)} | Win target: 13% | Max hold: 20 weeks',
        sep,
    ]
    bb_excl = sorted(set(bb_only) - set(standard) - set(high))
    if bb_excl:
        for t in bb_excl:
            lines.append(f'  {t}')
        lines.append(f'Total (excl. higher tiers): {len(bb_excl)} | All BB-only: {len(bb_only)}')
    else:
        lines.append('  No additional Tier 3B signals this week.')

    lines += ['', sep,
        '★ TIER 3C — STANDARD-MACD (BB Trigger + Stochastic + MACD divergence)',
        'BB Trigger + Stochastic divergence + MACD divergence (no VixFix required)',
        '',
        '  WHAT IT LOOKS FOR:',
        '  • BB Trigger candle: same as Tier 3 (green, low ≤ BB lower, confirmed).',
        '  • Stochastic divergence: price lower low + stoch higher low within',
        '    prior 10 bars.',
        '  • MACD divergence: MACD histogram or line/signal is higher now',
        '    vs 10 bars ago. (No VixFix pair required.)',
        '  • No-break rule and 85% of 52-week high filter applied.',
        '  Entry: close of BB trigger candle | Stop: low of BB trigger candle',
        '',
        f'  {wr_line(sm_wr, sm_signals, sm_spm)}',
        f'  Position: {pos_note(sm_wr)} | Win target: 13% | Max hold: 20 weeks',
        sep,
    ]
    sm_excl = sorted(set(stoch_macd) - set(high) - set(ultra))
    if sm_excl:
        for t in sm_excl:
            lines.append(f'  {t}')
        lines.append(f'Total (excl. higher tiers): {len(sm_excl)} | All Stoch+MACD: {len(stoch_macd)}')
    else:
        lines.append('  No additional Tier 3C signals this week.')

    lines += ['', sep, 'FULL SCAN SUMMARY', sep,
        f'  Tier 1  ULTRA         (BB+VixFix+MACD+Stoch): {len(ultra)}',
        f'  Tier 2  HIGH          (BB+VixFix+Stoch):       {len(high)}',
        f'  Tier 3  STANDARD      (BB+Stoch):              {len(standard)}',
        f'  Tier 3B STD-BB        (BB Trigger only):       {len(bb_only)}',
        f'  Tier 3C STD-MACD      (BB+Stoch+MACD):        {len(stoch_macd)}',
        sep,
    ]
    return '\n'.join(lines)

def send_email(report):
    msg = MIMEMultipart()
    msg['From']    = GMAIL_USER
    msg['To']      = TO_EMAIL
    msg['Subject'] = 'Weekly Stock Scan Results'
    msg.attach(MIMEText(report, 'plain'))
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as s:
            s.login(GMAIL_USER, GMAIL_PASSWORD)
            s.sendmail(GMAIL_USER, TO_EMAIL, msg.as_string())
        print('\nEmail sent successfully.')
    except Exception as e:
        print(f'\nEmail failed: {e}')

if __name__ == '__main__':
    tickers = get_all_tickers()
    ultra, high, standard, bb_only, stoch_macd = run_scans(tickers)

    # ── Plug in backtest stats below once you have them ────────────────────────
    # Format: wr = win rate (float), signals = total signals (int),
    #         spm = signals per month (float)
    report = build_report(
        ultra, high, standard, bb_only, stoch_macd,
        ultra_wr=None,  ultra_signals=None,  ultra_spm=None,
        high_wr=None,   high_signals=None,   high_spm=None,
        std_wr=None,    std_signals=None,    std_spm=None,
        bb_wr=None,     bb_signals=None,     bb_spm=None,
        sm_wr=None,     sm_signals=None,     sm_spm=None,
    )

    print('\n' + report)
    if GMAIL_USER:
        send_email(report)
