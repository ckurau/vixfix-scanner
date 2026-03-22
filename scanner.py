import requests
import pandas as pd
import numpy as np
import yfinance as yf
import smtplib
import time
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ── Your exact TOS settings ────────────────────────────────────────────────────
BB_LENGTH  = 20
BB_MULT    = 2.0
VF_PD      = 30
VF_BBL     = 20
VF_MULT    = 2.0
VF_LB      = 75
VF_PH      = 0.85
MAX_GAP    = 35
SCAN_DELAY = 3
VF_NEAR    = 2

# ── Email settings ─────────────────────────────────────────────────────────────
GMAIL_USER     = os.environ['GMAIL_USER']
GMAIL_PASSWORD = os.environ['GMAIL_PASSWORD']
TO_EMAIL       = 'bkcolby@yahoo.com'

# ── Step 1: Fetch tickers ──────────────────────────────────────────────────────
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
                    tickers.append(sym)
        except Exception as e:
            print(f'Error fetching {exchange}: {e}')
    tickers = list(set(tickers))
    print(f'Total tickers fetched: {len(tickers)}')
    return tickers

# ── Step 2: Compute indicators ─────────────────────────────────────────────────
def compute_indicators(df):
    close = df['Close']
    low   = df['Low']
    open_ = df['Open']

    # Bollinger Bands
    bb_mid   = close.rolling(BB_LENGTH).mean()
    bb_std   = close.rolling(BB_LENGTH).std(ddof=0)
    bb_lower = bb_mid - BB_MULT * bb_std

    # Trigger candle
    trigger = (close > open_) & (low <= bb_lower)

    # Confirmation candle
    next_green       = (close.shift(-1) > open_.shift(-1))
    next_open_above  = (open_.shift(-1) >= open_)
    next_close_above = (close.shift(-1) >= open_)
    confirmation     = next_green & next_open_above & next_close_above
    trigger_confirmed = trigger & confirmation

    # CMWilliams VixFix
    hc       = close.rolling(VF_PD).max()
    wvf      = (hc - low) / hc * 100
    vf_mid   = wvf.rolling(VF_BBL).mean()
    vf_std   = wvf.rolling(VF_BBL).std(ddof=0)
    vf_upper = vf_mid + VF_MULT * vf_std
    vf_range = wvf.rolling(VF_LB).max() * VF_PH
    is_green = (wvf >= vf_upper) | (wvf >= vf_range)

    # VixFix within VF_NEAR bars of trigger
    vf_near = pd.Series(False, index=df.index)
    for shift in range(VF_NEAR + 1):
        vf_near = vf_near | is_green.shift(shift).fillna(False).infer_objects(copy=False).astype(bool)
        if shift > 0:
            vf_near = vf_near | is_green.shift(-shift).fillna(False).infer_objects(copy=False).astype(bool)

    trigger_with_vf = trigger_confirmed & vf_near

    # Peak WVF near each trigger
    wvf_at_trigger = pd.Series(np.nan, index=df.index)
    for shift in range(-VF_NEAR, VF_NEAR + 1):
        shifted = wvf.shift(shift).fillna(0)
        wvf_at_trigger = wvf_at_trigger.combine(shifted, lambda a, b: b if np.isnan(a) else max(a, b))

    return trigger_with_vf, wvf_at_trigger, low

# ── Step 3: Check divergence ───────────────────────────────────────────────────
def check_divergence(trigger_with_vf, wvf_at_trigger, low):
    twvf  = trigger_with_vf.values
    wvf_v = wvf_at_trigger.values
    low_v = low.values
    n     = len(twvf)

    recent_idx = None
    for i in range(n - 1, max(n - SCAN_DELAY - 2, -1), -1):
        if twvf[i]:
            recent_idx = i
            break

    if recent_idx is None:
        return False

    recent_low = low_v[recent_idx]
    recent_wvf = wvf_v[recent_idx]

    if np.isnan(recent_low) or np.isnan(recent_wvf):
        return False

    search_start = max(recent_idx - MAX_GAP, 0)
    for j in range(recent_idx - 1, search_start - 1, -1):
        if not twvf[j]:
            continue
        prior_low = low_v[j]
        prior_wvf = wvf_v[j]
        if np.isnan(prior_low) or np.isnan(prior_wvf):
            continue
        if recent_low < prior_low and recent_wvf > prior_wvf:
            return True

    return False

# ── Step 4: Run scan ───────────────────────────────────────────────────────────
def run_scan(tickers):
    results = []
    total   = len(tickers)
    min_bars = VF_LB + MAX_GAP + 10

    print(f'Scanning {total} tickers...')

    for i, ticker in enumerate(tickers):
        try:
            df = yf.download(ticker, period='3y', interval='1wk', progress=False, auto_adjust=True)

            if df is None or len(df) < min_bars:
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            trigger_with_vf, wvf_at_trigger, low = compute_indicators(df)

            if check_divergence(trigger_with_vf, wvf_at_trigger, low):
                results.append(ticker)
                print(f'  ✓ {ticker}')

        except Exception as e:
            print(f'  Error on {ticker}: {e}')

        if (i + 1) % 100 == 0:
            print(f'  [{i + 1}/{total}] scanned — {len(results)} matches so far...')

        time.sleep(0.05)

    print(f'\nScan complete. {len(results)} matches found.')
    return results

# ── Step 5: Send email ─────────────────────────────────────────────────────────
def send_email(results):
    subject = 'BB + VixFix Divergence Scan Results'

    if results:
        body = 'Weekly BB + CMWilliams VixFix Divergence Scan Results\n'
        body += '=' * 45 + '\n'
        body += '\n'.join(sorted(results))
        body += f'\n\n{"=" * 45}'
        body += f'\nTotal matches: {len(results)}'
    else:
        body = 'Weekly BB + CMWilliams VixFix Divergence Scan\n'
        body += 'No matches found this week.'

    msg = MIMEMultipart()
    msg['From']    = GMAIL_USER
    msg['To']      = TO_EMAIL
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(GMAIL_USER, GMAIL_PASSWORD)
            server.sendmail(GMAIL_USER, TO_EMAIL, msg.as_string())
        print(f'Email sent to {TO_EMAIL}')
    except Exception as e:
        print(f'Email failed: {e}')

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    tickers = get_all_tickers()
    results = run_scan(tickers)
    send_email(results)
