import requests
import pandas as pd
import numpy as np
import yfinance as yf
import smtplib
import time
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ── Settings (matching original TOS scan exactly) ─────────────────────────────
BB_LENGTH      = 20
BB_MULT        = 2.0
LOOKBACK       = 10
STOCH_LOOKBACK = 25
STOCH_K        = 14   # StochasticFast default period
STOCH_D        = 3    # StochasticFast default smoothing
YEAR_HIGH_BARS = 52   # weekly bars = 1 year

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

# ── Step 2: Compute Stochastic Fast (FastK) ───────────────────────────────────
def stochastic_fast_k(high, low, close, period=14, smooth=3):
    lowest_low   = low.rolling(period).min()
    highest_high = high.rolling(period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    return k

# ── Step 3: Compute all indicators ────────────────────────────────────────────
def compute_indicators(df):
    close = df['Close']
    high  = df['High']
    low   = df['Low']
    open_ = df['Open']

    # Bollinger Bands
    bb_mid   = close.rolling(BB_LENGTH).mean()
    bb_std   = close.rolling(BB_LENGTH).std(ddof=0)
    bb_lower = bb_mid - BB_MULT * bb_std

    # Trigger candle: green + touched lower BB
    green       = close > open_
    touched     = low <= bb_lower
    trigger     = green & touched

    # Confirmation candle (validPair):
    # current candle is green, opens and closes at or above prior trigger open
    valid_pair = (
        trigger.shift(1).fillna(False) &
        green &
        (open_ >= open_.shift(1)) &
        (close >= open_.shift(1))
    )

    # Stochastic Fast (FastK)
    stoch_k = stochastic_fast_k(high, low, close, period=STOCH_K, smooth=STOCH_D)

    # Stochastic divergence:
    # price makes lower low vs prior STOCH_LOOKBACK bars
    # stochastic makes higher low vs prior STOCH_LOOKBACK bars
    price_low  = low < low.shift(1).rolling(STOCH_LOOKBACK).min()
    stoch_high = stoch_k > stoch_k.shift(1).rolling(STOCH_LOOKBACK).min()
    stoch_div  = price_low & stoch_high

    # No candle breaks below trigger low within LOOKBACK bars
    # Track the most recent trigger low
    trigger_low = low.where(trigger).ffill()
    no_break    = low.rolling(LOOKBACK).min() >= trigger_low

    # Not within 15% of 52-week high
    year_high       = high.rolling(YEAR_HIGH_BARS).max()
    below_high_limit = close <= 0.85 * year_high

    return valid_pair, stoch_div, no_break, below_high_limit

# ── Step 4: Check scan conditions ─────────────────────────────────────────────
def check_scan(valid_pair, stoch_div, no_break, below_high_limit):
    """
    Mirrors original TOS logic:
    - validPair occurred within last LOOKBACK bars
    - stochDiv occurred within last LOOKBACK bars
    - noBreak is true on current bar
    - belowHighLimit is true on current bar
    """
    vp   = valid_pair.values
    sd   = stoch_div.values
    nb   = no_break.values
    bhl  = below_high_limit.values
    n    = len(vp)

    if n < LOOKBACK + 1:
        return False

    # Check current bar conditions
    if not nb[-1] or not bhl[-1]:
        return False

    # Check if validPair occurred within last LOOKBACK bars
    valid_pair_recent = any(vp[max(0, n - LOOKBACK):n])
    if not valid_pair_recent:
        return False

    # Check if stochDiv occurred within last LOOKBACK bars
    stoch_div_recent = any(sd[max(0, n - LOOKBACK):n])
    if not stoch_div_recent:
        return False

    return True

# ── Step 5: Run scan ───────────────────────────────────────────────────────────
def run_scan(tickers):
    results  = []
    total    = len(tickers)
    min_bars = YEAR_HIGH_BARS + LOOKBACK + STOCH_LOOKBACK + 10

    print(f'Scanning {total} tickers (Stochastic Divergence scan)...')

    for i, ticker in enumerate(tickers):
        try:
            df = yf.download(ticker, period='3y', interval='1wk', progress=False, auto_adjust=True)

            if df is None or len(df) < min_bars:
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            valid_pair, stoch_div, no_break, below_high_limit = compute_indicators(df)

            if check_scan(valid_pair, stoch_div, no_break, below_high_limit):
                results.append(ticker)
                print(f'  ✓ {ticker}')

        except Exception as e:
            print(f'  Error on {ticker}: {e}')

        if (i + 1) % 100 == 0:
            print(f'  [{i + 1}/{total}] scanned — {len(results)} matches so far...')

        time.sleep(0.05)

    print(f'\nScan complete. {len(results)} matches found.')
    return results

# ── Step 6: Send email ─────────────────────────────────────────────────────────
def send_email(results):
    subject = 'BB + Stochastic Divergence Scan Results'

    if results:
        body  = 'Weekly BB + Stochastic Divergence Scan Results\n'
        body += '=' * 45 + '\n'
        body += '\n'.join(sorted(results))
        body += f'\n\n{"=" * 45}'
        body += f'\nTotal matches: {len(results)}'
    else:
        body  = 'Weekly BB + Stochastic Divergence Scan\n'
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
