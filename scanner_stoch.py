import requests
import pandas as pd
import numpy as np
import yfinance as yf
import smtplib
import time
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ── Settings ───────────────────────────────────────────────────────────────────
BB_LENGTH      = 20
BB_MULT        = 2.0
LOOKBACK       = 10
STOCH_LOOKBACK = 25
STOCH_K        = 14
YEAR_HIGH_BARS = 52

# ── Filters ────────────────────────────────────────────────────────────────────
MIN_PRICE      = 5.0
MIN_MARKET_CAP = 1_000_000_000  # $1B

# ── Email settings ─────────────────────────────────────────────────────────────
GMAIL_USER     = os.environ.get('GMAIL_USER', '')
GMAIL_PASSWORD = os.environ.get('GMAIL_PASSWORD', '')
SEND_EMAIL     = os.environ.get('SEND_EMAIL', 'false') == 'true'
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

def stochastic_k(high, low, close, period=14):
    lowest_low   = low.rolling(period).min()
    highest_high = high.rolling(period).max()
    return 100 * (close - lowest_low) / (highest_high - lowest_low)

def compute_indicators(df):
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

    stoch_k   = stochastic_k(high, low, close, period=STOCH_K)
    price_low  = low < low.shift(1).rolling(STOCH_LOOKBACK).min()
    stoch_high = stoch_k > stoch_k.shift(1).rolling(STOCH_LOOKBACK).min()
    stoch_div  = price_low & stoch_high

    trigger_low = low.where(trigger).ffill()
    no_break    = low.rolling(LOOKBACK).min() >= trigger_low

    year_high        = high.rolling(YEAR_HIGH_BARS).max()
    below_high_limit = close <= 0.85 * year_high

    return valid_pair, stoch_div, no_break, below_high_limit

def check_scan(valid_pair, stoch_div, no_break, below_high_limit):
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

def run_scan(tickers):
    results  = []
    min_bars = YEAR_HIGH_BARS + LOOKBACK + STOCH_LOOKBACK + 10

    print(f'Scanning {len(tickers)} tickers (Stochastic scan)...')

    for i, ticker in enumerate(tickers):
        try:
            df = yf.download(ticker, period='3y', interval='1wk', progress=False, auto_adjust=True)
            if df is None or len(df) < min_bars:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if float(df['Close'].iloc[-1]) < MIN_PRICE:
                continue

            try:
                mc = yf.Ticker(ticker).fast_info.market_cap
                if mc is not None and mc < MIN_MARKET_CAP:
                    continue
            except:
                pass

            valid_pair, stoch_div, no_break, below_high_limit = compute_indicators(df)

            if check_scan(valid_pair, stoch_div, no_break, below_high_limit):
                results.append(ticker)
                print(f'  ✓ {ticker}')

        except Exception as e:
            print(f'  Error on {ticker}: {e}')

        if (i + 1) % 100 == 0:
            print(f'  [{i + 1}/{len(tickers)}] scanned — {len(results)} matches so far...')

        time.sleep(0.05)

    print(f'\nStochastic scan complete. {len(results)} matches found.')
    return results

if __name__ == '__main__':
    tickers = get_all_tickers()
    results = run_scan(tickers)

    with open('results_stoch.txt', 'w') as f:
        f.write('\n'.join(sorted(results)))

    if SEND_EMAIL and GMAIL_USER:
        body  = 'BB + STOCHASTIC DIVERGENCE RESULTS\n' + '=' * 45 + '\n'
        body += '\n'.join(sorted(results)) if results else 'No matches.'
        body += f'\nTotal: {len(results)}'
        msg = MIMEMultipart()
        msg['From']    = GMAIL_USER
        msg['To']      = TO_EMAIL
        msg['Subject'] = 'BB + Stochastic Divergence Scan Results'
        msg.attach(MIMEText(body, 'plain'))
        try:
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as s:
                s.login(GMAIL_USER, GMAIL_PASSWORD)
                s.sendmail(GMAIL_USER, TO_EMAIL, msg.as_string())
            print(f'Email sent to {TO_EMAIL}')
        except Exception as e:
            print(f'Email failed: {e}')
    else:
        print('Manual run — no email sent.')
