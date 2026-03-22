import requests
import pandas as pd
import numpy as np
import yfinance as yf
import smtplib
import time
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ── Scan settings (must match scanner.py and scanner_stoch.py exactly) ─────────
BB_LENGTH      = 20
BB_MULT        = 2.0
VF_PD          = 30
VF_BBL         = 20
VF_MULT        = 2.0
VF_LB          = 75
VF_PH          = 0.85
MAX_GAP        = 35
SCAN_DELAY     = 3
VF_NEAR        = 2
STOCH_LOOKBACK = 25
STOCH_K        = 14
LOOKBACK       = 10
YEAR_HIGH_BARS = 52

# ── Backtest settings ──────────────────────────────────────────────────────────
HOLD_WEEKS     = 10      # Max holding period
WIN_TARGET     = 0.13    # 13% gain = win
MIN_PRICE      = 5.0
MIN_MARKET_CAP = 1_000_000_000

# ── Email settings ─────────────────────────────────────────────────────────────
GMAIL_USER     = os.environ.get('GMAIL_USER', '')
GMAIL_PASSWORD = os.environ.get('GMAIL_PASSWORD', '')
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

# ── Step 2: VixFix indicators ──────────────────────────────────────────────────
def compute_vixfix(df):
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

    return trigger_with_vf, wvf_at_trigger, low

def find_vixfix_signals(df):
    """Find all historical bars where VixFix divergence signal fired"""
    trigger_with_vf, wvf_at_trigger, low = compute_vixfix(df)
    twvf  = trigger_with_vf.values
    wvf_v = wvf_at_trigger.values
    low_v = low.values
    n     = len(twvf)
    signals = []

    for recent_idx in range(n):
        if not twvf[recent_idx]:
            continue
        recent_low = low_v[recent_idx]
        recent_wvf = wvf_v[recent_idx]
        if np.isnan(recent_low) or np.isnan(recent_wvf):
            continue
        for j in range(recent_idx - 1, max(recent_idx - MAX_GAP, 0) - 1, -1):
            if not twvf[j]:
                continue
            prior_low = low_v[j]
            prior_wvf = wvf_v[j]
            if np.isnan(prior_low) or np.isnan(prior_wvf):
                continue
            if recent_low < prior_low and recent_wvf > prior_wvf:
                signals.append({
                    'signal_idx': recent_idx,
                    'signal_date': df.index[recent_idx],
                    'entry_price': float(df['Close'].iloc[recent_idx]),
                    'stop_loss': float(low_v[recent_idx])  # low of second trigger
                })
                break

    return signals

# ── Step 3: Stochastic indicators ─────────────────────────────────────────────
def compute_stoch(df):
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

    return valid_pair, stoch_div, no_break, below_high_limit

def find_stoch_signal_bars(df):
    """Return set of bar indices where stoch scan is active"""
    valid_pair, stoch_div, no_break, below_high_limit = compute_stoch(df)
    vp  = valid_pair.values
    sd  = stoch_div.values
    nb  = no_break.values
    bhl = below_high_limit.values
    n   = len(vp)
    active = set()

    for i in range(LOOKBACK, n):
        if not nb[i] or not bhl[i]:
            continue
        if not any(vp[max(0, i - LOOKBACK):i]):
            continue
        if not any(sd[max(0, i - LOOKBACK):i]):
            continue
        active.add(i)

    return active

# ── Step 4: Backtest a single signal ──────────────────────────────────────────
def backtest_signal(df, signal):
    """
    Entry: close of signal bar
    Win:   price rises 13% above entry within HOLD_WEEKS
    Loss:  price closes below stop loss (second trigger low)
    Neutral: neither — return actual % change at week 10
    """
    idx        = signal['signal_idx']
    entry      = signal['entry_price']
    stop_loss  = signal['stop_loss']
    win_target = entry * (1 + WIN_TARGET)
    n          = len(df)

    result = 'NEUTRAL'
    exit_price = None
    exit_week  = None

    for w in range(1, HOLD_WEEKS + 1):
        future_idx = idx + w
        if future_idx >= n:
            break

        week_high  = float(df['High'].iloc[future_idx])
        week_low   = float(df['Low'].iloc[future_idx])
        week_close = float(df['Close'].iloc[future_idx])

        # Check win first (high touched target)
        if week_high >= win_target:
            result     = 'WIN'
            exit_price = win_target
            exit_week  = w
            break

        # Check stop loss (close below stop)
        if week_close < stop_loss:
            result     = 'LOSS'
            exit_price = week_close
            exit_week  = w
            break

    # Neutral — use close at week 10 or last available
    if result == 'NEUTRAL':
        last_idx   = min(idx + HOLD_WEEKS, n - 1)
        exit_price = float(df['Close'].iloc[last_idx])
        exit_week  = min(HOLD_WEEKS, n - 1 - idx)

    pct_return = (exit_price - entry) / entry * 100

    return {
        'result':     result,
        'entry':      entry,
        'stop_loss':  stop_loss,
        'exit_price': exit_price,
        'exit_week':  exit_week,
        'pct_return': pct_return
    }

# ── Step 5: Run full backtest ──────────────────────────────────────────────────
def run_backtest(tickers):
    all_trades  = []
    total       = len(tickers)
    min_bars    = max(VF_LB + MAX_GAP, YEAR_HIGH_BARS + LOOKBACK + STOCH_LOOKBACK) + HOLD_WEEKS + 10

    print(f'Backtesting {total} tickers...')

    for i, ticker in enumerate(tickers):
        try:
            df = yf.download(ticker, period='5y', interval='1wk', progress=False, auto_adjust=True)
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

            # Find VixFix signals
            vf_signals = find_vixfix_signals(df)
            if not vf_signals:
                continue

            # Find Stochastic active bars
            stoch_active = find_stoch_signal_bars(df)
            if not stoch_active:
                continue

            # Only keep signals where BOTH scans agree
            for signal in vf_signals:
                sidx = signal['signal_idx']
                # Check if stoch scan was active within SCAN_DELAY bars of vixfix signal
                stoch_match = any(
                    (sidx + offset) in stoch_active
                    for offset in range(-SCAN_DELAY, SCAN_DELAY + 1)
                )
                if not stoch_match:
                    continue

                # Need future bars to evaluate
                if sidx + 1 >= len(df):
                    continue

                trade = backtest_signal(df, signal)
                trade['ticker'] = ticker
                trade['date']   = str(signal['signal_date'].date())
                all_trades.append(trade)
                print(f'  {ticker} {trade["date"]} → {trade["result"]} ({trade["pct_return"]:+.1f}% in {trade["exit_week"]}w)')

        except Exception as e:
            print(f'  Error on {ticker}: {e}')

        if (i + 1) % 100 == 0:
            print(f'  [{i + 1}/{total}] scanned — {len(all_trades)} signals found so far...')

        time.sleep(0.05)

    return all_trades

# ── Step 6: Summarize results ──────────────────────────────────────────────────
def summarize(trades):
    if not trades:
        return 'No historical signals found where both scans agreed.'

    wins     = [t for t in trades if t['result'] == 'WIN']
    losses   = [t for t in trades if t['result'] == 'LOSS']
    neutrals = [t for t in trades if t['result'] == 'NEUTRAL']
    total    = len(trades)
    returns  = [t['pct_return'] for t in trades]

    win_rate    = len(wins) / total * 100
    avg_return  = np.mean(returns)
    avg_win     = np.mean([t['pct_return'] for t in wins]) if wins else 0
    avg_loss    = np.mean([t['pct_return'] for t in losses]) if losses else 0
    best_trade  = max(trades, key=lambda t: t['pct_return'])
    worst_trade = min(trades, key=lambda t: t['pct_return'])

    sep = '=' * 50
    lines = [
        sep,
        'BACKTEST REPORT — BB + VixFix + Stochastic',
        'Signals where BOTH scans agreed (NYSE + NASDAQ)',
        sep,
        f'Total signals:      {total}',
        f'Wins (>+13%):       {len(wins)} ({win_rate:.1f}%)',
        f'Losses (stop hit):  {len(losses)} ({len(losses)/total*100:.1f}%)',
        f'Neutral (10w hold): {len(neutrals)} ({len(neutrals)/total*100:.1f}%)',
        '',
        f'Avg return:         {avg_return:+.1f}%',
        f'Avg win:            {avg_win:+.1f}%',
        f'Avg loss:           {avg_loss:+.1f}%',
        '',
        f'Best trade:  {best_trade["ticker"]} {best_trade["date"]} ({best_trade["pct_return"]:+.1f}%)',
        f'Worst trade: {worst_trade["ticker"]} {worst_trade["date"]} ({worst_trade["pct_return"]:+.1f}%)',
        sep,
        'ALL SIGNALS',
        sep,
    ]

    # Sort by date
    for t in sorted(trades, key=lambda x: x['date']):
        lines.append(
            f'{t["ticker"]:<6} {t["date"]}  {t["result"]:<7}  {t["pct_return"]:+6.1f}%  '
            f'exit wk {t["exit_week"]}  entry ${t["entry"]:.2f}  stop ${t["stop_loss"]:.2f}'
        )

    lines.append(sep)
    return '\n'.join(lines)

# ── Step 7: Send email ─────────────────────────────────────────────────────────
def send_email(report):
    msg = MIMEMultipart()
    msg['From']    = GMAIL_USER
    msg['To']      = TO_EMAIL
    msg['Subject'] = 'Stock Scanner Backtest Report'
    msg.attach(MIMEText(report, 'plain'))

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as s:
            s.login(GMAIL_USER, GMAIL_PASSWORD)
            s.sendmail(GMAIL_USER, TO_EMAIL, msg.as_string())
        print(f'Backtest report emailed to {TO_EMAIL}')
    except Exception as e:
        print(f'Email failed: {e}')

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    tickers = get_all_tickers()
    trades  = run_backtest(tickers)
    report  = summarize(trades)
    print('\n' + report)
    if GMAIL_USER:
        send_email(report)
