import requests
import pandas as pd
import numpy as np
import yfinance as yf
import smtplib
import time
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ── Scan settings ──────────────────────────────────────────────────────────────
BB_LENGTH      = 20
BB_MULT        = 2.0
VF_PD          = 30
VF_BBL         = 20
VF_MULT        = 2.0
VF_LB          = 75
VF_PH          = 0.85
MAX_GAP        = 35
MAX_STOP_DIST  = 0.11        # Skip signals where stop is more than 11% below entry
SCAN_DELAY     = 3
VF_NEAR        = 2
STOCH_LOOKBACK = 25
STOCH_K        = 14
LOOKBACK       = 10
YEAR_HIGH_BARS = 52

# ── Backtest settings ──────────────────────────────────────────────────────────
HOLD_WEEKS     = 15
WIN_TARGET     = 0.13        # 13% gain = win
POSITION_SIZE  = 5000.0      # $5,000 per trade

# ── Filters ────────────────────────────────────────────────────────────────────
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

# ── Step 2: VixFix signals ─────────────────────────────────────────────────────
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
                # Skip if stop loss is more than MAX_STOP_DIST below entry
                entry_price = float(df['Close'].iloc[recent_idx])
                stop_price  = float(low_v[recent_idx])
                if (entry_price - stop_price) / entry_price > MAX_STOP_DIST:
                    break
                signals.append({
                    'signal_idx':  recent_idx,
                    'signal_date': df.index[recent_idx],
                    'entry_price': float(df['Close'].iloc[recent_idx]),
                    'stop_loss':   float(low_v[recent_idx])
                })
                break

    return signals

# ── Step 3: Stochastic signals ─────────────────────────────────────────────────
def find_stoch_signal_bars(df):
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
    idx        = signal['signal_idx']
    entry      = signal['entry_price']
    stop_loss  = signal['stop_loss']
    win_target = entry * (1 + WIN_TARGET)
    n          = len(df)

    # Position sizing
    shares     = POSITION_SIZE / entry
    cost_basis = shares * entry

    result     = 'NEUTRAL'
    exit_price = None
    exit_week  = None

    for w in range(1, HOLD_WEEKS + 1):
        future_idx = idx + w
        if future_idx >= n:
            break

        week_high  = float(df['High'].iloc[future_idx])
        week_low   = float(df['Low'].iloc[future_idx])
        week_close = float(df['Close'].iloc[future_idx])

        # Win: intraday high touched target
        if week_high >= win_target:
            result     = 'WIN'
            exit_price = win_target
            exit_week  = w
            break

        # Stop loss: intraday low pierced stop level
        if week_low <= stop_loss:
            result     = 'LOSS'
            exit_price = stop_loss
            exit_week  = w
            break

    # Neutral: use close at week 10 or last available bar
    if result == 'NEUTRAL':
        last_idx   = min(idx + HOLD_WEEKS, n - 1)
        exit_price = float(df['Close'].iloc[last_idx])
        exit_week  = min(HOLD_WEEKS, n - 1 - idx)

    pct_return    = (exit_price - entry) / entry * 100
    dollar_return = shares * (exit_price - entry)

    return {
        'result':        result,
        'entry':         entry,
        'stop_loss':     stop_loss,
        'exit_price':    exit_price,
        'exit_week':     exit_week,
        'pct_return':    pct_return,
        'shares':        shares,
        'cost_basis':    cost_basis,
        'dollar_return': dollar_return
    }

# ── Step 5: Run full backtest ──────────────────────────────────────────────────
def run_backtest(tickers):
    all_trades = []
    min_bars   = max(VF_LB + MAX_GAP, YEAR_HIGH_BARS + LOOKBACK + STOCH_LOOKBACK) + HOLD_WEEKS + 10

    print(f'Backtesting {len(tickers)} tickers...')

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

            vf_signals = find_vixfix_signals(df)
            if not vf_signals:
                continue

            stoch_active = find_stoch_signal_bars(df)
            if not stoch_active:
                continue

            for signal in vf_signals:
                sidx = signal['signal_idx']
                stoch_match = any(
                    (sidx + offset) in stoch_active
                    for offset in range(-SCAN_DELAY, SCAN_DELAY + 1)
                )
                if not stoch_match:
                    continue
                if sidx + 1 >= len(df):
                    continue

                trade = backtest_signal(df, signal)
                trade['ticker'] = ticker
                trade['date']   = str(signal['signal_date'].date())
                all_trades.append(trade)
                print(
                    f'  {ticker} {trade["date"]} → {trade["result"]:<7} '
                    f'{trade["pct_return"]:+.1f}%  ${trade["dollar_return"]:+.2f}'
                )

        except Exception as e:
            print(f'  Error on {ticker}: {e}')

        if (i + 1) % 100 == 0:
            print(f'  [{i + 1}/{len(tickers)}] scanned — {len(all_trades)} signals so far...')

        time.sleep(0.05)

    return all_trades

# ── Step 6: Summarize ──────────────────────────────────────────────────────────
def summarize(trades):
    if not trades:
        return 'No historical signals found where both scans agreed.'

    wins     = [t for t in trades if t['result'] == 'WIN']
    losses   = [t for t in trades if t['result'] == 'LOSS']
    neutrals = [t for t in trades if t['result'] == 'NEUTRAL']
    total    = len(trades)

    pct_returns    = [t['pct_return'] for t in trades]
    dollar_returns = [t['dollar_return'] for t in trades]

    win_rate       = len(wins) / total * 100
    avg_pct        = np.mean(pct_returns)
    avg_win_pct    = np.mean([t['pct_return'] for t in wins]) if wins else 0
    avg_loss_pct   = np.mean([t['pct_return'] for t in losses]) if losses else 0
    avg_dollar     = np.mean(dollar_returns)
    avg_win_dollar = np.mean([t['dollar_return'] for t in wins]) if wins else 0
    avg_loss_dollar= np.mean([t['dollar_return'] for t in losses]) if losses else 0

    total_capital  = total * POSITION_SIZE
    total_profit   = sum(dollar_returns)
    total_roi      = total_profit / total_capital * 100

    best_trade     = max(trades, key=lambda t: t['dollar_return'])
    worst_trade    = min(trades, key=lambda t: t['dollar_return'])

    sep = '=' * 55
    lines = [
        sep,
        'BACKTEST REPORT — BB + VixFix + Stochastic',
        'Signals where BOTH scans agreed (NYSE + NASDAQ)',
        f'Position size: ${POSITION_SIZE:,.0f} per trade',
        sep,
        f'Total signals:       {total}',
        f'Wins  (>+13%):       {len(wins)} ({win_rate:.1f}%)',
        f'Losses (stop hit):   {len(losses)} ({len(losses)/total*100:.1f}%)',
        f'Neutral (10w hold):  {len(neutrals)} ({len(neutrals)/total*100:.1f}%)',
        '',
        '── Per Trade ─────────────────────────────────────────',
        f'Avg return:          {avg_pct:+.1f}%  (${avg_dollar:+,.2f})',
        f'Avg win:             {avg_win_pct:+.1f}%  (${avg_win_dollar:+,.2f})',
        f'Avg loss:            {avg_loss_pct:+.1f}%  (${avg_loss_dollar:+,.2f})',
        '',
        '── Overall P&L ───────────────────────────────────────',
        f'Total capital deployed: ${total_capital:,.2f}',
        f'Total profit/loss:      ${total_profit:+,.2f}',
        f'Overall ROI:            {total_roi:+.1f}%',
        '',
        f'Best trade:   {best_trade["ticker"]} {best_trade["date"]} '
        f'({best_trade["pct_return"]:+.1f}%  ${best_trade["dollar_return"]:+,.2f})',
        f'Worst trade:  {worst_trade["ticker"]} {worst_trade["date"]} '
        f'({worst_trade["pct_return"]:+.1f}%  ${worst_trade["dollar_return"]:+,.2f})',
        '',
        sep,
        'ALL SIGNALS (sorted by date)',
        sep,
        f'{"Ticker":<6} {"Date":<12} {"Result":<8} {"Return%":>8} {"$Return":>10} '
        f'{"Wk":>3} {"Entry":>8} {"Stop":>8}',
        '-' * 55,
    ]

    for t in sorted(trades, key=lambda x: x['date']):
        lines.append(
            f'{t["ticker"]:<6} {t["date"]:<12} {t["result"]:<8} '
            f'{t["pct_return"]:>+7.1f}% {t["dollar_return"]:>+10,.2f} '
            f'{t["exit_week"]:>3}w  ${t["entry"]:>7.2f}  ${t["stop_loss"]:>7.2f}'
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
