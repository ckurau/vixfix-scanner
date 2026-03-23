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
SCAN_DELAY     = 5
VF_NEAR        = 2
STOCH_LOOKBACK = 25
STOCH_K        = 14
LOOKBACK       = 10
YEAR_HIGH_BARS = 52

# ── MACD settings (matching your TOS chart exactly) ───────────────────────────
MACD_FAST      = 12
MACD_SLOW      = 26
MACD_SIGNAL    = 9

# ── Backtest settings ──────────────────────────────────────────────────────────
HOLD_WEEKS_SHORT = 15
HOLD_WEEKS_LONG  = 30
WIN_TARGET       = 0.13
POSITION_SIZE    = 5000.0
YEARS_HISTORY    = 15    # Download up to 15 years of weekly data

# ── Filters ────────────────────────────────────────────────────────────────────
MIN_PRICE      = 5.0
MIN_MARKET_CAP = 1_000_000_000
MAX_STOP_DIST  = 0.11
NO_BREAK_BARS  = 10

# ── Email ──────────────────────────────────────────────────────────────────────
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
    """Compute MACD line, Signal line, and Histogram using EMA"""
    ema_fast   = close.ewm(span=MACD_FAST,   adjust=False).mean()
    ema_slow   = close.ewm(span=MACD_SLOW,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal     = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    histogram  = macd_line - signal
    return macd_line.values, signal.values, histogram.values

def no_break_before(low_values, idx, n_bars):
    trigger_low = low_values[idx]
    start = max(0, idx - n_bars)
    for j in range(start, idx):
        if low_values[j] < trigger_low:
            return False
    return True

def macd_divergence(prior_idx, recent_idx, macd_line, signal_line, histogram):
    """
    Check MACD divergence between two trigger candles.
    Price already confirmed lower low (recent < prior).
    We need: histogram higher OR (macd_line higher OR signal_line higher)

    Type A: histogram at recent > histogram at prior
            (bars getting less negative = bullish divergence)
    Type B: macd_line at recent > macd_line at prior
            OR signal_line at recent > signal_line at prior

    Returns True if either Type A or Type B is satisfied.
    """
    hist_prior   = histogram[prior_idx]
    hist_recent  = histogram[recent_idx]
    macd_prior   = macd_line[prior_idx]
    macd_recent  = macd_line[recent_idx]
    sig_prior    = signal_line[prior_idx]
    sig_recent   = signal_line[recent_idx]

    if any(np.isnan(v) for v in [hist_prior, hist_recent, macd_prior, macd_recent, sig_prior, sig_recent]):
        return False

    type_a = hist_recent > hist_prior
    type_b = (macd_recent > macd_prior) or (sig_recent > sig_prior)

    return type_a or type_b

def find_vixfix_signals(df, macd_line, signal_line, histogram):
    close   = df['Close']
    low     = df['Low']
    open_   = df['Open']
    close_v = close.values
    low_v   = low.values

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
    n     = len(twvf)
    signals = []
    skipped_nobreak = 0
    skipped_stopdist = 0
    skipped_macd = 0
    skipped_novixdiv = 0

    for recent_idx in range(n):
        if not twvf[recent_idx]:
            continue
        recent_low   = low_v[recent_idx]
        recent_close = close_v[recent_idx]
        recent_wvf   = wvf_v[recent_idx]

        if np.isnan(recent_low) or np.isnan(recent_wvf):
            continue

        if not no_break_before(low_v, recent_idx, NO_BREAK_BARS):
            skipped_nobreak += 1
            continue

        if (recent_close - recent_low) / recent_close > MAX_STOP_DIST:
            skipped_stopdist += 1
            continue

        found = False
        for j in range(recent_idx - 1, max(recent_idx - MAX_GAP, 0) - 1, -1):
            if not twvf[j]:
                continue
            prior_low = low_v[j]
            prior_wvf = wvf_v[j]
            if np.isnan(prior_low) or np.isnan(prior_wvf):
                continue
            if recent_low < prior_low and recent_wvf > prior_wvf:
                # Check MACD divergence
                if not macd_divergence(j, recent_idx, macd_line, signal_line, histogram):
                    skipped_macd += 1
                    break
                signals.append({
                    'signal_idx':  recent_idx,
                    'signal_date': df.index[recent_idx],
                    'entry_price': float(recent_close),
                    'stop_loss':   float(recent_low)
                })
                found = True
                break

        if not found and not any(twvf[max(recent_idx - MAX_GAP, 0):recent_idx]):
            skipped_novixdiv += 1

    return signals, {
        'skipped_nobreak':  skipped_nobreak,
        'skipped_stopdist': skipped_stopdist,
        'skipped_macd':     skipped_macd,
    }

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

def backtest_signal(df, signal, hold_weeks):
    idx        = signal['signal_idx']
    entry      = signal['entry_price']
    stop_loss  = signal['stop_loss']
    win_target = entry * (1 + WIN_TARGET)
    shares     = POSITION_SIZE / entry
    n          = len(df)

    result     = 'NEUTRAL'
    exit_price = None
    exit_week  = None

    for w in range(1, hold_weeks + 1):
        future_idx = idx + w
        if future_idx >= n:
            break
        week_high = float(df['High'].iloc[future_idx])
        week_low  = float(df['Low'].iloc[future_idx])

        if week_high >= win_target:
            result     = 'WIN'
            exit_price = win_target
            exit_week  = w
            break
        if week_low <= stop_loss:
            result     = 'LOSS'
            exit_price = stop_loss
            exit_week  = w
            break

    if result == 'NEUTRAL':
        last_idx   = min(idx + hold_weeks, n - 1)
        exit_price = float(df['Close'].iloc[last_idx])
        exit_week  = min(hold_weeks, n - 1 - idx)

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
        'dollar_return': dollar_return
    }

def run_backtest(tickers):
    all_trades_short = []
    all_trades_long  = []
    total_skipped_nobreak  = 0
    total_skipped_stopdist = 0
    total_skipped_macd     = 0

    cutoff_date = pd.Timestamp.now() - pd.DateOffset(years=YEARS_HISTORY)
    min_bars = max(VF_LB + MAX_GAP, YEAR_HIGH_BARS + LOOKBACK + STOCH_LOOKBACK) + HOLD_WEEKS_LONG + 10

    print(f'Backtesting {len(tickers)} tickers (up to {YEARS_HISTORY} years of weekly data)...')

    for i, ticker in enumerate(tickers):
        try:
            df = yf.download(ticker, period='max', interval='1wk', progress=False, auto_adjust=True)
            if df is None or len(df) < min_bars:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Trim to last 15 years
            df = df[df.index >= cutoff_date].copy()
            if len(df) < min_bars:
                continue

            if float(df['Close'].iloc[-1]) < MIN_PRICE:
                continue

            recent_close = float(df['Close'].iloc[-1])

            try:
                mc = yf.Ticker(ticker).fast_info.market_cap
                if mc is not None and mc < MIN_MARKET_CAP:
                    continue
            except:
                pass

            # Compute MACD
            macd_line, signal_line, histogram = compute_macd(df['Close'])

            vf_signals, skip_counts = find_vixfix_signals(df, macd_line, signal_line, histogram)
            total_skipped_nobreak  += skip_counts['skipped_nobreak']
            total_skipped_stopdist += skip_counts['skipped_stopdist']
            total_skipped_macd     += skip_counts['skipped_macd']

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

                # Price sanity check
                entry = signal['entry_price']
                if entry < recent_close * 0.15 or entry > recent_close * 6.0:
                    print(f'  Skipping {ticker} {signal["signal_date"].date()} — price mismatch (${entry:.2f} vs ${recent_close:.2f})')
                    continue

                trade_short = backtest_signal(df, signal, HOLD_WEEKS_SHORT)
                trade_long  = backtest_signal(df, signal, HOLD_WEEKS_LONG)

                for t in [trade_short, trade_long]:
                    t['ticker'] = ticker
                    t['date']   = str(signal['signal_date'].date())

                all_trades_short.append(trade_short)
                all_trades_long.append(trade_long)

                print(
                    f'  {ticker} {trade_short["date"]} → '
                    f'{trade_short["result"]:<7} {trade_short["pct_return"]:+.1f}% '
                    f'(${trade_short["dollar_return"]:+.2f}) | '
                    f'30w: {trade_long["result"]:<7} {trade_long["pct_return"]:+.1f}%'
                )

        except Exception as e:
            print(f'  Error on {ticker}: {e}')

        if (i + 1) % 100 == 0:
            print(f'  [{i + 1}/{len(tickers)}] — {len(all_trades_short)} signals so far...')

        time.sleep(0.05)

    print(f'\nFilter summary:')
    print(f'  Skipped (no-break violated):  {total_skipped_nobreak}')
    print(f'  Skipped (stop too wide >11%): {total_skipped_stopdist}')
    print(f'  Skipped (no MACD divergence): {total_skipped_macd}')

    return all_trades_short, all_trades_long

def safe_mean(values):
    """Mean that handles empty lists and NaN values"""
    clean = [v for v in values if v is not None and not np.isnan(v)]
    return np.mean(clean) if clean else 0.0

def summarize(trades, hold_weeks, label):
    if not trades:
        return f'No signals found for {label}.'

    wins     = [t for t in trades if t['result'] == 'WIN']
    losses   = [t for t in trades if t['result'] == 'LOSS']
    neutrals = [t for t in trades if t['result'] == 'NEUTRAL']
    total    = len(trades)

    pct_returns    = [t['pct_return'] for t in trades]
    dollar_returns = [t['dollar_return'] for t in trades]
    hold_times     = [t['exit_week'] for t in trades if t['exit_week'] is not None]

    win_rate        = len(wins) / total * 100
    avg_pct         = safe_mean(pct_returns)
    avg_win_pct     = safe_mean([t['pct_return'] for t in wins])
    avg_loss_pct    = safe_mean([t['pct_return'] for t in losses])
    avg_dollar      = safe_mean(dollar_returns)
    avg_win_dollar  = safe_mean([t['dollar_return'] for t in wins])
    avg_loss_dollar = safe_mean([t['dollar_return'] for t in losses])
    avg_hold        = safe_mean(hold_times)
    avg_hold_win    = safe_mean([t['exit_week'] for t in wins])
    avg_hold_loss   = safe_mean([t['exit_week'] for t in losses])

    total_capital = total * POSITION_SIZE
    total_profit  = sum(v for v in dollar_returns if not np.isnan(v))
    total_roi     = (total_profit / total_capital * 100) if total_capital > 0 else 0.0

    best_trade  = max(trades, key=lambda t: t['dollar_return'] if not np.isnan(t['dollar_return']) else -999999)
    worst_trade = min(trades, key=lambda t: t['dollar_return'] if not np.isnan(t['dollar_return']) else 999999)

    sep = '=' * 58
    lines = [
        sep,
        f'BACKTEST REPORT — {label}',
        f'BB + VixFix + Stochastic + MACD Divergence | $5,000/trade',
        f'Max hold: {hold_weeks} weeks | Last {YEARS_HISTORY} years of data',
        sep,
        f'Total signals:        {total}',
        f'Wins  (>+13%):        {len(wins)} ({win_rate:.1f}%)',
        f'Losses (stop hit):    {len(losses)} ({len(losses)/total*100:.1f}%)',
        f'Neutral ({hold_weeks}w hold):   {len(neutrals)} ({len(neutrals)/total*100:.1f}%)',
        '',
        '── Per Trade ───────────────────────────────────────────',
        f'Avg return:           {avg_pct:+.1f}%  (${avg_dollar:+,.2f})',
        f'Avg win:              {avg_win_pct:+.1f}%  (${avg_win_dollar:+,.2f})',
        f'Avg loss:             {avg_loss_pct:+.1f}%  (${avg_loss_dollar:+,.2f})',
        '',
        '── Hold Times ──────────────────────────────────────────',
        f'Avg hold (all):       {avg_hold:.1f} weeks',
        f'Avg hold (wins):      {avg_hold_win:.1f} weeks',
        f'Avg hold (losses):    {avg_hold_loss:.1f} weeks',
        '',
        '── Overall P&L ─────────────────────────────────────────',
        f'Total capital:        ${total_capital:,.2f}',
        f'Total profit/loss:    ${total_profit:+,.2f}',
        f'Overall ROI:          {total_roi:+.1f}%',
        '',
        f'Best:   {best_trade["ticker"]} {best_trade["date"]} ({best_trade["pct_return"]:+.1f}%  ${best_trade["dollar_return"]:+,.2f})',
        f'Worst:  {worst_trade["ticker"]} {worst_trade["date"]} ({worst_trade["pct_return"]:+.1f}%  ${worst_trade["dollar_return"]:+,.2f})',
        '',
        sep,
        f'ALL SIGNALS — {label}',
        sep,
        f'{"Ticker":<6} {"Date":<12} {"Result":<8} {"Ret%":>7} {"$Ret":>10} {"Wk":>4} {"Entry":>8} {"Stop":>8}',
        '-' * 58,
    ]

    for t in sorted(trades, key=lambda x: x['date']):
        ret_str  = f'{t["pct_return"]:+.1f}%' if not np.isnan(t["pct_return"]) else 'nan'
        dret_str = f'${t["dollar_return"]:+,.2f}' if not np.isnan(t["dollar_return"]) else 'nan'
        lines.append(
            f'{t["ticker"]:<6} {t["date"]:<12} {t["result"]:<8} '
            f'{ret_str:>7} {dret_str:>10} '
            f'{str(t["exit_week"])+"w":>4}  ${t["entry"]:>7.2f}  ${t["stop_loss"]:>7.2f}'
        )

    lines.append(sep)
    return '\n'.join(lines)

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
        print(f'Report emailed to {TO_EMAIL}')
    except Exception as e:
        print(f'Email failed: {e}')

if __name__ == '__main__':
    tickers = get_all_tickers()
    trades_short, trades_long = run_backtest(tickers)

    report_short = summarize(trades_short, HOLD_WEEKS_SHORT, f'{HOLD_WEEKS_SHORT}-WEEK HOLD')
    report_long  = summarize(trades_long,  HOLD_WEEKS_LONG,  f'{HOLD_WEEKS_LONG}-WEEK HOLD')

    sep = '=' * 58
    if trades_short and trades_long:
        wr_short  = len([t for t in trades_short if t['result'] == 'WIN']) / len(trades_short) * 100
        wr_long   = len([t for t in trades_long  if t['result'] == 'WIN']) / len(trades_long)  * 100
        pnl_short = sum(t['dollar_return'] for t in trades_short if not np.isnan(t['dollar_return']))
        pnl_long  = sum(t['dollar_return'] for t in trades_long  if not np.isnan(t['dollar_return']))
        comparison = '\n'.join([
            sep,
            '15-WEEK vs 30-WEEK HOLD COMPARISON',
            sep,
            f'{"Metric":<28} {"15 Weeks":>12} {"30 Weeks":>12}',
            '-' * 54,
            f'{"Win rate":<28} {wr_short:>11.1f}% {wr_long:>11.1f}%',
            f'{"Total P&L":<28} ${pnl_short:>+11,.2f} ${pnl_long:>+11,.2f}',
            f'{"Total signals":<28} {len(trades_short):>12} {len(trades_long):>12}',
            sep,
        ])
    else:
        comparison = 'Not enough data for comparison.'

    full_report = comparison + '\n\n' + report_short + '\n\n' + report_long
    print('\n' + full_report)
    if GMAIL_USER:
        send_email(full_report)
