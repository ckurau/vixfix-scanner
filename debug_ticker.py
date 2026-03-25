import pandas as pd
import numpy as np
import yfinance as yf

# ── Settings (must match scanner_combined.py exactly) ─────────────────────────
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
MIN_PRICE      = 5.0
MIN_MARKET_CAP = 1_000_000_000
MAX_STOP_DIST  = 0.11
NO_BREAK_BARS  = 10

WIN_TARGET     = 0.13
POSITION_SIZE  = 10_000.0

# ── Change this to any ticker you want to debug ────────────────────────────────
TICKER = 'CRM'

def compute_macd(close):
    ema_fast  = close.ewm(span=MACD_FAST,   adjust=False).mean()
    ema_slow  = close.ewm(span=MACD_SLOW,   adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal    = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    histogram = macd_line - signal
    return macd_line.values, signal.values, histogram.values

def no_break_before(low_values, idx, n_bars):
    trigger_low = low_values[idx]
    start = max(0, idx - n_bars)
    for j in range(start, idx):
        if low_values[j] < trigger_low:
            return False, j, low_values[j]
    return True, None, None

def run_debug(ticker):
    sep  = '=' * 60
    sep2 = '-' * 60
    print(f'\n{sep}')
    print(f'DEBUG REPORT — {ticker}')
    print(f'{sep}\n')

    df = yf.download(ticker, period='3y', interval='1wk', progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    print(f'Data: {len(df)} weekly bars ({df.index[0].date()} to {df.index[-1].date()})\n')

    close   = df['Close']
    low     = df['Low']
    high    = df['High']
    open_   = df['Open']
    close_v = close.values
    low_v   = low.values

    # ── [1] Price filter ──────────────────────────────────────────────────────
    current_price = float(close.iloc[-1])
    print(f'[1] PRICE FILTER')
    print(f'    Current price: ${current_price:.2f} (min ${MIN_PRICE})')
    if current_price < MIN_PRICE:
        print(f'    ✗ FAIL — price below minimum')
        return
    print(f'    ✓ PASS\n')

    # ── [2] Market cap filter ─────────────────────────────────────────────────
    try:
        mc = yf.Ticker(ticker).fast_info.market_cap
        print(f'[2] MARKET CAP FILTER')
        print(f'    Market cap: ${mc:,.0f} (min ${MIN_MARKET_CAP:,.0f})')
        if mc is not None and mc < MIN_MARKET_CAP:
            print(f'    ✗ FAIL — market cap below $1B')
            return
        print(f'    ✓ PASS\n')
    except:
        print(f'[2] MARKET CAP FILTER — could not retrieve, skipping\n')

    # ── [3] BB trigger + confirmation candles ─────────────────────────────────
    bb_mid   = close.rolling(BB_LENGTH).mean()
    bb_std   = close.rolling(BB_LENGTH).std(ddof=0)
    bb_lower = bb_mid - BB_MULT * bb_std
    trigger  = (close > open_) & (low <= bb_lower)

    next_green       = (close.shift(-1) > open_.shift(-1))
    next_open_above  = (open_.shift(-1) >= open_)
    next_close_above = (close.shift(-1) >= open_)
    trigger_confirmed = trigger & next_green & next_open_above & next_close_above

    confirmed_bars = [i for i, v in enumerate(trigger_confirmed.values) if v]
    print(f'[3] BB TRIGGER + CONFIRMATION CANDLES')
    print(f'    Found {len(confirmed_bars)} confirmed trigger candles total')
    for idx in confirmed_bars[-5:]:
        print(f'    {df.index[idx].date()} — close ${close_v[idx]:.2f}, low ${low_v[idx]:.2f}')
    print()

    # ── [4] VixFix ────────────────────────────────────────────────────────────
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

    vf_trigger_bars = [i for i, v in enumerate(twvf) if v]
    print(f'[4] VIXFIX TRIGGER CANDLES (trigger + VixFix spike within {VF_NEAR} bars)')
    print(f'    Found {len(vf_trigger_bars)} VixFix-confirmed trigger candles')
    for idx in vf_trigger_bars[-5:]:
        print(f'    {df.index[idx].date()} — close ${close_v[idx]:.2f}, low ${low_v[idx]:.2f}, WVF {wvf_v[idx]:.2f}')
    print()

    # ── [5] Most recent trigger within SCAN_DELAY ─────────────────────────────
    print(f'[5] MOST RECENT TRIGGER (within last {SCAN_DELAY} bars)')
    recent_idx = None
    for i in range(n - 1, max(n - SCAN_DELAY - 2, -1), -1):
        if twvf[i]:
            recent_idx = i
            break

    if recent_idx is None:
        print(f'    ✗ FAIL — no VixFix trigger candle within last {SCAN_DELAY} bars')
        if vf_trigger_bars:
            bars_ago = n - 1 - vf_trigger_bars[-1]
            print(f'    Most recent VixFix trigger: {df.index[vf_trigger_bars[-1]].date()} ({bars_ago} bars ago)')
        return
    print(f'    ✓ Found at {df.index[recent_idx].date()} — close ${close_v[recent_idx]:.2f}, low ${low_v[recent_idx]:.2f}\n')

    recent_low   = low_v[recent_idx]
    recent_close = close_v[recent_idx]
    recent_wvf   = wvf_v[recent_idx]

    # ── [6] No-break before trigger ───────────────────────────────────────────
    print(f'[6] NO-BREAK RULE (no prior {NO_BREAK_BARS} candles can have lower low than trigger)')
    ok, fail_j, fail_low = no_break_before(low_v, recent_idx, NO_BREAK_BARS)
    nobreak_pass = ok
    if not ok:
        print(f'    ✗ FAIL — candle at {df.index[fail_j].date()} had low ${fail_low:.2f}')
        print(f'             which is below trigger low ${recent_low:.2f}')
        start = max(0, recent_idx - NO_BREAK_BARS)
        for j in range(start, recent_idx):
            flag = ' ← VIOLATES' if low_v[j] < recent_low else ''
            print(f'      {df.index[j].date()} low ${low_v[j]:.2f}{flag}')
    else:
        print(f'    ✓ PASS\n')

    # ── [7] Stop distance ─────────────────────────────────────────────────────
    stop_dist = (recent_close - recent_low) / recent_close
    print(f'[7] STOP DISTANCE FILTER (max {MAX_STOP_DIST*100:.0f}%)')
    print(f'    Entry: ${recent_close:.2f}, Stop: ${recent_low:.2f}')
    print(f'    Distance: {stop_dist*100:.1f}%')
    stop_pass = stop_dist <= MAX_STOP_DIST
    if not stop_pass:
        print(f'    ✗ FAIL — stop is more than {MAX_STOP_DIST*100:.0f}% below entry\n')
    else:
        print(f'    ✓ PASS\n')

    # ── [8] VixFix divergence ─────────────────────────────────────────────────
    print(f'[8] VIXFIX DIVERGENCE (lower low + higher WVF vs prior trigger)')
    found_vf_div = False
    prior_vf_idx = None
    for j in range(recent_idx - 1, max(recent_idx - MAX_GAP, 0) - 1, -1):
        if not twvf[j]:
            continue
        prior_low = low_v[j]
        prior_wvf = wvf_v[j]
        if np.isnan(prior_low) or np.isnan(prior_wvf):
            continue
        print(f'    Comparing recent ({df.index[recent_idx].date()}) vs prior ({df.index[j].date()}):')
        print(f'      Price: ${recent_low:.2f} vs ${prior_low:.2f} — lower low: {recent_low < prior_low}')
        print(f'      WVF:   {recent_wvf:.2f} vs {prior_wvf:.2f} — higher WVF: {recent_wvf > prior_wvf}')
        if recent_low < prior_low and recent_wvf > prior_wvf:
            found_vf_div = True
            prior_vf_idx = j
            print(f'    ✓ VixFix divergence CONFIRMED\n')
        else:
            print(f'    ✗ VixFix divergence NOT met\n')
        break

    if not found_vf_div and not any(twvf[max(recent_idx - MAX_GAP, 0):recent_idx]):
        print(f'    ✗ FAIL — no prior VixFix trigger found within {MAX_GAP} bars\n')

    # ── [9] MACD divergence ───────────────────────────────────────────────────
    macd_line, signal_line, histogram = compute_macd(close)
    found_macd_div = False

    print(f'[9] MACD DIVERGENCE (Type A: histogram, or Type B: MACD/Signal line)')
    for j in range(recent_idx - 1, max(recent_idx - MAX_GAP, 0) - 1, -1):
        if not twvf[j]:
            continue
        hist_prior   = histogram[j]
        hist_recent  = histogram[recent_idx]
        macd_prior   = macd_line[j]
        macd_recent  = macd_line[recent_idx]
        sig_prior    = signal_line[j]
        sig_recent   = signal_line[recent_idx]

        type_a = hist_recent > hist_prior
        type_b = (macd_recent > macd_prior) or (sig_recent > sig_prior)

        print(f'    Prior trigger:  {df.index[j].date()}')
        print(f'    Recent trigger: {df.index[recent_idx].date()}')
        print(f'    Type A (histogram): {hist_recent:.4f} vs {hist_prior:.4f} — higher: {type_a}')
        print(f'    Type B (MACD line): {macd_recent:.4f} vs {macd_prior:.4f} — higher: {macd_recent > macd_prior}')
        print(f'    Type B (Signal):    {sig_recent:.4f} vs {sig_prior:.4f} — higher: {sig_recent > sig_prior}')

        if type_a or type_b:
            found_macd_div = True
            print(f'    ✓ MACD divergence CONFIRMED ({"Type A" if type_a else "Type B"})\n')
        else:
            print(f'    ✗ FAIL — no MACD divergence\n')
        break

    # ── [10] Stochastic scan ──────────────────────────────────────────────────
    print(f'[10] STOCHASTIC DIVERGENCE SCAN')
    trigger_v    = trigger.values
    lowest_low   = low.rolling(STOCH_K).min()
    highest_high = high.rolling(STOCH_K).max()
    stoch_k      = 100 * (close - lowest_low) / (highest_high - lowest_low)
    price_low_s  = low < low.shift(1).rolling(STOCH_LOOKBACK).min()
    stoch_high_s = stoch_k > stoch_k.shift(1).rolling(STOCH_LOOKBACK).min()
    stoch_div    = price_low_s & stoch_high_s
    trigger_low_s    = low.where(trigger).ffill()
    no_break_s       = low.rolling(LOOKBACK).min() >= trigger_low_s
    year_high        = high.rolling(YEAR_HIGH_BARS).max()
    below_high_limit = close <= 0.85 * year_high

    vp  = trigger_confirmed.values
    sd  = stoch_div.values
    nb  = no_break_s.values
    bhl = below_high_limit.values

    nb_result  = bool(nb[-1])
    bhl_result = bool(bhl[-1])
    vp_recent  = any(vp[max(0, n - LOOKBACK):n])
    sd_recent  = any(sd[max(0, n - LOOKBACK):n])

    print(f'    noBreak (current bar):          {nb_result}')
    print(f'    belowHighLimit (current bar):   {bhl_result}')
    print(f'    validPair (within {LOOKBACK} bars):      {vp_recent}')
    print(f'    stochDiv (within {LOOKBACK} bars):       {sd_recent}')

    stoch_pass = nb_result and bhl_result and vp_recent and sd_recent
    if stoch_pass:
        print(f'    ✓ PASS — {ticker} qualifies for Stochastic scan\n')
    else:
        print(f'    ✗ FAIL — {ticker} does not qualify for Stochastic scan\n')

    # ── TIER VERDICT + TRADE LEVELS ───────────────────────────────────────────
    all_filters_pass = nobreak_pass and stop_pass and found_vf_div

    if all_filters_pass and found_macd_div and stoch_pass:
        tier = 'TIER 1 ★★★ ULTRA  (BB + VixFix + MACD + Stoch)'
        position = POSITION_SIZE
    elif all_filters_pass and stoch_pass:
        tier = 'TIER 2 ★★  HIGH   (BB + VixFix + Stoch — no MACD)'
        position = POSITION_SIZE
    elif all_filters_pass:
        tier = 'VixFix div confirmed but Stochastic not active — no tier signal'
        position = 0.0
    else:
        fails = []
        if not nobreak_pass: fails.append('no-break rule')
        if not stop_pass:    fails.append(f'stop dist {stop_dist*100:.1f}% > {MAX_STOP_DIST*100:.0f}%')
        if not found_vf_div: fails.append('VixFix divergence')
        tier = f'NO SIGNAL — failing: {", ".join(fails)}'
        position = 0.0

    print(sep)
    print('TIER VERDICT')
    print(sep)
    print(f'  {tier}')
    print()

    if position > 0:
        entry      = recent_close
        stop       = recent_low
        target     = entry * (1 + WIN_TARGET)
        risk_per   = entry - stop          # per share downside
        reward_per = target - entry        # per share upside
        rr_ratio   = reward_per / risk_per if risk_per > 0 else 0
        shares     = int(position / entry)
        max_loss   = shares * risk_per
        max_gain   = shares * reward_per
        stop_pct   = (entry - stop) / entry * 100

        print(sep2)
        print('TRADE LEVELS')
        print(sep2)
        print(f'  Buy at:        ${entry:.2f}  '
              f'(close of BB trigger candle — enter at next session open)')
        print(f'  Stop loss:     ${stop:.2f}  ({stop_pct:.1f}% below entry)')
        print(f'  Win target:    ${target:.2f}  (+{int(WIN_TARGET*100)}% above entry)')
        print(f'  Risk/Reward:   1 : {rr_ratio:.2f}  '
              f'(risk ${risk_per:.2f}/share to make ${reward_per:.2f}/share)')
        print(f'  Position:      ${position:,.0f} → {shares} shares @ ${entry:.2f}')
        print(f'  Max loss:      -${max_loss:,.2f}  (if stopped out)')
        print(f'  Max gain:      +${max_gain:,.2f}  (if target hit)')
        print(sep)
    else:
        print(sep)

if __name__ == '__main__':
    run_debug(TICKER)
