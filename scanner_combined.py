"""
scanner_combined.py  —  Weekly scanner (interval=1wk)
Runs inline mini-backtest at startup to auto-populate win rates,
signal counts, signals/month, and best EV target per tier.
Sends HTML email with bold tier headers and red ticker symbols.
"""
import requests, pandas as pd, numpy as np, yfinance as yf
import smtplib, time, os, random
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

INTERVAL='1wk'
BB_LENGTH=20; BB_MULT=2.0; VF_PD=30; VF_BBL=20; VF_MULT=2.0; VF_LB=75; VF_PH=0.85
MAX_GAP=35; SCAN_DELAY=5; VF_NEAR=2; STOCH_LOOKBACK=25; STOCH_K=14; LOOKBACK=10
YEAR_HIGH_BARS=52; MACD_FAST=12; MACD_SLOW=26; MACD_SIGNAL=9
MIN_PRICE=10.0; MIN_MARKET_CAP=1_000_000_000; MAX_STOP_DIST=0.11; NO_BREAK_BARS=10
HOLD_BARS=20; WIN_TARGET=0.13; POSITION_HIGH=10000.0; POSITION_STD=5000.0; YEARS_HISTORY=15
EV_TARGETS=[i/100 for i in range(5,55,5)]; BACKTEST_SAMPLE=400
GMAIL_USER=os.environ.get('GMAIL_USER',''); GMAIL_PASSWORD=os.environ.get('GMAIL_PASSWORD','')
TO_EMAIL='bkcolby@yahoo.com'

def safe_mean(v):
    c=[x for x in v if x is not None and not np.isnan(x)]; return np.mean(c) if c else 0.0
def safe_sum(v):
    c=[x for x in v if x is not None and not np.isnan(x)]; return sum(c) if c else 0.0

def get_all_tickers():
    headers={'User-Agent':'Mozilla/5.0'}; tickers=[]
    for exchange in ['NYSE','NASDAQ']:
        url=f'https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=10000&exchange={exchange}'
        try:
            r=requests.get(url,headers=headers,timeout=15); rows=r.json()['data']['table']['rows']
            for row in rows:
                sym=row['symbol'].strip()
                if sym.isalpha() and len(sym)<=4:
                    try:
                        mc=float(str(row.get('marketCap','0')).replace(',',''))
                        if mc>0 and mc<MIN_MARKET_CAP: continue
                    except: pass
                    tickers.append(sym)
        except Exception as e: print(f'Error {exchange}: {e}')
    tickers=list(set(tickers)); print(f'Total tickers: {len(tickers)}'); return tickers

def compute_macd(close):
    ef=close.ewm(span=MACD_FAST,adjust=False).mean(); es=close.ewm(span=MACD_SLOW,adjust=False).mean()
    ml=ef-es; sl=ml.ewm(span=MACD_SIGNAL,adjust=False).mean(); return ml.values,sl.values,(ml-sl).values

def no_break_before(lv,idx,n):
    tl=lv[idx]
    for j in range(max(0,idx-n),idx):
        if lv[j]<tl: return False
    return True

def no_break_after(lv,idx,end):
    tl=lv[idx]
    for j in range(idx+1,end+1):
        if lv[j]<tl: return False
    return True

def macd_divergence(pi,ri,ml,sl,hist):
    vals=[hist[pi],hist[ri],ml[pi],ml[ri],sl[pi],sl[ri]]
    if any(np.isnan(v) for v in vals): return False
    return (hist[ri]>hist[pi]) or (ml[ri]>ml[pi]) or (sl[ri]>sl[pi])

def check_vixfix(df,ml,sl,hist):
    close,low,open_=df['Close'],df['Low'],df['Open']; cv,lv=close.values,low.values; n=len(df)
    bb_lo=close.rolling(BB_LENGTH).mean()-BB_MULT*close.rolling(BB_LENGTH).std(ddof=0)
    trig=(close>open_)&(low<=bb_lo)
    tc=(trig&(close.shift(-1)>open_.shift(-1))&(open_.shift(-1)>=open_)&(close.shift(-1)>=open_))
    hc=close.rolling(VF_PD).max(); wvf=(hc-low)/hc*100
    vf_up=wvf.rolling(VF_BBL).mean()+VF_MULT*wvf.rolling(VF_BBL).std(ddof=0)
    vf_rng=wvf.rolling(VF_LB).max()*VF_PH; is_grn=(wvf>=vf_up)|(wvf>=vf_rng)
    vf_near=pd.Series(False,index=df.index)
    for s in range(VF_NEAR+1):
        vf_near|=is_grn.shift(s).fillna(False).infer_objects(copy=False).astype(bool)
        if s>0: vf_near|=is_grn.shift(-s).fillna(False).infer_objects(copy=False).astype(bool)
    twvf_s=tc&vf_near; wat=pd.Series(np.nan,index=df.index)
    for s in range(-VF_NEAR,VF_NEAR+1):
        sh=wvf.shift(s).fillna(0); wat=wat.combine(sh,lambda a,b: b if np.isnan(a) else max(a,b))
    twvf,wvfv=twvf_s.values,wat.values
    recent_idx=None
    for i in range(n-1,max(n-SCAN_DELAY-2,-1),-1):
        if twvf[i]: recent_idx=i; break
    if recent_idx is None: return False,False
    rl,rc,rw=lv[recent_idx],cv[recent_idx],wvfv[recent_idx]
    if np.isnan(rl) or np.isnan(rw): return False,False
    if not no_break_before(lv,recent_idx,NO_BREAK_BARS): return False,False
    if (rc-rl)/rc>MAX_STOP_DIST: return False,False
    if not no_break_after(lv,recent_idx,n-1): return False,False
    for j in range(recent_idx-1,max(recent_idx-MAX_GAP,0)-1,-1):
        if not twvf[j]: continue
        pl,pw=lv[j],wvfv[j]
        if np.isnan(pl) or np.isnan(pw): continue
        if not no_break_before(lv,j,NO_BREAK_BARS): continue
        if rl<pl and rw>pw: return macd_divergence(j,recent_idx,ml,sl,hist),True
        break
    return False,False

def check_stoch(df):
    close,high,low,open_=df['Close'],df['High'],df['Low'],df['Open']
    bb_lo=close.rolling(BB_LENGTH).mean()-BB_MULT*close.rolling(BB_LENGTH).std(ddof=0)
    trig=(close>open_)&(low<=bb_lo)
    vp=(trig.shift(1).fillna(False)&(close>open_)&(open_>=open_.shift(1))&(close>=open_.shift(1)))
    ll=low.rolling(STOCH_K).min(); hh=high.rolling(STOCH_K).max(); sk=100*(close-ll)/(hh-ll)
    sd=((low<low.shift(1).rolling(STOCH_LOOKBACK).min())&(sk>sk.shift(1).rolling(STOCH_LOOKBACK).min()))
    tlo=low.where(trig).ffill(); nb=low.rolling(LOOKBACK).min()>=tlo
    bhl=close<=0.85*high.rolling(YEAR_HIGH_BARS).max()
    vpv,sdv,nbv,bhlv=vp.values,sd.values,nb.values,bhl.values; n=len(vpv)
    if n<LOOKBACK+1: return False
    if not nbv[-1] or not bhlv[-1]: return False
    if not any(vpv[max(0,n-LOOKBACK):n]): return False
    if not any(sdv[max(0,n-LOOKBACK):n]): return False
    return True

def check_bb_only(df):
    close,low,open_=df['Close'],df['Low'],df['Open']
    bb_lo=close.rolling(BB_LENGTH).mean()-BB_MULT*close.rolling(BB_LENGTH).std(ddof=0)
    trig=(close>open_)&(low<=bb_lo)
    tc=(trig&(close.shift(-1)>open_.shift(-1))&(open_.shift(-1)>=open_)&(close.shift(-1)>=open_))
    tlo=low.where(trig).ffill(); nb=low.rolling(LOOKBACK).min()>=tlo
    tcv,nbv=tc.values,nb.values; n=len(tcv)
    for i in range(n-1,max(n-SCAN_DELAY-2,-1),-1):
        if tcv[i] and nbv[i]:
            cl,lw=float(df['Close'].values[i]),float(df['Low'].values[i])
            if not np.isnan(cl) and not np.isnan(lw):
                if (cl-lw)/cl<=MAX_STOP_DIST: return True
    return False

def check_stoch_macd(df,ml,sl,hist):
    close,high,low,open_=df['Close'],df['High'],df['Low'],df['Open']
    bb_lo=close.rolling(BB_LENGTH).mean()-BB_MULT*close.rolling(BB_LENGTH).std(ddof=0)
    trig=(close>open_)&(low<=bb_lo)
    vp=(trig.shift(1).fillna(False)&(close>open_)&(open_>=open_.shift(1))&(close>=open_.shift(1)))
    ll=low.rolling(STOCH_K).min(); hh=high.rolling(STOCH_K).max(); sk=100*(close-ll)/(hh-ll)
    sd=((low<low.shift(1).rolling(STOCH_LOOKBACK).min())&(sk>sk.shift(1).rolling(STOCH_LOOKBACK).min()))
    tlo=low.where(trig).ffill(); nb=low.rolling(LOOKBACK).min()>=tlo
    bhl=close<=0.85*high.rolling(YEAR_HIGH_BARS).max()
    vpv,sdv,nbv,bhlv=vp.values,sd.values,nb.values,bhl.values; n=len(vpv)
    if n<LOOKBACK+2: return False
    if not nbv[-1] or not bhlv[-1]: return False
    if not any(vpv[max(0,n-LOOKBACK):n]): return False
    if not any(sdv[max(0,n-LOOKBACK):n]): return False
    i,pi=n-1,max(0,n-1-LOOKBACK)
    if np.isnan(hist[i]) or np.isnan(hist[pi]): return False
    return (hist[i]>hist[pi]) or (ml[i]>ml[pi]) or (sl[i]>sl[pi])

# ── Inline backtest signal finders ────────────────────────────────────────────
def _vixfix_bt(df):
    close,low,open_=df['Close'],df['Low'],df['Open']; cv,lv=close.values,low.values; n=len(df)
    bb_lo=close.rolling(BB_LENGTH).mean()-BB_MULT*close.rolling(BB_LENGTH).std(ddof=0)
    trig=(close>open_)&(low<=bb_lo)
    tc=(trig&(close.shift(-1)>open_.shift(-1))&(open_.shift(-1)>=open_)&(close.shift(-1)>=open_))
    hc=close.rolling(VF_PD).max(); wvf=(hc-low)/hc*100
    vf_up=wvf.rolling(VF_BBL).mean()+VF_MULT*wvf.rolling(VF_BBL).std(ddof=0)
    vf_rng=wvf.rolling(VF_LB).max()*VF_PH; is_grn=(wvf>=vf_up)|(wvf>=vf_rng)
    vf_near=pd.Series(False,index=df.index)
    for s in range(VF_NEAR+1):
        vf_near|=is_grn.shift(s).fillna(False).infer_objects(copy=False).astype(bool)
        if s>0: vf_near|=is_grn.shift(-s).fillna(False).infer_objects(copy=False).astype(bool)
    twvf_s=tc&vf_near; wat=pd.Series(np.nan,index=df.index)
    for s in range(-VF_NEAR,VF_NEAR+1):
        sh=wvf.shift(s).fillna(0); wat=wat.combine(sh,lambda a,b: b if np.isnan(a) else max(a,b))
    twvf,wvfv=twvf_s.values,wat.values; ml,sl,hist=compute_macd(close); pairs=[]
    for ri in range(n):
        if not twvf[ri]: continue
        rl,rc,rw=lv[ri],cv[ri],wvfv[ri]
        if np.isnan(rl) or np.isnan(rw): continue
        if not no_break_before(lv,ri,NO_BREAK_BARS): continue
        if (rc-rl)/rc>MAX_STOP_DIST: continue
        if not no_break_after(lv,ri,n-1): continue
        for j in range(ri-1,max(ri-MAX_GAP,0)-1,-1):
            if not twvf[j]: continue
            pl,pw=lv[j],wvfv[j]
            if np.isnan(pl) or np.isnan(pw): continue
            if not no_break_before(lv,j,NO_BREAK_BARS): continue
            if rl<pl and rw>pw:
                pairs.append({'signal_idx':ri,'signal_date':df.index[ri],
                               'entry_price':float(rc),'stop_loss':float(rl),
                               'has_macd':macd_divergence(j,ri,ml,sl,hist)}); break
    return pairs

def _stoch_active_bt(df):
    close,high,low,open_=df['Close'],df['High'],df['Low'],df['Open']
    bb_lo=close.rolling(BB_LENGTH).mean()-BB_MULT*close.rolling(BB_LENGTH).std(ddof=0)
    trig=(close>open_)&(low<=bb_lo)
    vp=(trig.shift(1).fillna(False)&(close>open_)&(open_>=open_.shift(1))&(close>=open_.shift(1)))
    ll=low.rolling(STOCH_K).min(); hh=high.rolling(STOCH_K).max(); sk=100*(close-ll)/(hh-ll)
    sd=((low<low.shift(1).rolling(STOCH_LOOKBACK).min())&(sk>sk.shift(1).rolling(STOCH_LOOKBACK).min()))
    tlo=low.where(trig).ffill(); nb=low.rolling(LOOKBACK).min()>=tlo
    bhl=close<=0.85*high.rolling(YEAR_HIGH_BARS).max()
    vpv,sdv,nbv,bhlv=vp.values,sd.values,nb.values,bhl.values; active=set()
    for i in range(LOOKBACK,len(vpv)):
        if not nbv[i] or not bhlv[i]: continue
        if not any(vpv[max(0,i-LOOKBACK):i]): continue
        if not any(sdv[max(0,i-LOOKBACK):i]): continue
        active.add(i)
    return active

def _stoch_bt(df):
    close,high,low,open_=df['Close'],df['High'],df['Low'],df['Open']; cv,lv=close.values,low.values; n=len(df)
    bb_lo=close.rolling(BB_LENGTH).mean()-BB_MULT*close.rolling(BB_LENGTH).std(ddof=0)
    trig=(close>open_)&(low<=bb_lo)
    vp=(trig.shift(1).fillna(False)&(close>open_)&(open_>=open_.shift(1))&(close>=open_.shift(1)))
    ll=low.rolling(STOCH_K).min(); hh=high.rolling(STOCH_K).max(); sk=100*(close-ll)/(hh-ll)
    sd=((low<low.shift(1).rolling(STOCH_LOOKBACK).min())&(sk>sk.shift(1).rolling(STOCH_LOOKBACK).min()))
    tlo=low.where(trig).ffill(); nb=low.rolling(LOOKBACK).min()>=tlo
    bhl=close<=0.85*high.rolling(YEAR_HIGH_BARS).max()
    vpv,sdv,nbv,bhlv,tv=vp.values,sd.values,nb.values,bhl.values,trig.values; sigs=[]
    for i in range(LOOKBACK,n-1):
        if not nbv[i] or not bhlv[i]: continue
        if not any(vpv[max(0,i-LOOKBACK):i]): continue
        if not any(sdv[max(0,i-LOOKBACK):i]): continue
        ti=next((k for k in range(i,max(i-LOOKBACK,-1),-1) if tv[k]),None)
        if ti is None: continue
        e,s=cv[ti],lv[ti]
        if np.isnan(e) or np.isnan(s): continue
        if (e-s)/e>MAX_STOP_DIST: continue
        if sigs and sigs[-1]['signal_idx']==ti: continue
        sigs.append({'signal_idx':ti,'signal_date':df.index[ti],'entry_price':float(e),'stop_loss':float(s)})
    return sigs

def _bb_only_bt(df):
    close,low,open_=df['Close'],df['Low'],df['Open']; cv,lv=close.values,low.values; n=len(df)
    bb_lo=close.rolling(BB_LENGTH).mean()-BB_MULT*close.rolling(BB_LENGTH).std(ddof=0)
    trig=(close>open_)&(low<=bb_lo)
    tc=(trig&(close.shift(-1)>open_.shift(-1))&(open_.shift(-1)>=open_)&(close.shift(-1)>=open_))
    tlo=low.where(trig).ffill(); nb=low.rolling(LOOKBACK).min()>=tlo; tcv,nbv=tc.values,nb.values; sigs=[]
    for i in range(LOOKBACK,n-1):
        if not tcv[i] or not nbv[i]: continue
        e,s=cv[i],lv[i]
        if np.isnan(e) or np.isnan(s): continue
        if (e-s)/e>MAX_STOP_DIST: continue
        sigs.append({'signal_idx':i,'signal_date':df.index[i],'entry_price':float(e),'stop_loss':float(s)})
    return sigs

def _stoch_macd_bt(df):
    close,high,low,open_=df['Close'],df['High'],df['Low'],df['Open']; cv,lv=close.values,low.values; n=len(df)
    bb_lo=close.rolling(BB_LENGTH).mean()-BB_MULT*close.rolling(BB_LENGTH).std(ddof=0)
    trig=(close>open_)&(low<=bb_lo)
    vp=(trig.shift(1).fillna(False)&(close>open_)&(open_>=open_.shift(1))&(close>=open_.shift(1)))
    ll=low.rolling(STOCH_K).min(); hh=high.rolling(STOCH_K).max(); sk=100*(close-ll)/(hh-ll)
    sd=((low<low.shift(1).rolling(STOCH_LOOKBACK).min())&(sk>sk.shift(1).rolling(STOCH_LOOKBACK).min()))
    tlo=low.where(trig).ffill(); nb=low.rolling(LOOKBACK).min()>=tlo
    bhl=close<=0.85*high.rolling(YEAR_HIGH_BARS).max(); ml,sl_a,hist=compute_macd(close)
    vpv,sdv,nbv,bhlv,tv=vp.values,sd.values,nb.values,bhl.values,trig.values; sigs=[]
    for i in range(LOOKBACK+1,n-1):
        if not nbv[i] or not bhlv[i]: continue
        if not any(vpv[max(0,i-LOOKBACK):i]): continue
        if not any(sdv[max(0,i-LOOKBACK):i]): continue
        pi=max(0,i-LOOKBACK)
        if np.isnan(hist[i]) or np.isnan(hist[pi]): continue
        if not ((hist[i]>hist[pi]) or (ml[i]>ml[pi]) or (sl_a[i]>sl_a[pi])): continue
        ti=next((k for k in range(i,max(i-LOOKBACK,-1),-1) if tv[k]),None)
        if ti is None: continue
        e,s=cv[ti],lv[ti]
        if np.isnan(e) or np.isnan(s): continue
        if (e-s)/e>MAX_STOP_DIST: continue
        if sigs and sigs[-1]['signal_idx']==ti: continue
        sigs.append({'signal_idx':ti,'signal_date':df.index[ti],'entry_price':float(e),'stop_loss':float(s)})
    return sigs

def _eval_bt(df,sig,pos,tgt):
    idx,entry,stop=sig['signal_idx'],sig['entry_price'],sig['stop_loss']
    n,win=len(df),entry*(1+tgt); shares=pos/entry
    result,exit_price,exit_bar='NEUTRAL',None,None
    for w in range(1,HOLD_BARS+1):
        fi=idx+w
        if fi>=n: break
        wh,wl=float(df['High'].iloc[fi]),float(df['Low'].iloc[fi])
        if wh>=win: result,exit_price,exit_bar='WIN',win,w; break
        if wl<=stop: result,exit_price,exit_bar='LOSS',stop,w; break
    if result=='NEUTRAL':
        last=min(idx+HOLD_BARS,n-1); exit_price=float(df['Close'].iloc[last]); exit_bar=min(HOLD_BARS,n-1-idx)
    pct=(exit_price-entry)/entry*100; dollar=shares*(exit_price-entry)
    return {'result':result,'pct_return':pct if not np.isnan(pct) else 0.0,
            'dollar_return':dollar if not np.isnan(dollar) else 0.0}

def _best_ev(trades_by_target):
    best_tgt,best_ev,best_wr=WIN_TARGET,0.0,0.0
    for tgt,trades in trades_by_target.items():
        if not trades: continue
        n=len(trades); wins=[t for t in trades if t['result']=='WIN']; loss=[t for t in trades if t['result']=='LOSS']
        wr=len(wins)/n*100; lr=len(loss)/n*100
        aw=safe_mean([t['pct_return'] for t in wins]); al=safe_mean([t['pct_return'] for t in loss])
        ev=(wr/100*aw)+(lr/100*al)
        if ev>best_ev: best_ev,best_tgt,best_wr=ev,tgt,wr
    return best_tgt,best_wr,best_ev

def run_inline_backtest(tickers):
    print(f'\nRunning inline backtest ({BACKTEST_SAMPLE} ticker sample)...')
    sample=random.sample(tickers,min(BACKTEST_SAMPLE,len(tickers)))
    cutoff=pd.Timestamp.now()-pd.DateOffset(years=YEARS_HISTORY)
    min_bars=max(VF_LB+MAX_GAP,YEAR_HIGH_BARS+LOOKBACK+STOCH_LOOKBACK)+HOLD_BARS+10
    tt={k:{tgt:[] for tgt in EV_TARGETS} for k in ['ultra','high','standard','bb_only','stoch_macd']}
    for ticker in sample:
        try:
            df=yf.download(ticker,period='max',interval=INTERVAL,progress=False,auto_adjust=True)
            if df is None or len(df)<min_bars: continue
            if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
            df=df[df.index>=cutoff].copy()
            if len(df)<min_bars: continue
            cc=df['Close'].dropna()
            if cc.empty: continue
            rc=float(cc.iloc[-1])
            if np.isnan(rc) or rc<MIN_PRICE: continue
            try:
                mc=yf.Ticker(ticker).fast_info.market_cap
                if mc is not None and mc<MIN_MARKET_CAP: continue
            except: pass
            vf=_vixfix_bt(df); sa=_stoch_active_bt(df)
            for sig in vf:
                si,e=sig['signal_idx'],sig['entry_price']
                if e<rc*0.15 or e>rc*6.0: continue
                if si+1>=len(df): continue
                hs=any((si+o) in sa for o in range(-SCAN_DELAY,SCAN_DELAY+1))
                for tgt in EV_TARGETS:
                    t=_eval_bt(df,sig,POSITION_HIGH,tgt)
                    if sig['has_macd'] and hs: tt['ultra'][tgt].append(t)
                    if hs: tt['high'][tgt].append(t)
            for sig in _stoch_bt(df):
                si,e=sig['signal_idx'],sig['entry_price']
                if e<rc*0.15 or e>rc*6.0: continue
                if si+1>=len(df): continue
                for tgt in EV_TARGETS: tt['standard'][tgt].append(_eval_bt(df,sig,POSITION_STD,tgt))
            for sig in _bb_only_bt(df):
                si,e=sig['signal_idx'],sig['entry_price']
                if e<rc*0.15 or e>rc*6.0: continue
                if si+1>=len(df): continue
                for tgt in EV_TARGETS: tt['bb_only'][tgt].append(_eval_bt(df,sig,POSITION_STD,tgt))
            for sig in _stoch_macd_bt(df):
                si,e=sig['signal_idx'],sig['entry_price']
                if e<rc*0.15 or e>rc*6.0: continue
                if si+1>=len(df): continue
                for tgt in EV_TARGETS: tt['stoch_macd'][tgt].append(_eval_bt(df,sig,POSITION_STD,tgt))
        except: pass
        time.sleep(0.03)
    out={}
    for tier in tt:
        td=tt[tier].get(WIN_TARGET,[]); n=len(td)
        if n==0:
            out[tier]={'wr':None,'signals':0,'spm':0.0,'best_target':WIN_TARGET,'best_target_wr':None,'best_ev':None}
            continue
        wins=[t for t in td if t['result']=='WIN']; loss=[t for t in td if t['result']=='LOSS']
        wr=len(wins)/n*100; lr=len(loss)/n*100
        aw=safe_mean([t['pct_return'] for t in wins]); al=safe_mean([t['pct_return'] for t in loss])
        ev=(wr/100*aw)+(lr/100*al); spm=n/(YEARS_HISTORY*12)
        bt,bwr,bev=_best_ev(tt[tier])
        out[tier]={'wr':wr,'signals':n,'spm':spm,'ev':ev,'best_target':bt,'best_target_wr':bwr,'best_ev':bev}
    print('Inline backtest complete.'); return out

def run_scans(tickers):
    ultra,high,standard,bb_only,stoch_macd=[],[],[],[],[]
    min_bars=max(VF_LB+MAX_GAP,YEAR_HIGH_BARS+LOOKBACK+STOCH_LOOKBACK)+10
    print(f'Scanning {len(tickers)} tickers...\n')
    for i,ticker in enumerate(tickers):
        try:
            df=yf.download(ticker,period='3y',interval=INTERVAL,progress=False,auto_adjust=True)
            if df is None or len(df)<min_bars: continue
            if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
            cc=df['Close'].dropna()
            if cc.empty: continue
            cp=float(cc.iloc[-1])
            if np.isnan(cp) or cp<MIN_PRICE: continue
            try:
                mc=yf.Ticker(ticker).fast_info.market_cap
                if mc is not None and mc<MIN_MARKET_CAP: continue
            except: pass
            ml,sl,hist=compute_macd(df['Close'])
            hm,hv=check_vixfix(df,ml,sl,hist); hs=check_stoch(df)
            hb=check_bb_only(df); hsm=check_stoch_macd(df,ml,sl,hist)
            if hm and hv and hs: ultra.append(ticker)
            if hv and hs: high.append(ticker)
            if hs: standard.append(ticker)
            if hb: bb_only.append(ticker)
            if hsm: stoch_macd.append(ticker)
            tags=[]
            if hm and hv and hs: tags.append('ULTRA')
            elif hv and hs: tags.append('HIGH')
            elif hs: tags.append('Stoch')
            if hb: tags.append('BB')
            if hsm: tags.append('Stoch+MACD')
            if tags: print(f'  ✓ {ticker} — {" | ".join(tags)}')
        except Exception as e: print(f'  Error {ticker}: {e}')
        if (i+1)%100==0: print(f'  [{i+1}/{len(tickers)}]')
        time.sleep(0.05)
    return sorted(ultra),sorted(high),sorted(standard),sorted(bb_only),sorted(stoch_macd)

def _tier_html(tier_name,title,description,tickers_this_tier,stats,pos_size,hide_already_in=None):
    wr=stats.get('wr'); n_sigs=stats.get('signals',0); spm=stats.get('spm',0.0)
    best_tgt=stats.get('best_target',WIN_TARGET); best_tgt_wr=stats.get('best_target_wr'); best_ev=stats.get('best_ev')
    pos_note=f'${pos_size:,.0f}/trade'
    if wr is not None and wr>80: pos_note='<strong>$10,000/trade</strong>'
    # Always show win rate if available; only fall back if truly no signals found in sample
    if wr is not None and n_sigs > 0:
        wr_str=(f'<strong>{wr:.1f}%</strong> win rate at {int(WIN_TARGET*100)}% target &nbsp;|&nbsp; '
                f'{n_sigs} signals found in sample &nbsp;|&nbsp; ~{spm:.1f}/mo')
    else:
        wr_str='No signals found in backtest sample — run backtest.py for full results'
    best_ev_str=''
    if best_tgt is not None and best_tgt_wr is not None and best_ev is not None:
        best_ev_str=(f'<br><em>Best EV exit target: <strong>{int(best_tgt*100)}%</strong> &nbsp;|&nbsp; '
                     f'Win rate at that target: <strong>{best_tgt_wr:.1f}%</strong> &nbsp;|&nbsp; '
                     f'EV: {best_ev:+.2f}%</em>')
    excl=set(hide_already_in) if hide_already_in else set()
    show=[t for t in tickers_this_tier if t not in excl]
    if show:
        ticker_html='<br>'.join(f'&nbsp;&nbsp;<span style="color:#cc0000;font-weight:bold;">{t}</span>' for t in show)
        ticker_html+=f'<br><em>Total: {len(show)}'
        if excl: ticker_html+=f' (excl. higher tiers) | All {tier_name}: {len(tickers_this_tier)}'
        ticker_html+='</em>'
    else:
        ticker_html=f'<em>No {tier_name} signals this week.</em>'
    return f'''
<div style="border:1px solid #ccc;border-radius:6px;padding:14px;margin-bottom:18px;background:#fafafa;">
  <h2 style="margin:0 0 6px 0;font-size:1.1em;color:#222;">{title}</h2>
  <p style="margin:2px 0;font-size:0.85em;color:#555;">{description}</p>
  <p style="margin:6px 0;font-size:0.85em;">
    <strong>Backtest ({YEARS_HISTORY}yr, weekly):</strong> {wr_str}{best_ev_str}
  </p>
  <p style="margin:4px 0;font-size:0.85em;">
    <strong>Position:</strong> {pos_note} &nbsp;|&nbsp;
    <strong>Win target:</strong> {int(WIN_TARGET*100)}% &nbsp;|&nbsp;
    <strong>Max hold:</strong> {HOLD_BARS} weeks &nbsp;|&nbsp;
    Entry: close of BB trigger candle &nbsp;|&nbsp; Stop: low of BB trigger candle
  </p>
  <hr style="border:none;border-top:1px solid #ddd;margin:8px 0;">
  <p style="margin:0;font-size:0.95em;line-height:1.9;">{ticker_html}</p>
</div>'''

def build_html_report(ultra,high,standard,bb_only,stoch_macd,bt_stats):
    from datetime import date
    today=date.today().strftime('%B %d, %Y')
    summary_rows=''.join(f'<tr><td>{lbl}</td><td style="text-align:center;">{n}</td></tr>\n'
                         for lbl,n in [('Tier 1 ULTRA',len(ultra)),('Tier 2 HIGH',len(high)),
                                       ('Tier 3 STANDARD',len(standard)),('Tier 3B STD-BB',len(bb_only)),
                                       ('Tier 3C STD-MACD',len(stoch_macd))])
    ts =_tier_html('ULTRA','★★★ TIER 1 — ULTRA CONFIDENCE',
                   'BB Trigger + VixFix div + MACD div + Stochastic div. All four confirmed.',
                   ultra,bt_stats.get('ultra',{}),POSITION_HIGH)
    ts+=_tier_html('HIGH','★★ TIER 2 — HIGH CONFIDENCE',
                   'BB Trigger + VixFix div + Stochastic div. MACD not required.',
                   high,bt_stats.get('high',{}),POSITION_HIGH,hide_already_in=ultra)
    ts+=_tier_html('STANDARD','★ TIER 3 — STANDARD',
                   'BB Trigger + Stochastic div. No VixFix or MACD.',
                   standard,bt_stats.get('standard',{}),POSITION_STD,hide_already_in=set(ultra)|set(high))
    ts+=_tier_html('STD-BB','★ TIER 3B — STANDARD-BB (BB Trigger only)',
                   'Confirmed BB touch trigger candle. No other indicators. Baseline.',
                   bb_only,bt_stats.get('bb_only',{}),POSITION_STD,
                   hide_already_in=set(ultra)|set(high)|set(standard))
    ts+=_tier_html('STD-MACD','★ TIER 3C — STANDARD-MACD',
                   'BB Trigger + Stochastic div + MACD div. No VixFix.',
                   stoch_macd,bt_stats.get('stoch_macd',{}),POSITION_STD,
                   hide_already_in=set(ultra)|set(high))
    return f'''<!DOCTYPE html><html><head><meta charset="utf-8">
<style>body{{font-family:Arial,sans-serif;font-size:14px;color:#222;max-width:680px;margin:0 auto;padding:20px;}}
table{{border-collapse:collapse;width:100%;margin-bottom:14px;}}th,td{{border:1px solid #ccc;padding:5px 10px;font-size:0.85em;}}th{{background:#f0f0f0;}}</style>
</head><body>
<h1 style="font-size:1.3em;margin-bottom:4px;">Weekly Stock Scan — {today}</h1>
<p style="margin:0 0 16px 0;font-size:0.85em;color:#666;">
Interval: Weekly &nbsp;|&nbsp; Universe: NYSE + NASDAQ &nbsp;|&nbsp;
Filters: Price &gt;$10, Mkt cap &gt;$1B, Stop dist &lt;11%</p>
<h3 style="margin:0 0 6px 0;font-size:0.95em;">Signal Count Summary</h3>
<table><tr><th>Tier</th><th>Signals This Week</th></tr>{summary_rows}</table>
{ts}
<p style="font-size:0.75em;color:#999;margin-top:20px;">
Stats from {BACKTEST_SAMPLE}-ticker sample. Run backtest.py for definitive numbers.</p>
</body></html>'''

def send_email(subject,html_body):
    msg=MIMEMultipart('alternative'); msg['From']=GMAIL_USER; msg['To']=TO_EMAIL; msg['Subject']=subject
    msg.attach(MIMEText(html_body,'html'))
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com',465) as s:
            s.login(GMAIL_USER,GMAIL_PASSWORD); s.sendmail(GMAIL_USER,TO_EMAIL,msg.as_string())
        print('Email sent.')
    except Exception as e: print(f'Email failed: {e}')

if __name__=='__main__':
    tickers=get_all_tickers()
    ultra,high,standard,bb_only,stoch_macd=run_scans(tickers)
    bt_stats=run_inline_backtest(tickers)
    html=build_html_report(ultra,high,standard,bb_only,stoch_macd,bt_stats)
    print(f'\n[Done] ULTRA:{len(ultra)} HIGH:{len(high)} STD:{len(standard)} BB:{len(bb_only)} SM:{len(stoch_macd)}')
    if GMAIL_USER:
        from datetime import date
        send_email(f'Weekly Stock Scan — {date.today().strftime("%b %d %Y")}',html)
