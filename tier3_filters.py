"""
tier3_filters.py — Tier 3 TEST quality filters
────────────────────────────────────────────────
Drop in repo root alongside the backtest and scanner scripts.

Three filters (ALL must pass — UNKNOWN = soft pass, no data = don't penalise):

  A. Fundamental quality  — positive trailing FCF OR positive YoY EPS growth.
     Distinguishes a genuinely-troubled business from a temporarily-oversold
     quality name (the SBH problem).

  B. Max drawdown cap     — reject if down >= 70% from rolling high.
     Extreme drawdowns almost always reflect real business deterioration.
     (The existing filter already requires price < 85% of high — this adds a
     hard floor that kills the truly broken names.)

  C. Earnings exclusion   — reject if signal is within 14 calendar days of a
     known earnings date (past or future).
     Avoids binary earnings risk; doesn't conflict with mean-reversion logic.
"""

from __future__ import annotations
import datetime
from functools import lru_cache
import yfinance as yf

# ── tuneable constants ────────────────────────────────────────────────────────
MAX_DRAWDOWN_FROM_HIGH = 0.70   # reject if price dropped >= 70% from rolling high
EARNINGS_WINDOW_DAYS   = 14     # calendar days either side of an earnings date

PASS    = "PASS"
FAIL    = "FAIL"
UNKNOWN = "UNKNOWN"   # no yfinance data — treated as soft PASS


# ── A. Fundamental quality ────────────────────────────────────────────────────

@lru_cache(maxsize=4096)
def _fundamentals(ticker: str):
    """Return (fcf, pos_eps_growth) for ticker. Cached per-process."""
    try:
        info = yf.Ticker(ticker).info
        fcf  = info.get("freeCashflow")      # trailing 12-month FCF in USD
        eg   = info.get("earningsGrowth")    # YoY EPS growth, e.g. 0.12 = +12%
        pos_eps = (eg is not None and eg > 0)
        return fcf, pos_eps
    except Exception:
        return None, None


def fundamental_filter(ticker: str) -> str:
    fcf, pos_eps = _fundamentals(ticker)
    if fcf is None and pos_eps is None:
        return UNKNOWN
    if (fcf is not None and fcf > 0) or pos_eps:
        return PASS
    return FAIL


# ── B. Max drawdown cap ───────────────────────────────────────────────────────

def drawdown_filter(current_price: float, year_high: float) -> str:
    """Reject if price has dropped >= MAX_DRAWDOWN_FROM_HIGH from rolling high."""
    if not year_high or year_high <= 0:
        return UNKNOWN
    if (1.0 - current_price / year_high) >= MAX_DRAWDOWN_FROM_HIGH:
        return FAIL
    return PASS


# ── C. Earnings exclusion window ──────────────────────────────────────────────

@lru_cache(maxsize=4096)
def _earnings_dates(ticker: str) -> tuple:
    """Return a tuple of datetime.date objects for all known earnings dates."""
    try:
        ed = yf.Ticker(ticker).earnings_dates
        if ed is None or ed.empty:
            return ()
        return tuple(ts.date() for ts in ed.index)
    except Exception:
        return ()


def earnings_filter(ticker: str, signal_date: datetime.date) -> str:
    """Fail if signal is within EARNINGS_WINDOW_DAYS of any known earnings date."""
    dates = _earnings_dates(ticker)
    if not dates:
        return UNKNOWN
    for ed in dates:
        if abs((signal_date - ed).days) <= EARNINGS_WINDOW_DAYS:
            return FAIL
    return PASS


# ── Combined gate ─────────────────────────────────────────────────────────────

def passes_tier3_filters(
    ticker:        str,
    signal_date:   datetime.date,
    current_price: float,
    year_high:     float,
) -> bool:
    """
    Return True if the signal passes all three Tier 3 quality filters.
    UNKNOWN (no yfinance data) is treated as a soft PASS so we don't
    silently discard signals just because yfinance returned nothing.
    """
    fa = fundamental_filter(ticker)
    fb = drawdown_filter(current_price, year_high)
    fc = earnings_filter(ticker, signal_date)
    return fa != FAIL and fb != FAIL and fc != FAIL
