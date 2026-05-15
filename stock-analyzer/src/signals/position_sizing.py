"""Position sizing based on market absolute momentum (Antonacci / AQR approach).

Determines total position size by measuring the broad market's own trend
over a 20-day window (approx 1 month in A-shares). This replaces the previous
MA20 deviation approach which had -0.206 correlation with next-day returns.

Reference:
  - Antonacci, "Risk Premia Harvesting Through Dual Momentum" (2012)
  - Faber, "A Quantitative Approach to Tactical Asset Allocation" (2007)
"""


def compute_position_pct(index_hist, lookback: int = 20) -> int:
    """Determine total position size (%) based on market absolute momentum.

    Uses the 沪深300 index return over `lookback` trading days.
    Returns 0-100.
    """
    if index_hist is None or len(index_hist) < lookback + 1:
        return 60  # conservative default when insufficient data

    close = float(index_hist["close"].iloc[-1])
    prev = float(index_hist["close"].iloc[-(lookback + 1)])

    if prev == 0:
        return 60

    ret_pct = (close / prev - 1) * 100

    # Map return to position %
    # ret:  +15% → 100%,  +5% → 80%,   0% → 60%,  -5% → 35%,  -10% → 10%
    if ret_pct >= 10:
        return 100
    elif ret_pct >= 5:
        return int(80 + (ret_pct - 5) * 4)     # 5-10% -> 80-100%
    elif ret_pct >= 0:
        return int(60 + ret_pct * 4)            # 0-5% -> 60-80%
    elif ret_pct >= -5:
        return int(35 + (ret_pct + 5) * 5)      # -5-0% -> 35-60%
    elif ret_pct >= -10:
        return int(10 + (ret_pct + 10) * 5)     # -10-(-5)% -> 10-35%
    else:
        return 10  # deeply negative → minimum exposure
