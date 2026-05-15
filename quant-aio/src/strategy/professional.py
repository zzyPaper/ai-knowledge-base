"""Professional utilities: regime detection, volatility scaling, market timing.

Shared by V3 engine and other professional strategies.
"""

import numpy as np
import pandas as pd
from typing import Optional

# Volatility targeting
TARGET_ANNUAL_VOL = 0.12  # 12% annual target
VOL_LOOKBACK = 63  # trading days
VOL_SCALE_MIN = 0.25
VOL_SCALE_MAX = 1.20


def detect_market_regime_pro(index_hist: Optional[pd.DataFrame]) -> str:
    """Detect trending vs ranging with MA structure (20/60/120 day).

    Trending: close > MA20 AND MA20 > MA60 (aligned upward structure).
    """
    if index_hist is None or len(index_hist) < 60:
        return "ranging"
    close = index_hist["close"].values
    ma20 = float(np.mean(close[-20:]))
    ma60 = float(np.mean(close[-60:]))
    if close[-1] > ma20 and ma20 > ma60:
        return "trending"
    return "ranging"


def negative_semi_volatility(index_hist: pd.DataFrame, lookback: int = VOL_LOOKBACK) -> float:
    """Negative semi-volatility: std of only negative daily returns."""
    if len(index_hist) < max(lookback, 2):
        return 0.01
    recent = index_hist.tail(lookback + 1).copy()
    rets = recent["close"].pct_change().dropna().values
    neg_rets = rets[rets < 0]
    if len(neg_rets) < 3:
        return float(np.std(rets))
    return float(np.std(neg_rets))


def volatility_scale(index_data: pd.DataFrame, date: pd.Timestamp,
                     lookback: int = VOL_LOOKBACK,
                     target_vol: float = TARGET_ANNUAL_VOL) -> float:
    """Compute volatility scaling factor.

    Returns scale in [VOL_SCALE_MIN, VOL_SCALE_MAX].
    """
    hist = index_data[(index_data["date"] <= date)].tail(lookback + 1)
    if len(hist) < 10:
        return 1.0
    semi_vol = negative_semi_volatility(hist, lookback)
    annual_vol = semi_vol * np.sqrt(252)
    if annual_vol < 0.005:
        return VOL_SCALE_MAX
    scale = target_vol / annual_vol
    return max(VOL_SCALE_MIN, min(scale, VOL_SCALE_MAX))


def index_trend_filter(index: pd.DataFrame, date: pd.Timestamp,
                       entry_lookback: int = 20, exit_lookback: int = 10) -> tuple[bool, float]:
    """Two-speed market timing: slow entry (20d), fast exit (10d).

    Entry requires 20d return > 0 AND 20d return accelerating (5d delta > 0).
    Exit triggers when 10d return < -0.5%.

    Returns (in_market, trend_strength_pct).
    """
    hist = index[(index["date"] <= date)]
    if len(hist) < entry_lookback + 6:
        return False, 0.0

    close = hist["close"].values
    ret_entry = float(close[-1] / close[-(entry_lookback + 1)] - 1) * 100  # pct
    ret_exit = float(close[-1] / close[-(exit_lookback + 1)] - 1) * 100    # pct

    # Entry: 20d return > 0.5% AND accelerating
    ret_entry_5d_ago = float(close[-6] / close[-(entry_lookback + 6)] - 1) * 100
    accelerating = ret_entry > ret_entry_5d_ago
    can_enter = ret_entry > 0.5 and accelerating

    # Exit: 10d return < -0.5%
    should_exit = ret_exit < -0.5

    return can_enter and not should_exit, ret_entry
