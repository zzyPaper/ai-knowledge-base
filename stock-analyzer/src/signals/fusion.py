"""Multi-signal fusion with Dual Momentum (Antonacci, 2012) approach.

Core methodology:
  1. Absolute momentum filter (time-series): only sectors with positive >= lookback return
  2. Relative momentum ranking (cross-sectional): rank qualifying sectors by momentum
  3. Congestion as secondary signal for tiebreaking

References:
  - Antonacci, "Risk Premia Harvesting Through Dual Momentum" (2012)
  - Moskowitz & Grinblatt, "Industry Momentum" (1999)
  - Jegadeesh & Titman, "Returns to Buying Winners" (1993)
"""

from typing import Optional
import pandas as pd
import numpy as np
from src.features.momentum import rank_momentum, calc_roc
from src.features.congestion import score_congestion

TRENDING_WEIGHTS = (0.80, 0.20)   # momentum + congestion
RANGING_WEIGHTS = (0.60, 0.40)


def absolute_momentum_filter(
    sectors_data: dict[str, pd.DataFrame],
    index_hist: Optional[pd.DataFrame] = None,
    lookback: int = 10,
    min_return_pct: float = 0.0,
    max_return_pct: float = 15.0,
) -> set[str]:
    """Apply absolute momentum filter (Antonacci Dual Momentum) with overbought cap.

    Sector qualifies only if:
      1. Its return over lookback is >= market_threshold (absolute momentum)
      2. Its return is NOT excessively high (avoids overbought sectors)

    Applies AQR-style skip-extreme logic: sectors that have run too far too fast
    tend to reverse (short-term reversal effect, Jegadeesh 1990).

    Returns set of sector names that pass the filter.
    """
    qualified = set()

    market_threshold = min_return_pct
    if index_hist is not None and len(index_hist) >= lookback:
        index_close = index_hist["close"].values
        index_ret = (index_close[-1] / index_close[-(lookback + 1)] - 1) * 100
        if index_ret < -7:
            market_threshold = 3.0
        elif index_ret < -3:
            market_threshold = 1.0

    for name, df in sectors_data.items():
        close = df["close"].values
        if len(close) < lookback + 1:
            continue
        ret_pct = (close[-1] / close[-(lookback + 1)] - 1) * 100
        # Lower bound: minimum trend filter
        if ret_pct < market_threshold:
            continue
        # Upper bound: skip overbought (short-term reversal protection)
        if ret_pct > max_return_pct:
            continue
        qualified.add(name)

    return qualified


def detect_market_regime(index_hist: Optional[pd.DataFrame]) -> str:
    """Detect trending vs ranging market based on index MA deviation."""
    if index_hist is None or len(index_hist) < 20:
        return "ranging"
    ma20 = index_hist["close"].rolling(20).mean().iloc[-1]
    close = index_hist["close"].iloc[-1]
    deviation = abs(close / ma20 - 1)
    return "trending" if deviation > 0.02 else "ranging"


def _min_max(scores: dict[str, float]) -> dict[str, float]:
    values = list(scores.values())
    mn, mx = min(values), max(values)
    if mx - mn < 1e-10:
        return {k: 0.5 for k in scores}
    return {k: (v - mn) / (mx - mn) for k, v in scores.items()}


def compute_sector_scores(
    sectors_data: dict[str, pd.DataFrame],
    index_hist: Optional[pd.DataFrame] = None,
    ma_period: int = 20,
    roc_period: int = 20,
    lookback: int = 10,
    trending_weights: tuple = TRENDING_WEIGHTS,
    ranging_weights: tuple = RANGING_WEIGHTS,
) -> pd.DataFrame:
    """Compute composite scores with Dual Momentum approach.

    1. Apply absolute momentum filter (only sectors with positive trend qualify)
    2. Rank qualifying sectors by relative momentum
    3. Congestion score used as secondary signal (tiebreaker)
    4. Return DataFrame with sector, momentum, congestion, composite, rank

    Returns empty DataFrame if no sectors pass the filter.
    """
    # Step 1: Absolute momentum filter
    qualified = absolute_momentum_filter(sectors_data, index_hist, lookback)

    if not qualified:
        # No sectors pass — return empty. The caller should hold cash.
        return pd.DataFrame(columns=["sector", "momentum", "congestion", "composite", "rank"])

    # Step 2: Relative momentum ranking among qualified sectors
    filtered_data = {name: sectors_data[name] for name in qualified}
    momentum_scores = rank_momentum(filtered_data, ma_period, roc_period)

    # Step 3: Congestion scores (secondary)
    congestion_scores = score_congestion(filtered_data)

    mom_norm = _min_max(momentum_scores)
    cong_norm = _min_max(congestion_scores) if len(congestion_scores) > 1 else {s: 0.5 for s in qualified}

    # Step 4: Regime-adaptive weighting
    regime = detect_market_regime(index_hist)
    w_mom, w_cong = trending_weights if regime == "trending" else ranging_weights

    composites = {}
    for s in qualified:
        composites[s] = w_mom * mom_norm[s] + w_cong * cong_norm.get(s, 0.5)

    result = pd.DataFrame({
        "sector": list(composites.keys()),
        "momentum": [mom_norm[s] for s in composites],
        "congestion": [cong_norm.get(s, 0.5) for s in composites],
        "composite": list(composites.values()),
    }).sort_values("composite", ascending=False).reset_index(drop=True)
    result["rank"] = range(1, len(result) + 1)
    return result
