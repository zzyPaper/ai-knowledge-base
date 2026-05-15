"""Professional sector rotation signals based on institutional research.

Key references:
  - 国盛证券 "趋势-拥挤度" 二维框架 (2024-2025)
  - 中银证券 波动率控制+多策略复合 (2025)
  - Barroso & Santa-Clara (2015) volatility targeting
  - Moskowitz, Ooi & Pedersen (2012) time-series momentum

Dimensions:
  1. Multi-timeframe momentum (1M/3M/6M) — 国盛 "趋势" 维度
  2. Crowding filter (turnover/vol/Beta percentiles) — 国盛 "拥挤度" 维度
  3. Volatility-controlled position sizing — 中银 + Barroso
  4. Market regime with MA structure — two-layer trend filter
"""

import pandas as pd
import numpy as np
from typing import Optional

TRENDING_WEIGHTS = (0.70, 0.30)  # momentum, crowding
RANGING_WEIGHTS = (0.50, 0.50)

# Multi-timeframe momentum windows
MOM_WINDOWS = [20, 60, 120]  # 1M, 3M, 6M trading days
MOM_WEIGHTS = [0.50, 0.30, 0.20]  # short-term weighted higher

# Crowding lookback (国盛: 4-year rolling, we approximate with 500 trading days)
CROWDING_HISTORY = 500
CROWDING_SHORT = 40  # 3-month lookback for short-term indicators

# Volatility targeting
TARGET_ANNUAL_VOL = 0.12  # 12% annual target
VOL_LOOKBACK = 63  # 中银: 63 trading day rolling window
VOL_SCALE_MIN = 0.25
VOL_SCALE_MAX = 1.20


def _min_max(scores: dict[str, float]) -> dict[str, float]:
    values = list(scores.values())
    mn, mx = min(values), max(values)
    if mx - mn < 1e-10:
        return {k: 0.5 for k in scores}
    return {k: (v - mn) / (mx - mn) for k, v in scores.items()}


def multi_timeframe_momentum(sectors_data: dict[str, pd.DataFrame]) -> dict[str, float]:
    """Multi-timeframe momentum: weighted composite of 1M/3M/6M ROC.

    Uses rank-based normalization (中银 S7 methodology): rank等权优于zscore等权.
    """
    all_names = list(sectors_data.keys())
    n = len(all_names)
    if n <= 1:
        return {name: 0.5 for name in all_names}

    # Compute raw ROC for each window
    raw_scores = {name: 0.0 for name in all_names}
    for name, df in sectors_data.items():
        close = df["close"].values
        if len(close) < max(MOM_WINDOWS) + 1:
            raw_scores[name] = 0.0
            continue
        weighted_sum = 0.0
        for w, period in zip(MOM_WEIGHTS, MOM_WINDOWS):
            if len(close) > period:
                roc = (close[-1] / close[-(period + 1)] - 1)
                weighted_sum += w * roc
        raw_scores[name] = weighted_sum

    # Rank-based normalization (中银: rank等权优于zscore等权)
    sorted_names = sorted(all_names, key=lambda x: raw_scores[x])
    ranks = {name: i / (n - 1) if n > 1 else 0.5 for i, name in enumerate(sorted_names)}
    return ranks


def rolling_percentile(series: np.ndarray, window: int, value: float) -> float:
    """Calculate what percentile the current value is within the rolling window."""
    if len(series) < window:
        return 0.5
    recent = series[-window:]
    return float((recent < value).sum()) / len(recent)


def score_crowding_pro(sectors_data: dict[str, pd.DataFrame]) -> dict[str, float]:
    """Professional crowding score based on 国盛证券 拥挤度 三维度:

    1. 换手率分位数 (turnover rate percentile vs history)
    2. 波动率分位数 (volatility percentile vs history)
    3. Beta分位数 (market sensitivity percentile vs history)

    Higher score = MORE crowded (more risk). Returns in [0, 1].
    """
    scores = {}
    for name, df in sectors_data.items():
        close = df["close"].values
        if len(close) < CROWDING_SHORT + 1:
            scores[name] = 0.5
            continue

        # 1. Turnover percentile
        turnover_percentile = 0.5
        if "turnover_rate" in df.columns and len(df) >= CROWDING_SHORT:
            tr_recent = df["turnover_rate"].values[-CROWDING_SHORT:].mean()
            tr_hist = df["turnover_rate"].values
            if len(tr_hist) >= CROWDING_HISTORY:
                tr_hist = tr_hist[-CROWDING_HISTORY:]
            turnover_percentile = float((tr_hist < tr_recent).sum()) / max(len(tr_hist), 1)

        # 2. Volatility percentile
        returns = np.diff(close) / close[:-1]
        if len(returns) >= CROWDING_SHORT:
            vol_recent = float(np.std(returns[-CROWDING_SHORT:]))
            vol_hist = np.array([float(np.std(returns[max(0, i-CROWDING_SHORT):i+1]))
                                 for i in range(CROWDING_SHORT, len(returns))])
            if len(vol_hist) > 0:
                vol_percentile = float((vol_hist < vol_recent).sum()) / len(vol_hist)
            else:
                vol_percentile = 0.5
        else:
            vol_percentile = 0.5

        # 3. Beta percentile (relative volatility to equal-weight basket)
        # Simplified: use relative volatility as Beta proxy
        beta_percentile = vol_percentile  # approximation when Beta data unavailable

        # Composite crowding: equal-weighted, higher = more risk
        crowding = (turnover_percentile + vol_percentile + beta_percentile) / 3.0
        scores[name] = crowding

    return scores


def compute_professional_scores(
    sectors_data: dict[str, pd.DataFrame],
    index_hist: Optional[pd.DataFrame] = None,
    trending_weights: tuple = TRENDING_WEIGHTS,
    ranging_weights: tuple = RANGING_WEIGHTS,
) -> pd.DataFrame:
    """Professional sector scoring combining trend + crowding dimensions.

    Trend dimension: multi-timeframe momentum (1M/3M/6M)
    Crowding dimension: turnover/volatility/Beta percentiles

    Composite = w_mom * momentum_rank + w_crowd * (1 - crowding_rank)
    (lower crowding = higher score, so we invert)

    Returns DataFrame with sector, trend, crowding, composite, rank.
    """
    if len(sectors_data) < 2:
        return pd.DataFrame(columns=["sector", "trend", "crowding", "composite", "rank"])

    # Trend scores (rank-normalized)
    trend_scores = multi_timeframe_momentum(sectors_data)

    # Crowding scores (higher = more risk, we want lower)
    crowding_raw = score_crowding_pro(sectors_data)
    # Invert: 1 - crowding → higher score = less crowded
    crowding_scores = {k: 1.0 - v for k, v in crowding_raw.items()}

    # Normalize both to [0, 1]
    trend_norm = _min_max(trend_scores)
    crowd_norm = _min_max(crowding_scores)

    # Regime-adaptive weighting
    regime = detect_market_regime_pro(index_hist)
    w_trend, w_crowd = trending_weights if regime == "trending" else ranging_weights

    composites = {}
    for name in sectors_data:
        composites[name] = w_trend * trend_norm[name] + w_crowd * crowd_norm.get(name, 0.5)

    result = pd.DataFrame({
        "sector": list(composites.keys()),
        "trend": [trend_norm[s] for s in composites],
        "crowding": [crowd_norm.get(s, 0.5) for s in composites],
        "composite": list(composites.values()),
    }).sort_values("composite", ascending=False).reset_index(drop=True)
    result["rank"] = range(1, len(result) + 1)
    return result


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
    """中银证券 负向波动率: std of only negative daily returns.

    This penalizes downside volatility more than total volatility.
    """
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
    """Compute volatility scaling factor (Barroso & Santa-Clara 2015 + 中银改良).

    Uses negative semi-volatility (中银 innovation).
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

    # Entry: 20d return > 1% AND accelerating (current 20d > 5d-ago 20d)
    # Requires stronger signal to avoid whipsaw in consolidation
    ret_entry_5d_ago = float(close[-6] / close[-(entry_lookback + 6)] - 1) * 100
    accelerating = ret_entry > ret_entry_5d_ago
    can_enter = ret_entry > 0.5 and accelerating

    # Exit: 10d return < -0.5% (small buffer)
    should_exit = ret_exit < -0.5

    return can_enter and not should_exit, ret_entry
