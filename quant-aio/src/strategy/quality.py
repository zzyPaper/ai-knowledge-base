"""Quality dimension: momentum path quality, Sharpe ratio, drawdown recovery.

New dimension for V3 to distinguish sectors with same momentum but different
risk-adjusted path characteristics.
"""

import numpy as np
import pandas as pd
from typing import Optional


def calc_up_day_ratio(close: np.ndarray, period: int = 20) -> float:
    """Fraction of up days in the period. Higher = smoother uptrend."""
    if len(close) < period + 1:
        return 0.5
    rets = np.diff(close[-(period + 1):]) / close[-(period + 1):-1]
    return float(np.mean(rets > 0))


def calc_sharpe_ratio(close: np.ndarray, period: int = 60) -> float:
    """Annualized Sharpe ratio over period."""
    if len(close) < period + 1:
        return 0.0
    rets = np.diff(close[-(period + 1):]) / close[-(period + 1):-1]
    if len(rets) < 2:
        return 0.0
    mean_ret = np.mean(rets)
    std_ret = np.std(rets)
    if std_ret < 1e-10:
        return 10.0 if mean_ret > 0 else -10.0
    return float(mean_ret / std_ret * np.sqrt(252))


def calc_drawdown_recovery(close: np.ndarray, period: int = 60) -> float:
    """How close is current price to recent high? Higher = better recovery.

    Returns 1.0 if at new high, 0.0 if at period low.
    """
    if len(close) < period:
        return 0.5
    window = close[-period:]
    peak = np.maximum.accumulate(window)
    current_dd = (window[-1] - peak[-1]) / peak[-1] if peak[-1] > 0 else 0
    # Also compute max DD in period
    drawdowns = (window - peak) / peak
    max_dd = np.min(drawdowns)
    if max_dd >= -1e-10:
        return 1.0  # no drawdown
    # Normalize: current_dd / max_dd, closer to 0 = better
    recovery = 1.0 - (current_dd / (max_dd - 1e-10))
    return float(np.clip(recovery, 0.0, 1.0))


def calc_max_consecutive_up(close: np.ndarray, period: int = 20) -> float:
    """Max consecutive up days / period. Higher = stronger trend."""
    if len(close) < period + 1:
        return 0.0
    rets = np.diff(close[-(period + 1):]) / close[-(period + 1):-1]
    max_streak = 0
    current = 0
    for r in rets:
        if r > 0:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return min(max_streak / 10.0, 1.0)  # normalize: 10 consecutive up = 1.0


def compute_quality_dimension(
    sectors_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Compute quality dimension scores for all sectors.

    Returns DataFrame with columns: sector, up_ratio, sharpe, recovery,
    max_streak, quality_score.
    """
    records = []
    for name, df in sectors_data.items():
        close = df["close"].values
        if len(close) < 21:
            records.append({
                "sector": name,
                "up_ratio": 0.5,
                "sharpe": 0.0,
                "recovery": 0.5,
                "max_streak": 0.0,
                "quality_raw": 0.0,
            })
            continue

        up_ratio = calc_up_day_ratio(close, 20)
        sharpe = calc_sharpe_ratio(close, min(60, len(close) - 1))
        recovery = calc_drawdown_recovery(close, min(60, len(close)))
        streak = calc_max_consecutive_up(close, 20)

        # Sharpe can be negative; normalize to [-1, 1] then [0, 1]
        sharpe_norm = np.clip(sharpe / 3.0, -1.0, 1.0)  # assume 3 is excellent
        sharpe_score = (sharpe_norm + 1.0) / 2.0

        # Composite quality
        quality_raw = (up_ratio * 0.25 + sharpe_score * 0.35 +
                       recovery * 0.25 + streak * 0.15)

        records.append({
            "sector": name,
            "up_ratio": up_ratio,
            "sharpe": sharpe,
            "recovery": recovery,
            "max_streak": streak,
            "quality_raw": quality_raw,
        })

    df = pd.DataFrame(records)
    if df.empty:
        return df

    # Rank-normalize quality_raw
    sorted_items = sorted(zip(df["sector"], df["quality_raw"]), key=lambda x: x[1])
    n = len(sorted_items)
    quality_norm = {name: i / (n - 1) if n > 1 else 0.5 for i, (name, _) in enumerate(sorted_items)}
    df["quality_score"] = df["sector"].map(quality_norm)
    return df.sort_values("quality_score", ascending=False).reset_index(drop=True)
