"""Advanced momentum signals: path-adjusted and idiosyncratic momentum.

References:
  - Path-adjusted momentum: penalize high-volatility paths (variance ratio)
  - Idiosyncratic momentum: Blitz, Huij & Martens (2011)
  - Moskowitz & Grinblatt (1999) industry momentum
"""

import numpy as np
import pandas as pd
from typing import Optional

# Multi-timeframe momentum windows — shortened for short-history data
MOM_WINDOWS = [10, 20, 40]
MOM_WEIGHTS = [0.40, 0.35, 0.25]


def calc_path_adjusted_momentum(close: np.ndarray, period: int) -> float:
    """Path-adjusted momentum: raw momentum penalized by path volatility.

    Two stocks with same 20-day return: one smooth uptrend, one whipsaw.
    The smooth one gets higher score.

    Formula: raw_mom / (1 + annualized_volatility_ratio)
    where vol_ratio = std(daily_returns) * sqrt(252) / |raw_mom| * |period|
    """
    if len(close) < period + 1:
        return 0.0

    raw_mom = close[-1] / close[-(period + 1)] - 1.0
    daily_rets = np.diff(close[-(period + 1):]) / close[-(period + 1):-1]
    if len(daily_rets) < 2:
        return raw_mom

    std_daily = float(np.std(daily_rets))
    if std_daily < 1e-10:
        return raw_mom  # perfectly smooth, no penalty

    # Annualized volatility
    ann_vol = std_daily * np.sqrt(252)
    # Volatility ratio: how much vol per unit return
    vol_ratio = ann_vol / (abs(raw_mom) + 1e-10)
    # Penalize high vol_ratio: cap penalty at 3x
    penalty = min(vol_ratio, 3.0)
    return raw_mom / (1.0 + penalty * 0.5)


def calc_idiosyncratic_momentum(
    close: np.ndarray,
    market_close: np.ndarray,
    period: int = 60,
) -> float:
    """Idiosyncratic momentum: residual momentum after removing market Beta.

    Steps:
      1. Compute daily returns for sector and market
      2. Rolling regression Beta over lookback
      3. Cumulative residual return over period

    Higher idio_mom = sector outperforming on its own merit, not just Beta.
    """
    if len(close) < period + 2 or len(market_close) < period + 2:
        return 0.0

    # Align lengths
    n = min(len(close), len(market_close))
    sec_ret = np.diff(close[-n:]) / close[-n:-1]
    mkt_ret = np.diff(market_close[-n:]) / market_close[-n:-1]

    if len(sec_ret) < period or len(mkt_ret) < period:
        return 0.0

    # Rolling regression Beta on last 60 days (or available)
    reg_window = min(60, len(sec_ret))
    x = mkt_ret[-reg_window:]
    y = sec_ret[-reg_window:]

    x_mean, y_mean = np.mean(x), np.mean(y)
    cov = np.mean((x - x_mean) * (y - y_mean))
    var = np.mean((x - x_mean) ** 2)
    beta = cov / (var + 1e-10)
    alpha = y_mean - beta * x_mean

    # Residual returns
    residuals = y[-period:] - (alpha + beta * mkt_ret[-period:])

    # Cumulative residual return
    cum_residual = float(np.prod(1.0 + residuals) - 1.0)
    return cum_residual


def multi_timeframe_path_momentum(
    close: np.ndarray,
    market_close: Optional[np.ndarray] = None,
    windows: list[int] = None,
    weights: list[float] = None,
) -> dict[str, float]:
    """Compute multi-timeframe path-adjusted + idiosyncratic momentum composite.

    Returns dict with keys: 'path_mom', 'idio_mom', 'composite'.
    """
    windows = windows or MOM_WINDOWS
    weights = weights or MOM_WEIGHTS

    # Path-adjusted momentum across timeframes
    path_scores = []
    for w, wt in zip(windows, weights):
        if len(close) >= w + 1:
            pm = calc_path_adjusted_momentum(close, w)
            path_scores.append(wt * pm)

    path_mom = sum(path_scores) if path_scores else 0.0

    # Idiosyncratic momentum (use longest window)
    idio_mom = 0.0
    if market_close is not None and len(market_close) > 0:
        idio_mom = calc_idiosyncratic_momentum(close, market_close, period=max(windows))

    # Composite: 70% path-adjusted + 30% idiosyncratic
    composite = 0.7 * path_mom + 0.3 * idio_mom

    return {
        "path_mom": path_mom,
        "idio_mom": idio_mom,
        "composite": composite,
    }


def rank_normalize_dict(scores: dict[str, float]) -> dict[str, float]:
    """Rank-normalize scores to [0, 1]."""
    if len(scores) <= 1:
        return {k: 0.5 for k in scores}
    sorted_items = sorted(scores.items(), key=lambda x: x[1])
    n = len(sorted_items)
    return {name: i / (n - 1) for i, (name, _) in enumerate(sorted_items)}


def compute_trend_dimension(
    sectors_data: dict[str, pd.DataFrame],
    index_hist: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Compute trend dimension scores for all sectors.

    Returns DataFrame with columns: sector, path_mom, idio_mom, trend_score.
    """
    market_close = None
    if index_hist is not None and not index_hist.empty and "close" in index_hist.columns:
        market_close = index_hist["close"].values

    records = []
    for name, df in sectors_data.items():
        close = df["close"].values
        if len(close) < max(MOM_WINDOWS) + 1:
            records.append({
                "sector": name,
                "path_mom": 0.0,
                "idio_mom": 0.0,
                "trend_raw": 0.0,
            })
            continue

        result = multi_timeframe_path_momentum(close, market_close)
        records.append({
            "sector": name,
            "path_mom": result["path_mom"],
            "idio_mom": result["idio_mom"],
            "trend_raw": result["composite"],
        })

    df = pd.DataFrame(records)
    if df.empty:
        return df

    # Rank-normalize trend_raw to [0, 1]
    trend_raw_dict = dict(zip(df["sector"], df["trend_raw"]))
    trend_norm = rank_normalize_dict(trend_raw_dict)
    df["trend_score"] = df["sector"].map(trend_norm)
    return df.sort_values("trend_score", ascending=False).reset_index(drop=True)
