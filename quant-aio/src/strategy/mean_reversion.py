"""
均值回归策略 —— 涨多了会跌，跌多了会涨。

核心逻辑：
1. 布林带偏离：价格触及上轨 → 超买，触及下轨 → 超卖
2. RSI 超买超卖：RSI > 70 超买，RSI < 30 超卖
3. 均值回归分 = 1 - 动量分（逆向操作）
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI (Relative Strength Index)。"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def calc_bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    """布林带。返回 (upper, middle, lower)。"""
    middle = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower


def score_mean_reversion(
    hist: pd.DataFrame,
    rsi_period: int = 14,
    boll_period: int = 20,
) -> float:
    """均值回归策略得分。

    超买 → 负分（建议卖出）
    超卖 → 正分（建议买入）
    """
    if hist is None or len(hist) < max(rsi_period, boll_period) + 1:
        return 0.0

    close = hist["close"]
    score = 0.0

    # RSI
    rsi = calc_rsi(close, rsi_period)
    last_rsi = rsi.iloc[-1]
    if not np.isnan(last_rsi):
        if last_rsi > 70:
            score -= (last_rsi - 70) / 30  # 超买程度
        elif last_rsi < 30:
            score += (30 - last_rsi) / 30  # 超卖程度

    # 布林带
    upper, middle, lower = calc_bollinger(close, boll_period)
    last_close = close.iloc[-1]
    last_upper = upper.iloc[-1]
    last_lower = lower.iloc[-1]
    last_mid = middle.iloc[-1]
    if not np.isnan(last_upper) and not np.isnan(last_lower):
        bandwidth = last_upper - last_lower
        if bandwidth > 0:
            position = (last_close - last_mid) / (bandwidth / 2)
            score -= np.clip(position, -1, 1) * 0.5  # 偏离中轨越远越看反向

    return float(np.clip(score, -1, 1))
