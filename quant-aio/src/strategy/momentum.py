"""
动量策略 —— 基于价格动量和均线偏离度。

核心逻辑：
1. 短期动量（5 日 ROC）：反映近期趋势强度
2. 中期均线偏离（价格 / MA10 - 1）：反映趋势持续性
3. 量价确认：放量上涨加分，缩量上涨减分
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def calc_roc(close: pd.Series, period: int = 5) -> pd.Series:
    """变化率 (Rate of Change)。"""
    return close.pct_change(periods=period) * 100


def calc_ma_ratio(close: pd.Series, period: int = 10) -> pd.Series:
    """价格 / 均线 - 1，衡量均线偏离度。"""
    ma = close.rolling(period).mean()
    return (close / ma - 1) * 100


def calc_volume_momentum(volume: pd.Series, period: int = 5) -> pd.Series:
    """量能动量：当前成交量 / 过去 N 日均量。"""
    avg_vol = volume.rolling(period).mean()
    return volume / avg_vol


def score_momentum(
    hist: pd.DataFrame,
    roc_period: int = 5,
    ma_period: int = 10,
    vol_period: int = 5,
) -> float:
    """计算动量策略综合得分。

    Returns
    -------
    float : [-1, 1] 区间，正值看涨，负值看跌。
    """
    if hist is None or len(hist) < ma_period + 1:
        return 0.0

    close = hist["close"]
    volume = hist.get("volume", pd.Series(dtype=float))

    roc = calc_roc(close, roc_period).iloc[-1]
    ma_ratio = calc_ma_ratio(close, ma_period).iloc[-1]

    # 基础动量分
    score = 0.0
    score += np.clip(roc / 5.0, -1, 1) * 0.4    # ROC 权重 40%
    score += np.clip(ma_ratio / 5.0, -1, 1) * 0.4  # 均线偏离权重 40%

    # 量价确认
    if not volume.empty and len(volume) >= vol_period:
        vol_mom = calc_volume_momentum(volume, vol_period).iloc[-1]
        if roc > 0 and vol_mom > 1.2:
            score += 0.2   # 放量上涨加分
        elif roc > 0 and vol_mom < 0.8:
            score -= 0.1   # 缩量上涨减分
        elif roc < 0 and vol_mom > 1.2:
            score -= 0.2   # 放量下跌减分
        elif roc < 0 and vol_mom < 0.8:
            score += 0.1   # 缩量下跌（可能见底）微加分

    return float(np.clip(score, -1, 1))


def generate_signal(score: float, buy_threshold: float = 0.3, sell_threshold: float = -0.3) -> str:
    """将得分转为买卖信号。"""
    if score > buy_threshold:
        return "BUY"
    elif score < sell_threshold:
        return "SELL"
    return "HOLD"
