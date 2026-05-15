"""
短期动量因子 —— 5日/10日 ROC 加权组合。

逻辑：5日ROC权重更大，更敏感地反映近期资金流向变化；
10日ROC提供中期确认。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def calc_roc(close: pd.Series, period: int = 5) -> float:
    """计算变化率。"""
    if close is None or len(close) < period + 1:
        return 0.0
    val = (close.iloc[-1] / close.iloc[-period - 1] - 1) * 100
    return float(val) if not np.isnan(val) else 0.0


def score_short_momentum(
    hist: pd.DataFrame,
    roc5_weight: float = 0.6,
    roc10_weight: float = 0.4,
) -> float:
    """计算短期动量策略得分。

    综合得分 = 0.6 × 5日ROC + 0.4 × 10日ROC
    归一化到 [-1, 1]（10%为满分）

    Returns
    -------
    float : [-1, 1] 区间
    """
    if hist is None or len(hist) < 11:
        return 0.0

    close = hist["close"]
    roc5 = calc_roc(close, 5)
    roc10 = calc_roc(close, 10)

    raw = roc5_weight * roc5 + roc10_weight * roc10
    # 归一化：10%的ROC → 得分1.0
    return float(np.clip(raw / 10.0, -1, 1))
