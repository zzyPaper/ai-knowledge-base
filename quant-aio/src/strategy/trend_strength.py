"""
趋势强度因子 —— 对数价格线性回归斜率 × R²。

逻辑：稳步上涨（R²≈0.9）得分远高于暴涨暴跌（R²≈0.3），
因为前者更可能延续趋势。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def calc_trend_strength(
    close: pd.Series,
    window: int = 25,
) -> float:
    """计算趋势强度得分。

    对收盘价取对数后做线性回归，用 (年化斜率 × R²) 衡量
    "涨得稳不稳"。

    Parameters
    ----------
    close : 收盘价序列
    window : 回归窗口（交易日数），默认25

    Returns
    -------
    float : 趋势得分，正值=上涨趋势，负值=下跌趋势
    """
    if close is None or len(close) < window:
        return 0.0

    closes = close.iloc[-window:].values
    if len(closes) < window:
        return 0.0

    # 去除 NaN
    closes = closes[~np.isnan(closes)]
    if len(closes) < max(window // 2, 10):
        return 0.0

    try:
        log_prices = np.log(closes)
        x = np.arange(len(log_prices))

        # 线性回归
        slope, intercept = np.polyfit(x, log_prices, 1)

        # 年化斜率
        annualized_return = np.exp(slope) ** 250 - 1

        # R² 决定系数
        y_pred = slope * x + intercept
        ss_tot = np.sum((log_prices - np.mean(log_prices)) ** 2)
        ss_res = np.sum((log_prices - y_pred) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        score = annualized_return * r_squared
        # 归一化到 [-1, 1] 区间（年化50%为满分）
        return float(np.clip(score / 0.5, -1, 1))
    except Exception:
        return 0.0


def score_trend_strength(hist: pd.DataFrame, window: int = 25) -> float:
    """计算趋势强度策略得分。

    Returns
    -------
    float : [-1, 1] 区间
    """
    if hist is None or len(hist) < window:
        return 0.0
    return calc_trend_strength(hist["close"], window)
