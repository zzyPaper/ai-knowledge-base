"""
均线策略 —— 基于均线交叉（MA 金叉 / 死叉）。

核心逻辑：
1. 短期均线（MA5）上穿长期均线（MA20）→ 金叉买入
2. 短期均线下穿长期均线 → 死叉卖出
3. 结合 MACD 辅助确认
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def calc_ema(series: pd.Series, period: int) -> pd.Series:
    """指数移动平均。"""
    return series.ewm(span=period, adjust=False).mean()


def calc_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """计算 MACD。返回 (DIF, DEA, MACD柱)。"""
    ema_fast = calc_ema(close, fast)
    ema_slow = calc_ema(close, slow)
    dif = ema_fast - ema_slow
    dea = calc_ema(dif, signal)
    macd_bar = 2 * (dif - dea)
    return dif, dea, macd_bar


def score_ma_crossover(
    hist: pd.DataFrame,
    short_period: int = 5,
    long_period: int = 20,
) -> float:
    """均线交叉策略得分。

    Returns
    -------
    float : [-1, 1]
    """
    if hist is None or len(hist) < long_period + 2:
        return 0.0

    close = hist["close"]
    ma_short = close.rolling(short_period).mean()
    ma_long = close.rolling(long_period).mean()

    # 当前和前一天的 MA 差值
    diff_curr = ma_short.iloc[-1] - ma_long.iloc[-1]
    diff_prev = ma_short.iloc[-2] - ma_long.iloc[-2]

    score = 0.0

    # 金叉 / 死叉判断
    if diff_prev <= 0 and diff_curr > 0:
        score = 0.8  # 金叉
    elif diff_prev >= 0 and diff_curr < 0:
        score = -0.8  # 死叉
    elif diff_curr > 0:
        # 多头排列，趋势延续
        score = 0.3 + 0.2 * min(diff_curr / close.iloc[-1] * 100, 1.0)
    else:
        # 空头排列
        score = -0.3 - 0.2 * min(abs(diff_curr) / close.iloc[-1] * 100, 1.0)

    # MACD 辅助
    if len(close) >= 35:
        dif, dea, macd_bar = calc_macd(close)
        if not macd_bar.empty:
            last_macd = macd_bar.iloc[-1]
            prev_macd = macd_bar.iloc[-2]
            if last_macd > 0 and prev_macd <= 0:
                score += 0.2   # MACD 金叉确认
            elif last_macd < 0 and prev_macd >= 0:
                score -= 0.2   # MACD 死叉确认

    return float(np.clip(score, -1, 1))
