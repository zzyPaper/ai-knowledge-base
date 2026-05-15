"""
量能确认因子 —— ln(5日均量 / 20日均量)。

逻辑：放量上涨→指标为正→趋势含金量高；
缩量上涨→指标为负→发出风险警告。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def calc_volume_ratio(
    volume: pd.Series,
    fast: int = 5,
    slow: int = 20,
) -> float:
    """计算量能比：ln(快均量 / 慢均量)。"""
    if volume is None or len(volume) < slow:
        return 0.0
    vol = volume.iloc[-slow:].values
    # 过滤0值
    vol = vol[vol > 0]
    if len(vol) < fast:
        return 0.0

    vol_fast = np.mean(vol[-fast:])
    vol_slow = np.mean(vol)

    if vol_slow <= 0:
        return 0.0
    ratio = vol_fast / vol_slow
    return float(np.log(ratio)) if ratio > 0 else 0.0


def score_volume_confirm(
    hist: pd.DataFrame,
    fast: int = 5,
    slow: int = 20,
) -> float:
    """计算量能确认策略得分。

    Returns
    -------
    float : [-1, 1] 区间
        ln(1.5) ≈ 0.4 → 得分接近1（强放量）
        ln(0.5) ≈ -0.7 → 得分接近-1（强缩量）
    """
    if hist is None or len(hist) < slow:
        return 0.0

    volume = hist.get("volume", pd.Series(dtype=float))
    if volume.empty or volume.sum() == 0:
        # 无成交量数据时，用成交额替代
        volume = hist.get("amount", pd.Series(dtype=float))
    if volume.empty:
        return 0.0

    raw = calc_volume_ratio(volume, fast, slow)
    # 归一化：ln(1.5) ≈ 0.4 为满分
    return float(np.clip(raw / 0.4, -1, 1))
