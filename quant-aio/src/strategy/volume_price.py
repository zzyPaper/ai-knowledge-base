"""
量价策略 —— 基于成交量与价格的协同关系。

核心逻辑：
1. 量价齐升 → 强势
2. 放量滞涨 → 见顶信号
3. 缩量下跌 → 可能见底
4. 量价背离 → 趋势即将反转

参考经典量价关系八大形态。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _volume_ratio(volume: pd.Series, lookback: int = 5) -> pd.Series:
    """量比：当前量 / N日均量。"""
    return volume / volume.rolling(lookback).mean()


def _amplitude_ratio(close: pd.Series, lookback: int = 5) -> pd.Series:
    """振幅比：当前振幅 / N日均振幅（用涨跌幅近似）。"""
    pct = close.pct_change() * 100
    avg_amp = pct.abs().rolling(lookback).mean()
    return pct.abs() / avg_amp.replace(0, np.nan)


# 八种经典量价形态
PATTERNS = {
    "bullish_surge":        {"vr": 1.5, "rr": 1.5, "sign": +1.0, "desc": "放量大涨"},
    "bullish_consolidation":{"vr": 0.7, "rr": 1.5, "sign": +0.5, "desc": "缩量上涨"},
    "bearish_dump":         {"vr": 1.5, "rr": 1.5, "sign": -1.0, "desc": "放量大跌"},
    "bearish_drift":        {"vr": 0.7, "rr": 1.5, "sign": -0.5, "desc": "缩量阴跌"},
    "divergence":           {"vr": 1.5, "rr": 0.0, "sign": -0.3, "desc": "量价背离"},
    "stabilization":        {"vr": 0.7, "rr": 0.0, "sign": +0.3, "desc": "缩量企稳"},
    "climax":               {"vr": 3.0, "rr": 3.0, "sign": +0.2, "desc": "天量天价"},
    "exhaustion":           {"vr": 0.3, "rr": 0.0, "sign": +0.1, "desc": "地量地价"},
}


def detect_pattern(vr: float, pct: float, avg_amp: float) -> tuple[str, float]:
    """检测当日量价形态。

    Parameters
    ----------
    vr : 量比
    pct : 当日涨跌幅 (%)
    avg_amp : 近 N 日平均振幅 (%)

    Returns
    -------
    (形态名称, 得分)
    """
    rr = abs(pct) / avg_amp if avg_amp > 0 else 0
    is_up = pct > 0

    if vr > 1.5 and rr > 1.5:
        return ("bullish_surge" if is_up else "bearish_dump", 1.0 if is_up else -1.0)
    if vr < 0.7 and rr > 1.5:
        return ("bullish_consolidation" if is_up else "bearish_drift", 0.5 if is_up else -0.5)
    if vr > 1.5 and rr <= 1.5:
        return ("divergence", -0.3)
    if vr < 0.7 and rr <= 1.5:
        return ("stabilization", 0.3)
    if vr > 3.0 and rr > 3.0:
        return ("climax", 0.2)
    if vr < 0.3:
        return ("exhaustion", 0.1)

    return ("normal", 0.0)


def score_volume_price(hist: pd.DataFrame, lookback: int = 10) -> float:
    """量价策略综合得分。

    逐日检测量价形态并累加，最后归一化到 [-1, 1]。
    """
    if hist is None or len(hist) < lookback + 1:
        return 0.0

    close = hist["close"]
    volume = hist.get("volume", pd.Series(dtype=float))
    if volume.empty:
        return 0.0

    pct = close.pct_change() * 100
    vr = _volume_ratio(volume, lookback)
    avg_amp = pct.abs().rolling(lookback).mean()

    total_score = 0.0
    window = hist.tail(lookback)
    for i in range(len(window)):
        idx = window.index[i]
        if pd.isna(vr.get(idx)) or pd.isna(avg_amp.get(idx)):
            continue
        _, s = detect_pattern(vr[idx], pct[idx], avg_amp[idx])
        total_score += s

    # 归一化
    max_possible = lookback * 1.0
    return float(np.clip(total_score / max_possible if max_possible > 0 else 0, -1, 1))
