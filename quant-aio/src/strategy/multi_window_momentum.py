"""
多窗口动量因子 —— 1月/3月/6月/12月 ROC 共振。

逻辑：
- 单一窗口动量容易受噪音干扰
- 多窗口共振确认趋势更可靠
- 1月=短期，3月=中期，6月/12月=长期
- 四窗口同向共振时信号最强

参考：
- Moskowitz & Grinblatt (1999): 行业动量效应
- Jegadeesh & Titman (1993): 动量策略需多窗口确认
- Asness, Moskowitz & Pedersen (2013): 价值与动量在多资产多周期有效
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def calc_roc(close: pd.Series, period: int) -> float:
    """计算变化率（%）。"""
    if close is None or len(close) < period + 1:
        return np.nan
    val = (close.iloc[-1] / close.iloc[-period - 1] - 1) * 100
    return float(val) if not np.isnan(val) else np.nan


def score_multi_window_momentum(
    hist: pd.DataFrame,
    windows: tuple[int, ...] = (22, 66, 132, 264),
    weights: tuple[float, ...] = (0.40, 0.30, 0.20, 0.10),
) -> float:
    """多窗口动量共振得分。

    Parameters
    ----------
    hist : 板块日K线
    windows : 各窗口交易日数 (1月≈22, 3月≈66, 6月≈132, 12月≈264)
    weights : 各窗口权重（短>长，短期动量信号更强）

    Returns
    -------
    float : [-1, 1] 区间
        四窗口同向共振时得分最高
    """
    if hist is None or len(hist) < 22:
        return 0.0

    close = hist["close"]
    rocs = []
    valid_weights = []

    for w, wt in zip(windows, weights):
        roc = calc_roc(close, w)
        if not np.isnan(roc):
            rocs.append(roc)
            valid_weights.append(wt)

    if not rocs:
        return 0.0

    # 归一化权重
    wsum = sum(valid_weights)
    valid_weights = [w / wsum for w in valid_weights]

    # ── 加权动量得分 ──
    weighted_roc = sum(r * w for r, w in zip(rocs, valid_weights))

    # 归一化：15% ROC → 满分1.0（A股板块年化15%已是不错的动量）
    base_score = np.clip(weighted_roc / 15.0, -1, 1)

    # ── 共振加成 ──
    # 四窗口同向（全部正或全部负）→ 加成50%
    # 三窗口同向 → 加成25%
    positive_count = sum(1 for r in rocs if r > 0)
    negative_count = sum(1 for r in rocs if r < 0)
    total = len(rocs)

    resonance_bonus = 0.0
    if total >= 3:
        if positive_count == total or negative_count == total:
            resonance_bonus = 0.5  # 全同向
        elif positive_count >= total - 1 or negative_count >= total - 1:
            resonance_bonus = 0.25  # 近乎同向

    # 共振加成叠加在方向上
    if base_score > 0:
        final = base_score * (1 + resonance_bonus)
    elif base_score < 0:
        final = base_score * (1 + resonance_bonus)
    else:
        final = 0.0

    return float(np.clip(final, -1, 1))


def get_momentum_detail(
    hist: pd.DataFrame,
    windows: tuple[int, ...] = (22, 66, 132, 264),
) -> dict:
    """获取各窗口动量明细。"""
    close = hist["close"]
    result = {}
    names = ["1月", "3月", "6月", "12月"]

    for name, w in zip(names, windows):
        roc = calc_roc(close, w)
        result[name] = round(roc, 2) if not np.isnan(roc) else None

    # 共振状态
    valid_rocs = [v for v in result.values() if v is not None]
    if valid_rocs:
        positive = sum(1 for v in valid_rocs if v > 0)
        if positive == len(valid_rocs):
            result["resonance"] = "全多"
        elif positive == 0:
            result["resonance"] = "全空"
        elif positive >= len(valid_rocs) - 1:
            result["resonance"] = "偏多"
        elif positive <= 1:
            result["resonance"] = "偏空"
        else:
            result["resonance"] = "分歧"
    else:
        result["resonance"] = "无数据"

    return result
