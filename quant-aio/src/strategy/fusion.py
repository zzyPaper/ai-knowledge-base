"""
策略融合兼容层 —— 旧代码通过 fusion.py 调用，自动重定向到 V2ThreeFactor。

新代码请直接使用：
  from src.strategy import get_strategy
  strategy = get_strategy("v2")
"""
from __future__ import annotations

# 为了向后兼容，保留旧函数签名
import numpy as np
import pandas as pd

from src.strategy.v2_three_factor import V2ThreeFactor


# 单例
_v2_instance = V2ThreeFactor()


def detect_market_regime(index_hist: pd.DataFrame, **kwargs) -> str:
    """兼容旧接口：检测市场状态。"""
    return _v2_instance.detect_regime(index_hist)


def compute_composite_score(
    sector_hist: pd.DataFrame,
    index_hist: pd.DataFrame | None = None,
    regime: str | None = None,
) -> dict:
    """兼容旧接口：计算板块综合策略得分。"""
    s = _v2_instance.score_sector(sector_hist, index_hist, regime)
    return {
        "trend_strength": s.factors.get("trend_strength", 0),
        "short_momentum": s.factors.get("short_momentum", 0),
        "volume_confirm": s.factors.get("volume_confirm", 0),
        "composite": s.composite,
        "signal": s.signal,
        "position": s.position,
        "regime": s.regime,
        "ma60_pass": s.factors.get("ma60_pass", False),
        "atr_pct": s.factors.get("atr_pct", 0),
        "tr_atr_ratio": s.factors.get("tr_atr_ratio", 0),
    }


def rank_sectors(
    sector_scores: dict[str, dict],
    top_n: int = 5,
) -> list[tuple[str, float]]:
    """兼容旧接口：截面排名。"""
    ranked = []
    for name, scores in sector_scores.items():
        composite = scores.get("composite", 0)
        if composite <= 0:
            continue
        ranked.append((name, composite))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked[:top_n]


# 也暴露 ATR 计算函数（其他模块可能依赖）
from src.strategy.v2_three_factor import V2ThreeFactor as _V2

calc_atr = _V2._calc_atr
calc_atr_pct = _V2._calc_atr_pct
