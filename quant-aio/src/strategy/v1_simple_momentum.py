"""
策略V1 —— 简单动量（Dual Momentum, Antonacci 2012）。

这是最早回测验证过的策略，逻辑简单但有效：
1. 绝对动量过滤：只选近期涨幅 > 0 的板块（趋势确认）
2. 相对动量排名：在合格板块中按 ROC 排名
3. MA 均线过滤：价格低于均线 → 排除
4. 超涨过滤：涨幅过大 → 跳过（防追高）

与 V2 的区别：
- V1 只看价格动量，不看量能和趋势拟合
- V1 的 ROC 窗口更长（20日），V2 更短（5/10日）
- V1 更保守，V2 更激进
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yaml
from pathlib import Path

from src.strategy.base import BaseStrategy, SectorScore


def _load_params() -> dict:
    params_path = Path(__file__).resolve().parent.parent.parent / "config" / "strategy_params.yaml"
    if params_path.exists():
        with open(params_path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


class V1SimpleMomentum(BaseStrategy):
    """策略V1：简单双动量。"""

    name = "simple_momentum"
    version = "1.0"
    description = "Dual Momentum (Antonacci 2012) - 绝对动量过滤 + 相对动量排名"

    def __init__(self):
        self._params = _load_params()

    @property
    def params(self) -> dict:
        return self._params

    def detect_regime(self, index_hist: pd.DataFrame) -> str:
        """检测市场状态：基于指数与MA20偏离度。"""
        if index_hist is None or len(index_hist) < 20:
            return "range"
        close = index_hist["close"]
        ma20 = close.rolling(20).mean().iloc[-1]
        if np.isnan(ma20) or ma20 <= 0:
            return "range"
        deviation = abs(close.iloc[-1] / ma20 - 1)
        return "trend" if deviation > 0.02 else "range"

    def score_sector(
        self,
        sector_hist: pd.DataFrame,
        index_hist: pd.DataFrame | None = None,
        regime: str | None = None,
    ) -> SectorScore:
        """对单个板块评分。"""
        params = self._params.get("v1", {})
        roc_period = params.get("roc_period", 20)
        ma_period = params.get("ma_period", 20)
        min_return_pct = params.get("min_return_pct", 0.0)
        max_return_pct = params.get("max_return_pct", 15.0)

        if regime is None:
            regime = self.detect_regime(index_hist) if index_hist is not None else "range"

        close = sector_hist["close"]
        factors = {}

        # ── 绝对动量（ROC）──
        roc = self._calc_roc(close, roc_period)
        factors["roc"] = roc
        factors["roc_period"] = roc_period

        # ── MA均线过滤 ──
        ma_pass = self._check_ma(close, ma_period)
        factors["ma_pass"] = ma_pass

        # ── 市场环境调节阈值 ──
        market_threshold = min_return_pct
        if index_hist is not None and len(index_hist) >= roc_period + 1:
            index_close = index_hist["close"].values
            index_ret = (index_close[-1] / index_close[-(roc_period + 1)] - 1) * 100
            if index_ret < -7:
                market_threshold = 3.0
            elif index_ret < -3:
                market_threshold = 1.0

        # ── 综合评分 ──
        # 基础分 = ROC 归一化到 [-1, 1]（10% 为满分）
        composite = np.clip(roc / 10.0, -1, 1)

        # 绝对动量过滤：涨幅不够 → 强制低分
        if roc < market_threshold:
            composite = min(composite, -0.5)

        # 超涨过滤：涨幅过大 → 打折（防追高）
        if roc > max_return_pct:
            composite *= 0.3
            factors["overbought"] = True

        # MA过滤：低于均线 → 仓位减半
        if not ma_pass:
            composite *= 0.5

        composite = float(np.clip(composite, -1, 1))

        # ── 信号与仓位 ──
        buy_threshold = params.get("buy_threshold", 0.10)
        sell_threshold = params.get("sell_threshold", -0.10)

        if composite > buy_threshold:
            signal = "BUY"
            position = float(np.clip(0.3 + composite * 0.7, 0.3, 1.0))
        elif composite < sell_threshold:
            signal = "SELL"
            position = 0.0
        else:
            signal = "HOLD"
            position = float(np.clip(0.1 + composite * 0.3, 0.0, 0.3))

        # MA不通过 → 仓位减半
        if not ma_pass:
            position *= 0.5

        return SectorScore(
            sector="",  # 由 score_all_sectors 填充
            composite=composite,
            signal=signal,
            position=position,
            factors=factors,
            regime=regime,
        )

    @staticmethod
    def _calc_roc(close: pd.Series, period: int = 20) -> float:
        """计算变化率（%）。"""
        if close is None or len(close) < period + 1:
            return 0.0
        val = (close.iloc[-1] / close.iloc[-period - 1] - 1) * 100
        return float(val) if not np.isnan(val) else 0.0

    @staticmethod
    def _check_ma(close: pd.Series, period: int = 20) -> bool:
        """均线过滤：价格 >= MA → 通过。"""
        if close is None or len(close) < period:
            return False
        ma = close.rolling(period).mean().iloc[-1]
        if np.isnan(ma) or ma <= 0:
            return False
        return close.iloc[-1] >= ma
