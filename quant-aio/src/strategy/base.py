"""
策略基类 —— 所有策略必须实现的接口。

设计原则：
- 策略只负责"评分和信号"，不负责交易执行
- 输入：板块历史数据 + 指数历史数据
- 输出：板块综合得分 + 买卖信号 + 建议仓位
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class SectorScore:
    """单个板块的评分结果。"""
    sector: str
    composite: float           # 综合得分 [-1, 1]
    signal: str                # BUY / SELL / HOLD
    position: float            # 建议仓位 [0, 1]
    factors: dict = field(default_factory=dict)  # 各因子得分明细
    regime: str = "unknown"     # 市场状态
    extra: dict = field(default_factory=dict)     # 扩展信息


class BaseStrategy(ABC):
    """策略基类。"""

    # 子类必须设置
    name: str = "base"
    version: str = "0.1"
    description: str = ""

    @abstractmethod
    def score_sector(
        self,
        sector_hist: pd.DataFrame,
        index_hist: Optional[pd.DataFrame] = None,
        regime: Optional[str] = None,
    ) -> SectorScore:
        """对单个板块评分。

        Parameters
        ----------
        sector_hist : 板块日K线历史
        index_hist : 基准指数日K线历史
        regime : 市场状态（如已知）

        Returns
        -------
        SectorScore
        """
        ...

    @abstractmethod
    def detect_regime(self, index_hist: pd.DataFrame) -> str:
        """检测市场状态。"""
        ...

    def score_all_sectors(
        self,
        sectors_data: dict[str, pd.DataFrame],
        index_hist: Optional[pd.DataFrame] = None,
    ) -> list[SectorScore]:
        """批量评分所有板块，按综合分降序排列。"""
        regime = self.detect_regime(index_hist) if index_hist is not None else "unknown"
        scores = []
        for name, hist in sectors_data.items():
            try:
                # 附加板块名，供策略查询资金/估值等外部数据
                hist_copy = hist.copy()
                hist_copy._sector_name = name
                s = self.score_sector(hist_copy, index_hist, regime)
                s.sector = name
                scores.append(s)
            except Exception:
                continue
        scores.sort(key=lambda x: x.composite, reverse=True)
        return scores

    def rank_sectors(
        self,
        sectors_data: dict[str, pd.DataFrame],
        index_hist: Optional[pd.DataFrame] = None,
        top_n: int = 5,
        min_score: float = 0.0,
    ) -> list[SectorScore]:
        """排名板块，只返回正分的前N个。"""
        scores = self.score_all_sectors(sectors_data, index_hist)
        return [s for s in scores if s.composite >= min_score][:top_n]

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} v{self.version}: {self.name}>"
