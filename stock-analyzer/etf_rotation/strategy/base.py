"""策略基类与工具

借鉴QLib的BaseStrategy设计思路，但更轻量，专为基金定投/择时场景
"""
from typing import Dict, List, Optional, Callable
import pandas as pd
import numpy as np
from dataclasses import dataclass

from ..backtest.account import Account
from ..backtest.engine import Signal


# 策略函数签名
# def strategy_fn(date: str, account: Account, nav_data: Dict,
#                 fund_universe: List[str]) -> List[Signal]:


def apply_weight_signal(date: str, account: Account, nav_data: Dict,
                         fund_universe: List[str],
                         weights: Dict[str, float],
                         total_amount: float) -> List[Signal]:
    """将目标权重表转化为买卖信号（再平衡）

    Args:
        weights: {code: target_weight} 目标权重，总和<=1
        total_amount: 当前总资产用于计算目标市值

    Returns:
        买卖信号列表
    """
    signals = []
    current_val = account.cash + sum(
        account.positions.get(c, type("", (), {"market_value": lambda n: 0})()).market_value(
            _get_latest_nav(nav_data, c, date)
        )
        for c in fund_universe
    )

    for code, target_weight in weights.items():
        target_value = current_val * target_weight
        current_pos = account.positions.get(code)
        current_nav = _get_latest_nav(nav_data, code, date)

        if current_pos and current_nav:
            current_value = current_pos.shares * current_nav
            diff = target_value - current_value

            if diff > current_value * 0.01:  # >1%偏差触发买入
                signals.append(Signal(
                    code=code, date=date, action="buy",
                    amount=diff, reason=f"再平衡买入({target_weight*100:.0f}%)"
                ))
            elif diff < -current_value * 0.01:  # >1%偏差触发卖出
                signals.append(Signal(
                    code=code, date=date, action="sell",
                    amount=-diff, reason=f"再平衡卖出({target_weight*100:.0f}%)"
                ))
        elif target_weight > 0:
            amount = total_amount * target_weight
            signals.append(Signal(
                code=code, date=date, action="buy",
                amount=amount, reason=f"新建仓位({target_weight*100:.0f}%)"
            ))

    return signals


def _get_latest_nav(nav_data: Dict[str, pd.DataFrame], code: str, date: str) -> float:
    """获取某基金最新净值（回测中辅助用）"""
    df = nav_data.get(code)
    if df is None or df.empty:
        return 0.0
    row = df[df["date"] <= date]
    if row.empty:
        return 0.0
    return row.iloc[-1]["nav"]
