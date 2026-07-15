"""自定义策略示例 - 教你如何写自己的策略

策略函数签名:
    def strategy(date: str, account: Account, nav_data: Dict,
                 fund_universe: List[str]) -> List[Signal]:
        ...

返回:
    List[Signal] - 买卖信号列表
"""
import pandas as pd
import numpy as np
from typing import Dict, List

from ..backtest.engine import Signal
from ..backtest.account import Account


def my_etf_strategy(date: str, account: Account, nav_data: Dict,
                     fund_universe: List[str]) -> List[Signal]:
    """我的第一个策略: 简单双均线 + RSI过滤

    规则:
    1. 只买第一只基金
    2. 计算20日均线和60日均线
    3. 计算RSI(14)
    4. 金叉且RSI<70 → 买入
    5. 死叉或RSI>80 → 卖出
    """
    signals = []
    if not fund_universe:
        return signals

    code = fund_universe[0]
    df = nav_data.get(code)
    if df is None or len(df) < 60:
        return signals

    # 取到当前日期的历史数据
    hist = df[df["date"] <= date].copy()
    if len(hist) < 60:
        return signals

    # 计算均线
    hist["ma20"] = hist["nav"].rolling(20).mean()
    hist["ma60"] = hist["nav"].rolling(60).mean()

    # 计算RSI
    delta = hist["nav"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    hist["rsi"] = 100 - (100 / (1 + rs))

    current = hist.iloc[-1]
    prev = hist.iloc[-2] if len(hist) >= 2 else current

    pos = account.positions.get(code)

    # 买入条件: 金叉且RSI不过热
    if (current["ma20"] > current["ma60"] and
        prev["ma20"] <= prev["ma60"] and
        current["rsi"] < 70 and
        account.cash > 0):
        buy_amount = account.cash * 0.5  # 半仓买入
        signals.append(Signal(
            code=code, date=date, action="buy",
            amount=buy_amount,
            reason=f"金叉买入(MA20={current['ma20']:.4f},RSI={current['rsi']:.1f})"
        ))

    # 卖出条件: 死叉 或 RSI超买
    elif pos and (
        (current["ma20"] < current["ma60"] and
         prev["ma20"] >= prev["ma60"]) or
        current["rsi"] > 80
    ):
        signals.append(Signal(
            code=code, date=date, action="sell",
            amount=999999,  # 全部卖出
            reason=f"卖出(MA20={current['ma20']:.4f},RSI={current['rsi']:.1f})"
        ))

    return signals


def multi_etf_rotation(date: str, account: Account, nav_data: Dict,
                        fund_universe: List[str]) -> List[Signal]:
    """多ETF轮动策略

    每月初，在所有基金中选择过去20日涨幅最好的持有
    """
    signals = []

    # 获取交易日列表
    all_dates = sorted(set(
        d for df in nav_data.values()
        for d in df["date"].astype(str)
    ))

    if date not in all_dates:
        return signals

    date_idx = all_dates.index(date)

    # 每月初才调仓
    dt = pd.Timestamp(date)
    if dt.day > 5:  # 每月5号后不操作
        return signals

    if date_idx < 20:
        return signals

    # 计算过去20日涨幅
    returns = {}
    for code in fund_universe:
        df = nav_data.get(code)
        if df is None or len(df) < 20:
            continue
        hist = df[df["date"] <= date]
        if len(hist) < 20:
            continue
        ret = (hist["nav"].iloc[-1] / hist["nav"].iloc[-20] - 1) * 100
        returns[code] = ret

    if not returns:
        return signals

    # 选最强的那只
    best_code = max(returns, key=returns.get)

    # 卖出其他所有
    for code in list(account.positions.keys()):
        if code != best_code:
            signals.append(Signal(
                code=code, date=date, action="sell",
                amount=999999,
                reason=f"轮出(收益{returns.get(code, 0):.1f}%)"
            ))

    # 买入最强的那只
    if best_code and account.cash > 0:
        signals.append(Signal(
            code=best_code, date=date, action="buy",
            amount=account.cash * 0.9,
            reason=f"轮入(收益{returns.get(best_code, 0):.1f}%)"
        ))

    return signals
