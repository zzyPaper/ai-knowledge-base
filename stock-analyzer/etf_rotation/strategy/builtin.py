"""内置策略 - 适合小散的简单有效策略

包含：
1. 定投策略 (DCA) - 固定日期固定金额买入
2. 均线策略 - 突破均线买入，跌破卖出
3. 股债平衡 - 定期再平衡
4. 动量策略 - 追涨杀跌(N日收益率排名)
5. 网格策略 - 低买高卖
6. 估值定投 - 低估多买，高估少买/卖出
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable
from datetime import datetime

from ..backtest.account import Account
from ..backtest.engine import Signal
from .base import _get_latest_nav


def dca_strategy(
    invest_dates: List[str] = None,
    invest_amount: float = 1000,
    fund_code: str = None,
    day_of_week: int = 4,  # 默认每周四定投（周定投）
) -> Callable:
    """定投策略 - 固定日期固定金额买入

    Args:
        invest_dates: 指定定投日期列表（优先级高）
        invest_amount: 每期定投金额
        fund_code: 定投基金代码
        day_of_week: 周几定投(0=周一, 4=周五), 默认周四

    Returns:
        策略函数
    """
    invest_set = set()
    if invest_dates:
        invest_set = set(invest_dates)

    def _strategy(date: str, account: Account, nav_data: Dict,
                  fund_universe: List[str]) -> List[Signal]:
        signals = []

        if fund_universe:
            codes = [fund_code] if fund_code else [fund_universe[0]]
        else:
            return signals

        code = codes[0]

        # 判断是否定投日
        should_invest = False
        if invest_set:
            should_invest = date in invest_set
        else:
            dt = pd.Timestamp(date)
            should_invest = dt.weekday() == day_of_week

        if should_invest and account.cash >= invest_amount:
            signals.append(Signal(
                code=code, date=date, action="buy",
                amount=invest_amount,
                reason=f"定投({invest_amount}元)"
            ))

        return signals

    return _strategy


def ma_strategy(
    fund_code: str = None,
    short_window: int = 5,
    long_window: int = 20,
    max_position_pct: float = 0.8,
) -> Callable:
    """均线策略 - 短线上穿长线买入，下穿卖出

    经典: 5日线上穿20日线买入，下穿卖出
    适合ETF
    """
    def _strategy(date: str, account: Account, nav_data: Dict,
                  fund_universe: List[str]) -> List[Signal]:
        signals = []
        code = fund_code or (fund_universe[0] if fund_universe else None)
        if not code:
            return signals

        df = nav_data.get(code)
        if df is None or len(df) < long_window:
            return signals

        # 计算均线
        prices = df[df["date"] <= date].tail(long_window + short_window)
        if len(prices) < long_window:
            return signals

        ma_short = prices["nav"].tail(short_window).mean()
        ma_long = prices["nav"].tail(long_window).mean()

        # 获取前一天数据
        prev_prices = df[df["date"] < date].tail(long_window + short_window)
        if len(prev_prices) >= long_window:
            prev_ma_s = prev_prices["nav"].tail(short_window).mean()
            prev_ma_l = prev_prices["nav"].tail(long_window).mean()
        else:
            prev_ma_s, prev_ma_l = ma_short, ma_long

        pos = account.positions.get(code)
        current_nav = _get_latest_nav(nav_data, code, date)

        if ma_short > ma_long and prev_ma_s <= prev_ma_l:
            # 金叉买入
            if account.cash > 0:
                buy_amount = account.cash * max_position_pct
                signals.append(Signal(
                    code=code, date=date, action="buy",
                    amount=buy_amount,
                    reason=f"金叉(MA{short_window}={ma_short:.4f}>MA{long_window}={ma_long:.4f})"
                ))
        elif ma_short < ma_long and prev_ma_s >= prev_ma_l:
            # 死叉卖出
            if pos:
                signals.append(Signal(
                    code=code, date=date, action="sell",
                    amount=999999,
                    reason=f"死叉(MA{short_window}={ma_short:.4f}<MA{long_window}={ma_long:.4f})"
                ))

        return signals

    return _strategy


def balanced_strategy(
    etf_code: str = "510300",   # 沪深300ETF
    bond_code: str = "511520",  # 债券ETF（国债）
    stock_pct: float = 0.6,    # 股票仓位60%
    rebalance_days: int = 20,   # 约每月再平衡
    threshold: float = 0.05,   # 偏离5%触发再平衡
) -> Callable:
    """股债平衡策略 - 经典的60/40组合，定期再平衡

    借鉴达利欧全天候策略思路，简化版
    """
    def _strategy(date: str, account: Account, nav_data: Dict,
                  fund_universe: List[str]) -> List[Signal]:
        signals = []

        # 检查是否是再平衡日（简化：按交易天数）
        trading_dates = _get_common_dates(nav_data)
        if date not in trading_dates:
            return signals

        date_idx = trading_dates.index(date)
        is_rebalance_day = (date_idx % rebalance_days == 0)

        if not is_rebalance_day:
            return signals

        total_value = account.cash
        # 计算当前持仓市值
        current_stock_value = 0
        current_bond_value = 0
        etf_pos = account.positions.get(etf_code)
        bond_pos = account.positions.get(bond_code)

        etf_nav = _get_latest_nav(nav_data, etf_code, date)
        bond_nav = _get_latest_nav(nav_data, bond_code, date)

        if etf_pos and etf_nav:
            current_stock_value = etf_pos.shares * etf_nav
        if bond_pos and bond_nav:
            current_bond_value = bond_pos.shares * bond_nav

        total_value += current_stock_value + current_bond_value

        target_stock = total_value * stock_pct
        target_bond = total_value * (1 - stock_pct)

        # 偏离阈值检测
        stock_diff = current_stock_value - target_stock
        if abs(stock_diff) / max(total_value, 1) > threshold:
            if stock_diff > 0:
                # 股票超配，卖出股票
                signals.append(Signal(
                    code=etf_code, date=date, action="sell",
                    amount=stock_diff,
                    reason=f"再平衡: 股票超配, 卖出{stock_diff:.0f}元"
                ))
                signals.append(Signal(
                    code=bond_code, date=date, action="buy",
                    amount=stock_diff,
                    reason=f"再平衡: 买入债券{stock_diff:.0f}元"
                ))
            else:
                # 股票低配，买入股票
                signals.append(Signal(
                    code=etf_code, date=date, action="buy",
                    amount=-stock_diff,
                    reason=f"再平衡: 股票低配, 买入{-stock_diff:.0f}元"
                ))
                signals.append(Signal(
                    code=bond_code, date=date, action="sell",
                    amount=-stock_diff,
                    reason=f"再平衡: 卖出债券{-stock_diff:.0f}元"
                ))

        return signals

    return _strategy


def momentum_strategy(
    lookback: int = 20,
    top_n: int = 3,
    rebalance_freq: int = 5,
    hold_pct: float = 0.3,
) -> Callable:
    """动量策略 - 过去N日涨幅最多的几只基金等权持有

    适合ETF轮动
    """
    def _strategy(date: str, account: Account, nav_data: Dict,
                  fund_universe: List[str]) -> List[Signal]:
        signals = []
        trading_dates = _get_common_dates(nav_data)
        if date not in trading_dates:
            return signals

        date_idx = trading_dates.index(date)
        if date_idx < lookback:
            return signals
        if date_idx % rebalance_freq != 0:
            return signals

        # 计算各基金过去N日收益
        returns = {}
        for code in fund_universe:
            df = nav_data.get(code)
            if df is None or len(df) < lookback:
                continue
            hist = df[df["date"] <= date]
            if len(hist) < lookback:
                continue
            start_nav = hist.iloc[-lookback]["nav"]
            end_nav = hist.iloc[-1]["nav"]
            if start_nav > 0:
                returns[code] = (end_nav / start_nav - 1) * 100

        if not returns:
            return signals

        # 选出动量最强的top_n
        sorted_codes = sorted(returns.items(), key=lambda x: x[1], reverse=True)
        top_codes = [c[0] for c in sorted_codes[:top_n]]

        # 先卖出不在top_n中的持仓
        for code in list(account.positions.keys()):
            if code not in top_codes:
                signals.append(Signal(
                    code=code, date=date, action="sell",
                    amount=999999,
                    reason=f"轮动调出({returns.get(code, 0):.1f}%)"
                ))

        # 等权买入top_n
        total_cash = account.cash
        if total_cash > 0 and top_codes:
            amount_per = total_cash / len(top_codes)
            for code in top_codes:
                pos = account.positions.get(code)
                if pos:
                    # 已有持仓，检查是否需要加仓
                    pass
                signals.append(Signal(
                    code=code, date=date, action="buy",
                    amount=amount_per,
                    reason=f"动量买入({returns.get(code, 0):.1f}%)"
                ))

        return signals

    return _strategy


def grid_strategy(
    fund_code: str = None,
    grid_size: float = 0.03,  # 网格3%
    base_price: float = None,  # 基准价
    max_grids: int = 5,
    cash_per_grid: float = 2000,
) -> Callable:
    """网格策略 - 跌X%买入一格，涨X%卖出一格

    适合震荡市，不适合单边行情
    """
    def _strategy(date: str, account: Account, nav_data: Dict,
                  fund_universe: List[str]) -> List[Signal]:
        signals = []
        code = fund_code or (fund_universe[0] if fund_universe else None)
        if not code:
            return signals

        df = nav_data.get(code)
        if df is None or df.empty:
            return signals

        current_nav = _get_latest_nav(nav_data, code, date)
        if current_nav <= 0:
            return signals

        # 以第一个净值为基准
        first_nav = df.iloc[0]["nav"]
        ref_price = base_price or first_nav

        # 计算网格位置
        diff_pct = (current_nav - ref_price) / ref_price
        grid_level = round(diff_pct / grid_size)

        # 获取上次网格操作记录
        pos = account.positions.get(code)

        if grid_level < -1:  # 跌超一格，买入
            grids_down = abs(grid_level) // 1
            for _ in range(int(min(grids_down, max_grids))):
                if account.cash >= cash_per_grid:
                    signals.append(Signal(
                        code=code, date=date, action="buy",
                        amount=cash_per_grid,
                        reason=f"网格买入(跌{abs(diff_pct)*100:.1f}%)"
                    ))
        elif grid_level > 0 and pos:  # 涨超一格，卖出
            sell_pct = min(grid_level / 3, 0.5)  # 每涨一格卖一部分
            sell_amount = (pos.shares * current_nav) * sell_pct
            signals.append(Signal(
                code=code, date=date, action="sell",
                amount=sell_amount,
                reason=f"网格卖出(涨{diff_pct*100:.1f}%)"
            ))

        return signals

    return _strategy


def _get_common_dates(nav_data: Dict) -> List[str]:
    """获取所有基金共同的交易日序列"""
    all_dates = set()
    for df in nav_data.values():
        if df is not None and not df.empty:
            all_dates.update(df["date"].astype(str))
    return sorted(all_dates)
