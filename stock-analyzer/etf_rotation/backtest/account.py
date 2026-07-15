"""账户与持仓管理

核心逻辑：持仓少于7天卖出将收取1.5%惩罚性赎回费
借鉴QLib的Account/Position设计，但简化针对基金场景
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# 赎回费率表（持有期限 → 费率）
# 实际规则: <7天 1.5%惩罚, >=7天 0%（免赎回费）
REDEMPTION_FEE_RATES = [
    (7, 0.015),      # <7天: 1.5%惩罚费
    (float("inf"), 0.0),  # >=7天: 0%
]


def calc_redemption_fee(hold_days: int) -> float:
    """根据持有天数计算赎回费率

    Args:
        hold_days: 持有自然日天数

    Returns:
        费率 (如 0.015 表示 1.5%)
    """
    for days, rate in REDEMPTION_FEE_RATES:
        if hold_days < days:
            return rate
    return 0.0


@dataclass
class Position:
    """持仓信息"""
    code: str                    # 基金代码
    name: str = ""               # 基金名称
    shares: float = 0.0          # 持有份额
    cost: float = 0.0            # 持仓成本（总金额）
    entry_date: str = ""         # 最早买入日期（用于判断持有期）
    last_buy_date: str = ""      # 最近买入日期

    @property
    def avg_cost(self) -> float:
        return self.cost / self.shares if self.shares > 0 else 0.0

    @property
    def market_value(self, current_nav: float = None) -> float:
        """市值 = 份额 × 最新净值"""
        if current_nav:
            return self.shares * current_nav
        return self.cost


@dataclass
class Order:
    """交易指令"""
    code: str
    amount: float          # 金额（买入）/ 份额或金额（卖出）
    order_type: str        # "buy" / "sell"
    price: float = 0.0     # 成交净值价
    date: str = ""         # 交易日
    fee: float = 0.0       # 交易费用

    def __post_init__(self):
        assert self.order_type in ("buy", "sell"), f"未知订单类型: {self.order_type}"


@dataclass
class TradeRecord:
    """成交记录"""
    code: str
    date: str
    order_type: str        # "buy" / "sell"
    amount: float          # 交易金额
    shares: float          # 交易份额
    nav: float             # 成交净值
    fee: float             # 费用
    hold_days: int = 0     # 卖出时的持有天数


class Account:
    """交易账户 - 管理资金和持仓"""

    def __init__(self, init_cash: float = 100000):
        self.init_cash = init_cash
        self.cash = init_cash
        self.positions: Dict[str, Position] = {}  # code -> Position
        self.trade_history: List[TradeRecord] = []
        self.daily_values: List[dict] = []  # 每日账户价值快照

    def buy(self, code: str, amount: float, nav: float, date: str,
            name: str = "", buy_fee_rate: float = 0.001) -> Order:
        """买入基金

        买入费率通常0.1%-0.15%（C类免申购费）
        """
        # 计算可买入金额
        fee = amount * buy_fee_rate
        actual_amount = amount - fee

        if actual_amount <= 0:
            logger.warning(f"{date} 买入金额不足，跳过")
            return None

        if actual_amount > self.cash:
            actual_amount = self.cash
            fee = actual_amount / (1 - buy_fee_rate) - actual_amount  # 调整
            actual_amount = self.cash * (1 - buy_fee_rate)

        shares = actual_amount / nav

        # 更新持仓
        if code not in self.positions:
            self.positions[code] = Position(code=code, name=name)

        pos = self.positions[code]
        # 加权平均成本计算
        total_cost = pos.cost + actual_amount
        total_shares = pos.shares + shares

        pos.shares = total_shares
        pos.cost = total_cost
        pos.last_buy_date = date
        if not pos.entry_date or date < pos.entry_date:
            pos.entry_date = date

        # 扣除现金
        self.cash -= amount

        # 记录交易
        trade = TradeRecord(
            code=code, date=date, order_type="buy",
            amount=actual_amount, shares=shares,
            nav=nav, fee=fee,
        )
        self.trade_history.append(trade)

        return Order(code=code, amount=actual_amount, order_type="buy",
                     price=nav, date=date, fee=fee)

    def sell(self, code: str, amount_or_shares: float, nav: float, date: str,
             is_share: bool = False) -> Order:
        """卖出基金

        核心逻辑: 不足7天收取1.5%惩罚性赎回费

        Args:
            code: 基金代码
            amount_or_shares: 卖出金额或份额
            nav: 卖出净值
            date: 交易日
            is_share: amount_or_shares 是否为份额
        """
        if code not in self.positions:
            logger.warning(f"{date} 基金 {code} 无持仓，跳过卖出")
            return None

        pos = self.positions[code]

        if is_share:
            sell_shares = min(amount_or_shares, pos.shares)
        else:
            sell_shares = min(amount_or_shares / nav, pos.shares)

        if sell_shares <= 0:
            return None

        sell_amount = sell_shares * nav

        # 计算持有天数
        hold_days = (pd.Timestamp(date) - pd.Timestamp(pos.entry_date)).days
        redemption_fee_rate = calc_redemption_fee(hold_days)
        redemption_fee = sell_amount * redemption_fee_rate

        actual_amount = sell_amount - redemption_fee

        # 更新持仓
        if sell_shares >= pos.shares:
            # 全部卖出
            del self.positions[code]
        else:
            pos.shares -= sell_shares
            pos.cost *= (pos.shares / (pos.shares + sell_shares))

        # 增加现金
        self.cash += actual_amount

        # 记录交易
        trade = TradeRecord(
            code=code, date=date, order_type="sell",
            amount=actual_amount, shares=sell_shares,
            nav=nav, fee=redemption_fee,
            hold_days=hold_days,
        )
        self.trade_history.append(trade)

        if redemption_fee_rate >= 0.015:
            logger.warning(f"{date} 卖出 {code} 持有仅 {hold_days}天，"
                          f"产生 {redemption_fee_rate*100:.1f}% 惩罚费!"
                          f" 费用={redemption_fee:.2f}")

        return Order(code=code, amount=actual_amount, order_type="sell",
                     price=nav, date=date, fee=redemption_fee)

    def can_sell(self, code: str, current_date: str) -> bool:
        """检查某基金是否可以卖出（持有是否>=7天）

        Returns:
            True = 可以卖出（无惩罚）
            False = 不足7天
            "penalty" = 不足7天但用户仍可强制卖出（会扣费）
        """
        if code not in self.positions:
            return False
        pos = self.positions[code]
        hold_days = (pd.Timestamp(current_date) - pd.Timestamp(pos.entry_date)).days
        return hold_days >= 7

    def snapshot(self, date: str, nav_map: Dict[str, float]) -> dict:
        """记录每日账户快照"""
        total_position_value = 0.0
        holdings = []

        for code, pos in self.positions.items():
            nav = nav_map.get(code, 0)
            mv = pos.shares * nav if nav > 0 else pos.cost
            total_position_value += mv
            hold_days = (pd.Timestamp(date) - pd.Timestamp(pos.entry_date)).days
            holdings.append({
                "code": code,
                "name": pos.name,
                "shares": pos.shares,
                "cost": pos.cost,
                "nav": nav,
                "market_value": mv,
                "profit": mv - pos.cost,
                "profit_pct": (mv / pos.cost - 1) * 100 if pos.cost > 0 else 0,
                "hold_days": hold_days,
                "can_sell": hold_days >= 7,
            })

        total_value = self.cash + total_position_value

        snapshot = {
            "date": date,
            "cash": self.cash,
            "position_value": total_position_value,
            "total_value": total_value,
            "daily_return": 0.0,  # 由外部计算
            "holdings": holdings,
            "position_count": len(self.positions),
        }

        self.daily_values.append(snapshot)
        return snapshot

    @property
    def total_value(self) -> float:
        """当前总资产（需要先调用snapshot更新）"""
        if self.daily_values:
            return self.daily_values[-1]["total_value"]
        return self.cash

    def summary(self) -> dict:
        """生成账户交易总结"""
        if not self.daily_values:
            return {}

        first_val = self.daily_values[0]["total_value"]
        last_val = self.daily_values[-1]["total_value"]

        total_return = (last_val / self.init_cash - 1) * 100
        buy_trades = sum(1 for t in self.trade_history if t.order_type == "buy")
        sell_trades = sum(1 for t in self.trade_history if t.order_type == "sell")
        total_fees = sum(t.fee for t in self.trade_history)
        penalty_fees = sum(t.fee for t in self.trade_history
                          if t.order_type == "sell" and getattr(t, "hold_days", 999) < 7)

        return {
            "init_cash": self.init_cash,
            "final_value": last_val,
            "total_return_pct": round(total_return, 2),
            "total_profit": round(last_val - self.init_cash, 2),
            "buy_count": buy_trades,
            "sell_count": sell_trades,
            "total_fees": round(total_fees, 2),
            "penalty_fees": round(penalty_fees, 2),
            "remaining_cash": self.cash,
            "positions": len(self.positions),
        }
