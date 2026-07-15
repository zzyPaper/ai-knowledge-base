"""交易执行模块 - 尾盘买卖执行

目前提供回测模拟执行 + 未来对接券商API的接口规范
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExecutedTrade:
    """成交结果"""
    code: str
    date: str
    order_type: str  # "buy" / "sell"
    amount: float
    shares: float
    price: float
    fee: float
    status: str = "filled"  # filled / pending / failed
    message: str = ""


class Broker:
    """交易券商接口抽象

    回测模式: 按净值价格成交，自动计算费用
    实盘模式: 需要对接具体券商API（如华泰、中信等）

    参考QLib的Exchange/Executor设计思路
    """

    def __init__(self, mode: str = "backtest", config: dict = None):
        """
        Args:
            mode: "backtest" / "paper" / "live"
            config: 券商配置（live模式需要）
        """
        self.mode = mode
        self.config = config or {}
        self.trade_history: List[ExecutedTrade] = []

    def buy(self, code: str, amount: float, nav: float = None,
            date: str = None, fund_name: str = "") -> ExecutedTrade:
        """买入基金

        Args:
            code: 基金代码
            amount: 买入金额
            nav: 基金净值（回测模式用，实盘模式从市场获取）
            date: 交易日
            fund_name: 基金名称

        Returns:
            成交记录
        """
        if self.mode == "backtest":
            fee = amount * 0.001  # 0.1%申购费
            shares = (amount - fee) / nav if nav and nav > 0 else 0
            trade = ExecutedTrade(
                code=code, date=date, order_type="buy",
                amount=amount - fee, shares=shares,
                price=nav or 0, fee=fee,
                status="filled", message="回测模式成交"
            )
        elif self.mode == "paper":
            # 模拟交易（待实现）
            trade = ExecutedTrade(
                code=code, date=date, order_type="buy",
                amount=amount, shares=0,
                price=0, fee=0,
                status="pending", message="模拟交易-待处理"
            )
        else:
            # 实盘模式（待对接券商API）
            trade = self._live_buy(code, amount)
            trade.date = date

        self.trade_history.append(trade)
        logger.info(f"{date} [{self.mode}] 买入 {code} {amount:.2f}元")
        return trade

    def sell(self, code: str, shares: float, nav: float = None,
             date: str = None, hold_days: int = 999) -> ExecutedTrade:
        """卖出基金

        自动计算赎回费（含不足7天惩罚）

        Args:
            code: 基金代码
            shares: 卖出份额
            nav: 净值
            date: 交易日
            hold_days: 持有天数

        Returns:
            成交记录
        """
        from ..backtest.account import calc_redemption_fee

        if self.mode == "backtest":
            amount = shares * nav if nav and nav > 0 else 0
            fee_rate = calc_redemption_fee(hold_days)
            fee = amount * fee_rate
            trade = ExecutedTrade(
                code=code, date=date, order_type="sell",
                amount=amount - fee, shares=shares,
                price=nav or 0, fee=fee,
                status="filled",
                message=f"回测成交(持有{hold_days}天,费率{fee_rate*100:.2f}%)"
            )
            if fee_rate >= 0.015:
                logger.warning(f"{date} 卖出 {code} 持有{hold_days}天不足7天!"
                              f" 扣除惩罚费 {fee:.2f}元")
        elif self.mode == "paper":
            trade = ExecutedTrade(
                code=code, date=date, order_type="sell",
                amount=0, shares=shares,
                price=0, fee=0,
                status="pending", message="模拟交易-待处理"
            )
        else:
            trade = self._live_sell(code, shares)
            trade.date = date

        self.trade_history.append(trade)
        logger.info(f"{date} [{self.mode}] 卖出 {code} {shares:.2f}份")
        return trade

    def get_realtime_nav(self, code: str) -> float:
        """获取基金实时估值（实盘模式用）

        ETF有盘中实时价格
        场外基金只有盘后净值
        """
        if self.mode != "live":
            return 0.0

        try:
            import akshare as ak
            # ETF实时价格
            df = ak.fund_etf_spot_em()
            row = df[df["代码"] == code]
            if not row.empty:
                return float(row["最新价"].values[0])

            # 场外基金（返回最新净值，可能有滞后）
            df = ak.fund_open_fund_info_em(symbol=code, indicator="单位净值走势")
            return float(df["单位净值"].iloc[-1])
        except Exception as e:
            logger.error(f"获取 {code} 实时净值失败: {e}")
            return 0.0

    def _live_buy(self, code: str, amount: float) -> ExecutedTrade:
        """实盘买入 - 需要对接券商API"""
        # TODO: 对接券商API
        logger.warning(f"实盘交易尚未实现: 买入 {code} {amount}元")
        return ExecutedTrade(
            code=code, date="", order_type="buy",
            amount=amount, shares=0, price=0, fee=0,
            status="failed", message="实盘交易API未对接"
        )

    def _live_sell(self, code: str, shares: float) -> ExecutedTrade:
        """实盘卖出 - 需要对接券商API"""
        logger.warning(f"实盘交易尚未实现: 卖出 {code} {shares}份")
        return ExecutedTrade(
            code=code, date="", order_type="sell",
            amount=0, shares=shares, price=0, fee=0,
            status="failed", message="实盘交易API未对接"
        )
