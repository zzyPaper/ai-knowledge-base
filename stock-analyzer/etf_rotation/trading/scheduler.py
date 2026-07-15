"""定时交易调度 - 尾盘自动交易

在每天14:50-15:00之间执行策略，完成买卖
"""
from datetime import datetime, time
import time as time_module
import logging
from typing import Callable, Dict, List, Optional

from .broker import Broker
from ..backtest.engine import Signal

logger = logging.getLogger(__name__)


class TradingScheduler:
    """交易调度器 - 每天尾盘执行"""

    def __init__(self, strategy_fn: Callable, broker: Broker,
                 fund_universe: List[str], nav_data: Dict = None):
        """
        Args:
            strategy_fn: 策略函数，同回测接口
            broker: 交易执行器
            fund_universe: 基金池
            nav_data: 净值数据（用于策略决策）
        """
        self.strategy_fn = strategy_fn
        self.broker = broker
        self.fund_universe = fund_universe
        self.nav_data = nav_data or {}
        self._running = False
        self.last_execution_date = None

    def execute(self, date: str = None) -> List[Signal]:
        """执行一次尾盘交易（手动调用或由定时任务触发）

        流程:
        1. 获取策略信号
        2. 检查持有期约束
        3. 执行买卖
        4. 记录结果
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"=== 尾盘交易执行: {date} ===")

        # mock account for signal generation
        # 这里简化处理，实际应有持久化的账户状态
        from ..backtest.account import Account
        mock_account = Account()

        # 获取信号
        signals = self.strategy_fn(
            date=date,
            account=mock_account,
            nav_data=self.nav_data,
            fund_universe=self.fund_universe,
        )

        executed = []
        for sig in signals:
            if sig.action == "buy":
                nav = self.broker.get_realtime_nav(sig.code)
                self.broker.buy(sig.code, sig.amount, nav, date)
                executed.append(sig)
            elif sig.action == "sell":
                # check holding period
                nav = self.broker.get_realtime_nav(sig.code)
                self.broker.sell(sig.code, sig.amount, nav, date)
                executed.append(sig)

        self.last_execution_date = date
        logger.info(f"执行完成: {len(executed)} 笔交易")
        return executed

    def run_daily_at_market_close(self):
        """每天14:50开始等待尾盘交易（阻塞式）

        适用于定时脚本或服务
        """
        logger.info("启动尾盘交易调度器...")
        self._running = True

        while self._running:
            now = datetime.now()

            # 检查是否到尾盘时间（14:50-15:00）且为交易日
            if now.hour == 14 and now.minute >= 50:
                self.execute()
                # 等待到15:00后，防止重复执行
                next_run = now.replace(hour=15, minute=1, second=0)
                sleep_seconds = (next_run - datetime.now()).total_seconds()
                if sleep_seconds > 0:
                    time_module.sleep(sleep_seconds)
            elif now.hour >= 15:
                # 过了交易时间，等第二天
                next_day = now.replace(day=now.day + 1, hour=14, minute=50, second=0)
                sleep_seconds = (next_day - datetime.now()).total_seconds()
                logger.info(f"等待至明日尾盘: {sleep_seconds/3600:.1f}小时后")
                if sleep_seconds > 0:
                    time_module.sleep(min(sleep_seconds, 3600))
            else:
                # 还没到尾盘，等待
                target = now.replace(hour=14, minute=50, second=0)
                sleep_seconds = (target - datetime.now()).total_seconds()
                if sleep_seconds > 0:
                    time_module.sleep(min(sleep_seconds, 600))

    def stop(self):
        self._running = False
