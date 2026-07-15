"""回测引擎

核心功能:
1. 按交易日迭代，触发策略信号
2. 执行买卖，自动处理1周持有期约束
3. 每日记录账户快照，计算绩效指标
4. 输出回测报告

借鉴QLib的backtest_loop + Exchange设计思路，简化适配基金场景
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass
import logging

from .account import Account, calc_redemption_fee

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """策略信号：在某个日期对某只基金的买卖建议"""
    code: str
    date: str
    action: str           # "buy" / "sell" / "hold"
    amount: float = 0.0   # 买卖金额
    reason: str = ""      # 信号原因


class BacktestEngine:
    """回测引擎主力"""

    def __init__(self, init_cash: float = 100000.0,
                 buy_fee_rate: float = 0.0,  # 申购费0%
                 min_hold_days: int = 7):
        self.init_cash = init_cash
        self.buy_fee_rate = buy_fee_rate
        self.min_hold_days = min_hold_days
        self.account = Account(init_cash)

        # 回测结果
        self.daily_data: pd.DataFrame = None        # 所有基金日线数据
        self.trading_dates: List[str] = []          # 交易日序列
        self.fund_universe: List[str] = []          # 备选基金池
        self.nav_data: Dict[str, pd.DataFrame] = {}  # code -> DataFrame[date, nav]

    def load_data(self, nav_data: Dict[str, pd.DataFrame],
                  fund_names: Dict[str, str] = None):
        """加载回测数据

        Args:
            nav_data: code -> DataFrame(date, nav, acc_nav, daily_return)
            fund_names: code -> name
        """
        self.nav_data = {}
        self.fund_universe = list(nav_data.keys())
        self.fund_names = fund_names or {}

        # 将所有日期统一为字符串格式 YYYY-MM-DD
        for code, df in nav_data.items():
            if df is not None and not df.empty:
                clean = df.copy()
                clean["date"] = pd.to_datetime(clean["date"]).dt.strftime("%Y-%m-%d")
                self.nav_data[code] = clean

        # 构建统一的交易日序列
        all_dates = set()
        for code, df in self.nav_data.items():
            if df is not None and not df.empty:
                all_dates.update(df["date"].tolist())
        self.trading_dates = sorted(all_dates)
        logger.info(f"回测交易日范围: {self.trading_dates[0]} ~ {self.trading_dates[-1]}, "
                   f"共 {len(self.trading_dates)} 天")

    def get_nav(self, code: str, date: str) -> float:
        """获取某基金在指定日期的净值"""
        df = self.nav_data.get(code)
        if df is None:
            return 0.0
        row = df[df["date"] == date]
        if row.empty:
            return 0.0
        return row["nav"].values[0]

    def run(self, strategy_fn: Callable, verbose: bool = True) -> dict:
        """运行回测

        Args:
            strategy_fn: 策略函数
                def strategy(date: str, account: Account, nav_data: dict,
                           fund_universe: list) -> List[Signal]:
                ...
            verbose: 是否打印进度

        Returns:
            回测结果dict
        """
        self.account = Account(self.init_cash)

        prev_total = self.init_cash

        # 按交易日循环 - 借鉴QLib的backtest_loop
        for i, date in enumerate(self.trading_dates):
            # 1. 获取策略信号
            signals = strategy_fn(
                date=date,
                account=self.account,
                nav_data=self.nav_data,
                fund_universe=self.fund_universe,
            )

            # 2. 执行交易
            nav_map = {}
            for sig in signals:
                nav = self.get_nav(sig.code, date)

                if nav <= 0:
                    continue
                nav_map[sig.code] = nav

                if sig.action == "buy":
                    # 买入
                    name = self.fund_names.get(sig.code, "")
                    self.account.buy(
                        code=sig.code,
                        amount=min(sig.amount, self.account.cash),
                        nav=nav,
                        date=date,
                        name=name,
                        buy_fee_rate=self.buy_fee_rate,
                    )

                elif sig.action == "sell":
                    # 卖出 — 检查持有期约束
                    if self.account.can_sell(sig.code, date):
                        self.account.sell(
                            code=sig.code,
                            amount_or_shares=sig.amount if sig.amount > 0 else 999999,
                            nav=nav,
                            date=date,
                        )
                    else:
                        logger.debug(f"{date} {sig.code} 持有不足7天，跳过卖出")
                # "hold" = 无操作

            # 3. 补充所有持仓的最新净值到nav_map
            for code in self.account.positions:
                if code not in nav_map:
                    nav_map[code] = self.get_nav(code, date)

            # 4. 记录快照
            snap = self.account.snapshot(date, nav_map)

            # 5. 计算日收益率
            if i > 0:
                daily_ret = (snap["total_value"] / prev_total - 1) * 100
                self.account.daily_values[-1]["daily_return"] = daily_ret
            prev_total = snap["total_value"]

            if verbose and (i % 20 == 0 or i == len(self.trading_dates) - 1):
                logger.info(f"{date} 市值={snap['total_value']:.2f} "
                           f"现金={snap['cash']:.2f} 持仓={snap['position_count']}只"
                           f" 日收益={snap['daily_return']:.2f}%")

        return self._generate_report()

    def _generate_report(self) -> dict:
        """生成回测报告"""
        summary = self.account.summary()

        if len(self.account.daily_values) < 2:
            return summary

        # 计算更多绩效指标
        df = pd.DataFrame(self.account.daily_values)

        # 计算年化收益率
        days = (pd.Timestamp(self.trading_dates[-1]) -
                pd.Timestamp(self.trading_dates[0])).days
        years = max(days / 365.0, 0.01)
        total_return = df["total_value"].iloc[-1] / self.init_cash
        annual_return = (total_return ** (1 / years) - 1) * 100

        # 最大回撤
        peak = df["total_value"].expanding().max()
        drawdown = (df["total_value"] - peak) / peak * 100
        max_drawdown = drawdown.min()

        # 夏普比率（简化：无风险利率=2%）
        daily_returns = df["daily_return"] / 100
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            excess_returns = daily_returns.mean() * 252 - 0.02
            sharpe = excess_returns / (daily_returns.std() * np.sqrt(252))
        else:
            sharpe = 0

        # 胜率
        win_days = (daily_returns > 0).sum()
        total_days = len(daily_returns)
        win_rate = win_days / total_days * 100 if total_days > 0 else 0

        summary.update({
            "trading_days": total_days,
            "annual_return_pct": round(annual_return, 2),
            "max_drawdown_pct": round(max_drawdown, 2),
            "sharpe_ratio": round(sharpe, 2),
            "win_rate_pct": round(win_rate, 2),
            "daily_snapshots": df,
        })

        # 交易统计
        sell_trades = [t for t in self.account.trade_history if t.order_type == "sell"]
        if sell_trades:
            pen_trades = [t for t in sell_trades if getattr(t, "hold_days", 999) < 7]
            summary["penalty_trade_count"] = len(pen_trades)
            summary["penalty_fee_total"] = round(sum(t.fee for t in pen_trades), 2)

        return summary


class MultiFundBacktest:
    """多基金同时回测比较"""

    def __init__(self, init_cash: float = 100000):
        self.init_cash = init_cash
        self.results = {}

    def run_strategies(self, nav_data: Dict[str, pd.DataFrame],
                       strategies: Dict[str, Callable],
                       fund_names: Dict[str, str] = None):
        """运行多个策略并比较

        Args:
            nav_data: 基金净值数据
            strategies: {strategy_name: strategy_function}
        """
        for name, strategy_fn in strategies.items():
            logger.info(f"\n===== 运行策略: {name} =====")
            engine = BacktestEngine(init_cash=self.init_cash)
            engine.load_data(nav_data, fund_names)
            result = engine.run(strategy_fn, verbose=False)
            self.results[name] = result
        return self.results

    def compare(self) -> pd.DataFrame:
        """生成策略对比表"""
        rows = []
        for name, result in self.results.items():
            rows.append({
                "策略": name,
                "最终资产": result.get("final_value", 0),
                "总收益率%": result.get("total_return_pct", 0),
                "年化收益%": result.get("annual_return_pct", 0),
                "最大回撤%": result.get("max_drawdown_pct", 0),
                "夏普比率": result.get("sharpe_ratio", 0),
                "胜率%": result.get("win_rate_pct", 0),
                "交易次数": result.get("buy_count", 0),
                "总费用": result.get("total_fees", 0),
            })
        return pd.DataFrame(rows).sort_values("总收益率%", ascending=False)
