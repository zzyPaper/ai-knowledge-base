"""
个股级别回测引擎 —— Barra CNE5 风格多因子选股。

与板块轮动引擎完全独立，核心差异：
1. 从 ~30 个板块 → 全 A 股票池（经流动性/市值筛选后约 500-1000 只）
2. 月频调仓，每月初评分 + 换仓
3. 行业中性约束：单行业最大权重 20%
4. 个股集中度约束：单票最大 5%
5. 止损/止盈
6. 交易成本含印花税
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import (
    BACKTEST_INITIAL_CASH,
    BACKTEST_DIR,
    BENCHMARK_INDEX,
    STOCK_UNIVERSE_MIN_MARKET_CAP,
    STOCK_UNIVERSE_MIN_AMOUNT,
    STOCK_TOP_N,
    STOCK_MAX_INDUSTRY_WEIGHT,
)
from src.data.stock_fetcher import (
    get_stock_list,
    get_stock_history,
    get_stock_industry_map,
    batch_get_stock_history,
)
from src.data.fetcher import get_index_history

logger = logging.getLogger(__name__)


@dataclass
class StockTrade:
    """个股交易记录。"""
    date: str
    code: str           # baostock 代码 e.g. sh.600519
    name: str           # 股票名称
    action: str         # BUY / SELL
    price: float
    shares: int
    cash_after: float
    position_value: float
    reason: str = ""


@dataclass
class StockBacktestResult:
    """个股回测结果。"""
    start_date: str
    end_date: str
    initial_cash: float
    final_value: float
    total_return: float
    benchmark_return: float
    excess_return: float
    strategy_name: str = ""
    strategy_version: str = ""
    trades: list[StockTrade] = field(default_factory=list)
    daily_values: list[dict] = field(default_factory=list)
    num_stocks_scored: int = 0
    num_stocks_held: int = 0

    @property
    def is_passing(self) -> bool:
        return self.excess_return > 0.10


@dataclass
class StockScore:
    """个股评分结果。"""
    code: str
    name: str
    industry: str
    composite: float          # 综合得分 [-1, 1]
    signal: str               # BUY / SELL / HOLD
    position: float           # 建议仓位比例 [0, 1]
    factors: dict = field(default_factory=dict)


class StockBacktestEngine:
    """个股级别回测引擎。"""

    def __init__(
        self,
        scoring_fn,                   # 评分函数 (stock_hist, index_hist, industry_map) -> StockScore
        initial_cash: float = BACKTEST_INITIAL_CASH,
        top_n: int = STOCK_TOP_N,
        max_position_pct: float = 0.05,  # 单票最大5%
        max_industry_pct: float = STOCK_MAX_INDUSTRY_WEIGHT,
        stop_loss: float = -0.07,         # 7% 止损
        take_profit: float = 0.20,        # 20% 止盈
        commission_rate: float = 0.0003,  # 万三佣金
        stamp_tax: float = 0.001,          # 千一印花税（卖出）
        slippage: float = 0.002,           # 0.2% 滑点
        rebalance_freq: str = "monthly",
        min_market_cap: float = STOCK_UNIVERSE_MIN_MARKET_CAP,
        min_daily_amount: float = STOCK_UNIVERSE_MIN_AMOUNT,
    ):
        self.scoring_fn = scoring_fn
        self.initial_cash = initial_cash
        self.top_n = top_n
        self.max_position_pct = max_position_pct
        self.max_industry_pct = max_industry_pct
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.commission_rate = commission_rate
        self.stamp_tax = stamp_tax
        self.slippage = slippage
        self.rebalance_freq = rebalance_freq
        self.min_market_cap = min_market_cap
        self.min_daily_amount = min_daily_amount

    def _build_universe(self, date_str: str) -> pd.DataFrame:
        """构建股票池：按市值 + 流动性过滤。"""
        stock_list = get_stock_list(date_str)
        if stock_list.empty:
            return pd.DataFrame()

        df = stock_list.copy()

        # 市值过滤
        if "market_cap" in df.columns:
            df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce")
            df = df[df["market_cap"] >= self.min_market_cap]

        # 成交额过滤
        if "amount" in df.columns:
            df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
            df = df[df["amount"] >= self.min_daily_amount]

        # 排除 ST
        if "name" in df.columns:
            df = df[~df["name"].astype(str).str.contains("ST|退", na=False)]

        # 排除北交所
        if "code" in df.columns:
            df = df[~df["code"].astype(str).str.startswith(("4", "8"))]

        return df.reset_index(drop=True)

    def run(
        self,
        start_date: str,
        end_date: str,
    ) -> StockBacktestResult:
        """执行回测。"""
        cash = self.initial_cash
        positions: dict[str, dict] = {}  # code -> {shares, cost_price, buy_date, name, industry}
        trades: list[StockTrade] = []
        daily_values: list[dict] = []

        # 获取基准指数
        index_hist = get_index_history(BENCHMARK_INDEX, start=start_date, end=end_date)
        if index_hist.empty:
            return StockBacktestResult(start_date, end_date, self.initial_cash, self.initial_cash, 0, 0, 0)

        benchmark_start = index_hist["close"].iloc[0]
        benchmark_end = index_hist["close"].iloc[-1]

        # 获取行业映射
        industry_map = get_stock_industry_map()

        # 构建股票池（在回测开始时构建一次，避免逐月频繁请求）
        logger.info("构建股票池...")
        universe_df = self._build_universe(start_date)
        if universe_df.empty:
            logger.warning("股票池为空，无法回测")
            return StockBacktestResult(start_date, end_date, self.initial_cash, self.initial_cash, 0, 0, 0)

        # 提取代码列表
        if "bs_code" in universe_df.columns:
            codes = universe_df["bs_code"].dropna().unique().tolist()
        elif "code" in universe_df.columns:
            codes = universe_df["code"].dropna().unique().tolist()
        else:
            return StockBacktestResult(start_date, end_date, self.initial_cash, self.initial_cash, 0, 0, 0)

        # 代码→名称 映射
        name_map = {}
        if "bs_code" in universe_df.columns and "name" in universe_df.columns:
            name_map = dict(zip(universe_df["bs_code"], universe_df["name"]))
        elif "code" in universe_df.columns and "name" in universe_df.columns:
            name_map = dict(zip(universe_df["code"], universe_df["name"]))

        logger.info(f"股票池: {len(codes)} 只")

        # 批量预加载历史K线（start前推60天）
        preload_start = (datetime.strptime(start_date, "%Y%m%d") - timedelta(days=75)).strftime("%Y%m%d")
        logger.info(f"批量下载K线: {preload_start} → {end_date}, {len(codes)} 只...")
        stock_histories = batch_get_stock_history(codes, preload_start, end_date, show_progress=True)
        logger.info(f"成功下载: {len(stock_histories)} 只")

        num_scored = len(stock_histories)
        last_rebalance_month = -1

        # 逐日回测
        trading_dates = index_hist["date"].astype(str).tolist()
        for date_str in trading_dates:
            date_val = pd.Timestamp(date_str)
            current_month = date_val.month

            # ── 止损止盈（每日检查）──
            codes_to_sell = []
            for code, pos in list(positions.items()):
                if code not in stock_histories:
                    continue
                h = stock_histories[code]
                day_data = h[h["date"].astype(str) <= date_str]
                if day_data.empty:
                    continue
                latest_price = float(day_data.iloc[-1]["close"])
                if latest_price <= 0 or pos["cost_price"] <= 0:
                    continue
                pnl_pct = (latest_price - pos["cost_price"]) / pos["cost_price"]

                if pnl_pct <= self.stop_loss:
                    codes_to_sell.append((code, "止损"))
                elif pnl_pct >= self.take_profit:
                    codes_to_sell.append((code, "止盈"))

            for code, reason in codes_to_sell:
                if code not in positions:
                    continue
                pos = positions[code]
                h = stock_histories[code]
                day_data = h[h["date"].astype(str) <= date_str]
                if day_data.empty:
                    continue
                sell_price = float(day_data.iloc[-1]["close"]) * (1 - self.slippage)
                sell_value = pos["shares"] * sell_price
                commission = sell_value * self.commission_rate
                stamp = sell_value * self.stamp_tax
                cash += sell_value - commission - stamp
                trades.append(StockTrade(
                    date=date_str, code=code, name=pos.get("name", ""),
                    action="SELL", price=sell_price, shares=pos["shares"],
                    cash_after=cash, position_value=0, reason=reason,
                ))
                del positions[code]

            # ── 月频调仓 ──
            should_rebalance = (current_month != last_rebalance_month)
            if not should_rebalance:
                # 记录当日资产
                pv = 0.0
                for code, pos in positions.items():
                    if code in stock_histories:
                        h = stock_histories[code]
                        day_data = h[h["date"].astype(str) <= date_str]
                        if not day_data.empty:
                            pv += pos["shares"] * float(day_data.iloc[-1]["close"])
                total_value = cash + pv
                daily_values.append({
                    "date": date_str,
                    "cash": round(cash, 2),
                    "position_value": round(pv, 2),
                    "total_value": round(total_value, 2),
                    "num_positions": len(positions),
                })
                continue

            last_rebalance_month = current_month

            # 用评分函数给所有股票打分
            index_to_date = index_hist[index_hist["date"].astype(str) <= date_str]
            scores: list[StockScore] = []

            for code, hist in stock_histories.items():
                hist_to_date = hist[hist["date"].astype(str) <= date_str].copy()
                if len(hist_to_date) < 20:
                    continue
                try:
                    ind = industry_map.get(code, "未知")
                    score = self.scoring_fn(
                        code=code,
                        name=name_map.get(code, code),
                        industry=ind,
                        stock_hist=hist_to_date,
                        index_hist=index_to_date,
                    )
                    if score and score.composite > 0:
                        scores.append(score)
                except Exception:
                    continue

            # 按综合得分排名
            scores.sort(key=lambda s: s.composite, reverse=True)

            # 行业集中度约束
            target_scores = self._apply_industry_constraint(scores)

            # 取 top_n
            target_scores = target_scores[:self.top_n]

            # 卖出不在目标的持仓
            target_codes = {s.code for s in target_scores}
            for code in list(positions.keys()):
                if code not in target_codes:
                    pos = positions[code]
                    h = stock_histories[code]
                    day_data = h[h["date"].astype(str) <= date_str]
                    if day_data.empty:
                        continue
                    sell_price = float(day_data.iloc[-1]["close"]) * (1 - self.slippage)
                    sell_value = pos["shares"] * sell_price
                    commission = sell_value * self.commission_rate
                    stamp = sell_value * self.stamp_tax
                    cash += sell_value - commission - stamp
                    trades.append(StockTrade(
                        date=date_str, code=code, name=pos.get("name", ""),
                        action="SELL", price=sell_price, shares=pos["shares"],
                        cash_after=cash, position_value=0, reason="轮动换仓",
                    ))
                    del positions[code]

            # 买入目标股票（等权重）
            position_size = min(self.max_position_pct, 1.0 / max(len(target_scores), 1))
            for rank_idx, score_obj in enumerate(target_scores):
                code = score_obj.code
                if code in positions:
                    continue
                if code not in stock_histories:
                    continue
                h = stock_histories[code]
                day_data = h[h["date"].astype(str) <= date_str]
                if day_data.empty:
                    continue

                buy_price = float(day_data.iloc[-1]["close"]) * (1 + self.slippage)
                invest = cash * position_size
                commission = invest * self.commission_rate
                actual_invest = invest - commission
                if actual_invest < 5000:
                    continue

                # A 股必须 100 股整数倍
                shares = int(actual_invest / (buy_price * 100)) * 100
                if shares <= 0:
                    continue

                cost = shares * buy_price
                cash -= cost + commission

                positions[code] = {
                    "shares": shares,
                    "cost_price": buy_price,
                    "buy_date": date_str,
                    "name": score_obj.name,
                    "industry": score_obj.industry,
                }

                reason_parts = [f"排名#{rank_idx+1}", f"综合={score_obj.composite:.3f}"]
                for k, v in score_obj.factors.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        reason_parts.append(f"{k}={v:.2f}" if abs(v) < 100 else f"{k}={v:.0f}")
                trades.append(StockTrade(
                    date=date_str, code=code, name=score_obj.name,
                    action="BUY", price=buy_price, shares=shares,
                    cash_after=cash, position_value=shares * buy_price,
                    reason=" ".join(reason_parts),
                ))

            # 记录当日资产
            pv = 0.0
            for code, pos in positions.items():
                if code in stock_histories:
                    h = stock_histories[code]
                    day_data = h[h["date"].astype(str) <= date_str]
                    if not day_data.empty:
                        pv += pos["shares"] * float(day_data.iloc[-1]["close"])

            total_value = cash + pv
            daily_values.append({
                "date": date_str,
                "cash": round(cash, 2),
                "position_value": round(pv, 2),
                "total_value": round(total_value, 2),
                "num_positions": len(positions),
            })

        # 计算结果
        final_value = daily_values[-1]["total_value"] if daily_values else self.initial_cash
        total_return = (final_value - self.initial_cash) / self.initial_cash
        benchmark_return = (benchmark_end - benchmark_start) / benchmark_start
        excess_return = total_return - benchmark_return

        result = StockBacktestResult(
            start_date=start_date,
            end_date=end_date,
            initial_cash=self.initial_cash,
            final_value=round(final_value, 2),
            total_return=round(total_return, 4),
            benchmark_return=round(benchmark_return, 4),
            excess_return=round(excess_return, 4),
            strategy_name="stock_multifactor",
            strategy_version="1.0",
            trades=trades,
            daily_values=daily_values,
            num_stocks_scored=num_scored,
            num_stocks_held=len(positions),
        )

        self._save_result(result)
        return result

    def _apply_industry_constraint(self, scores: list[StockScore]) -> list[StockScore]:
        """行业集中度约束：单行业最多占 top_n * max_industry_pct 个。"""
        max_per_industry = max(1, int(self.top_n * self.max_industry_pct))
        industry_count: dict[str, int] = {}
        result = []
        for s in scores:
            ind = s.industry
            if industry_count.get(ind, 0) >= max_per_industry:
                continue
            industry_count[ind] = industry_count.get(ind, 0) + 1
            result.append(s)
        return result

    def _save_result(self, result: StockBacktestResult) -> None:
        out_path = BACKTEST_DIR / f"stock_backtest_{result.start_date}_{result.end_date}.json"
        data = {
            "strategy": f"{result.strategy_name} v{result.strategy_version}",
            "start_date": result.start_date,
            "end_date": result.end_date,
            "initial_cash": result.initial_cash,
            "final_value": result.final_value,
            "total_return": result.total_return,
            "benchmark_return": result.benchmark_return,
            "excess_return": result.excess_return,
            "num_stocks_scored": result.num_stocks_scored,
            "num_trades": len(result.trades),
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
