"""
回测引擎 v4 —— ATR动态止损 + 参数从YAML读取 + Bear空仓。

核心改进（v4）：
1. ATR动态止损：止损=买入价-2×ATR，追踪止盈=最高价回撤2×ATR
2. 参数从strategy_params.yaml读取，不再用硬编码默认值
3. Bear市场状态时自动空仓
4. 板块持仓跟踪最高价（用于追踪止盈）
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from config.settings import (
    BACKTEST_INITIAL_CASH,
    BACKTEST_DIR,
    BENCHMARK_INDEX,
)
from src.data.fetcher import get_sector_history, get_sectors_list, get_index_history
from src.strategy.base import BaseStrategy, SectorScore


def _load_params() -> dict:
    """从YAML加载策略参数。"""
    params_path = Path(__file__).resolve().parent.parent.parent / "config" / "strategy_params.yaml"
    if params_path.exists():
        with open(params_path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


@dataclass
class Trade:
    """一笔交易记录。"""
    date: str
    sector: str
    action: str          # BUY / SELL
    price: float
    shares: int
    cash_after: float
    position_value: float
    reason: str = ""     # 交易原因


@dataclass
class BacktestResult:
    """回测结果。"""
    start_date: str
    end_date: str
    initial_cash: float
    final_value: float
    total_return: float
    benchmark_return: float
    excess_return: float
    strategy_name: str = ""
    strategy_version: str = ""
    trades: list[Trade] = field(default_factory=list)
    daily_values: list[dict] = field(default_factory=list)

    @property
    def is_passing(self) -> bool:
        """是否超过基准 10%。"""
        return self.excess_return > 0.10


class BacktestEngine:
    """回测引擎 v4（ATR动态止损 + YAML参数）。"""

    def __init__(
        self,
        strategy: BaseStrategy,
        initial_cash: float | None = None,
        top_n: int | None = None,
        position_per_sector: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        rebalance_freq: str | None = None,
        commission_rate: float | None = None,
        slippage: float | None = None,
        use_atr_stop: bool = True,
        atr_stop_multiplier: float = None,
        atr_trailing_multiplier: float = None,
    ):
        self.strategy = strategy

        # 从YAML读取默认参数
        params = _load_params()
        strategy_params = params.get("v2", params)

        self.initial_cash = initial_cash if initial_cash is not None else BACKTEST_INITIAL_CASH
        self.top_n = top_n if top_n is not None else strategy_params.get("top_n", 5)
        self.position_per_sector = position_per_sector if position_per_sector is not None else strategy_params.get("position_per_sector", 0.20)
        self.stop_loss = stop_loss if stop_loss is not None else strategy_params.get("stop_loss", -0.05)
        self.take_profit = take_profit if take_profit is not None else strategy_params.get("take_profit", 0.15)
        self.rebalance_freq = rebalance_freq if rebalance_freq is not None else strategy_params.get("rebalance_freq", "monthly")
        self.commission_rate = commission_rate if commission_rate is not None else strategy_params.get("commission_rate", 0.0003)
        self.slippage = slippage if slippage is not None else strategy_params.get("slippage", 0.001)

        # ATR动态止损参数
        self.use_atr_stop = use_atr_stop
        self.atr_stop_multiplier = atr_stop_multiplier if atr_stop_multiplier is not None else strategy_params.get("atr_stop_multiplier", 3.0)
        self.atr_trailing_multiplier = atr_trailing_multiplier if atr_trailing_multiplier is not None else strategy_params.get("atr_trailing_multiplier", 3.0)

    def run(
        self,
        start_date: str,
        end_date: str,
        **kwargs,
    ) -> BacktestResult:
        """执行回测。"""
        cash = self.initial_cash
        # positions: {sector: {shares, cost_price, buy_date, atr_at_buy, highest_price}}
        positions: dict[str, dict] = {}
        trades: list[Trade] = []
        daily_values: list[dict] = []

        # 通知策略回测范围（用于历史因子缓存）
        if hasattr(self.strategy, 'set_backtest_range'):
            self.strategy.set_backtest_range(
                (datetime.strptime(start_date, "%Y%m%d") - timedelta(days=365)).strftime("%Y%m%d"),
                end_date,
            )

        # 预下载历史因子数据
        try:
            from src.data.factor_cache import preload_all_factors
            extended_start = (datetime.strptime(start_date, "%Y%m%d") - timedelta(days=365)).strftime("%Y%m%d")
            preload_result = preload_all_factors(extended_start, end_date)
            import logging
            logging.getLogger(__name__).info(f"因子数据预下载: {preload_result}")
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"因子数据预下载失败（将使用实时API fallback）: {e}")

        # 获取基准
        index_hist = get_index_history(BENCHMARK_INDEX, start=start_date, end=end_date)
        if index_hist.empty:
            return BacktestResult(start_date, end_date, self.initial_cash, self.initial_cash, 0, 0, 0,
                                  strategy_name=self.strategy.name, strategy_version=self.strategy.version)

        benchmark_start = index_hist["close"].iloc[0]
        benchmark_end = index_hist["close"].iloc[-1]

        # 获取板块列表
        from src.data.fetcher import _SECTOR_REPRESENTATIVE_STOCKS
        try:
            sectors_df = get_sectors_list()
            name_col = "板块名称" if "板块名称" in sectors_df.columns else sectors_df.columns[0]
            all_names = sectors_df[name_col].tolist()
            import re
            all_names = [
                n for n in all_names
                if isinstance(n, str) and not re.match(r'^[A-Z]\d+', n) and not n.isdigit()
            ]
            mapped = [n for n in all_names if n in _SECTOR_REPRESENTATIVE_STOCKS]
            unmapped = [n for n in all_names if n not in _SECTOR_REPRESENTATIVE_STOCKS]
            if "成交额" in sectors_df.columns:
                amount_col = "成交额"
                amount_map = dict(zip(sectors_df[name_col], sectors_df[amount_col]))
                unmapped.sort(key=lambda x: float(amount_map.get(x, 0)), reverse=True)
            sector_names = mapped + unmapped[:max(0, 30 - len(mapped))]
        except Exception:
            sector_names = list(_SECTOR_REPRESENTATIVE_STOCKS.keys())

        # 预加载板块历史（前推120天，确保ATR和多窗口动量有足够数据）
        preload_start = (datetime.strptime(start_date, "%Y%m%d") - timedelta(days=120)).strftime("%Y%m%d")
        sector_histories: dict[str, pd.DataFrame] = {}
        for name in sector_names:
            try:
                h = get_sector_history(name, start=preload_start, end=end_date)
                if not h.empty and len(h) >= 10:
                    sector_histories[name] = h
            except Exception:
                continue

        if not sector_histories:
            return BacktestResult(start_date, end_date, self.initial_cash, self.initial_cash, 0, 0, 0,
                                  strategy_name=self.strategy.name, strategy_version=self.strategy.version)

        # 跟踪上次调仓月份/周
        last_rebalance_month = -1
        self._last_week = -1
        self._last_iso_year = -1

        # 逐日回测
        for date_str in index_hist["date"].astype(str):
            date_val = pd.Timestamp(date_str)
            current_month = date_val.month

            # ── 调仓检查 ──
            should_rebalance = False
            if self.rebalance_freq == "monthly":
                should_rebalance = (current_month != last_rebalance_month)
            elif self.rebalance_freq == "weekly":
                # 使用 ISO year + week 避免跨年bug
                iso_year, iso_week, _ = date_val.isocalendar()
                current_week_key = (iso_year, iso_week)
                last_week_key = (self._last_iso_year, self._last_week)
                should_rebalance = (current_week_key != last_week_key)

            # ── 止损止盈检查（每日）──
            sectors_to_sell = []
            for sector, pos in list(positions.items()):
                if sector not in sector_histories:
                    continue
                h = sector_histories[sector]
                day_data = h[h["date"] <= date_val]
                if day_data.empty:
                    continue
                latest_price = day_data.iloc[-1]["close"]
                pnl_pct = (latest_price - pos["cost_price"]) / pos["cost_price"]

                sell_reason = None

                if self.use_atr_stop and pos.get("atr_at_buy", 0) > 0:
                    # ── ATR动态止损 ──
                    atr = pos["atr_at_buy"]
                    dynamic_stop = pos["cost_price"] - self.atr_stop_multiplier * atr
                    # 追踪止盈：从最高价回撤 atr_trailing_multiplier * ATR
                    highest = pos.get("highest_price", latest_price)
                    trailing_stop = highest - self.atr_trailing_multiplier * atr

                    if latest_price <= dynamic_stop:
                        sell_reason = f"ATR止损(成本{pos['cost_price']:.2f}-2×ATR={dynamic_stop:.2f})"
                    elif latest_price <= trailing_stop and highest > pos["cost_price"]:
                        sell_reason = f"追踪止盈(最高{highest:.2f}-2×ATR={trailing_stop:.2f},盈利{(highest-pos['cost_price'])/pos['cost_price']:.1%})"

                    # 更新最高价
                    if latest_price > pos.get("highest_price", 0):
                        positions[sector]["highest_price"] = latest_price

                # 静态止损止盈作为安全网
                if sell_reason is None:
                    if pnl_pct <= self.stop_loss:
                        sell_reason = f"静态止损({pnl_pct:.1%})"
                    elif pnl_pct >= self.take_profit:
                        sell_reason = f"静态止盈({pnl_pct:.1%})"

                if sell_reason:
                    sectors_to_sell.append((sector, sell_reason))

            # 执行止损止盈卖出
            for sector, reason in sectors_to_sell:
                if sector not in positions:
                    continue
                pos = positions[sector]
                h = sector_histories[sector]
                day_data = h[h["date"] <= date_val]
                if day_data.empty:
                    continue
                sell_price = day_data.iloc[-1]["close"] * (1 - self.slippage)
                sell_value = pos["shares"] * sell_price
                commission = sell_value * self.commission_rate
                cash += sell_value - commission
                trades.append(Trade(
                    date=date_str, sector=sector, action="SELL",
                    price=sell_price, shares=pos["shares"],
                    cash_after=cash, position_value=0,
                    reason=reason,
                ))
                del positions[sector]

            # ── 调仓 ──
            if should_rebalance:
                last_rebalance_month = current_month
                if self.rebalance_freq == "weekly":
                    iso_year, iso_week, _ = date_val.isocalendar()
                    self._last_week = iso_week
                    self._last_iso_year = iso_year

                # 用策略评分所有板块
                index_to_date = index_hist[index_hist["date"] <= date_val]

                try:
                    sectors_data = {}
                    for sector, hist in sector_histories.items():
                        hist_to_date = hist[hist["date"] <= date_val].copy()
                        if len(hist_to_date) >= 10:
                            hist_to_date._sector_name = sector
                            sectors_data[sector] = hist_to_date

                    ranked_scores = self.strategy.score_all_sectors(sectors_data, index_to_date)
                    ranked = [s for s in ranked_scores if s.composite > 0][:self.top_n]
                except Exception:
                    sector_scores_map: dict[str, SectorScore] = {}
                    for sector, hist in sector_histories.items():
                        hist_to_date = hist[hist["date"] <= date_val].copy()
                        if len(hist_to_date) < 10:
                            continue
                        try:
                            hist_to_date._sector_name = sector
                            s = self.strategy.score_sector(hist_to_date, index_to_date)
                            s.sector = sector
                            sector_scores_map[sector] = s
                        except Exception:
                            continue
                    ranked = sorted(
                        [s for s in sector_scores_map.values() if s.composite > 0],
                        key=lambda x: x.composite,
                        reverse=True,
                    )[:self.top_n]

                # Bear状态：全部清仓
                regime = ranked[0].regime if ranked else "range"
                if regime == "bear":
                    for sector in list(positions.keys()):
                        pos = positions[sector]
                        h = sector_histories[sector]
                        day_data = h[h["date"] <= date_val]
                        if day_data.empty:
                            continue
                        sell_price = day_data.iloc[-1]["close"] * (1 - self.slippage)
                        sell_value = pos["shares"] * sell_price
                        commission = sell_value * self.commission_rate
                        cash += sell_value - commission
                        trades.append(Trade(
                            date=date_str, sector=sector, action="SELL",
                            price=sell_price, shares=pos["shares"],
                            cash_after=cash, position_value=0,
                            reason="Bear空仓",
                        ))
                        del positions[sector]
                    # 不买入任何新板块
                    # 记录当日总资产
                    pv = sum(
                        pos["shares"] * sector_histories[s].set_index("date").loc[:date_val].iloc[-1]["close"]
                        for s, pos in positions.items() if s in sector_histories
                    ) if positions else 0.0
                    total_value = cash + pv
                    daily_values.append({
                        "date": date_str,
                        "cash": round(cash, 2),
                        "position_value": round(pv, 2),
                        "total_value": round(total_value, 2),
                        "num_positions": len(positions),
                        "regime": regime,
                    })
                    continue

                target_sectors = {s.sector for s in ranked}

                # 卖出不在目标中的持仓
                for sector in list(positions.keys()):
                    if sector not in target_sectors:
                        pos = positions[sector]
                        h = sector_histories[sector]
                        day_data = h[h["date"] <= date_val]
                        if day_data.empty:
                            continue
                        sell_price = day_data.iloc[-1]["close"] * (1 - self.slippage)
                        sell_value = pos["shares"] * sell_price
                        commission = sell_value * self.commission_rate
                        cash += sell_value - commission
                        trades.append(Trade(
                            date=date_str, sector=sector, action="SELL",
                            price=sell_price, shares=pos["shares"],
                            cash_after=cash, position_value=0,
                            reason="轮动换仓",
                        ))
                        del positions[sector]

                # 买入新的目标板块
                for rank_idx, score_obj in enumerate(ranked):
                    sector = score_obj.sector
                    if sector in positions:
                        continue
                    h = sector_histories[sector]
                    day_data = h[h["date"] <= date_val]
                    if day_data.empty:
                        continue
                    buy_price = day_data.iloc[-1]["close"] * (1 + self.slippage)
                    pos_multiplier = score_obj.position
                    invest = cash * self.position_per_sector * pos_multiplier
                    commission = invest * self.commission_rate
                    actual_invest = invest - commission
                    if actual_invest < 1000:
                        continue
                    shares = int(actual_invest / buy_price)
                    if shares <= 0:
                        continue
                    cost = shares * buy_price
                    cash -= cost + commission

                    # 获取ATR用于动态止损
                    atr_at_buy = score_obj.factors.get("atr_abs", 0.0)
                    if atr_at_buy == 0:
                        # 重新计算
                        hist_to_date = day_data.copy()
                        hist_to_date._sector_name = sector
                        from src.strategy.v2_three_factor import V2ThreeFactor
                        atr_at_buy = V2ThreeFactor._calc_atr(hist_to_date, period=20)

                    positions[sector] = {
                        "shares": shares,
                        "cost_price": buy_price,
                        "buy_date": date_str,
                        "atr_at_buy": atr_at_buy,
                        "highest_price": buy_price,
                    }

                    reason_parts = [f"排名#{rank_idx+1} 综合={score_obj.composite:.3f}"]
                    reason_parts.append(f"仓位={pos_multiplier:.0%}")
                    reason_parts.append(f"ATR={atr_at_buy:.2f}")
                    for k, v in score_obj.factors.items():
                        if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in ("weights", "risk_discounts"):
                            if abs(v) < 100:
                                reason_parts.append(f"{k}={v:.2f}")
                    trades.append(Trade(
                        date=date_str, sector=sector, action="BUY",
                        price=buy_price, shares=shares,
                        cash_after=cash, position_value=shares * buy_price,
                        reason=" ".join(reason_parts),
                    ))

            # ── 记录当日总资产 ──
            pv = 0.0
            for sector, pos in positions.items():
                if sector in sector_histories:
                    h = sector_histories[sector]
                    day_data = h[h["date"] <= date_val]
                    if not day_data.empty:
                        price = day_data.iloc[-1]["close"]
                        pv += pos["shares"] * price
                        # 更新最高价
                        if price > pos.get("highest_price", 0):
                            positions[sector]["highest_price"] = price

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

        result = BacktestResult(
            start_date=start_date,
            end_date=end_date,
            initial_cash=self.initial_cash,
            final_value=round(final_value, 2),
            total_return=round(total_return, 4),
            benchmark_return=round(benchmark_return, 4),
            excess_return=round(excess_return, 4),
            strategy_name=self.strategy.name,
            strategy_version=self.strategy.version,
            trades=trades,
            daily_values=daily_values,
        )

        _save_result(result)
        return result


def _save_result(result: BacktestResult) -> None:
    """保存回测结果。"""
    out_path = BACKTEST_DIR / f"backtest_{result.strategy_name}_{result.start_date}_{result.end_date}.json"
    data = {
        "strategy": f"{result.strategy_name} v{result.strategy_version}",
        "start_date": result.start_date,
        "end_date": result.end_date,
        "initial_cash": result.initial_cash,
        "final_value": result.final_value,
        "total_return": result.total_return,
        "benchmark_return": result.benchmark_return,
        "excess_return": result.excess_return,
        "is_passing": result.is_passing,
        "num_trades": len(result.trades),
        "daily_values": result.daily_values,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
