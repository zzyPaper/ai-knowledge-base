#!/usr/bin/env python3
"""V5 个股多因子选股回测脚本 — 与 V1/V2/V3 板块轮动做同期对比。"""
from __future__ import annotations
import argparse
import warnings
import logging
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))

from src.backtest.stock_engine import StockBacktestEngine
from src.strategy.v5_stock_multifactor import score_stock


def run_stock_backtest(start: str, end: str) -> dict:
    engine = StockBacktestEngine(
        scoring_fn=score_stock,
        top_n=30,
        max_position_pct=0.05,
        stop_loss=-0.07,
        take_profit=0.20,
    )

    result = engine.run(start_date=start, end_date=end)

    cumulative = result.total_return * 100
    bench_cum = result.benchmark_return * 100
    excess = result.excess_return * 100

    # 计算夏普和最大回撤
    if result.daily_values:
        df = pd.DataFrame(result.daily_values)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        nav = df["total_value"] / result.initial_cash
        daily_ret = nav.pct_change().dropna()
        sharpe = daily_ret.mean() / daily_ret.std() * (252**0.5) if daily_ret.std() > 0 else 0
        cum_max = nav.cummax()
        max_dd = ((nav - cum_max) / cum_max).min() * 100
        daily_win = (daily_ret > 0).mean() * 100
    else:
        sharpe = max_dd = daily_win = 0

    print(f"\n{'='*60}")
    print(f"V5 个股多因子回测: {start} → {end}")
    print(f"{'='*60}")
    print(f"策略累计收益 : {cumulative:+.2f}%")
    print(f"基准累计收益 : {bench_cum:+.2f}%")
    print(f"超额收益     : {excess:+.2f}%")
    print(f"夏普比率     : {sharpe:.3f}")
    print(f"最大回撤     : {max_dd:.2f}%")
    print(f"日胜率       : {daily_win:.1f}%")
    print(f"交易次数     : {len(result.trades)}")
    print(f"评分股票数   : {result.num_stocks_scored}")

    # 打印交易明细
    print(f"\n{'Date':12s} {'Action':6s} {'Code':12s} {'Name':8s} {'Price':>8s} {'Shares':>8s} Reason")
    print("-" * 80)
    for t in result.trades[:30]:  # 只打印前30笔
        print(f"{t.date[:10]:12s} {t.action:6s} {t.code:12s} {t.name[:6]:8s} {t.price:8.2f} {t.shares:8d} {t.reason[:40]}")
    if len(result.trades) > 30:
        print(f"  ... and {len(result.trades) - 30} more trades")

    return {
        "window": f"{start}-{end}",
        "cumulative": cumulative,
        "benchmark": bench_cum,
        "excess": excess,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "trades": len(result.trades),
        "stocks_scored": result.num_stocks_scored,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    args = parser.parse_args()
    run_stock_backtest(args.start, args.end)
