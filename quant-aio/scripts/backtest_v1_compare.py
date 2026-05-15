#!/usr/bin/env python3
"""V1 回测脚本 — 用于同期对比."""
from __future__ import annotations
import argparse
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))

from src.backtest.engine import BacktestEngine
from src.strategy.v1_simple_momentum import V1SimpleMomentum

def run_v1_backtest(start: str, end: str) -> dict:
    strategy = V1SimpleMomentum()
    engine = BacktestEngine(strategy=strategy, initial_cash=100_000)
    result = engine.run(start_date=start, end_date=end)

    cumulative = result.total_return * 100
    bench_cum = result.benchmark_return * 100
    excess = result.excess_return * 100

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

    print(f"\n{'='*50}")
    print(f"V1 回测: {start} → {end}")
    print(f"{'='*50}")
    print(f"策略累计收益 : {cumulative:+.2f}%")
    print(f"基准累计收益 : {bench_cum:+.2f}%")
    print(f"超额收益     : {excess:+.2f}%")
    print(f"夏普比率     : {sharpe:.3f}")
    print(f"最大回撤     : {max_dd:.2f}%")
    print(f"日胜率       : {daily_win:.1f}%")
    print(f"交易次数     : {len(result.trades)}")

    return {
        "window": f"{start}-{end}",
        "cumulative": cumulative,
        "benchmark": bench_cum,
        "excess": excess,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "trades": len(result.trades),
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    args = parser.parse_args()
    run_v1_backtest(args.start, args.end)
