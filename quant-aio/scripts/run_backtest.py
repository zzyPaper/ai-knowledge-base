"""
回测单轮执行脚本 —— 供 AI Agent 在自循环训练中调用。

用法：
  python scripts/run_backtest.py --strategy v2 --start 20250101 --end 20250301

v3.0 改进：
- BacktestEngine 从 YAML 读取参数，不再需要手动传递
- 支持 ATR 动态止损
- 自动预下载历史因子数据
"""
from __future__ import annotations

import json
import sys
import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from src.strategy import get_strategy
from src.backtest.engine import BacktestEngine, Trade
from config.settings import BACKTEST_DIR


def load_params() -> dict:
    """从 YAML 加载当前策略参数。"""
    params_path = PROJECT_ROOT / "config" / "strategy_params.yaml"
    with open(params_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_one_round(start: str, end: str, strategy_name: str = "v2") -> dict:
    """执行一轮回测，返回结构化结果。"""
    params = load_params()
    strategy = get_strategy(strategy_name)

    # BacktestEngine 现在自动从YAML读取参数，只需传strategy
    engine = BacktestEngine(strategy=strategy)

    result = engine.run(start, end)

    # 构建输出
    trades_summary = []
    for t in result.trades:
        trades_summary.append({
            "date": t.date,
            "sector": t.sector,
            "action": t.action,
            "price": round(t.price, 2),
            "shares": t.shares,
            "reason": t.reason,
        })

    buy_count = sum(1 for t in result.trades if t.action == "BUY")
    sell_count = sum(1 for t in result.trades if t.action == "SELL")

    # 分析持仓分布
    avg_positions = 0
    if result.daily_values:
        avg_positions = sum(d.get("num_positions", 0) for d in result.daily_values) / len(result.daily_values)

    output = {
        "strategy": f"{strategy.name} v{strategy.version}",
        "window": f"{start}-{end}",
        "params_used": params,
        "performance": {
            "total_return": result.total_return,
            "benchmark_return": result.benchmark_return,
            "excess_return": result.excess_return,
            "initial_cash": result.initial_cash,
            "final_value": result.final_value,
            "is_passing": result.is_passing,
            "avg_positions": round(avg_positions, 1),
        },
        "trade_stats": {
            "total_trades": len(result.trades),
            "buy_count": buy_count,
            "sell_count": sell_count,
        },
        "trades": trades_summary,
        "daily_values_tail": result.daily_values[-5:] if result.daily_values else [],
    }

    # 保存
    out_path = BACKTEST_DIR / "runner_latest.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)

    return output


def main():
    parser = argparse.ArgumentParser(description="回测单轮执行")
    parser.add_argument("--strategy", default="v2", help="策略: v1 或 v2")
    parser.add_argument("--start", required=True, help="起始日期 YYYYMMDD")
    parser.add_argument("--end", required=True, help="结束日期 YYYYMMDD")
    args = parser.parse_args()

    result = run_one_round(args.start, args.end, args.strategy)
    print(f"策略:   {result['strategy']}")
    print(f"收益率: {result['performance']['total_return']:.2%}")
    print(f"基准:   {result['performance']['benchmark_return']:.2%}")
    print(f"超额:   {result['performance']['excess_return']:.2%}")
    print(f"达标:   {'✅' if result['performance']['is_passing'] else '❌'}")
    print(f"平均持仓: {result['performance']['avg_positions']:.1f}个板块")
    print(f"结果已保存: {BACKTEST_DIR / 'runner_latest.json'}")


if __name__ == "__main__":
    main()
