"""
quant-aio 主入口

运行模式：
1. daily    : 每日分析（独立运行，读训练好的参数）
2. backtest : 单次回测
3. train    : 自循环训练（AI Agent 驱动，逐轮迭代）
4. strategies: 列出可用策略
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def cmd_daily(args):
    """每日分析 —— 独立运行，不需要 AI。"""
    from src.strategy import get_strategy
    from src.engine.daily import run_daily_analysis
    strategy = get_strategy(args.strategy)
    result = run_daily_analysis(strategy, top_n=args.top_n)
    return result


def cmd_backtest(args):
    """单次回测。"""
    from src.strategy import get_strategy
    from src.backtest.engine import BacktestEngine
    strategy = get_strategy(args.strategy)
    engine = BacktestEngine(
        strategy=strategy,
        initial_cash=args.cash,
        stop_loss=args.stop_loss,
        take_profit=args.take_profit,
    )
    result = engine.run(args.start, args.end)
    from rich.console import Console
    console = Console()
    console.print(f"\n[bold]回测结果 | 策略: {strategy}[/bold]")
    console.print(f"  区间: {result.start_date} ~ {result.end_date}")
    console.print(f"  收益率: {result.total_return:.2%}")
    console.print(f"  基准:   {result.benchmark_return:.2%}")
    console.print(f"  超额:   {result.excess_return:.2%}")
    console.print(f"  达标:   {'✅' if result.is_passing else '❌'}")
    return result


def cmd_train(args):
    """自循环训练 —— 打印训练计划和工具函数，由 AI Agent 驱动循环。"""
    from src.strategy import get_strategy, list_strategies
    from src.backtest.optimizer import (
        generate_time_windows,
        load_strategy_params,
        save_strategy_params,
    )
    from config.settings import TRAINING_MONTHS, BACKTEST_WINDOW_MONTHS

    strategy = get_strategy(args.strategy)
    windows = generate_time_windows(TRAINING_MONTHS, BACKTEST_WINDOW_MONTHS)
    current_params = load_strategy_params()

    from rich.console import Console
    console = Console()

    console.print(f"[bold cyan]═══ 自循环训练模式 | 策略: {strategy} ═══[/bold cyan]")
    console.print(f"[green]训练覆盖: 最近 {TRAINING_MONTHS} 个月[/green]")
    console.print(f"[green]窗口大小: {BACKTEST_WINDOW_MONTHS} 个月[/green]")
    console.print(f"[green]时间窗口: {len(windows)} 个[/green]")
    console.print()

    for i, (start, end) in enumerate(windows):
        console.print(f"  窗口 {i+1}: {start} ~ {end}")

    console.print(f"\n[yellow]当前策略参数:[/yellow]")
    import json
    console.print(json.dumps(current_params, ensure_ascii=False, indent=2))

    console.print(f"\n[bold]AI Agent 自循环流程:[/bold]")
    console.print("  1. 读取 config/strategy_params.yaml")
    console.print(f"  2. 执行: python scripts/run_backtest.py --strategy {args.strategy} --start START --end END")
    console.print("  3. 读取 data/backtest_results/runner_latest.json 分析结果")
    console.print("  4. 调整 config/strategy_params.yaml 中的参数")
    console.print("  5. 重复直到所有窗口达标")


def cmd_strategies(args):
    """列出可用策略。"""
    from src.strategy import list_strategies
    from rich.console import Console
    from rich.table import Table

    console = Console()
    strategies = list_strategies()

    table = Table(title="可用策略", show_lines=True)
    table.add_column("Key", style="cyan", width=18)
    table.add_column("名称", width=18)
    table.add_column("版本", width=8)
    table.add_column("描述", width=50)

    for s in strategies:
        table.add_row(s["key"], s["name"], s["version"], s["description"])

    console.print(table)


def main():
    parser = argparse.ArgumentParser(
        description="quant-aio: A股热门板块实时分析 & 量化买卖策略系统"
    )
    sub = parser.add_subparsers(dest="command", help="运行模式")

    # ── 通用策略参数 ──
    strategy_help = "策略名称: v1/simple_momentum 或 v2/three_factor"

    # daily
    p_daily = sub.add_parser("daily", help="每日分析（独立运行）")
    p_daily.add_argument("--strategy", default="v2", help=strategy_help)
    p_daily.add_argument("--top-n", type=int, default=5, help="热门板块数量")
    p_daily.set_defaults(func=cmd_daily)

    # backtest
    p_bt = sub.add_parser("backtest", help="单次回测")
    p_bt.add_argument("--strategy", default="v2", help=strategy_help)
    p_bt.add_argument("--start", required=True, help="起始日期 YYYYMMDD")
    p_bt.add_argument("--end", required=True, help="结束日期 YYYYMMDD")
    p_bt.add_argument("--cash", type=float, default=1_000_000, help="初始资金")
    p_bt.add_argument("--stop-loss", type=float, default=-0.05, help="止损线")
    p_bt.add_argument("--take-profit", type=float, default=0.15, help="止盈线")
    p_bt.set_defaults(func=cmd_backtest)

    # train
    p_train = sub.add_parser("train", help="自循环训练（AI Agent 驱动）")
    p_train.add_argument("--strategy", default="v2", help=strategy_help)
    p_train.set_defaults(func=cmd_train)

    # strategies
    p_strat = sub.add_parser("strategies", help="列出可用策略")
    p_strat.set_defaults(func=cmd_strategies)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
