"""
自循环优化器 —— AI Agent 驱动版本。

训练阶段：AI Agent 循环调用 run_backtest.py，分析结果，调整参数。
此模块提供时间窗口生成和参数文件读写工具函数。
运行阶段：daily.py 直接读 strategy_params.yaml，不需要 AI。
"""
from __future__ import annotations

import json
import yaml
from datetime import datetime, timedelta
from pathlib import Path

from rich.console import Console
from rich.table import Table

from config.settings import (
    BACKTEST_BENCHMARK_THRESHOLD,
    BACKTEST_WINDOW_MONTHS,
    BACKTEST_MAX_ROUNDS,
    BACKTEST_DIR,
    PROJECT_DIR,
)

console = Console()


def generate_time_windows(months: int = 12, window_months: int = 2) -> list[tuple[str, str]]:
    """生成回测时间窗口列表（从远到近）。"""
    windows = []
    end = datetime.now()
    for i in range(0, months, window_months):
        window_end = end - timedelta(days=i * 30)
        window_start = window_end - timedelta(days=window_months * 30)
        windows.append((
            window_start.strftime("%Y%m%d"),
            window_end.strftime("%Y%m%d"),
        ))
    windows.reverse()
    return windows


def load_strategy_params() -> dict:
    """读取当前策略参数。"""
    params_path = PROJECT_DIR / "config" / "strategy_params.yaml"
    with open(params_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_strategy_params(params: dict) -> None:
    """保存策略参数。"""
    params_path = PROJECT_DIR / "config" / "strategy_params.yaml"
    with open(params_path, "w", encoding="utf-8") as f:
        yaml.dump(params, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


def load_backtest_result() -> dict | None:
    """读取最近一次回测结果。"""
    result_path = BACKTEST_DIR / "runner_latest.json"
    if not result_path.exists():
        return None
    with open(result_path, encoding="utf-8") as f:
        return json.load(f)


def print_optimization_summary(all_results: list[dict]) -> None:
    """输出全部窗口优化汇总。"""
    console.print("\n[bold cyan]═══ 自循环优化汇总 ═══[/bold cyan]")

    table = Table(title="各窗口回测结果", show_lines=True)
    table.add_column("窗口", width=22)
    table.add_column("迭代轮次", width=8)
    table.add_column("是否达标", width=8)
    table.add_column("最终超额", width=10)

    for r in all_results:
        style = "green" if r["passed"] else "red"
        table.add_row(
            r["window"],
            str(r["rounds_needed"]),
            f"[{style}]{'✅' if r['passed'] else '❌'}[/{style}]",
            f"{r['final_excess']:.2%}",
        )

    console.print(table)
    pass_count = sum(1 for r in all_results if r["passed"])
    console.print(f"\n[bold]总达标率: {pass_count}/{len(all_results)}[/bold]")
