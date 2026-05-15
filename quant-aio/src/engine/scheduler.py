"""
调度器 —— 每天 14:30 自动运行每日分析。

使用 schedule 库实现定时调度。
"""
from __future__ import annotations

import schedule
import time
import signal
import sys

from rich.console import Console

from config.settings import DAILY_RUN_TIME
from src.engine.daily import run_daily_analysis

console = Console()
_running = True


def _signal_handler(sig, frame):
    global _running
    console.print("\n[yellow]收到终止信号，停止调度...[/yellow]")
    _running = False


def start_scheduler() -> None:
    """启动定时调度。"""
    signal.signal(signal.SIGINT, _signal_handler)

    schedule.every().day.at(DAILY_RUN_TIME).do(run_daily_analysis)

    console.print(f"[bold green]调度器已启动，每日 {DAILY_RUN_TIME} 运行[/bold green]")
    console.print("[dim]按 Ctrl+C 停止[/dim]")

    # 立即运行一次（可选）
    console.print("[cyan]先执行一次分析...[/cyan]")
    run_daily_analysis()

    while _running:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    start_scheduler()
