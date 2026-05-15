"""
每日分析引擎 v3 —— 策略无关，支持 V1/V2 任意策略。

输出：
1. 所有板块排名（含因子明细）
2. 建议买入板块及仓位
3. 当前持仓建议
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from config.settings import (
    HOT_SECTOR_TOP_N,
    RESULTS_DIR,
    BENCHMARK_INDEX,
)
from src.data.fetcher import (
    get_sectors_list,
    get_sector_history,
    get_index_history,
)
from src.strategy.base import BaseStrategy, SectorScore

console = Console()


def run_daily_analysis(
    strategy: BaseStrategy,
    top_n: int = HOT_SECTOR_TOP_N,
) -> dict:
    """执行每日分析，返回结果字典。"""
    console.print(f"[bold cyan]═══ 每日板块分析 | 策略: {strategy} ═══[/bold cyan]")
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d %H:%M")

    # 1. 获取基准指数
    index_hist = get_index_history(BENCHMARK_INDEX)
    regime = strategy.detect_regime(index_hist)
    regime_cn = "趋势市" if regime == "trend" else "震荡市"
    console.print(f"[yellow]市场状态: {regime_cn}[/yellow]")

    # 2. 获取板块列表
    sectors_df = get_sectors_list()
    import re
    valid_mask = sectors_df.iloc[:, 0].apply(
        lambda x: not bool(re.match(r'^[A-Z]\d+', str(x))) if isinstance(x, str) else True
    )
    sector_names = sectors_df[valid_mask].iloc[:, 0].tolist()
    console.print(f"[green]共 {len(sector_names)} 个板块[/green]")

    # 3. 用策略评分
    sectors_data = {}
    for name in sector_names:
        try:
            hist = get_sector_history(name)
            if hist.empty or len(hist) < 20:
                continue
            sectors_data[name] = hist
        except Exception:
            continue

    ranked = strategy.rank_sectors(sectors_data, index_hist, top_n=top_n, min_score=0.0)

    # 4. 输出结果
    _print_results(strategy, ranked, date_str, regime_cn)
    _save_results(strategy, ranked, date_str)

    return {
        "date": date_str,
        "strategy": f"{strategy.name} v{strategy.version}",
        "regime": regime,
        "ranked": [(s.sector, s.composite) for s in ranked],
        "sectors": [_sector_score_to_dict(s) for s in ranked],
    }


def _sector_score_to_dict(s: SectorScore) -> dict:
    """SectorScore → dict。"""
    return {
        "sector": s.sector,
        "composite": s.composite,
        "signal": s.signal,
        "position": s.position,
        "regime": s.regime,
        "factors": s.factors,
    }


def _print_results(
    strategy: BaseStrategy,
    ranked: list[SectorScore],
    date_str: str,
    regime_cn: str,
) -> None:
    """Rich 表格输出。"""
    console.print(f"\n[bold]📅 {date_str} | 市场状态: {regime_cn} | 策略: {strategy}[/bold]\n")

    table = Table(title=f"🔥 板块排名 ({strategy.name} v{strategy.version})", show_lines=True)
    table.add_column("排名", style="cyan", width=4)
    table.add_column("板块", style="white", width=12)
    table.add_column("综合", style="bold", width=6)
    table.add_column("信号", width=6)
    table.add_column("仓位", width=6)
    table.add_column("因子明细", width=40)

    for i, s in enumerate(ranked):
        signal_style = {"BUY": "green", "SELL": "red", "HOLD": "yellow"}.get(s.signal, "white")
        factor_str = " ".join(
            f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in s.factors.items()
            if not k.startswith("_")
        )
        table.add_row(
            str(i + 1),
            s.sector,
            f"{s.composite:+.2f}",
            f"[{signal_style}]{s.signal}[/{signal_style}]",
            f"{s.position:.0%}",
            factor_str,
        )

    console.print(table)

    # 操作建议
    buy_sectors = [s for s in ranked if s.signal == "BUY"]
    if buy_sectors:
        console.print("\n[bold]📋 建议买入:[/bold]")
        for s in buy_sectors:
            console.print(f"  [green]{s.sector}[/green]: 综合={s.composite:.3f} 仓位={s.position:.0%}")


def _save_results(
    strategy: BaseStrategy,
    ranked: list[SectorScore],
    date_str: str,
) -> None:
    """保存分析结果到 JSON。"""
    date_file = date_str.split()[0].replace("-", "")
    out_path = RESULTS_DIR / f"daily_{strategy.name}_{date_file}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "strategy": f"{strategy.name} v{strategy.version}",
            "date": date_str,
            "ranked": [(s.sector, s.composite) for s in ranked],
            "sectors": [_sector_score_to_dict(s) for s in ranked],
        }, f, ensure_ascii=False, indent=2, default=str)
    console.print(f"\n[dim]结果已保存: {out_path}[/dim]")
