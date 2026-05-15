"""Daily report — shows trades and market overview with Dual Momentum approach."""

from typing import Optional
import pandas as pd
from config.sector_etf_map import get_etf

FULL_CAPITAL = 4000


def generate_report(
    date_str: str,
    scores_df: pd.DataFrame,
    regime: str,
    top_sectors: list[str],
    today_data: dict,
    index_recent: Optional[pd.DataFrame] = None,
    position_pct: int = 100,
    ma20_level: float = 0,
    trades: Optional[list[dict]] = None,
    total_invest: int = 0,
    remain_cash: int = 0,
    target_display: Optional[list[dict]] = None,
    rebalance_note: str = "",
) -> str:
    lines = []
    lines.append(f"# A股板块分析 & 基金操作建议 — {date_str}")
    lines.append("")

    t = "📈 趋势市" if regime == "trending" else "📊 震荡市"
    lines.append(f"## {t}")
    lines.append("")

    if index_recent is not None and not index_recent.empty:
        last = index_recent.iloc[-1]
        lines.append(f"- 沪深300: {float(last.get('close', 0)):.0f}  ({float(last.get('pct_chg', 0)):+.2f}%)")
        lines.append(f"- 近5日: {_fmt_change(index_recent['close'].iloc[-1], index_recent['close'].iloc[0])}")
        if ma20_level > 0:
            lines.append(f"- MA20: {ma20_level:.0f}")
        lines.append("")

    lines.append("## 评分排名")
    lines.append("")

    if scores_df.empty:
        lines.append("> 所有板块均未通过绝对动量过滤（60日趋势为负），建议持有现金观望。")
        lines.append("")
    else:
        lines.append("| 排名 | 板块 | 基金 | 综合分 |")
        lines.append("|------|------|------|-------|")
        for _, row in scores_df.head(10).iterrows():
            lines.append(f"| {int(row['rank'])} | {row['sector']} | {get_etf(row['sector'])} | {float(row['composite']):.3f} |")
        lines.append("")

    total_capital = FULL_CAPITAL
    lines.append("# ⭐ 今日操作指令")
    lines.append("")
    lines.append(f"**账户总资金**: {total_capital}元")
    lines.append(f"**建议总仓位**: {position_pct}%{rebalance_note}")
    lines.append("")

    holdings_text = "空仓（等待入场信号）" if not target_display else "、".join(
        f"{h['code']}({h['amount']}元)" for h in target_display
    )
    lines.append(f"**目标持仓**: {holdings_text}")
    lines.append("")

    if rebalance_note and not trades:
        lines.append(f"> 今日非调仓日{rebalance_note}，持仓不变，继续持有。")
        lines.append("")
    elif trades:
        has_action = any(abs(t.get("amount", 0)) > 0 for t in trades)
        if has_action:
            lines.append("| 操作 | 基金代码 | 基金名称 | 金额 |")
            lines.append("|------|---------|---------|------|")
            for t in trades:
                a = "**买入**" if t["action"] == "buy" else "**卖出**"
                lines.append(f"| {a} | **{t['code']}** | {t['name']} | **{t['amount']}元** |")
            lines.append("")
    else:
        lines.append("✅ **无需操作**，持仓不变")
        lines.append("")

    if target_display:
        total_hold = sum(h["amount"] for h in target_display)
        lines.append(f"> 持仓合计 {total_hold}元，现金 {remain_cash}元")
        lines.append("")

    if trades and any(abs(t.get("amount", 0)) > 0 for t in trades):
        lines.append("### ⚠️ 卖出规则")
        lines.append("")
        lines.append(f"1. 持有板块单日跌超 **-5%** → 全卖，等明天报告")
        lines.append(f"2. 沪深300跌破 {ma20_level:.0f}（20日均线）→ 全卖")
        lines.append("3. 调仓日（满5天）重新评估，中间不动")
        lines.append("")

    lines.append("---")
    lines.append(f"*报告: {date_str} 14:30 | Dual Momentum*")
    return "\n".join(lines)


def _fmt_change(a: float, b: float) -> str:
    if b == 0:
        return "N/A"
    return f"{(a / b - 1) * 100:+.2f}%"
