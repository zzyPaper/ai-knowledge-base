"""回测报告与可视化

借鉴QLib的分析模块思路，提供简洁的图表和绩效指标
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# 尝试导入matplotlib，如果没有就降级
try:
    import matplotlib
    matplotlib.use("Agg")  # 无头模式
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    logger.warning("matplotlib未安装，图表功能不可用")

try:
    import seaborn as sns
    HAS_SNS = True
except ImportError:
    HAS_SNS = False


def report(backtest_result: dict, save_path: str = None) -> str:
    """生成回测报告文本

    Args:
        backtest_result: BacktestEngine.run() 返回的结果
        save_path: 报告保存路径（可选）

    Returns:
        报告文本
    """
    lines = []
    lines.append("=" * 60)
    lines.append("📊 基金量化回测报告")
    lines.append("=" * 60)

    lines.append(f"\n【账户概览】")
    lines.append(f"  初始资金: {backtest_result.get('init_cash', 0):,.2f} 元")
    lines.append(f"  最终资产: {backtest_result.get('final_value', 0):,.2f} 元")
    lines.append(f"  总收益: {backtest_result.get('total_profit', 0):+,.2f} 元")
    lines.append(f"  总收益率: {backtest_result.get('total_return_pct', 0):+.2f}%")

    lines.append(f"\n【绩效指标】")
    lines.append(f"  年化收益率: {backtest_result.get('annual_return_pct', 0):+.2f}%")
    lines.append(f"  最大回撤: {backtest_result.get('max_drawdown_pct', 0):.2f}%")
    lines.append(f"  夏普比率: {backtest_result.get('sharpe_ratio', 0):.2f}")
    lines.append(f"  胜率(日): {backtest_result.get('win_rate_pct', 0):.1f}%")
    lines.append(f"  交易天数: {backtest_result.get('trading_days', 0)} 天")

    lines.append(f"\n【交易统计】")
    lines.append(f"  买入次数: {backtest_result.get('buy_count', 0)}")
    lines.append(f"  卖出次数: {backtest_result.get('sell_count', 0)}")
    lines.append(f"  总手续费: {backtest_result.get('total_fees', 0):.2f} 元")

    penalty = backtest_result.get("penalty_fee_total", 0)
    if penalty > 0:
        lines.append(f"  ⚠️ 惩罚费(不足7天): {penalty:.2f} 元")
        lines.append(f"  惩罚交易笔数: {backtest_result.get('penalty_trade_count', 0)}")

    lines.append(f"\n【持仓】")
    lines.append(f"  剩余现金: {backtest_result.get('remaining_cash', 0):,.2f} 元")
    lines.append(f"  持仓数量: {backtest_result.get('positions', 0)} 只")

    lines.append("\n" + "=" * 60)

    report_text = "\n".join(lines)

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        logger.info(f"报告已保存至: {save_path}")

    return report_text


def plot_equity_curve(backtest_result: dict, save_path: str = "equity_curve.png",
                      title: str = "账户净值曲线"):
    """绘制净值曲线

    Args:
        backtest_result: 回测结果dict
        save_path: 图片保存路径
        title: 标题
    """
    if not HAS_MPL:
        logger.warning("matplotlib未安装，跳过绘图")
        return

    df = backtest_result.get("daily_snapshots")
    if df is None or df.empty:
        logger.warning("无净值数据，跳过绘图")
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # 1. 净值曲线
    ax = axes[0]
    ax.plot(df["date"], df["total_value"], label="组合净值", color="#1f77b4", linewidth=2)
    ax.fill_between(df["date"], backtest_result.get("init_cash", 0),
                    df["total_value"], alpha=0.1, color="#1f77b4")
    ax.axhline(y=backtest_result.get("init_cash", 0), color="gray",
               linestyle="--", alpha=0.5, label="初始本金")
    ax.set_ylabel("账户总值 (元)")
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 格式化y轴为万元
    def millions(x, pos):
        return f"{x/10000:.1f}万"
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(millions))

    # 2. 每日收益率
    ax = axes[1]
    daily_ret = df["daily_return"]
    colors = ["#e74c3c" if r < 0 else "#2ecc71" for r in daily_ret]
    ax.bar(df["date"], daily_ret, color=colors, alpha=0.7, width=0.8)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_ylabel("日收益率 (%)")
    ax.grid(True, alpha=0.3)

    # 3. 回撤曲线
    ax = axes[2]
    peak = df["total_value"].expanding().max()
    drawdown = (df["total_value"] - peak) / peak * 100
    ax.fill_between(df["date"], drawdown, 0, color="#e74c3c", alpha=0.5)
    ax.set_ylabel("回撤 (%)")
    ax.set_xlabel("日期")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"净值曲线已保存至: {save_path}")


def plot_compare(results: Dict[str, dict], save_path: str = "strategy_compare.png"):
    """多策略对比图

    Args:
        results: {strategy_name: backtest_result}
        save_path: 保存路径
    """
    if not HAS_MPL:
        logger.warning("matplotlib未安装，跳过绘图")
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
              "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    ax = axes[0]
    for i, (name, result) in enumerate(results.items()):
        df = result.get("daily_snapshots")
        if df is not None and not df.empty:
            norm = df["total_value"] / result.get("init_cash", 1) * 100
            ax.plot(df["date"], norm, label=name, color=colors[i % len(colors)],
                   linewidth=1.5)

    ax.set_ylabel("净值 (初始=100)")
    ax.set_title("多策略净值对比", fontsize=14)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for i, (name, result) in enumerate(results.items()):
        df = result.get("daily_snapshots")
        if df is not None and not df.empty:
            peak = df["total_value"].expanding().max()
            drawdown = (df["total_value"] - peak) / peak * 100
            ax.plot(df["date"], drawdown, label=name, color=colors[i % len(colors)],
                   linewidth=1.5)

    ax.set_ylabel("回撤 (%)")
    ax.set_xlabel("日期")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"策略对比图已保存至: {save_path}")


def print_trade_history(backtest_result: dict, max_rows: int = 50) -> pd.DataFrame:
    """打印交易记录

    Returns:
        交易记录DataFrame
    """
    # 从backtest_result中并没有直接包含trade_history
    # 需要通过DailySnapshot的holdings来重建
    # 这里留作外部使用，实际需要从Account对象获取

    trades = []
    snapshots = backtest_result.get("daily_snapshots")
    if snapshots is not None and not snapshots.empty:
        for _, row in snapshots.iterrows():
            holdings = row.get("holdings", [])
            if isinstance(holdings, list):
                for h in holdings:
                    trades.append(h)

    if trades:
        df = pd.DataFrame(trades)
        print(f"\n=== 交易记录 (最近{max_rows}条) ===")
        print(df.tail(max_rows).to_string(index=False))
        return df
    return pd.DataFrame()


def summary_table(results: Dict[str, dict]) -> pd.DataFrame:
    """生成策略对比表"""
    rows = []
    for name, r in results.items():
        rows.append({
            "策略": name,
            "总收益率%": r.get("total_return_pct", 0),
            "年化收益%": r.get("annual_return_pct", 0),
            "最大回撤%": r.get("max_drawdown_pct", 0),
            "夏普": r.get("sharpe_ratio", 0),
            "胜率%": r.get("win_rate_pct", 0),
            "交易次数": r.get("buy_count", 0) + r.get("sell_count", 0),
            "手续费": round(r.get("total_fees", 0), 2),
        })
    df = pd.DataFrame(rows)
    return df.sort_values("总收益率%", ascending=False)
