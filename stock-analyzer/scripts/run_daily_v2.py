#!/usr/bin/env python3
"""Daily operation pipeline — Professional Sector Rotation strategy.

Designed for 2:40 PM workflow: analyzes data, outputs buy/sell actions
before 3:00 PM market close.

Usage:
  python scripts/run_daily_v2.py                    # live mode: update data + today's actions
  python scripts/run_daily_v2.py --date 2026-05-11  # historical backtest
  python scripts/run_daily_v2.py --brief             # short output
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import pandas as pd
import numpy as np
import requests

from src.engine.executor import ForwardExecutor
from src.engine.config import StrategyConfig
from config.sector_etf_map import get_etf, get_etf_code

SECTORS_PKL = BASE_DIR / "data" / "sectors_full.pkl"
INDEX_PKL = BASE_DIR / "data" / "index_full.pkl"
TENCENT_URL = "http://qt.gtimg.cn/q="
INITIAL_CAPITAL = 5000


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _find_pkl(path: Path, fallback: Path) -> Path:
    return path if path.exists() else fallback


def load_data():
    """Load cached pickle data."""
    sp = _find_pkl(SECTORS_PKL, BASE_DIR / "data" / "sectors_2023_2025.pkl")
    ip = _find_pkl(INDEX_PKL, BASE_DIR / "data" / "index_2023_2025.pkl")

    sectors = pd.read_pickle(sp)
    index_data = pd.read_pickle(ip)

    for df in sectors.values():
        df["date"] = pd.to_datetime(df["date"])
    index_data["date"] = pd.to_datetime(index_data["date"])

    for k, df in sectors.items():
        if "pct_chg" not in df.columns:
            df["pct_chg"] = df["close"].pct_change(1).fillna(0) * 100
    if "pct_chg" not in index_data.columns:
        index_data["pct_chg"] = index_data["close"].pct_change(1).fillna(0) * 100

    return sectors, index_data


def latest_date(index_data: pd.DataFrame) -> str:
    return index_data["date"].max().strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Live data (Tencent real-time quotes)
# ---------------------------------------------------------------------------

def _tencent_prefix(code: str) -> str:
    """Convert ETF code like '512480' to Tencent format 'sh512480'."""
    code = code.strip()
    if code.startswith(("sh", "sz")):
        return code
    return f"sh{code}" if code.startswith(("5", "6", "9")) else f"sz{code}"


def fetch_live_quotes(etf_codes: list[str]) -> dict[str, dict]:
    """Fetch real-time ETF + index quotes from Tencent.

    Returns dict keyed by raw code (e.g. 'sh512480') with fields:
      name, price, change_pct, amount
    """
    codes = [_tencent_prefix(c) for c in etf_codes]
    codes.append("sh000300")
    results = {}

    for i in range(0, len(codes), 50):
        batch = codes[i:i + 50]
        try:
            resp = requests.get(
                f"{TENCENT_URL}{','.join(batch)}",
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10,
            )
        except Exception:
            continue

        for line in resp.text.strip().split(";"):
            if not line.strip():
                continue
            parts = line.split("~")
            if len(parts) < 40:
                continue
            try:
                raw_code = parts[0].split("=")[0].replace("v_", "").strip()
                results[raw_code] = {
                    "name": parts[1],
                    "price": float(parts[3]) if parts[3] else 0.0,
                    "change_pct": float(parts[32]) if parts[32] else 0.0,
                }
            except (ValueError, IndexError):
                continue
    return results


def fetch_today_index() -> dict:
    """Get CSI 300 live quote."""
    quotes = fetch_live_quotes([])  # sh000300 is always fetched
    return quotes.get("sh000300", {"price": 0, "change_pct": 0})


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(date_str: str, warmup_days: int = 250,
                 initial_capital: float = 5000.0,
                 auto_update: bool = True) -> Optional[dict]:
    """Run ForwardExecutor through date_str and return the last day's record.
    
    warmup_days: 预热天数，用于计算信号指标。默认250天。
    initial_capital: 初始资金，默认5000元。
    auto_update: 自动更新数据到最新。默认True。
    """
    # 自动更新最新数据
    if auto_update:
        _update_data_live()
    
    sectors, index_data = load_data()

    target_dt = pd.Timestamp(date_str)
    all_dates = sorted(index_data["date"].unique())
    indices = [i for i, d in enumerate(all_dates) if d >= target_dt]
    if not indices:
        return None
    warmup_start_idx = max(0, indices[0] - warmup_days)
    warmup_start = all_dates[warmup_start_idx].strftime("%Y-%m-%d")

    config = StrategyConfig()
    # reset_nav: 账户从指定金额开始，不累积历史收益
    # warmup_days: 用历史数据算信号，但跳过 warmup 天的实际交易
    executor = ForwardExecutor(config, verbose=False, reset_nav=initial_capital)
    result_df = executor.run(sectors, index_data, warmup_start, date_str,
                            warmup_days=warmup_days)

    if result_df.empty:
        return None

    last = result_df.iloc[-1]
    return {
        "date": last["date"],
        "nav": last["nav"],
        "cash": last["cash"],
        "invested": last["invested"],
        "daily_pnl": last["daily_pnl"],
        "daily_return_pct": last["daily_return_pct"],
        "index_return_pct": last["index_return_pct"],
        "in_market": last["in_market"],
        "regime": last["regime"],
        "position_cap": last["position_cap"],
        "vol_scale": last["vol_scale"],
        "is_rebalance": last["is_rebalance"],
        "trades": last["trades"],
        "positions": last["positions"],
        "df": result_df,  # full history for signal extraction
    }


def run_live_pipeline() -> dict:
    """Live 2:40 PM pipeline: start fresh each day.

    Does NOT simulate history. Computes current market state, signals,
    and builds target portfolio directly. Day 1 = always an entry day.

    Steps:
      1. Update cached data to yesterday (AKShare)
      2. Compute market regime + timing (should we enter?)
      3. Compute sector signals + build target portfolio
      4. Fetch today's real-time index + ETF quotes
      5. Check crash stops (on target positions)
      6. Output buy instructions (always entry on fresh start)
    """
    today_str = datetime.now().strftime("%Y-%m-%d")
    print(f"[运行] 实时模式 — {today_str} 14:40")

    # Step 1: update data
    _update_data_live()

    # Step 2: load data and compute state
    sectors, index_data = load_data()
    yesterday = index_data["date"].max()
    yesterday_str = yesterday.strftime("%Y-%m-%d")
    print(f"[数据] 最新日期: {yesterday_str}")

    # Market regime
    from src.engine.regime import RegimeDetector
    regime_detector = RegimeDetector()
    regime_result = regime_detector.detect(index_data, yesterday)

    # Market timing
    idx_hist = index_data[index_data["date"] <= yesterday]
    idx_close = idx_hist["close"]
    ret_20d = float(idx_close.iloc[-1] / idx_close.iloc[-21] - 1) * 100 if len(idx_close) >= 21 else 0
    ret_10d = float(idx_close.iloc[-1] / idx_close.iloc[-11] - 1) * 100 if len(idx_close) >= 11 else 0
    ret_60d = float(idx_close.iloc[-1] / idx_close.iloc[-61] - 1) * 100 if len(idx_close) >= 61 else 0

    # Adaptive exit threshold
    if ret_60d > 5.0:
        adapt_exit = -3.0
    elif ret_60d > 3.0:
        adapt_exit = -2.0
    elif ret_60d > 1.0:
        adapt_exit = -1.0
    elif ret_60d > 0.0:
        adapt_exit = -0.8
    else:
        adapt_exit = -0.5

    can_enter = ret_20d > 0.5
    should_exit = ret_10d < adapt_exit
    in_market = can_enter and not should_exit

    # Step 3: compute signals and portfolio
    signals = _compute_signals_for_live(sectors, index_data, yesterday_str, regime_result.regime)
    trades = []
    target_positions = {}

    if in_market and signals is not None and not signals.empty:
        nav = INITIAL_CAPITAL
        pos_cap = regime_result.position_cap
        targets = _build_targets(signals, nav, pos_cap)

        for t in targets:
            target_positions[t["sector"]] = t["amount"]
            trades.append({
                "action": "buy", "reason": "建仓",
                "sector": t["sector"], "etf": t["etf"], "amount": t["amount"],
            })

        total_invest = sum(t["amount"] for t in targets)
        cash = INITIAL_CAPITAL - total_invest
    else:
        pos_cap = 0
        cash = INITIAL_CAPITAL

    # Step 4: today's real-time data
    print("[实时] 获取实时行情...")
    etf_codes = list(set(t["etf"] for t in trades)) if trades else []
    live_quotes = fetch_live_quotes(etf_codes)
    idx_quote = live_quotes.get("sh000300", {})
    idx_change_today = idx_quote.get("change_pct", 0.0)
    idx_price = idx_quote.get("price", 0.0)

    # Step 5: crash stop check on target positions
    for sector, amount in list(target_positions.items()):
        code = get_etf_code(sector)
        if code == "—":
            continue
        tencent_code = _tencent_prefix(code)
        quote = live_quotes.get(tencent_code, {})
        pct = quote.get("change_pct", 0.0)
        if pct <= -5.0:
            # Remove from targets
            trades = [t for t in trades if t["sector"] != sector]
            del target_positions[sector]
            cash += amount
            pos_cap = max(0, pos_cap - int(amount / INITIAL_CAPITAL * 100))
            print(f"  [警告] {sector} 盘中跌停 {pct:+.1f}% → 从买入清单移除")

    # Check market exit: if index drops sharply today, warn
    exit_warning = ""
    if in_market and idx_change_today <= -2.0:
        exit_warning = f"⚠️ 沪深300 盘中大跌 {idx_change_today:+.1f}%，建议观望，可暂缓买入"

    return {
        "date": today_str,
        "nav": INITIAL_CAPITAL,
        "cash": cash,
        "invested": sum(target_positions.values()),
        "daily_pnl": 0,
        "daily_return_pct": 0,
        "index_return_pct": idx_change_today,
        "index_price": idx_price,
        "in_market": in_market,
        "regime": regime_result.regime,
        "position_cap": pos_cap,
        "vol_scale": 1.0,
        "is_rebalance": True,  # day 1 = always entry
        "trades": trades,
        "positions": target_positions,
        "days_since_rebalance": 0,
        "exit_warning": exit_warning,
        "ret_20d": ret_20d,
        "ret_10d": ret_10d,
        "ret_60d": ret_60d,
        "adapt_exit": adapt_exit,
    }


# ---------------------------------------------------------------------------
# Live helpers
# ---------------------------------------------------------------------------

def _update_data_live():
    """Run full data update for live pipeline."""
    import subprocess
    update_script = BASE_DIR / "scripts" / "update_data.py"
    if update_script.exists():
        print("[更新] 刷新行情数据...")
        t0 = time.time()
        result = subprocess.run(
            [sys.executable, str(update_script)],
            capture_output=True, text=True, timeout=180,
        )
        elapsed = time.time() - t0
        print(f"[更新] 完成 ({elapsed:.0f}s)")
        for line in result.stdout.split("\n"):
            if "ERROR" in line or "WARN" in line:
                print(f"  {line}")


def _collect_etf_codes(positions: dict) -> list[str]:
    """Get ETF codes for current positions."""
    codes = []
    for sector in positions:
        code = get_etf_code(sector)
        if code != "—":
            codes.append(code)
    return codes


def _build_etf_sector_map() -> dict[str, str]:
    """Build {etf_code: sector_name} mapping."""
    try:
        from config.sector_etf_map import SECTOR_ETF_MAP
        return {v["code"]: k for k, v in SECTOR_ETF_MAP.items()}
    except Exception:
        return {}


def _count_trading_days(index_data: pd.DataFrame, from_date: str, to_date: str) -> int:
    mask = (index_data["date"] >= pd.Timestamp(from_date)) & (index_data["date"] <= pd.Timestamp(to_date))
    return len(index_data[mask])


def _compute_signals_for_live(sectors: dict, index_data: pd.DataFrame,
                               yesterday: str, regime: str):
    """Compute signal scores using data through yesterday."""
    from src.engine.signals import SignalPipeline
    from src.engine.config import SignalConfig

    date = pd.Timestamp(yesterday)
    start_dt = date - timedelta(days=365)

    sectors_data = {}
    for name, df in sectors.items():
        subset = df[(df["date"] >= start_dt) & (df["date"] <= date)].copy()
        if len(subset) >= 20:
            sectors_data[name] = subset

    if not sectors_data:
        return None

    pipeline = SignalPipeline(SignalConfig())
    result = pipeline.compute(sectors_data, regime)
    return result.scores_df if result else None


def _build_targets(scores_df, nav: float, position_cap: int) -> list[dict]:
    """Build target portfolio from signal scores."""
    from src.engine.portfolio import PortfolioBuilder
    from src.engine.config import PortfolioConfig

    builder = PortfolioBuilder(PortfolioConfig())
    plan = builder.build(scores_df, nav, position_cap)

    targets = []
    for t in plan.targets:
        targets.append({
            "sector": t.sector,
            "etf": t.etf,
            "amount": round(t.amount),
            "weight": t.weight,
        })
    return targets


# ---------------------------------------------------------------------------
# Historical pipeline
# ---------------------------------------------------------------------------

def _regime_label(regime: str) -> str:
    labels = {"BULL": "牛市", "RECOVERY": "修复", "NEUTRAL": "中性",
              "CORRECTION": "回调", "BEAR": "熊市"}
    return labels.get(regime, regime)


def _regime_emoji(regime: str) -> str:
    emojis = {"BULL": "🔥", "RECOVERY": "📈", "NEUTRAL": "📊",
              "CORRECTION": "⚠️", "BEAR": "🧊"}
    return emojis.get(regime, "")


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def format_report(result: dict) -> str:
    lines = []
    date_str = result["date"]
    lines.append(f"# 每日操作建议 — {date_str} 14:40")
    lines.append("")

    if "error" in result:
        lines.append(f"**错误**: {result['error']}")
        return "\n".join(lines)

    regime = result["regime"]
    lines.append(f"## 市场状态: {_regime_emoji(regime)} {_regime_label(regime)} ({regime})")
    lines.append("")

    lines.append("| 指标 | 数值 |")
    lines.append("|------|------|")
    lines.append(f"| 策略净值 | {result['nav']:.2f}元 |")
    if result.get("daily_pnl", 0) != 0:
        lines.append(f"| 当日收益 | {result['daily_pnl']:+.2f}元 ({result['daily_return_pct']:+.3f}%) |")
    if result.get("index_price"):
        lines.append(f"| 沪深300 | {result['index_price']:.2f} ({result['index_return_pct']:+.2f}%) |")
    else:
        lines.append(f"| 沪深300 | {result['index_return_pct']:+.2f}% |")
    lines.append(f"| 仓位上限 | {result['position_cap']}% |")
    lines.append(f"| 是否调仓日 | {'是' if result.get('is_rebalance') else '否'} |")
    lines.append(f"| 在场内 | {'是' if result['in_market'] else '否'} |")
    if result.get('ret_20d') is not None:
        lines.append(f"| 20日动量 | {result['ret_20d']:+.2f}% |")
        lines.append(f"| 10日动量 | {result['ret_10d']:+.2f}% |")
        lines.append(f"| 60日动量 | {result['ret_60d']:+.2f}% |")
        lines.append(f"| 离场阈值 | {result['adapt_exit']:+.1f}% |")
    lines.append("")

    if result.get("exit_warning"):
        lines.append(f"> **{result['exit_warning']}**")
        lines.append("")

    positions = result["positions"]
    if positions:
        lines.append("## 当前持仓")
        lines.append("")
        lines.append("| 板块 | ETF | 金额 |")
        lines.append("|------|-----|------|")
        for sector, amount in positions.items():
            code = get_etf_code(sector)
            etf_info = get_etf(sector)
            name = etf_info.split("(")[1].rstrip(")") if "(" in etf_info else code
            lines.append(f"| {sector} | {code} {name} | {amount:,.0f}元 |")
        lines.append("")
        lines.append(f"**持仓合计**: {result['invested']:,.0f}元 | **现金**: {result['cash']:,.0f}元")
    else:
        lines.append("## 当前持仓: 空仓")
        lines.append(f"**现金**: {result['cash']:,.0f}元")

    lines.append("")

    trades = result["trades"]
    if trades:
        buys = [t for t in trades if t["action"] == "buy"]
        sells = [t for t in trades if t["action"] == "sell"]
        lines.append("## ⭐ 今日操作指令（3点前执行）")
        lines.append("")

        if sells:
            lines.append("### 卖出")
            lines.append("")
            lines.append("| 板块 | ETF | 金额 | 原因 |")
            lines.append("|------|-----|------|------|")
            for t in sells:
                lines.append(f"| {t['sector']} | **{t['etf']}** | **{t['amount']:,.0f}元** | {t.get('reason', '调仓')} |")
            lines.append("")

        if buys:
            lines.append("### 买入")
            lines.append("")
            lines.append("| 板块 | ETF | 金额 |")
            lines.append("|------|-----|------|")
            for t in buys:
                lines.append(f"| {t['sector']} | **{t['etf']}** | **{t['amount']:,.0f}元** |")
            lines.append("")

        total_buy = sum(t["amount"] for t in buys)
        total_sell = sum(t["amount"] for t in sells)
        net = total_buy - total_sell
        lines.append(f"**买入合计**: {total_buy:,.0f}元 | **卖出合计**: {total_sell:,.0f}元 | **净变动**: {net:+,.0f}元")
    else:
        lines.append("## ⭐ 今日操作: 无")
        lines.append("")
        if not result["in_market"]:
            lines.append("> 策略处于离场状态，等待入场信号。")
        elif not result.get("is_rebalance"):
            dsr = result.get("days_since_rebalance", 0)
            lines.append(f"> 距上次调仓 {dsr} 天，还需 {5 - dsr} 天。持仓不变。")

    lines.append("")
    lines.append("---")
    lines.append(f"*报告生成: {date_str} 14:40 | Professional Sector Rotation v1.0*")
    return "\n".join(lines)


def print_brief(result: dict):
    trades = result["trades"]
    regime = result["regime"]
    date_str = result["date"]

    print(f"{'='*60}")
    print(f"  {date_str} 14:40 | {_regime_label(regime)} | 仓位上限: {result['position_cap']}%")
    print(f"{'='*60}")

    if "error" in result:
        print(f"  错误: {result['error']}")
        return

    idx_str = ""
    if result.get("index_price"):
        idx_str = f" | 沪深300: {result['index_price']:.0f} ({result['index_return_pct']:+.2f}%)"

    # Show market timing analysis
    print(f"  20日: {result.get('ret_20d', 0):+.2f}%  10日: {result.get('ret_10d', 0):+.2f}%  60日: {result.get('ret_60d', 0):+.2f}%")
    print(f"  入场条件(20d>0.5%): {'满足' if result['in_market'] else '不满足'}  离场阈值: {result.get('adapt_exit', -0.5):+.1f}%")
    print()

    if not trades:
        if result["in_market"]:
            print(f"  持仓不变（非调仓日）{idx_str}")
            for sector, amount in result["positions"].items():
                print(f"    持有 {sector}: {amount:,.0f}元")
        else:
            print(f"  建议空仓观望（不满足入场条件）{idx_str}")
    else:
        print(f"  ⭐ 买入指令（初始资金 {INITIAL_CAPITAL}元）:")
        for t in trades:
            print(f"    → {t['etf']} {t['sector']}: {t['amount']:,.0f}元")

    print(f"  投入: {result['invested']:,.0f}元 | 现金保留: {result['cash']:,.0f}元{idx_str}")

    if result.get("exit_warning"):
        print(f"  {result['exit_warning']}")

    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="每日操作建议 (Professional Sector Rotation v2)")
    parser.add_argument("--date", type=str, default=None,
                        help="历史回测日期 (YYYY-MM-DD)，默认实时模式")
    parser.add_argument("--brief", action="store_true", help="简短输出")
    parser.add_argument("--live", action="store_true", default=True,
                        help="实时模式 (默认)")
    args = parser.parse_args()

    if args.date:
        # Historical mode
        print(f"[运行] 历史回测 {args.date}")
        result = run_pipeline(args.date)
        if result is None:
            print(f"[错误] 无法分析 {args.date}")
            sys.exit(1)
        # Clean up internal df
        result.pop("df", None)
    else:
        # Live mode
        result = run_live_pipeline()

    if args.brief:
        print()
        print_brief(result)
    else:
        report = format_report(result)
        print()
        print(report)

    # Save report
    from config.settings import RESULTS_DIR
    date_label = result["date"]
    result_path = RESULTS_DIR / f"{date_label}_v2_report.md"
    result_path.write_text(format_report(result), encoding="utf-8")
    print(f"\n[保存] {result_path}")


if __name__ == "__main__":
    main()
