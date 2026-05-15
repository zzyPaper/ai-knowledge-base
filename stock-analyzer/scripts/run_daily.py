#!/usr/bin/env python3
"""Daily analysis: run at 14:30 for live analysis, or pass --date for history.

Rebalances weekly (every 5 trading days) — uses Dual Momentum approach:
  - Absolute momentum filter (10d): only positively trending sectors qualify
  - Overbought cap (15%): skip extreme winners to avoid reversal
  - Relative momentum ranking (20d) among qualifiers

References: Antonacci (2012), Jegadeesh & Titman (1993), AQR methodology

Usage:
  python scripts/run_daily.py                                 # today's live analysis
  python scripts/run_daily.py --date 2026-04-09               # historical analysis
  python scripts/run_daily.py --brief                          # short output
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import pandas as pd
from src.signals.fusion import compute_sector_scores, detect_market_regime
from src.signals.position_sizing import compute_position_pct
from src.daily.reporter import generate_report, FULL_CAPITAL
from src.daily.position_tracker import load_position, save_position, compute_trades, should_rebalance
from config.sector_etf_map import get_etf, get_etf_code


def _count_trading_days(index_data: pd.DataFrame, from_date: str, to_date: str) -> int:
    """Count trading days between two dates (inclusive of to_date)."""
    mask = (index_data["date"] >= pd.Timestamp(from_date)) & (index_data["date"] <= pd.Timestamp(to_date))
    return len(index_data[mask])


def run_live_analysis(date_str: str, brief: bool = False) -> str:
    """Fetch live AKShare data and generate report."""
    from src.data.fetcher import get_sectors_list_safe as get_live_sectors_safe
    from src.data.fetcher import get_sector_history_safe, get_index_history_safe

    today = pd.Timestamp(date_str)
    start_120d = (today - timedelta(days=120)).strftime("%Y-%m-%d")

    live = get_live_sectors_safe()
    if live is None or live.empty:
        return "# 报告生成失败：无法获取行情数据"

    if "成交额" in live.columns:
        top_sectors = live.nlargest(20, "成交额")["板块名称"].tolist()
    else:
        top_sectors = live["板块名称"].head(20).tolist()

    sectors_data = {}
    for s in top_sectors:
        df = get_sector_history_safe(s, start_120d, date_str)
        if df is not None and not df.empty:
            sectors_data[s] = df

    index_data = get_index_history_safe("沪深300", start_120d, date_str)

    return _build_report(date_str, sectors_data, index_data, top_sectors[:5])


def run_cached_analysis(date_str: str, brief: bool = False) -> str:
    """Use cached pickle data for historical analysis."""
    sectors_pkl = BASE_DIR / "data" / "sectors_full.pkl"
    if not sectors_pkl.exists():
        sectors_pkl = BASE_DIR / "data" / "sectors_2023_2025.pkl"
    index_pkl = BASE_DIR / "data" / "index_full.pkl"
    if not index_pkl.exists():
        index_pkl = BASE_DIR / "data" / "index_2023_2025.pkl"

    sectors = pd.read_pickle(sectors_pkl)
    index_data = pd.read_pickle(index_pkl)

    for df in sectors.values():
        df["date"] = pd.to_datetime(df["date"])
    index_data["date"] = pd.to_datetime(index_data["date"])

    target_dt = pd.Timestamp(date_str)
    start_120d = (target_dt - timedelta(days=120)).strftime("%Y-%m-%d")

    # Sector amount ranking
    sector_amounts = {}
    for name, df in sectors.items():
        row = df[df["date"] == target_dt]
        if not row.empty:
            sector_amounts[name] = float(row.iloc[0].get("amount", 0))

    sorted_sectors = sorted(sector_amounts.items(), key=lambda x: x[1], reverse=True)
    top_sectors = [s[0] for s in sorted_sectors[:20]]

    sectors_data = {}
    for name in top_sectors:
        if name in sectors:
            subset = sectors[name][
                (sectors[name]["date"] >= pd.Timestamp(start_120d)) & (sectors[name]["date"] <= target_dt)
            ].copy()
            if len(subset) >= 10:
                sectors_data[name] = subset

    idx_subset = index_data[
        (index_data["date"] >= pd.Timestamp(start_120d)) & (index_data["date"] <= target_dt)
    ].copy()
    idx_subset["pct_chg"] = idx_subset["close"].pct_change(1).fillna(0) * 100

    return _build_report(date_str, sectors_data, idx_subset, top_sectors[:5], index_data)


def _build_report(
    date_str: str,
    sectors_data: dict,
    index_data: pd.DataFrame,
    top_sectors: list[str],
    full_index_data: pd.DataFrame = None,
) -> str:
    if index_data is not None and not index_data.empty:
        index_data["pct_chg"] = index_data["close"].pct_change(1).fillna(0) * 100

    regime = detect_market_regime(index_data) if index_data is not None and len(index_data) >= 20 else "unknown"

    if index_data is not None and not index_data.empty:
        scores_df = compute_sector_scores(sectors_data, index_data)
    else:
        scores_df = compute_sector_scores(sectors_data)

    today_data = {}
    for s, df in sectors_data.items():
        row = df[df["date"] == date_str] if "date" in df.columns else df.iloc[-1:]
        if not row.empty:
            latest = row.iloc[-1]
            today_data[s] = {
                "close": float(latest.get("close", 0)),
                "pct_chg": float(latest.get("pct_chg", 0)),
                "amount": float(latest.get("amount", 0)),
                "turnover_rate": float(latest.get("turnover_rate", 0)),
            }

    position_pct = compute_position_pct(index_data) if index_data is not None else 50
    ma20_level = float(index_data["close"].rolling(20).mean().iloc[-1]) if index_data is not None and len(index_data) >= 20 else 0

    # Load old position to check rebalance timing
    old_pos = load_position()
    last_rebalance_date = old_pos.get("last_rebalance_date") if old_pos else None

    # Count trading days since last rebalance
    days_since_rebalance = 0
    if last_rebalance_date and full_index_data is not None:
        days_since_rebalance = _count_trading_days(full_index_data, last_rebalance_date, date_str)
        days_since_rebalance -= 1  # exclude the rebalance day itself

    rebalance_now = should_rebalance(days_since_rebalance, last_rebalance_date, date_str)

    # Compute target buys
    total_capital = FULL_CAPITAL

    # When not rebalancing, position_pct reflects current exposure
    current_nav_for_sizing = total_capital
    if old_pos and "total" in old_pos:
        current_nav_for_sizing = old_pos.get("total", total_capital)

    total_ratio = position_pct / 100.0
    invest_amount = round(current_nav_for_sizing * total_ratio)
    remain_cash = current_nav_for_sizing - invest_amount

    if scores_df.empty:
        # No sectors pass absolute momentum filter → hold cash
        target_buys = []
        target_top = pd.DataFrame()
    elif regime == "trending":
        target_top = scores_df.head(2)
    else:
        target_top = scores_df.head(3)
    n = len(target_top)

    target_buys = []
    for i, (_, row) in enumerate(target_top.iterrows()):
        code = get_etf_code(row["sector"])
        if code == "—":
            continue  # skip sectors without ETF mapping
        mapping = get_etf(row["sector"])
        name = mapping.split("(")[1].rstrip(")") if "(" in mapping else f"{row['sector']}ETF"
        # Weight by rank among actually-bought sectors
        bought_rank = len(target_buys) + 1
        rel_weight = (n - bought_rank + 1) / sum(range(1, n + 1))
        amount = round(invest_amount * rel_weight)
        target_buys.append({"action": "buy", "code": code, "name": name, "sector": row["sector"], "amount": amount})

    # Merge duplicate ETF codes
    from collections import defaultdict
    merged = defaultdict(lambda: {"name": "", "sector": "", "amount": 0})
    for b in target_buys:
        merged[b["code"]]["name"] = b["name"]
        merged[b["code"]]["sector"] = b["sector"]
        merged[b["code"]]["amount"] += b["amount"]
    target_buys_dedup = [{"action": "buy", "code": k, "name": v["name"], "amount": v["amount"]}
                         for k, v in merged.items()]

    if rebalance_now:
        # Compute trades against old position
        trades = compute_trades(old_pos, target_buys_dedup, remain_cash, total_capital)
        cash_after_trade = remain_cash
        # Save new position with rebalance flag
        save_position(date_str, target_buys_dedup, cash_after_trade, total_capital, rebalanced=True)
    else:
        # Hold current position, no trades
        trades = []
        # Keep old position data
        if old_pos:
            save_position(date_str, target_buys_dedup, old_pos.get("cash", remain_cash),
                          old_pos.get("total", total_capital), rebalanced=False)
        else:
            save_position(date_str, target_buys_dedup, remain_cash, total_capital, rebalanced=True)

    rebalance_note = ""
    if not rebalance_now:
        rebalance_note = f"（距上次调仓 {days_since_rebalance} 天，满5天再调）"

    return generate_report(
        date_str=date_str,
        scores_df=scores_df,
        regime=regime,
        top_sectors=top_sectors[:5],
        today_data=today_data,
        index_recent=index_data.tail(5) if index_data is not None else None,
        position_pct=position_pct,
        ma20_level=ma20_level,
        trades=trades,
        total_invest=invest_amount,
        remain_cash=remain_cash,
        target_display=target_buys_dedup,
        rebalance_note=rebalance_note,
    )


def main():
    parser = argparse.ArgumentParser(description="每日板块分析与基金操作建议 (Dual Momentum)")
    parser.add_argument("--date", type=str, default=None, help="分析指定日期 (YYYY-MM-DD)，默认当日")
    parser.add_argument("--brief", action="store_true", help="简短输出仅操作建议")
    args = parser.parse_args()

    date_str = args.date or datetime.now().strftime("%Y-%m-%d")
    is_historical = args.date is not None

    if is_historical:
        print(f"[分析] 使用缓存数据回测 {date_str}")
        report = run_cached_analysis(date_str, args.brief)
    else:
        print(f"[分析] 获取实时数据 {date_str}")
        report = run_live_analysis(date_str, args.brief)

    print()
    print(report)

    from config.settings import RESULTS_DIR
    result_path = RESULTS_DIR / f"{date_str}_report.md"
    result_path.write_text(report, encoding="utf-8")
    print(f"\n[保存] {result_path}")


if __name__ == "__main__":
    main()
