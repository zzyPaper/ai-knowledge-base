#!/usr/bin/env python3
"""Forward test: simulate daily trade execution with 5000 initial capital.

Models the user's actual workflow:
  1. 14:30: run analysis, get buy/sell signals
  2. Before 15:00: execute trades at close price
  3. Returns start from NEXT trading day
  4. Sell also settled at close price

Difference from BacktestEngine: P&L is computed for OLD positions BEFORE
rebalancing, so new positions get 0 return on their first day.
"""

import argparse
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import pandas as pd
from src.signals.fusion import compute_sector_scores, detect_market_regime
from src.signals.position_sizing import compute_position_pct
from config.sector_etf_map import get_etf_code, get_etf

INITIAL_CAPITAL = 5000
REBALANCE_FREQ = 5  # trading days
MIN_HISTORY = 120    # days of historical data for signal calculation


def load_data():
    """Load cached sector and index data."""
    sectors = pd.read_pickle(BASE_DIR / "data" / "sectors_full.pkl")
    index_data = pd.read_pickle(BASE_DIR / "data" / "index_full.pkl")

    for df in sectors.values():
        df["date"] = pd.to_datetime(df["date"])
    index_data["date"] = pd.to_datetime(index_data["date"])

    # Fix missing columns
    for k, df in sectors.items():
        if "pct_chg" not in df.columns:
            df["pct_chg"] = df["close"].pct_change(1).fillna(0) * 100
    if "pct_chg" not in index_data.columns:
        index_data["pct_chg"] = index_data["close"].pct_change(1).fillna(0) * 100

    return sectors, index_data


def get_sector_pct_chg(sectors: dict, sector_name: str, date: pd.Timestamp) -> float:
    """Get sector's daily pct_chg for a specific date."""
    df = sectors.get(sector_name)
    if df is None:
        return 0.0
    row = df[df["date"] == date]
    if row.empty:
        return 0.0
    return float(row["pct_chg"].iloc[0])


def get_sector_close(sectors: dict, sector_name: str, date: pd.Timestamp) -> Optional[float]:
    """Get sector's close price for a specific date."""
    df = sectors.get(sector_name)
    if df is None:
        return None
    row = df[df["date"] == date]
    if row.empty:
        return None
    return float(row["close"].iloc[0])


class ForwardTestEngine:
    """Day-by-day portfolio simulator with correct trade execution timing."""

    def __init__(self, capital: int = INITIAL_CAPITAL):
        self.cash = float(capital)
        self.positions: dict[str, float] = {}  # sector_name -> dollar_value_at_last_close
        self.trades_log: list[dict] = []
        self.daily_log: list[dict] = []
        self.last_rebalance_day = 0  # trading day index of last rebalance

    def run(self, sectors: dict, index: pd.DataFrame,
            start_date: str, end_date: str):
        """Run forward simulation day by day."""
        trading_days = sorted(index["date"].unique())
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)
        trading_days = [d for d in trading_days if start_dt <= d <= end_dt]

        # Force rebalance on day 0
        self.last_rebalance_day = -REBALANCE_FREQ

        for day_idx, day in enumerate(trading_days):
            day_str = day.strftime("%Y-%m-%d")

            # ---- Step 1: Snapshot NAV before today's P&L ----
            nav_before = self.cash + sum(self.positions.values())

            # ---- Step 2: P&L for CURRENT positions (D-1 close → D close) ----
            daily_pnl = 0.0
            for sector_name, value in list(self.positions.items()):
                pct = get_sector_pct_chg(sectors, sector_name, day)
                pnl = value * pct / 100.0
                daily_pnl += pnl
                self.positions[sector_name] = value + pnl

            # ---- Step 3: Rebalance check ----
            is_rebalance = (day_idx - self.last_rebalance_day) >= REBALANCE_FREQ

            trades_today = []
            if is_rebalance:
                # a) Sell all at close
                for sector_name, value in list(self.positions.items()):
                    trades_today.append({
                        "date": day_str, "action": "sell",
                        "sector": sector_name,
                        "etf": get_etf_code(sector_name),
                        "amount": round(value),
                    })
                    self.cash += value
                    del self.positions[sector_name]

                # b) Compute new targets
                targets = self._compute_targets(sectors, index, day)
                total_buy = sum(t["amount"] for t in targets)

                # c) Buy new positions at close
                for t in targets:
                    self.positions[t["sector"]] = float(t["amount"])
                    trades_today.append({
                        "date": day_str, "action": "buy",
                        "sector": t["sector"],
                        "etf": t["etf"],
                        "amount": t["amount"],
                    })
                self.cash -= total_buy
                self.last_rebalance_day = day_idx

            # ---- Step 4: Record daily snapshot ----
            nav = self.cash + sum(self.positions.values())

            idx_row = index[index["date"] == day]
            idx_return = float(idx_row["pct_chg"].iloc[0]) if not idx_row.empty else 0.0

            self.daily_log.append({
                "date": day_str,
                "nav": nav,
                "cash": self.cash,
                "invested": sum(self.positions.values()),
                "daily_pnl": daily_pnl,
                "daily_return": (nav / nav_before - 1) * 100 if nav_before > 0 else 0,
                "index_return": idx_return,
                "positions": {s: round(v) for s, v in self.positions.items()},
                "is_rebalance": is_rebalance,
                "trades": trades_today,
            })

            self.trades_log.append({
                "date": day_str, "is_rebalance": is_rebalance, "trades": trades_today,
            })

        return self._summary(start_date, end_date, index)

    def _active_sectors(self, sectors: dict, date: pd.Timestamp) -> list[str]:
        """Sectors with enough data up to this date."""
        result = []
        for name in sectors:
            df = sectors[name]
            hist = df[df["date"] <= date]
            if len(hist) >= MIN_HISTORY:
                result.append(name)
        return result

    def _compute_targets(self, sectors: dict, index: pd.DataFrame,
                         date: pd.Timestamp) -> list[dict]:
        """Compute target ETF allocations for a rebalance day."""
        date_str = date.strftime("%Y-%m-%d")
        start_120d = (date - timedelta(days=180)).strftime("%Y-%m-%d")

        # Build sectors_data for signal computation
        sectors_data = {}
        for name in sectors:
            df = sectors[name]
            subset = df[(df["date"] >= pd.Timestamp(start_120d)) & (df["date"] <= date)].copy()
            if len(subset) >= 10:
                sectors_data[name] = subset

        idx_subset = index[(index["date"] >= pd.Timestamp(start_120d)) & (index["date"] <= date)].copy()

        if len(sectors_data) == 0 or idx_subset.empty:
            return []

        # Compute scores (same pipeline as run_daily.py)
        scores_df = compute_sector_scores(sectors_data, idx_subset)
        regime = detect_market_regime(idx_subset)
        pos_pct = compute_position_pct(idx_subset)

        if scores_df.empty:
            return []

        effective_top_n = 2 if regime == "trending" else 3
        top_sectors = scores_df.head(effective_top_n)

        # ETF-safe filtering
        qualified = []
        for _, row in top_sectors.iterrows():
            code = get_etf_code(row["sector"])
            if code != "—":
                qualified.append({"sector": row["sector"], "code": code,
                                  "rank": len(qualified) + 1})

        if not qualified:
            return []

        n = len(qualified)
        invest_amount = round(self._total_nav() * pos_pct / 100.0)
        if invest_amount <= 0:
            return []

        # Rank-weighted allocation (same as run_daily.py)
        targets = []
        rank_sum = sum(range(1, n + 1))
        for q in qualified:
            rel_weight = (n - q["rank"] + 1) / rank_sum
            amount = round(invest_amount * rel_weight)
            if amount > 0:
                targets.append({
                    "sector": q["sector"], "etf": q["code"], "amount": amount,
                })
        return targets

    def _total_nav(self) -> float:
        return self.cash + sum(self.positions.values())

    def _summary(self, start_date: str, end_date: str, index: pd.DataFrame):
        df = pd.DataFrame(self.daily_log)
        if df.empty:
            return df

        # Cumulative returns
        start_nav = INITIAL_CAPITAL
        df["port_cumulative"] = df["nav"] / start_nav
        idx_start = index[index["date"] == pd.Timestamp(start_date)]
        idx_end = index[index["date"] == pd.Timestamp(end_date)]
        if not idx_start.empty and not idx_end.empty:
            idx_start_close = float(idx_start["close"].iloc[0])
            idx_end_close = float(idx_end["close"].iloc[0])
            idx_return = (idx_end_close / idx_start_close - 1) * 100
        else:
            idx_return = 0.0

        port_return = (df["nav"].iloc[-1] / start_nav - 1) * 100
        excess = port_return - idx_return

        # Max drawdown
        cumulative = df["port_cumulative"].values
        peak = pd.Series(cumulative).cummax().values
        drawdowns = (cumulative - peak) / peak
        max_dd = float(drawdowns.min()) * 100 if len(drawdowns) > 0 else 0

        df["max_dd_sofar"] = [float(drawdowns[:i+1].min()) * 100 for i in range(len(drawdowns))]

        print(f"\n{'='*80}")
        print(f"Forward Test: {start_date} → {end_date}")
        print(f"Initial capital: {INITIAL_CAPITAL}元")
        print(f"Final NAV: {df['nav'].iloc[-1]:.2f}元")
        print(f"Portfolio return: {port_return:+.2f}%")
        print(f"沪深300 return: {idx_return:+.2f}%")
        print(f"Excess return: {excess:+.2f}%")
        print(f"Max drawdown: {max_dd:.2f}%")
        print(f"{'='*80}")

        return df


def print_daily_report(df: pd.DataFrame, sectors: dict):
    """Print day-by-day portfolio statement."""
    print(f"\n{'日期':<12} {'NAV':>8} {'日收益':>8} {'仓位':>8} {'操作':<45} {'沪深300':>8}")
    print("-" * 100)

    for _, row in df.iterrows():
        date = row["date"]
        nav = row["nav"]
        daily_ret = row["daily_return"]
        invested = row["invested"]
        pct = invested / nav * 100 if nav > 0 else 0
        idx_ret = row["index_return"]

        # Format trades
        trade_str = ""
        if row["is_rebalance"] and row["trades"]:
            parts = []
            for t in row["trades"]:
                a = "B" if t["action"] == "buy" else "S"
                parts.append(f"{a} {t['etf']}({t['amount']}元)")
            trade_str = " | ".join(parts)

        print(f"{date:<12} {nav:>8.0f} {daily_ret:>+7.2f}% {pct:>6.0f}%  {trade_str:<45} {idx_ret:>+7.2f}%")

    print("-" * 100)


def print_trade_summary(df: pd.DataFrame):
    """Print all trades in a readable format."""
    print(f"\n{'='*80}")
    print("成交记录")
    print(f"{'='*80}")
    for _, row in df.iterrows():
        if row["trades"]:
            parts = []
            for t in row["trades"]:
                a = "买入" if t["action"] == "buy" else "卖出"
                parts.append(f"{a} {t['etf']} {t['sector']} {t['amount']}元")
            print(f"  {row['date']}: {'; '.join(parts)}")


def main():
    parser = argparse.ArgumentParser(description="Forward test simulation")
    parser.add_argument("--start", type=str, default="2026-02-24")
    parser.add_argument("--end", type=str, default="2026-04-07")
    parser.add_argument("--capital", type=int, default=INITIAL_CAPITAL)
    parser.add_argument("--rebalance-freq", type=int, default=REBALANCE_FREQ)
    args = parser.parse_args()

    print(f"Loading data...")
    sectors, index_data = load_data()
    print(f"Loaded {len(sectors)} sectors, {len(index_data)} index days")

    engine = ForwardTestEngine(capital=args.capital)
    df = engine.run(sectors, index_data, args.start, args.end)

    print_daily_report(df, sectors)
    print_trade_summary(df)

    # Save results
    from config.settings import RESULTS_DIR
    out_path = RESULTS_DIR / f"forward_{args.start}_{args.end}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
