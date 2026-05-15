#!/usr/bin/env python3
"""Professional Sector Rotation Strategy Runner.

Unified entry point for forward testing the professional strategy.

Architecture:
  StrategyConfig → RegimeDetector → SignalPipeline → RiskManager
  → PortfolioBuilder → ForwardExecutor → Results

Usage:
  python scripts/run_strategy.py --start 2026-02-24 --end 2026-03-31
  python scripts/run_strategy.py --start 2026-03-31 --end 2026-04-30
  python scripts/run_strategy.py --start 2026-02-24 --end 2026-04-30 --save
"""

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import pandas as pd

from src.engine.config import (
    StrategyConfig, MarketTimingConfig, RegimeConfig,
    SignalConfig, RiskConfig, PortfolioConfig,
)
from src.engine.executor import ForwardExecutor, DailyRecord


def load_data():
    """Load cached sector and index data."""
    sectors = pd.read_pickle(BASE_DIR / "data" / "sectors_full.pkl")
    index_data = pd.read_pickle(BASE_DIR / "data" / "index_full.pkl")

    for df in sectors.values():
        df["date"] = pd.to_datetime(df["date"])
    index_data["date"] = pd.to_datetime(index_data["date"])

    for k, df in sectors.items():
        if "pct_chg" not in df.columns:
            df["pct_chg"] = df["close"].pct_change(1).fillna(0) * 100
    if "pct_chg" not in index_data.columns:
        index_data["pct_chg"] = index_data["close"].pct_change(1).fillna(0) * 100

    return sectors, index_data


def print_daily_report(df: pd.DataFrame, max_trades: int = 50):
    """Print day-by-day portfolio statement."""
    print(f"\n{'日期':<12} {'NAV':>8} {'日收益':>8} {'仓位':>7} {'市场':>6} {'状态':>10} {'操作':<{max_trades}} {'沪深300':>8}")
    print("-" * (max_trades + 73))

    for _, row in df.iterrows():
        date = row["date"]
        nav = row["nav"]
        invested = row["invested"]
        pct = invested / nav * 100 if nav > 0 else 0
        idx_ret = row["index_return_pct"]
        in_mkt = "IN" if row.get("in_market") else "OUT"
        regime = row.get("regime", "?")[:10]

        trade_str = ""
        if row.get("trades"):
            parts = []
            for t in row["trades"]:
                a = "B" if t["action"] == "buy" else "S"
                r = t.get("reason", "")[:3]
                parts.append(f"{a}({r}){t['etf']}({t['amount']}元)")
            trade_str = " | ".join(parts)
        if len(trade_str) > max_trades:
            trade_str = trade_str[:max_trades - 3] + "..."

        print(f"{date:<12} {nav:>8.0f} {row['daily_return_pct']:>+7.2f}% {pct:>6.0f}% "
              f" {in_mkt:<6} {regime:<10} {trade_str:<{max_trades}} {idx_ret:>+7.2f}%")
    print("-" * (max_trades + 73))


def main():
    parser = argparse.ArgumentParser(
        description="Professional Sector Rotation Forward Test")
    parser.add_argument("--start", type=str, default="2026-02-24")
    parser.add_argument("--end", type=str, default="2026-03-31")
    parser.add_argument("--capital", type=int, default=5000)
    parser.add_argument("--save", action="store_true", help="Save results to CSV")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print daily report")
    args = parser.parse_args()

    # Build configuration
    config = StrategyConfig(
        timing=MarketTimingConfig(
            entry_lookback=20, entry_threshold=1.0,  # was 0.5 — stronger entry
            exit_lookback=10, exit_threshold=-0.5,
            exit_override_20d_threshold=3.0,  # was 2.0 — stronger trend hold
            reversal_single_day=2.0, reversal_2d_cumulative=2.5,
            reversal_enabled=True,
        ),
        regime=RegimeConfig(
            ma_short=10, ma_medium=20, ma_long=60,
            bull_deviation=2.0, bear_deviation=-2.0,
            slope_rising=0.3, slope_falling=-0.3,
        ),
        signals=SignalConfig(
            momentum_windows=(20, 60, 120),
            momentum_weights=(0.50, 0.30, 0.20),
            crowding_history=500, crowding_short=40,
            data_lookback_days=365, min_data_points=20,
        ),
        risk=RiskConfig(
            target_annual_vol=0.12, vol_lookback=63,
            vol_scale_min=0.25, vol_scale_max=1.20,
            position_cap_min=10, position_cap_max=90,
            crash_stop_threshold=-5.0, max_dd_threshold=-10.0,
        ),
        portfolio=PortfolioConfig(
            base_position_pct=80, min_position_pct=10, max_position_pct=90,
            min_sectors=3, max_sectors=5,
            sector_concentration_cap=0.40,
            rebalance_freq=5, initial_capital=float(args.capital),
        ),
        name="professional-sector-rotation", version="1.0.0",
    )

    print(f"Loading data...")
    sectors, index_data = load_data()
    print(f"Loaded {len(sectors)} sectors, {len(index_data)} index days")

    print(f"\n>>> Professional Sector Rotation v{config.version} <<<")
    print(f"  Regime: MA{config.regime.ma_short}/{config.regime.ma_medium}/{config.regime.ma_long}")
    print(f"  Trend: {config.signals.momentum_windows} multi-timeframe")
    print(f"  Crowding: turnover/vol/Beta {config.signals.crowding_short}d %ile")
    print(f"  Risk: {config.risk.target_annual_vol*100:.0f}% vol target + neg-semi vol")
    print(f"  Timing: {config.timing.entry_lookback}d entry + {config.timing.exit_lookback}d exit")
    print(f"  Portfolio: {config.portfolio.min_sectors}-{config.portfolio.max_sectors} sectors, "
          f"{config.portfolio.rebalance_freq}d rebalance")

    executor = ForwardExecutor(config=config)
    df = executor.run(sectors, index_data, args.start, args.end)

    if args.verbose:
        print_daily_report(df)

    if args.save:
        from config.settings import RESULTS_DIR
        out_path = RESULTS_DIR / f"strategy_{args.start}_{args.end}.csv"
        # Convert trades to string for CSV
        df_save = df.copy()
        df_save["trades"] = df_save["trades"].apply(str)
        df_save["positions"] = df_save["positions"].apply(str)
        df_save.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
