#!/usr/bin/env python3
"""Backtest entry point: run self-looping backtest and print results.

Usage:
  python scripts/run_backtest.py                                              # real data, all defaults
  python scripts/run_backtest.py --use-synthetic                             # synthetic data
  python scripts/run_backtest.py --ma-period 20 --roc-period 20 --lookback 10
  python scripts/run_backtest.py --start-date 2025-06-01 --end-date 2026-05-01
"""

import argparse
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest.engine import StrategyConfig
from src.backtest.loop import run_self_loop_backtest, split_windows
from config.settings import BACKTEST_RESULTS_DIR

import pandas as pd


def _synthetic_data():
    """Generate realistic synthetic sector data for training."""
    import numpy as np

    np.random.seed(42)
    n_days = 400
    dates = pd.date_range("2025-01-01", periods=n_days, freq="B")[:n_days]

    sectors = {}
    for name, trend, noise, reversal_day in [
        ("半导体", 0.04, 0.05, None),
        ("新能源", 0.03, 0.06, None),
        ("消费电子", 0.02, 0.05, 200),
        ("医药生物", 0.01, 0.05, None),
        ("金融", 0.00, 0.04, None),
        ("房地产", -0.01, 0.06, None),
        ("钢铁", -0.02, 0.05, 250),
        ("煤炭", -0.03, 0.04, None),
    ]:
        prices = [100.0]
        current_trend = trend
        for i in range(1, n_days):
            if reversal_day and i == reversal_day:
                current_trend = -current_trend * 0.5
            ret = current_trend + np.random.normal(0, noise)
            prices.append(prices[-1] * (1 + ret))
        sectors[name] = pd.DataFrame({
            "close": prices,
            "open": [p * (1 - abs(np.random.normal(0, 0.008))) for p in prices],
            "high": [p * (1 + abs(np.random.normal(0, 0.015))) for p in prices],
            "low": [p * (1 - abs(np.random.normal(0, 0.015))) for p in prices],
            "volume": [max(500, int(1000 + np.random.normal(0, 400))) for _ in range(n_days)],
            "amount": [p * max(500, int(1000 + np.random.normal(0, 400))) for p in prices],
            "turnover_rate": [max(0.1, abs(np.random.normal(1, 0.5))) for _ in range(n_days)],
            "pct_chg": [0] + [(prices[i] / prices[i - 1] - 1) * 100 for i in range(1, n_days)],
            "date": dates,
        })

    np.random.seed(1)
    index_rets = [0.0]
    for i in range(1, n_days):
        daily_ret = np.random.normal(0.025 / 252, 0.008)
        index_rets.append(daily_ret)
    index_prices = [100 * (1 + r) for r in index_rets]
    index_data = pd.DataFrame({
        "close": index_prices,
        "pct_chg": [r * 100 for r in index_rets],
        "date": dates,
    })

    return sectors, index_data


def _real_data():
    """Load sector and index pickle data from disk."""
    base_dir = Path(__file__).resolve().parent.parent

    sectors_pkl = base_dir / "data" / "sectors_full.pkl"
    if not sectors_pkl.exists():
        sectors_pkl = base_dir / "data" / "sectors_2023_2025.pkl"
    index_pkl = base_dir / "data" / "index_full.pkl"
    if not index_pkl.exists():
        index_pkl = base_dir / "data" / "index_2023_2025.pkl"

    if not sectors_pkl.exists() or not index_pkl.exists():
        print("[ERROR] No pickle data found. Run data collection first.")
        sys.exit(1)

    sectors = pd.read_pickle(sectors_pkl)
    index_data = pd.read_pickle(index_pkl)

    for df in sectors.values():
        df["date"] = pd.to_datetime(df["date"])
    index_data["date"] = pd.to_datetime(index_data["date"])

    if "pct_chg" not in index_data.columns:
        index_data["pct_chg"] = index_data["close"].pct_change(1).fillna(0) * 100

    min_days = 60
    sectors_clean = {}
    for name, df in sectors.items():
        if len(df) >= min_days:
            sectors_clean[name] = df.sort_values("date").reset_index(drop=True)

    print(f"Loaded {len(sectors_clean)} sectors with {min_days}+ days of data")
    print(f"Index data: {len(index_data)} trading days")
    return sectors_clean, index_data


def main():
    parser = argparse.ArgumentParser(description="Run self-loop backtest")
    parser.add_argument("--use-synthetic", action="store_true", help="Use synthetic data instead of real pickle data")
    parser.add_argument("--max-iterations", type=int, default=10, help="Max iterations per window")
    parser.add_argument("--target-excess", type=float, default=10.0, help="Target excess return (pct)")
    parser.add_argument("--top-n", type=int, default=3, help="Number of sectors to hold in ranging markets")
    parser.add_argument("--top-n-trending", type=int, default=2, help="Number of sectors to hold in trending markets")
    parser.add_argument("--rebalance-freq", type=int, default=5, help="Rebalance frequency (trading days)")
    parser.add_argument("--ma-period", type=int, default=20, help="MA period for relative momentum")
    parser.add_argument("--roc-period", type=int, default=20, help="ROC period for relative momentum")
    parser.add_argument("--lookback", type=int, default=10, help="Absolute momentum filter window")
    parser.add_argument("--start-date", type=str, default=None, help="Restrict backtest start (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, help="Restrict backtest end (YYYY-MM-DD)")
    args = parser.parse_args()

    print(f"Backtest starting at {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Config: top_n={args.top_n}, top_n_trending={args.top_n_trending}, rebalance_freq={args.rebalance_freq}")
    print(f"Signals: ma_period={args.ma_period}, roc_period={args.roc_period}, lookback={args.lookback}")
    print(f"Target: beat 沪深300 by {args.target_excess}%")
    print()

    if args.use_synthetic:
        print("Using synthetic data...")
        sectors_data, index_data = _synthetic_data()
    else:
        print("Loading real data from pickle files...")
        sectors_data, index_data = _real_data()

    if args.start_date and args.end_date:
        # Use the exact date range as a single window
        windows = [(args.start_date, args.end_date)]
        print(f"Single window: {args.start_date} → {args.end_date}")
    else:
        end_dt = datetime.strptime(args.end_date, "%Y-%m-%d") if args.end_date else datetime.now()
        windows = split_windows(end_date=end_dt)
        print(f"Windows: {len(windows)}")
        for ws, we in windows:
            print(f"  {ws} → {we}")
    print()

    config = StrategyConfig(
        top_n=args.top_n,
        top_n_trending=args.top_n_trending,
        rebalance_freq=args.rebalance_freq,
        ma_period=args.ma_period,
        roc_period=args.roc_period,
        lookback=args.lookback,
    )

    results = run_self_loop_backtest(
        sectors_data=sectors_data,
        index_data=index_data,
        windows=windows,
        initial_config=config,
        max_iterations=args.max_iterations,
        target_excess=args.target_excess,
    )

    print()
    print(f"{'Window':<30} {'Result':<8} {'Iter':<6} {'Port Ret':<10} {'Idx Ret':<10} {'Excess':<8} {'Drawdown':<10}")
    print("-" * 90)
    for r in results:
        passed = "PASS" if r.passed else "FAIL"
        print(
            f"{r.window_label:<30} {passed:<8} {r.iterations:<6} "
            f"{r.metrics.total_return:>+8.2%} {r.metrics.index_return:>+8.2%} "
            f"{r.metrics.excess_return:>+6.1f}% {r.metrics.max_drawdown:>+8.2%}"
        )

    passed_windows = sum(1 for r in results if r.passed)
    print(f"\nPassed: {passed_windows}/{len(results)} windows")
    print(f"Done. Results directory: {BACKTEST_RESULTS_DIR}")


if __name__ == "__main__":
    main()
