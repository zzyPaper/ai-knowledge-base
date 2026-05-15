#!/usr/bin/env python3
"""Forward test v11 — V4 Hybrid: V2 stock-picking + V3 risk overlay.

Usage:
  python scripts/forward_test_v11.py --start 20250501 --end 20260501
"""

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.backtest.engine import BacktestEngine
from src.strategy.v4_hybrid import V4HybridStrategy


def main():
    parser = argparse.ArgumentParser(description="V4 Hybrid Sector Rotation")
    parser.add_argument("--start", type=str, default="20250501", help="Start date YYYYMMDD")
    parser.add_argument("--end", type=str, default="20260501", help="End date YYYYMMDD")
    parser.add_argument("--cash", type=float, default=1_000_000, help="Initial cash")
    parser.add_argument("--top-n", type=int, default=5, help="Top N sectors to hold")
    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"V4 Hybrid Sector Rotation")
    print(f"{args.start} → {args.end}")
    print(f"{'='*80}")
    print(f"Stock-picking: V2 seven-factor model (trend + momentum + volume + ...)")
    print(f"Risk overlay:  V3 lifecycle (Buildup/Sustain/Exhaust/Avoid)")
    print(f"Timing:        decoupled — caps position, never full exit")
    print(f"{'='*80}\n")

    strategy = V4HybridStrategy()
    engine = BacktestEngine(
        strategy=strategy,
        initial_cash=args.cash,
        top_n=args.top_n,
        position_per_sector=0.20,
        stop_loss=-0.05,
        take_profit=0.15,
        rebalance_freq="monthly",
        commission_rate=0.0003,
        slippage=0.001,
    )

    result = engine.run(start_date=args.start, end_date=args.end)

    print(f"\n{'='*80}")
    print(f"Result: {result.strategy_name} v{result.strategy_version}")
    print(f"Initial: {result.initial_cash:,.0f} | Final: {result.final_value:,.0f}")
    print(f"Portfolio: {result.total_return:+.2%} | Benchmark: {result.benchmark_return:+.2%} | Excess: {result.excess_return:+.2%}")
    print(f"Trades: {len(result.trades)}")
    print(f"Pass: {'YES' if result.is_passing else 'NO'} (threshold: +10%)")
    print(f"{'='*80}")

    if result.trades:
        print(f"\n{'Date':<12} {'Action':<6} {'Sector':<12} {'Price':>8} {'Shares':>8} {'Reason'}")
        print("-" * 80)
        for t in result.trades[:20]:
            print(f"{t.date:<12} {t.action:<6} {t.sector:<12} {t.price:>8.2f} {t.shares:>8} {t.reason}")
        if len(result.trades) > 20:
            print(f"... and {len(result.trades) - 20} more trades")


if __name__ == "__main__":
    main()
