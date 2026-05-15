#!/usr/bin/env python3
"""Daily analysis entry point - run at 2:30 PM on trading days."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.daily.pipeline import run_daily_analysis


def main():
    parser = argparse.ArgumentParser(description="Run daily A-share hot sector analysis")
    parser.add_argument("--date", type=str, default=None, help="Analysis date (YYYY-MM-DD)")
    parser.add_argument("--top-k", type=int, default=20, help="Number of sectors to analyze")
    args = parser.parse_args()

    print(f"Running daily analysis for {args.date or 'today'}...")
    report = run_daily_analysis(date_str=args.date, top_k=args.top_k)
    print(report)


if __name__ == "__main__":
    main()
