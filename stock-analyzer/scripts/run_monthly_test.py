#!/usr/bin/env python3
"""Monthly batch test: run strategy on each month of the past year.

Produces per-month comparison vs 沪深300 with daily granularity.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import pandas as pd
import numpy as np

from src.engine.config import (
    StrategyConfig, MarketTimingConfig, RegimeConfig,
    SignalConfig, RiskConfig, PortfolioConfig,
)
from src.engine.executor import ForwardExecutor

# 12 months: 2025-05 through 2026-04
MONTHS = [
    ("2025-05-06", "2025-05-30", "2025-05 May"),
    ("2025-06-02", "2025-06-30", "2025-06 Jun"),
    ("2025-07-01", "2025-07-31", "2025-07 Jul"),
    ("2025-08-01", "2025-08-29", "2025-08 Aug"),
    ("2025-09-01", "2025-09-30", "2025-09 Sep"),
    ("2025-10-06", "2025-10-31", "2025-10 Oct"),
    ("2025-11-03", "2025-11-28", "2025-11 Nov"),
    ("2025-12-01", "2025-12-31", "2025-12 Dec"),
    ("2026-01-02", "2026-01-30", "2026-01 Jan"),
    ("2026-02-02", "2026-02-27", "2026-02 Feb"),
    ("2026-03-02", "2026-03-31", "2026-03 Mar"),
    ("2026-04-01", "2026-04-30", "2026-04 Apr"),
]


def get_default_config(capital: int = 5000) -> StrategyConfig:
    return StrategyConfig(
        timing=MarketTimingConfig(
            entry_lookback=20, entry_threshold=1.0,
            exit_lookback=10, exit_threshold=-0.5,
            exit_override_20d_threshold=3.0,
            reversal_single_day=2.0, reversal_2d_cumulative=2.5,
            reversal_enabled=True, reversal_grace_days=10,
        ),
        regime=RegimeConfig(
            ma_short=10, ma_medium=20, ma_long=60,
            bull_deviation=2.0, bear_deviation=-2.0,
            slope_rising=0.3, slope_falling=-0.3,
            hysteresis_days=2,
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
            rebalance_freq=5, initial_capital=float(capital),
        ),
        name="professional-sector-rotation", version="1.0.0",
    )


def load_data():
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


def main():
    print("Loading data...")
    sectors, index_data = load_data()

    config = get_default_config()

    monthly_results = []
    all_daily_rows = []

    for start, end, label in MONTHS:
        executor = ForwardExecutor(config=config)
        df = executor.run(sectors, index_data, start, end)

        if df.empty:
            monthly_results.append({"month": label, "port_ret": 0, "idx_ret": 0,
                                    "excess": 0, "max_dd": 0, "days_in": 0,
                                    "days_total": 0, "pass": False})
            continue

        start_nav = config.portfolio.initial_capital
        idx_start = index_data[index_data["date"] == pd.Timestamp(start)]
        idx_end = index_data[index_data["date"] == pd.Timestamp(end)]
        idx_return = 0.0
        if not idx_start.empty and not idx_end.empty:
            idx_return = (float(idx_end["close"].iloc[0]) /
                          float(idx_start["close"].iloc[0]) - 1) * 100
        port_return = (df["nav"].iloc[-1] / start_nav - 1) * 100
        excess = port_return - idx_return

        cumulative = df["nav"].values / start_nav
        peak_series = pd.Series(cumulative).cummax().values
        drawdowns = (cumulative - peak_series) / peak_series
        max_dd = float(drawdowns.min()) * 100 if len(drawdowns) > 0 else 0

        days_in = (df["in_market"] == True).sum()

        # Regime distribution
        regime_dist = df["regime"].value_counts().to_dict() if "regime" in df.columns else {}

        monthly_results.append({
            "month": label,
            "port_ret": round(port_return, 2),
            "idx_ret": round(idx_return, 2),
            "excess": round(excess, 2),
            "max_dd": round(max_dd, 2),
            "days_in": days_in,
            "days_total": len(df),
            "dominant_regime": max(regime_dist, key=regime_dist.get) if regime_dist else "?",
            "pass": excess >= 5.0,
            "pass_soft": excess >= 0.0,
        })

        # Collect daily rows
        for _, row in df.iterrows():
            all_daily_rows.append({
                "month": label,
                "date": row["date"],
                "nav": row["nav"],
                "daily_ret": row["daily_return_pct"],
                "idx_ret": row["index_return_pct"],
                "in_market": row.get("in_market", False),
                "regime": row.get("regime", "?"),
                "position_cap": row.get("position_cap", 0),
                "invested_pct": row["invested"] / row["nav"] * 100 if row["nav"] > 0 else 0,
            })

    # ---- Print Monthly Summary ----
    print(f"\n{'='*100}")
    print(f"Past Year Monthly Strategy Test — {MONTHS[0][0]} → {MONTHS[-1][1]}")
    print(f"Strategy: {config.name} v{config.version}")
    print(f"{'='*100}")
    print(f"{'Month':<18} {'Portfolio':>9} {'沪深300':>8} {'Excess':>8} {'Max DD':>7} {'In Mkt':>8} {'Dominant':>12} {'Result':>8}")
    print(f"{'-'*100}")

    n_pass = 0
    n_soft = 0
    total_excess = 0
    for r in monthly_results:
        status = "PASS" if r["pass"] else ("OK" if r["pass_soft"] else "FAIL")
        if r["pass"]: n_pass += 1
        if r["pass_soft"]: n_soft += 1
        total_excess += r["excess"]
        print(f"{r['month']:<18} {r['port_ret']:>+8.2f}% {r['idx_ret']:>+7.2f}% "
              f"{r['excess']:>+7.2f}% {r['max_dd']:>6.2f}% "
              f"{r['days_in']:>3}/{r['days_total']:<3} {r['dominant_regime']:>12} {status:>8}")

    print(f"{'-'*100}")
    print(f"{'AVERAGE':<18} {'':>9} {'':>8} {total_excess/len(monthly_results):>+7.2f}%")
    print(f"Hard pass (excess >= +5%): {n_pass}/{len(monthly_results)}")
    print(f"Soft pass (excess >= 0%):  {n_soft}/{len(monthly_results)}")
    print(f"{'='*100}")

    # ---- Monthly daily detail (compact) ----
    print(f"\n{'='*100}")
    print(f"Daily Return Tracking (Portfolio vs 沪深300)")
    print(f"{'='*100}")

    daily_df = pd.DataFrame(all_daily_rows)
    for month_label in [m[2] for m in MONTHS]:
        mdf = daily_df[daily_df["month"] == month_label]
        if mdf.empty:
            continue
        print(f"\n--- {month_label} ---")
        print(f"{'Date':<12} {'Prt Ret':>7} {'Idx Ret':>7} {'Diff':>7} {'In?':>4} {'Regime':>12} {'Pos%':>6}")
        for _, row in mdf.iterrows():
            diff = row["daily_ret"] - row["idx_ret"]
            in_mkt = "IN" if row["in_market"] else "OUT"
            print(f"{row['date']:<12} {row['daily_ret']:>+6.2f}% {row['idx_ret']:>+6.2f}% "
                  f"{diff:>+6.2f}% {in_mkt:>4} {row['regime']:>12} {row['invested_pct']:>5.0f}%")

    # Save results
    out_dir = BASE_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    pd.DataFrame(monthly_results).to_csv(out_dir / "monthly_summary.csv", index=False, encoding="utf-8-sig")
    daily_df.to_csv(out_dir / "monthly_daily.csv", index=False, encoding="utf-8-sig")
    print(f"\n\nResults saved to {out_dir}/monthly_summary.csv and monthly_daily.csv")


if __name__ == "__main__":
    main()
