#!/usr/bin/env python3
"""Multi-period batch test runner for professional strategy.

Design: 7 windows covering past year with distinct market regimes:
  1. 2025-05-15 → 2025-06-30  — mild uptrend (spring)
  2. 2025-07-01 → 2025-09-01  — strong bull (summer)
  3. 2025-09-01 → 2025-10-31  — ranging/choppy (fall)
  4. 2025-11-01 → 2025-12-31  — year-end flat
  5. 2026-01-01 → 2026-02-28  — new year sideways
  6. 2026-02-24 → 2026-04-30  — correction + V-recovery
  7. 2025-05-15 → 2026-04-30  — full year composite

Success criteria: Excess >= +5% on >= 4/7 periods, excess >= 0% on all.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import pandas as pd

from src.engine.config import (
    StrategyConfig, MarketTimingConfig, RegimeConfig,
    SignalConfig, RiskConfig, PortfolioConfig,
)
from src.engine.executor import ForwardExecutor


TEST_WINDOWS = [
    ("2025-05-15", "2025-06-30", "Spring mild up"),
    ("2025-07-01", "2025-09-01", "Summer bull"),
    ("2025-09-01", "2025-10-31", "Fall ranging"),
    ("2025-11-01", "2025-12-31", "Year-end flat"),
    ("2026-01-01", "2026-02-28", "New year sideways"),
    ("2026-02-24", "2026-04-30", "Correction + V-recovery"),
    ("2025-05-15", "2026-04-30", "FULL YEAR"),
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


def run_one(config: StrategyConfig, sectors: dict, index: pd.DataFrame,
            start: str, end: str) -> dict:
    executor = ForwardExecutor(config=config)
    df = executor.run(sectors, index, start, end)
    if df.empty:
        return {"start": start, "end": end, "port_ret": 0, "idx_ret": 0,
                "excess": 0, "max_dd": 0, "days_in": 0, "days_total": 0,
                "final_nav": 0, "pass": False}
    start_nav = config.portfolio.initial_capital
    idx_start = index[index["date"] == pd.Timestamp(start)]
    idx_end = index[index["date"] == pd.Timestamp(end)]
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
    regime_dist = df["regime"].value_counts().to_dict() if "regime" in df.columns else {}
    dominant_regime = max(regime_dist, key=regime_dist.get) if regime_dist else "?"

    return {
        "start": start, "end": end,
        "port_ret": round(port_return, 2),
        "idx_ret": round(idx_return, 2),
        "excess": round(excess, 2),
        "max_dd": round(max_dd, 2),
        "days_in": days_in,
        "days_total": len(df),
        "final_nav": round(df["nav"].iloc[-1], 2),
        "dominant_regime": dominant_regime,
        "pass": excess >= 5.0,
        "pass_soft": excess >= 0.0,
    }


def main():
    print("Loading data...")
    sectors, index_data = load_data()
    print(f"  {len(sectors)} sectors, {len(index_data)} index days\n")

    config = get_default_config()
    results = []

    for start, end, label in TEST_WINDOWS:
        print(f"[{label}] {start} → {end} ...", end=" ", flush=True)
        try:
            r = run_one(config, sectors, index_data, start, end)
            r["label"] = label
            results.append(r)
            status = "PASS" if r["pass"] else ("OK" if r["pass_soft"] else "FAIL")
            print(f" port={r['port_ret']:+.2f}% idx={r['idx_ret']:+.2f}% "
                  f"excess={r['excess']:+.2f}% DD={r['max_dd']:.2f}% "
                  f"in={r['days_in']}/{r['days_total']} [{status}]")
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"label": label, "start": start, "end": end,
                           "port_ret": 0, "idx_ret": 0, "excess": 0,
                           "max_dd": 0, "days_in": 0, "days_total": 0,
                           "pass": False, "pass_soft": False, "error": str(e)})

    # Summary
    print(f"\n{'='*90}")
    print(f"{'Window':<28} {'Portfolio':>9} {'沪深300':>8} {'Excess':>8} {'Max DD':>7} {'In Mkt':>8} {'Result':>8}")
    print(f"{'-'*90}")
    n_pass = 0
    n_soft = 0
    for r in results:
        label = r.get("label", "?")
        status = "PASS" if r["pass"] else ("OK" if r["pass_soft"] else "FAIL")
        if r["pass"]:
            n_pass += 1
        if r["pass_soft"]:
            n_soft += 1
        print(f"{label:<28} {r['port_ret']:>+8.2f}% {r['idx_ret']:>+7.2f}% "
              f"{r['excess']:>+7.2f}% {r['max_dd']:>6.2f}% "
              f"{r['days_in']:>3}/{r['days_total']:<3} {status:>8}")
    print(f"{'-'*90}")
    print(f"Hard pass (excess >= +5%): {n_pass}/{len(results)}")
    print(f"Soft pass (excess >= 0%):  {n_soft}/{len(results)}")
    print(f"{'='*90}")

    # Save results
    df_out = pd.DataFrame(results)
    out_path = BASE_DIR / "results" / "batch_test_results.csv"
    out_path.parent.mkdir(exist_ok=True)
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
