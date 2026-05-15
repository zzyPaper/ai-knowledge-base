"""Parameter sweep for failed backtest windows on A-share sector rotation."""

import sys
import pickle
from itertools import product

import pandas as pd

sys.path.insert(0, "/Users/zhenzhiyuan/AI知识库/stock-analyzer")
from src.backtest.engine import BacktestEngine, StrategyConfig

# ── Load data ─────────────────────────────────────────────────────────────
print("Loading data...")
with open("/Users/zhenzhiyuan/AI知识库/stock-analyzer/data/sectors_full.pkl", "rb") as f:
    sectors_data = pickle.load(f)
with open("/Users/zhenzhiyuan/AI知识库/stock-analyzer/data/index_full.pkl", "rb") as f:
    idx = pickle.load(f)
idx["pct_chg"] = idx["close"].pct_change(1).fillna(0) * 100
print(f"Loaded {len(sectors_data)} sectors, index has {len(idx)} rows.\n")

# ── Windows to sweep ──────────────────────────────────────────────────────
windows = {
    "w4_low_dispersion": ("2025-09-13", "2025-11-12"),
    "w5_high_dispersion": ("2025-07-15", "2025-09-13"),
}

# ── Parameter grid ────────────────────────────────────────────────────────
top_n_values = [2, 3, 4, 5]
rebalance_freq_values = [3, 5, 10]
ma_period_values = [5, 10, 20]
trending_weights_values = [
    (0.50, 0.25, 0.25),  # default
    (0.60, 0.20, 0.20),  # momentum-heavy
    (0.70, 0.15, 0.15),  # more momentum-heavy
    (0.40, 0.30, 0.30),  # balanced
]

total_combos = (
    len(top_n_values)
    * len(rebalance_freq_values)
    * len(ma_period_values)
    * len(trending_weights_values)
)

# ── Baseline (original default config) for comparison ─────────────────────
def run_baseline(window_sectors, window_index, win_start, win_end, label):
    config = StrategyConfig()
    engine = BacktestEngine(config)
    result = engine.run(window_sectors, window_index, win_start, win_end)
    m = result.metrics
    print(
        f"  Baseline: port={m.total_return:.1%}  idx={m.index_return:.1%}  "
        f"excess={m.excess_return:.1f}%  "
        f"dd={m.max_drawdown:.1%}  sharpe={m.sharpe_ratio:.2f}  trades={m.trade_count}"
    )

# ── Sweep ─────────────────────────────────────────────────────────────────
for win_name, (win_start, win_end) in windows.items():
    print(f"\n{'=' * 90}")
    print(f"WINDOW: {win_name}  ({win_start} → {win_end})")
    print(f"{'=' * 90}")

    # Subset data for this window
    window_sectors = {}
    for name, df in sectors_data.items():
        subset = df[(df["date"] >= win_start) & (df["date"] <= win_end)].copy()
        if len(subset) > 5:
            window_sectors[name] = subset
    window_index = idx[(idx["date"] >= win_start) & (idx["date"] <= win_end)].copy()
    print(f"Sectors: {len(window_sectors)}, Trading days: {len(window_index)}")
    run_baseline(window_sectors, window_index, win_start, win_end, win_name)

    # Sweep
    print(f"\nSweeping {total_combos} parameter combinations...")
    rows = []
    for top_n, rebal_freq, ma_period, trend_w in product(
        top_n_values, rebalance_freq_values, ma_period_values, trending_weights_values
    ):
        config = StrategyConfig(
            top_n=top_n,
            rebalance_freq=rebal_freq,
            ma_period=ma_period,
            signal_weights_trending=trend_w,
        )
        engine = BacktestEngine(config)
        result = engine.run(window_sectors, window_index, win_start, win_end)
        m = result.metrics
        rows.append(
            {
                "top_n": top_n,
                "rebal_freq": rebal_freq,
                "ma_period": ma_period,
                "w_mom": trend_w[0],
                "w_vp": trend_w[1],
                "w_cong": trend_w[2],
                "excess": round(m.excess_return, 1),
                "port_ret": round(m.total_return * 100, 1),
                "idx_ret": round(m.index_return * 100, 1),
                "sharpe": round(m.sharpe_ratio, 2),
                "dd": round(m.max_drawdown * 100, 1),
                "trades": m.trade_count,
            }
        )

    df = pd.DataFrame(rows)

    # ── Results ────────────────────────────────────────────────────────
    winners = df[df["excess"] >= 10.0].sort_values("excess", ascending=False)
    print(f"\nTotal combos tried: {len(df)}")
    print(f"Configs achieving >= 10% excess return: {len(winners)}")

    if len(winners) > 0:
        print(f"\n--- Top 25 configs (by excess return) ---")
        cols = ["top_n", "rebal_freq", "ma_period", "w_mom", "w_vp", "w_cong",
                "excess", "port_ret", "idx_ret", "sharpe", "dd", "trades"]
        print(winners.head(25)[cols].to_string(index=False))
    else:
        print("\n--- NO config achieved 10%+ excess ---")
        print("\nTop 15 closest configs:")
        cols = ["top_n", "rebal_freq", "ma_period", "w_mom", "w_vp", "w_cong",
                "excess", "port_ret", "idx_ret", "sharpe", "dd", "trades"]
        print(df.sort_values("excess", ascending=False).head(15)[cols].to_string(index=False))

    # ── Summary stats ──────────────────────────────────────────────────
    print(f"\n--- Sweep summary statistics ---")
    print(f"  excess range: {df['excess'].min():.1f}% to {df['excess'].max():.1f}%")
    print(f"  excess median: {df['excess'].median():.1f}%")
    print(f"  excess mean:   {df['excess'].mean():.1f}%")

    best_idx = df["excess"].idxmax()
    best = df.loc[best_idx]
    print(
        f"  BEST: top_n={best['top_n']} rebal_freq={best['rebal_freq']} "
        f"ma={best['ma_period']} w=({best['w_mom']},{best['w_vp']},{best['w_cong']}) "
        f"→ excess={best['excess']:.1f}%  port={best['port_ret']:.1f}%  "
        f"sharpe={best['sharpe']:.2f}  dd={best['dd']:.1f}%"
    )

    # ── Best per top_n ─────────────────────────────────────────────────
    print(f"\n--- Best config per top_n value ---")
    for tn in top_n_values:
        sub = df[df["top_n"] == tn].sort_values("excess", ascending=False)
        if not sub.empty:
            r = sub.iloc[0]
            print(
                f"  top_n={tn}: excess={r['excess']:.1f}%  "
                f"rebal={r['rebal_freq']}  ma={r['ma_period']}  "
                f"w=({r['w_mom']},{r['w_vp']},{r['w_cong']})  "
                f"sharpe={r['sharpe']:.2f}  dd={r['dd']:.1f}%"
            )
