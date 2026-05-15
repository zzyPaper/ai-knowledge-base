"""End-to-end test: full backtest pipeline with synthetic data."""

import pandas as pd
import numpy as np
from src.backtest.engine import StrategyConfig
from src.backtest.loop import run_self_loop_backtest, split_windows


def _synthetic_sectors(n_sectors=8, n_days=400) -> tuple[dict, pd.DataFrame]:
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=n_days, freq="B")[:n_days]

    sectors = {}
    trends = [0.15, 0.10, 0.08, 0.05, 0.03, -0.02, -0.05, -0.03][:n_sectors]
    noises = [0.03, 0.04, 0.035, 0.03, 0.02, 0.04, 0.03, 0.025][:n_sectors]
    names = [f"sector_{i}" for i in range(n_sectors)]

    for name, trend, noise in zip(names, trends, noises):
        prices = [100.0]
        for i in range(1, n_days):
            ret = trend + np.random.normal(0, noise)
            prices.append(prices[-1] * (1 + ret))
        sectors[name] = pd.DataFrame({
            "close": prices,
            "open": [p * 0.99 for p in prices],
            "high": [p * 1.02 for p in prices],
            "low": [p * 0.98 for p in prices],
            "volume": [1000] * n_days,
            "amount": [p * 1000 for p in prices],
            "turnover_rate": [1.0] * n_days,
            "pct_chg": [0] + [(prices[i] / prices[i - 1] - 1) * 100 for i in range(1, n_days)],
            "date": dates,
        })

    idx = pd.DataFrame({
        "close": [100 * (1 + 0.05 / 252) ** i for i in range(n_days)],
        "pct_chg": [0.05 / 252 * 100] * n_days,
        "date": dates,
    })
    return sectors, idx


def test_end_to_end_backtest():
    """Full pipeline: synthetic data → 2 windows → verify results."""
    sectors, idx = _synthetic_sectors(n_sectors=6, n_days=200)
    windows = split_windows(end_date=pd.Timestamp("2025-07-15"), num_windows=3)

    config = StrategyConfig(top_n=2, rebalance_freq=10)
    results = run_self_loop_backtest(
        sectors_data=sectors,
        index_data=idx,
        windows=windows,
        initial_config=config,
        max_iterations=5,
        target_excess=5.0,
    )

    assert len(results) == 3, f"Expected 3 window results, got {len(results)}"
    for r in results:
        assert isinstance(r.metrics.total_return, float)
        assert r.iterations > 0, "At least 1 iteration per window"

    assert results[0].history is not None and len(results[0].history) > 0
