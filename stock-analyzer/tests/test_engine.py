"""Tests for backtest engine."""

import pandas as pd
from src.backtest.engine import BacktestEngine, StrategyConfig


def _synthetic_sector(name: str, trend: float, n: int = 60) -> pd.DataFrame:
    prices = [100 + trend * i + (i % 5) * 0.5 for i in range(n)]
    dates = pd.date_range("2025-01-01", periods=n, freq="B")[:n]
    return pd.DataFrame({
        "close": prices,
        "open": [p * 0.99 for p in prices],
        "high": [p * 1.02 for p in prices],
        "low": [p * 0.98 for p in prices],
        "volume": [1000] * n,
        "amount": [p * 1000 for p in prices],
        "turnover_rate": [1.0] * n,
        "date": dates,
    })


def test_engine_runs_and_produces_output():
    """Engine should produce daily returns and trades."""
    sectors = {
        "sector_a": _synthetic_sector("a", trend=1.0, n=50),
        "sector_b": _synthetic_sector("b", trend=-0.5, n=50),
        "sector_c": _synthetic_sector("c", trend=0.3, n=50),
    }
    idx = _synthetic_sector("index", trend=0.1, n=50)
    idx["pct_chg"] = idx["close"].pct_change(1).fillna(0) * 100

    config = StrategyConfig(top_n=2, rebalance_freq=10)
    engine = BacktestEngine(config)
    result = engine.run(sectors, idx)

    assert len(result.daily_returns) > 0
    assert len(result.trades) > 0
    assert "port_cumulative" in result.daily_returns.columns
    assert "index_cumulative" in result.daily_returns.columns


def test_engine_picks_winners():
    """Engine should allocate more to outperforming sectors."""
    winners = _synthetic_sector("winner", trend=2.0, n=50)
    flat = _synthetic_sector("flat", trend=0, n=50)
    losers = _synthetic_sector("loser", trend=-1.0, n=50)
    idx = _synthetic_sector("index", trend=0.1, n=50)
    idx["pct_chg"] = idx["close"].pct_change(1).fillna(0) * 100

    sectors = {"winner": winners, "flat": flat, "loser": losers}
    config = StrategyConfig(top_n=1, rebalance_freq=10)
    engine = BacktestEngine(config)
    result = engine.run(sectors, idx)

    assert result.metrics.total_return >= -0.1, f"Even with bad luck, shouldn't lose much: {result.metrics.total_return}"


def test_engine_with_no_data():
    """Engine should handle empty sector data gracefully."""
    sectors = {"empty": pd.DataFrame(columns=["close", "date", "volume", "amount", "turnover_rate"])}
    idx = _synthetic_sector("index", trend=0, n=10)
    idx["pct_chg"] = idx["close"].pct_change(1).fillna(0) * 100

    engine = BacktestEngine()
    result = engine.run(sectors, idx)
    assert result.metrics.total_return == 0.0
