"""Tests for performance metrics."""

import pandas as pd
from src.backtest.metrics import compute_metrics


def _make_result(daily_rets: list[float], index_rets: list[float], trades: list | None = None):
    from src.backtest.engine import BacktestResult, StrategyConfig
    df = pd.DataFrame({
        "port_return": daily_rets,
        "index_return": index_rets,
    })
    df["port_cumulative"] = (1 + df["port_return"]).cumprod()
    df["index_cumulative"] = (1 + df["index_return"]).cumprod()
    return BacktestResult(daily_returns=df, trades=trades or [], config=StrategyConfig())


def test_metrics_known_returns():
    """With known return series, verify metrics are correct."""
    daily_rets = [0.01] * 252  # 1% per day for a year
    index_rets = [0.005] * 252
    result = _make_result(daily_rets, index_rets)
    metrics = compute_metrics(result)

    assert abs(metrics.total_return - ((1.01**252) - 1)) < 1.0
    assert metrics.excess_return > 0
    assert metrics.win_rate == 1.0
    assert metrics.trade_count == 0


def test_metrics_negative_returns():
    """Negative returns should yield negative metrics."""
    daily_rets = [-0.01] * 252
    index_rets = [0.001] * 252
    result = _make_result(daily_rets, index_rets)
    metrics = compute_metrics(result)
    assert metrics.total_return < 0
    assert metrics.excess_return < 0


def test_metrics_max_drawdown():
    """Drawdown should reflect peak-to-trough."""
    daily_rets = [0.10, -0.10, 0.05, -0.20, 0.10]
    index_rets = [0.01] * 5
    result = _make_result(daily_rets, index_rets)
    metrics = compute_metrics(result)
    assert metrics.max_drawdown < 0


def test_metrics_empty():
    """Empty returns should yield zeros."""
    result = _make_result([], [])
    metrics = compute_metrics(result)
    assert metrics.total_return == 0
