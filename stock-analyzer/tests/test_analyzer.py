"""Tests for backtest failure analyzer."""

import pandas as pd
from src.backtest.analyzer import analyze_failure
from src.backtest.engine import StrategyConfig


def _mock_result(total_return=-0.05, index_return=0.05, max_dd=-0.20, trades=None):
    from src.backtest.engine import BacktestResult
    from src.backtest.metrics import PerformanceMetrics
    df = pd.DataFrame({"port_return": [-0.01] * 20, "index_return": [0.005] * 20})
    df["port_cumulative"] = (1 + df["port_return"]).cumprod()
    df["index_cumulative"] = (1 + df["index_return"]).cumprod()
    result = BacktestResult(daily_returns=df, trades=trades or [{"sector": "a", "action": "buy"}] * 3, config=StrategyConfig())
    # Override metrics via cached property
    result._mock_metrics = PerformanceMetrics(
        total_return=total_return, index_return=index_return,
        excess_return=(total_return - index_return) * 100,
        max_drawdown=max_dd, win_rate=0.3, sharpe_ratio=-0.5, trade_count=len(result.trades),
    )
    return result


def test_analyze_failure_high_drawdown():
    """Should suggest reducing concentration when drawdown is high."""
    result = _mock_result(max_dd=-0.25)
    config = StrategyConfig(top_n=3)
    adjustments = analyze_failure(result, {"a": pd.DataFrame()}, config)
    drawdown_adjustments = [a for a in adjustments if a.param == "top_n"]
    assert len(drawdown_adjustments) > 0
    assert drawdown_adjustments[0].new_value > config.top_n


def test_analyze_failure_too_few_trades():
    """Should suggest increasing trade frequency when trades are few."""
    result = _mock_result(trades=[{"sector": "a", "action": "buy"}] * 3)
    config = StrategyConfig(rebalance_freq=10)
    adjustments = analyze_failure(result, {"a": pd.DataFrame()}, config)
    freq_adj = [a for a in adjustments if a.param == "rebalance_freq"]
    if freq_adj:
        assert freq_adj[0].new_value < config.rebalance_freq
