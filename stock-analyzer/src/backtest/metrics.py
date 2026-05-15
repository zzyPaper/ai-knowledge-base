"""Performance metrics computation for backtest results."""

from dataclasses import dataclass
import numpy as np


@dataclass
class PerformanceMetrics:
    total_return: float  # portfolio total return (decimal)
    index_return: float  # benchmark total return (decimal)
    excess_return: float  # portfolio - benchmark (percentage points)
    max_drawdown: float  # maximum peak-to-trough drawdown (decimal)
    win_rate: float  # fraction of days with positive portfolio return
    sharpe_ratio: float  # annualized Sharpe (risk-free ~ 0)
    trade_count: int


def compute_metrics(result) -> PerformanceMetrics:
    """Compute comprehensive performance metrics from a BacktestResult."""
    df = result.daily_returns

    if df.empty:
        return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0)

    total_return = float(df["port_cumulative"].iloc[-1] - 1) if not df.empty else 0.0
    index_return = float(df["index_cumulative"].iloc[-1] - 1) if not df.empty else 0.0
    excess_return = (total_return - index_return) * 100  # percentage points

    cumulative = df["port_cumulative"].values
    peak = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - peak) / peak
    max_drawdown = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

    daily_rets = df["port_return"].values
    win_rate = float(np.mean(daily_rets > 0)) if len(daily_rets) > 0 else 0.0

    if len(daily_rets) > 1 and np.std(daily_rets) > 0:
        annual_factor = np.sqrt(252)
        sharpe = float(np.mean(daily_rets) / np.std(daily_rets) * annual_factor)
    else:
        sharpe = 0.0

    return PerformanceMetrics(
        total_return=total_return,
        index_return=index_return,
        excess_return=excess_return,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        sharpe_ratio=sharpe,
        trade_count=len(result.trades),
    )
