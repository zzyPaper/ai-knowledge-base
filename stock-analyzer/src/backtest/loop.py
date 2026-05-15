"""Self-loop backtest orchestrator: iterative optimization across time windows."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Dict

import pandas as pd
from src.backtest.engine import BacktestEngine, StrategyConfig
from src.backtest.metrics import PerformanceMetrics
from src.backtest.analyzer import analyze_failure, apply_adjustments, Adjustment


@dataclass
class WindowResult:
    window_label: str
    config: StrategyConfig
    metrics: PerformanceMetrics
    passed: bool
    iterations: int
    history: Optional[List[dict]] = None


def split_windows(end_date: datetime = None, window_months: int = 3, num_windows: int = 4) -> List[Tuple[str, str]]:
    """Split past N months into windows with 60-day warmup prepended.

    Returns list of (window_start, window_end) date strings.
    """
    if end_date is None:
        end_date = datetime.now()
    windows = []
    for i in range(num_windows):
        w_end = end_date - timedelta(days=window_months * 30 * i)
        w_start = w_end - timedelta(days=window_months * 30)
        windows.append((
            w_start.strftime("%Y-%m-%d"),
            w_end.strftime("%Y-%m-%d"),
        ))
    return windows


def run_self_loop_backtest(
    sectors_data: Dict[str, pd.DataFrame],
    index_data: pd.DataFrame,
    windows: Optional[List[Tuple[str, str]]] = None,
    initial_config: Optional[StrategyConfig] = None,
    max_iterations: int = 10,
    target_excess: float = 10.0,
) -> List[WindowResult]:
    """Run self-looping backtest across time windows.

    For each window:
      - Run backtest with current config
      - If excess_return >= target_excess: mark pass, move on
      - Else: analyze_failure -> adjust config -> retry
    """
    if windows is None:
        windows = split_windows()

    config = initial_config or StrategyConfig()
    results = []

    for win_start, win_end in windows:
        warmup_start = (pd.Timestamp(win_start) - timedelta(days=60)).strftime("%Y-%m-%d")

        window_sectors = {}
        for name, df in sectors_data.items():
            subset = df[(df["date"] >= warmup_start) & (df["date"] <= win_end)].copy()
            if len(subset) > 0:
                window_sectors[name] = subset

        window_index = index_data[
            (index_data["date"] >= warmup_start) & (index_data["date"] <= win_end)
        ].copy()

        if len(window_sectors) == 0 or window_index.empty:
            results.append(WindowResult(
                window_label=f"{win_start}_{win_end}",
                config=config, metrics=PerformanceMetrics(0, 0, 0, 0, 0, 0, 0),
                passed=False, iterations=0,
            ))
            continue

        iteration = 0
        passed = False
        best_metrics = None
        best_config = config
        history = []
        stale_count = 0
        prev_excess = None

        while iteration < max_iterations:
            engine = BacktestEngine(config)
            result = engine.run(window_sectors, window_index, start_date=win_start, end_date=win_end)

            if best_metrics is None or result.metrics.excess_return > best_metrics.excess_return:
                best_metrics = result.metrics
                best_config = StrategyConfig(
                    top_n=config.top_n,
                    top_n_trending=config.top_n_trending,
                    rebalance_freq=config.rebalance_freq,
                    ma_period=config.ma_period,
                    roc_period=config.roc_period,
                    lookback=config.lookback,
                    trending_weights=config.trending_weights,
                    ranging_weights=config.ranging_weights,
                    regime_adaptive=config.regime_adaptive,
                )

            history.append({
                "iteration": iteration,
                "excess_return": result.metrics.excess_return,
                "total_return": result.metrics.total_return,
                "max_drawdown": result.metrics.max_drawdown,
                "config_params": {
                    "top_n": config.top_n,
                    "ma_period": config.ma_period,
                    "rebalance_freq": config.rebalance_freq,
                },
            })

            if result.metrics.excess_return >= target_excess:
                passed = True
                break

            # Convergence: stop if excess return hasn't improved after 3 adjustments
            if prev_excess is not None and result.metrics.excess_return <= prev_excess:
                stale_count += 1
            else:
                stale_count = 0
            prev_excess = result.metrics.excess_return
            if stale_count >= 3:
                break

            adjustments = analyze_failure(result, window_sectors, config)
            if not adjustments:
                # Smarter fallback: try different params instead of always reducing lookback
                if config.rebalance_freq > 3:
                    adjustments = [Adjustment("rebalance_freq", max(config.rebalance_freq - 2, 3), "Fallback: increase rebalance frequency")]
                elif config.ma_period > 10:
                    adjustments = [Adjustment("ma_period", max(config.ma_period - 5, 10), "Fallback: shorten MA period")]
                elif config.roc_period > 10:
                    adjustments = [Adjustment("roc_period", max(config.roc_period - 5, 10), "Fallback: shorten ROC period")]
                else:
                    adjustments = [Adjustment("lookback", max(config.lookback - 2, 5), "Fallback: reduce absolute filter window")]
            config = apply_adjustments(config, adjustments)
            iteration += 1

        results.append(WindowResult(
            window_label=f"{win_start}_{win_end}",
            config=best_config, metrics=best_metrics,
            passed=passed, iterations=iteration + 1,
            history=history,
        ))

    return results
