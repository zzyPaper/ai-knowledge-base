"""Failure analysis and strategy parameter adjustment for Dual Momentum system.

Heuristics:
  - Win rate < 40% → adjust ma_period / roc_period
  - Max drawdown > 15% → increase diversification (top_n)
  - Too few trades → rebalance more frequently
  - Few sectors pass absolute filter → decrease lookback
"""

from dataclasses import dataclass
from typing import Optional
from src.backtest.engine import StrategyConfig


@dataclass
class Adjustment:
    param: str
    new_value: object
    reason: str


def compute_signal_win_rate(result, sectors_data: dict) -> dict[str, float]:
    """Estimate sector win rate by comparing selected sector returns."""
    from collections import Counter
    sector_day_count = Counter()
    sector_win_count = Counter()
    for trade in result.trades:
        if trade["action"] == "buy":
            sector = trade["sector"]
            for row in result.daily_returns.itertuples():
                sector_day_count[sector] += 1
                if row.port_return > 0:
                    sector_win_count[sector] += 1

    rates = {}
    for s, total in sector_day_count.items():
        rates[s] = sector_win_count.get(s, 0) / max(total, 1)
    return rates


def analyze_failure(result, sectors_data: dict, config: StrategyConfig) -> list[Adjustment]:
    """Analyze backtest failure and propose parameter adjustments (Dual Momentum)."""
    adjustments = []
    metrics = result.metrics

    signal_win_rates = compute_signal_win_rate(result, sectors_data)
    avg_mom_win = sum(signal_win_rates.values()) / max(len(signal_win_rates), 1)

    # 1a. Momentum win rate too low — adjust ma_period
    if avg_mom_win < 0.4:
        if avg_mom_win < 0.3:
            adjustments.append(Adjustment(
                param="ma_period",
                new_value=min(config.ma_period + 5, 40),
                reason="MA period too short for weak momentum signal",
            ))
        else:
            adjustments.append(Adjustment(
                param="ma_period",
                new_value=max(config.ma_period - 5, 10),
                reason="MA period too long, too slow to react",
            ))

    # 1b. Also try adjusting roc_period orthogonally
    if avg_mom_win < 0.35:
        adjustments.append(Adjustment(
            param="roc_period",
            new_value=max(min(config.roc_period + 5, 40), 10),
            reason="ROC period adjustment for weak momentum",
        ))

    # 2. Max drawdown too high — increase diversification
    if metrics.max_drawdown < -0.15:
        adjustments.append(Adjustment(
            param="top_n",
            new_value=min(config.top_n + 1, 5),
            reason=f"Reduce concentration risk (DD={metrics.max_drawdown:.1%})",
        ))

    # 3. Too few rebalances — insufficient exploration
    if len(result.trades) < 4:
        adjustments.append(Adjustment(
            param="rebalance_freq",
            new_value=max(config.rebalance_freq - 2, 3),
            reason="Too few trades, increase rebalance frequency",
        ))

    # 4. Few qualified sectors — absolute filter may be too strict
    qualified_count = _count_qualified_sectors(result, sectors_data)
    if qualified_count is not None and qualified_count < 3:
        new_val = max(config.lookback - 3, 5)
        if new_val != config.lookback:
            adjustments.append(Adjustment(
                param="lookback",
                new_value=new_val,
                reason=f"Only {qualified_count} sectors passed absolute filter, reducing lookback",
            ))

    return adjustments


def _count_qualified_sectors(result, sectors_data: dict) -> Optional[int]:
    """Count sectors that passed absolute momentum filter at first rebalance."""
    if not result.sector_snapshots:
        return None
    snap = result.sector_snapshots[0]
    return len(snap.get("top_sectors", []))


def apply_adjustments(config: StrategyConfig, adjustments: list[Adjustment]) -> StrategyConfig:
    """Apply parameter adjustments to a config copy."""
    new_config = StrategyConfig(
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

    for adj in adjustments:
        if adj.param == "ma_period":
            new_config.ma_period = adj.new_value
        elif adj.param == "top_n":
            new_config.top_n = adj.new_value
        elif adj.param == "rebalance_freq":
            new_config.rebalance_freq = adj.new_value
        elif adj.param == "trending_weights":
            new_config.trending_weights = adj.new_value
        elif adj.param == "ranging_weights":
            new_config.ranging_weights = adj.new_value
        elif adj.param == "roc_period":
            new_config.roc_period = adj.new_value
        elif adj.param == "lookback":
            new_config.lookback = adj.new_value

    return new_config
