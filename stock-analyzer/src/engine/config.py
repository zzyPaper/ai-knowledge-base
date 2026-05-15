"""Strategy configuration — single source of truth for all parameters.

All tunable parameters are declared here. The executor, signal pipeline,
risk manager, and portfolio builder all read from this config.

References:
  - 国盛证券 "趋势-拥挤度" 二维框架 (2024-2025)
  - 中银证券 波动率控制 + 多策略复合 (2025)
  - Barroso & Santa-Clara (2015) volatility targeting
  - Antonacci (2012) Dual Momentum
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class MarketTimingConfig:
    """Market entry/exit rules (index-level absolute momentum)."""

    # Entry: index N-day return must exceed this threshold (pct)
    entry_lookback: int = 20
    entry_threshold: float = 0.5  # pct

    # Exit: index N-day return below this threshold (pct)
    exit_lookback: int = 10
    exit_threshold: float = -0.5  # pct

    # Minimum holding days after entry before exit is allowed (anti-whipsaw)
    min_hold_days: int = 3

    # Exit override: don't exit on short-term signal if long-term trend is strong
    exit_override_20d_threshold: float = 2.0  # pct — if 20d > 2%, ignore exit signal

    # Reversal detection: single-day or cumulative bounce
    reversal_single_day: float = 2.0  # pct
    reversal_2d_cumulative: float = 2.5  # pct
    reversal_enabled: bool = True

    # Grace period after reversal lock expires: position cap linearly decays
    # from lock-era cap to current regime cap over this many trading days.
    # Should be >= rebalance_freq to ensure at least one rebalance falls within grace.
    reversal_grace_days: int = 10

    # Breadth filter: when > this fraction of sectors have negative N-day returns,
    # position cap is cut by 50% to protect against broad market declines.
    breadth_lookback: int = 5
    breadth_danger_threshold: float = 0.5


@dataclass(frozen=True)
class RegimeConfig:
    """Market regime classification parameters."""

    # MA structure for regime detection
    ma_short: int = 10
    ma_medium: int = 20
    ma_long: int = 60

    # Deviation thresholds from MA
    bull_deviation: float = 2.0  # price > MA20 + 2%
    bear_deviation: float = -2.0  # price < MA20 - 2%

    # MA slope thresholds (% change over 5 days)
    slope_rising: float = 0.3
    slope_falling: float = -0.3

    # Hysteresis: require N consecutive days of new regime before switching
    hysteresis_days: int = 2


@dataclass(frozen=True)
class SignalConfig:
    """Multi-dimension signal computation parameters."""

    # Trend dimension: multi-timeframe momentum
    momentum_windows: tuple = (20, 60, 120)  # 1M, 3M, 6M
    momentum_weights: tuple = (0.50, 0.30, 0.20)

    # Crowding dimension
    crowding_history: int = 500  # trading days for percentile calculation
    crowding_short: int = 40  # recent window for indicators

    # Composite weights by regime
    trending_trend_weight: float = 0.70
    trending_crowd_weight: float = 0.30
    ranging_trend_weight: float = 0.50
    ranging_crowd_weight: float = 0.50

    # Data lookback for signal computation (calendar days)
    data_lookback_days: int = 365

    # Minimum data points required per sector
    min_data_points: int = 20


@dataclass(frozen=True)
class RiskConfig:
    """Risk management parameters."""

    # Volatility targeting (Barroso & Santa-Clara 2015 + 中银改良)
    target_annual_vol: float = 0.12  # 12% annual
    vol_lookback: int = 63  # trading days (中银)
    vol_scale_min: float = 0.25  # minimum scale
    vol_scale_max: float = 1.20  # maximum scale

    # Position cap boundaries (pct)
    position_cap_min: int = 10
    position_cap_max: int = 90

    # Crash stop: single-day sector loss exceeding this → immediate exit
    crash_stop_threshold: float = -5.0  # pct

    # Maximum drawdown control (reduce position if DD exceeds threshold)
    max_dd_threshold: float = -10.0  # pct from peak


@dataclass(frozen=True)
class PortfolioConfig:
    """Portfolio construction constraints."""

    base_position_pct: int = 80  # base position before vol scaling
    min_position_pct: int = 10
    max_position_pct: int = 90

    min_sectors: int = 3
    max_sectors: int = 5

    # Single sector concentration limit (pct of invested amount)
    sector_concentration_cap: float = 0.40

    rebalance_freq: int = 5  # trading days
    initial_capital: float = 5000.0

    # Turnover control: max turnover per rebalance (pct of NAV)
    max_turnover_pct: float = 1.0  # 1.0 = no limit


@dataclass(frozen=True)
class StrategyConfig:
    """Master configuration aggregating all sub-configs."""

    timing: MarketTimingConfig = field(default_factory=MarketTimingConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    signals: SignalConfig = field(default_factory=SignalConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)

    # Strategy metadata
    name: str = "professional-sector-rotation"
    version: str = "1.0.0"
