"""Market regime detection with multi-timeframe MA structure.

Classifies market into 5 states based on price position relative to MAs
and MA slope direction. More nuanced than binary in/out.

Regime hierarchy (risk order):
  BULL      — price > MA20 > MA60, both rising
  RECOVERY  — price > MA10 rising, post-crash bounce
  NEUTRAL   — mixed signals, moderate exposure
  CORRECTION — price < MA20, within broader trend
  BEAR      — price < MA20 < MA60, both falling

Reference: 国盛证券 MA结构 + Dual Momentum regime adaptation
"""

from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np

from src.engine.config import RegimeConfig


class Regime:
    """Market regime enum with risk-ordered levels."""
    BULL = "bull"
    RECOVERY = "recovery"
    NEUTRAL = "neutral"
    CORRECTION = "correction"
    BEAR = "bear"

    # Position caps by regime (pct of NAV)
    CAPS = {
        BULL: 90,
        RECOVERY: 70,
        NEUTRAL: 50,
        CORRECTION: 30,
        BEAR: 0,
    }

    # Signal weights by regime: (trend_weight, crowding_weight)
    SIGNAL_WEIGHTS = {
        BULL: (0.70, 0.30),
        RECOVERY: (0.60, 0.40),
        NEUTRAL: (0.50, 0.50),
        CORRECTION: (0.40, 0.60),
        BEAR: (0.30, 0.70),
    }

    @classmethod
    def cap(cls, regime: str) -> int:
        return cls.CAPS.get(regime, 50)

    @classmethod
    def signal_weights(cls, regime: str) -> tuple:
        return cls.SIGNAL_WEIGHTS.get(regime, (0.50, 0.50))


@dataclass
class RegimeResult:
    """Output of regime detection."""
    regime: str
    position_cap: int
    deviation_pct: float  # price vs MA20 deviation
    ma_slope_pct: float   # MA20 5d slope
    details: dict = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


class RegimeDetector:
    """Multi-timeframe MA structure regime classifier with hysteresis."""

    def __init__(self, config: RegimeConfig = RegimeConfig()):
        self.config = config
        self._last_regime: str = Regime.NEUTRAL
        self._pending_regime: str = ""
        self._pending_count: int = 0

    def detect(self, index: pd.DataFrame, date: pd.Timestamp) -> RegimeResult:
        """Classify market regime at a given date.

        Requires at least ma_long + 10 data points.
        """
        hist = index[(index["date"] <= date)].tail(self.config.ma_long + 20)
        if len(hist) < self.config.ma_medium + 10:
            return RegimeResult(Regime.NEUTRAL, Regime.cap(Regime.NEUTRAL), 0.0, 0.0)

        close = hist["close"].values
        price = float(close[-1])

        # Compute MAs
        ma10 = float(np.mean(close[-self.config.ma_short:]))
        ma20 = float(np.mean(close[-self.config.ma_medium:]))
        ma60 = float(np.mean(close[-min(self.config.ma_long, len(close)):]))

        # MA20 slope (5d change)
        if len(close) >= self.config.ma_medium + 5:
            ma20_now = float(np.mean(close[-self.config.ma_medium:]))
            ma20_5d = float(np.mean(close[-(self.config.ma_medium + 5):-5]))
            ma_slope = (ma20_now / ma20_5d - 1) * 100 if ma20_5d > 0 else 0
        else:
            ma_slope = 0.0

        # MA10 slope
        if len(close) >= self.config.ma_short + 5:
            ma10_now = float(np.mean(close[-self.config.ma_short:]))
            ma10_5d = float(np.mean(close[-(self.config.ma_short + 5):-5]))
            ma10_slope = (ma10_now / ma10_5d - 1) * 100 if ma10_5d > 0 else 0
        else:
            ma10_slope = 0.0

        # Deviation from MA20
        deviation = (price / ma20 - 1) * 100 if ma20 > 0 else 0

        # Regime classification with hysteresis
        raw_regime = self._classify(price, ma10, ma20, ma60, deviation, ma_slope, ma10_slope)
        regime = self._apply_hysteresis(raw_regime)
        cap = Regime.cap(regime)

        return RegimeResult(
            regime=regime,
            position_cap=cap,
            deviation_pct=round(deviation, 2),
            ma_slope_pct=round(ma_slope, 2),
            details={
                "ma10": round(ma10, 2),
                "ma20": round(ma20, 2),
                "ma60": round(ma60, 2),
                "ma10_slope": round(ma10_slope, 2),
            },
        )

    def _classify(self, price: float, ma10: float, ma20: float, ma60: float,
                  deviation: float, ma20_slope: float, ma10_slope: float) -> str:
        """Classify regime based on MA structure and slopes."""
        cfg = self.config

        # Bear: price below all MAs, MA20 declining, MA20 below MA60
        if deviation < cfg.bear_deviation and ma20_slope < cfg.slope_falling and ma20 < ma60:
            return Regime.BEAR

        # Correction: price below MA20 but not extreme
        if deviation < 0 and ma20_slope < 0:
            return Regime.CORRECTION

        # Recovery: price above MA10, MA10 turning up (V-shape)
        if price > ma10 and ma10_slope > cfg.slope_rising:
            if deviation < 1.0:  # still below MA20 or barely above
                return Regime.RECOVERY

        # Bull: price above MA20, MA20 above MA60, both rising
        if deviation > cfg.bull_deviation and ma20 > ma60 and ma20_slope > cfg.slope_rising:
            return Regime.BULL

        # Neutral: mixed signals
        if deviation > 0:
            if ma20_slope > 0:
                return Regime.NEUTRAL  # above MA20, MA20 rising but not bull-level
            return Regime.CORRECTION  # above MA20 but MA20 falling

        return Regime.NEUTRAL

    def _apply_hysteresis(self, raw_regime: str) -> str:
        """Apply hysteresis: require N consecutive days before switching regime.

        Filters out noise in regime classification, especially the rapid
        bull/neutral flips that occur when price hovers near deviation thresholds.
        """
        hyst = self.config.hysteresis_days
        if hyst <= 1:
            return raw_regime

        if raw_regime == self._last_regime:
            self._pending_regime = ""
            self._pending_count = 0
            return self._last_regime

        if raw_regime == self._pending_regime:
            self._pending_count += 1
        else:
            self._pending_regime = raw_regime
            self._pending_count = 1

        if self._pending_count >= hyst:
            self._last_regime = raw_regime
            self._pending_regime = ""
            self._pending_count = 0

        return self._last_regime
