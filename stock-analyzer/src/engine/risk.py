"""Risk management layer.

Two components:
  1. Volatility targeting (Barroso & Santa-Clara 2015, 中银改良)
  2. Crash stop + drawdown control

The vol targeting uses negative semi-volatility (中银 innovation),
which penalizes downside volatility more than symmetric volatility.

Reference:
  - Barroso & Santa-Clara, "Momentum Has Its Moments" (2015)
  - 中银证券 波动率控制多策略复合 (2025)
"""

from dataclasses import dataclass
import pandas as pd
import numpy as np

from src.engine.config import RiskConfig


@dataclass
class RiskAssessment:
    """Output of risk evaluation for a given day."""
    vol_scale: float
    annual_vol: float
    position_cap: int  # adjusted position cap (%)
    crash_triggered: bool
    drawdown_pct: float  # current drawdown from peak


class RiskManager:
    """Multi-layer risk management."""

    def __init__(self, config: RiskConfig = RiskConfig()):
        self.config = config
        self._peak_nav: float = 0.0

    def update_peak(self, nav: float):
        self._peak_nav = max(self._peak_nav, nav)

    def current_drawdown(self, nav: float) -> float:
        if self._peak_nav <= 0:
            return 0.0
        return (nav / self._peak_nav - 1) * 100

    def assess(self, index: pd.DataFrame, date: pd.Timestamp,
               nav: float, base_cap: int) -> RiskAssessment:
        """Evaluate risk and return adjusted position cap.

        Args:
            index: full index DataFrame
            date: current trading day
            nav: current portfolio NAV
            base_cap: position cap from regime (0-100)
        """
        # 1. Volatility scaling
        vol_scale = self._compute_vol_scale(index, date)

        # 2. Adjust cap
        adjusted_cap = int(base_cap * vol_scale)
        adjusted_cap = max(self.config.position_cap_min,
                          min(adjusted_cap, self.config.position_cap_max))

        # 3. Drawdown check
        dd = self.current_drawdown(nav)
        if dd < self.config.max_dd_threshold:
            # Halve position on significant drawdown
            adjusted_cap = max(0, adjusted_cap // 2)

        # 4. Annual vol estimate
        annual_vol = self._estimate_annual_vol(index, date)

        return RiskAssessment(
            vol_scale=round(vol_scale, 3),
            annual_vol=round(annual_vol * 100, 1),
            position_cap=adjusted_cap,
            crash_triggered=dd < self.config.max_dd_threshold,
            drawdown_pct=round(dd, 2),
        )

    def check_crash_stop(self, sector_name: str, daily_pct: float) -> bool:
        """Check if sector triggered crash stop (single-day > threshold)."""
        return daily_pct < self.config.crash_stop_threshold

    def _compute_vol_scale(self, index: pd.DataFrame, date: pd.Timestamp) -> float:
        """Barroso & Santa-Clara (2015) vol scaling + 中银 negative semi-vol.

        scale = target_vol / realized_vol, clamped to [min, max].
        Uses negative semi-volatility (std of only negative daily returns).
        """
        hist = index[(index["date"] <= date)].tail(self.config.vol_lookback + 1)
        if len(hist) < 15:
            return 1.0

        rets = hist["close"].pct_change().dropna().values
        if len(rets) < 5:
            return 1.0

        # Negative semi-volatility (中银)
        neg_rets = rets[rets < 0]
        if len(neg_rets) >= 3:
            daily_vol = float(np.std(neg_rets))
        else:
            daily_vol = float(np.std(rets))

        annual_vol = daily_vol * np.sqrt(252)
        if annual_vol < 0.005:
            return self.config.vol_scale_max

        scale = self.config.target_annual_vol / annual_vol
        return max(self.config.vol_scale_min, min(scale, self.config.vol_scale_max))

    def _estimate_annual_vol(self, index: pd.DataFrame, date: pd.Timestamp) -> float:
        """Estimate current annualized volatility."""
        hist = index[(index["date"] <= date)].tail(self.config.vol_lookback + 1)
        if len(hist) < 10:
            return 0.15
        rets = hist["close"].pct_change().dropna().values
        return float(np.std(rets)) * np.sqrt(252)
