"""V3 Sector Rotation Strategy: 4-dimension signals + lifecycle + timing decoupling.

Inherits from BaseStrategy to plug into the existing BacktestEngine.

Core improvements over V2:
  1. Four signal dimensions: Trend (path-adjusted + idio), Crowding (true Beta),
     Quality (path quality, Sharpe), Flow (volume-price coordination)
  2. Momentum lifecycle: Buildup/Sustain/Exhaust/Avoid with weight multipliers
  3. Timing only affects position multiplier, never forces full exit
"""

import numpy as np
import pandas as pd
from typing import Optional

from src.strategy.base import BaseStrategy, SectorScore
from src.strategy.path_momentum import compute_trend_dimension
from src.strategy.crowding_pro import score_crowding_pro
from src.strategy.quality import compute_quality_dimension
from src.strategy.flow import compute_flow_dimension
from src.strategy.lifecycle import compute_all_lifecycles, Lifecycle


# Signal weights by regime: (trend, crowding_inv, quality, flow)
REGIME_WEIGHTS = {
    "trending": (0.45, 0.20, 0.20, 0.15),
    "ranging":  (0.30, 0.30, 0.25, 0.15),
}

# Lifecycle weight multipliers
LIFECYCLE_MULT = {
    Lifecycle.BUILDUP: 1.25,
    Lifecycle.SUSTAIN: 1.00,
    Lifecycle.EXHAUST: 0.55,
    Lifecycle.AVOID: 0.00,
}


class V3Strategy(BaseStrategy):
    """V3 strategy: 4-dimension scoring with momentum lifecycle."""

    name = "v3_four_dimension"
    version = "3.0"
    description = "四维信号(趋势/拥挤度/质量/资金) + 动量生命周期 + 择时选股解耦"

    def detect_regime(self, index_hist: pd.DataFrame) -> str:
        """Detect market regime using MA structure."""
        if index_hist is None or len(index_hist) < 60:
            return "ranging"
        close = index_hist["close"].values
        ma20 = float(np.mean(close[-20:]))
        ma60 = float(np.mean(close[-60:]))
        if close[-1] > ma20 and ma20 > ma60:
            return "trending"
        return "ranging"

    def score_sector(
        self,
        sector_hist: pd.DataFrame,
        index_hist: Optional[pd.DataFrame] = None,
        regime: Optional[str] = None,
    ) -> SectorScore:
        """Score a single sector using 4-dimension + lifecycle framework.

        Note: This is called per-sector by BacktestEngine. For efficiency,
        we cache dimension results across sectors in score_all_sectors.
        """
        # For single-sector scoring, we do a simplified version
        # The full cross-section scoring happens in score_all_sectors
        close = sector_hist["close"].values
        if len(close) < 21:
            return SectorScore(
                sector=getattr(sector_hist, "_sector_name", "unknown"),
                composite=0.0,
                signal="HOLD",
                position=0.0,
                factors={},
                regime=regime or "unknown",
            )

        # Simple trend score for single sector
        from src.strategy.path_momentum import calc_path_adjusted_momentum
        path_mom = calc_path_adjusted_momentum(close, 20)

        # Normalize to [-1, 1] roughly
        trend_score = np.clip(path_mom * 10, -1.0, 1.0)

        # Position based on trend strength
        position = max(0.0, min(trend_score, 1.0))
        signal = "BUY" if position > 0.3 else "HOLD"

        return SectorScore(
            sector=getattr(sector_hist, "_sector_name", "unknown"),
            composite=position,
            signal=signal,
            position=position,
            factors={"path_mom": path_mom},
            regime=regime or "unknown",
        )

    def score_all_sectors(
        self,
        sectors_data: dict[str, pd.DataFrame],
        index_hist: Optional[pd.DataFrame] = None,
    ) -> list[SectorScore]:
        """Batch score all sectors with full 4-dimension + lifecycle framework."""
        regime = self.detect_regime(index_hist) if index_hist is not None else "ranging"

        # Compute four dimensions (cross-section, more accurate)
        trend_df = compute_trend_dimension(sectors_data, index_hist)
        crowd_df = score_crowding_pro(sectors_data, index_hist)
        quality_df = compute_quality_dimension(sectors_data)
        flow_df = compute_flow_dimension(sectors_data)

        # Detect lifecycle phases
        lifecycles = compute_all_lifecycles(trend_df, crowd_df, flow_df, quality_df)

        # Build lookup maps
        trend_map = dict(zip(trend_df["sector"], trend_df["trend_score"])) if not trend_df.empty else {}
        crowd_map = dict(zip(crowd_df["sector"], crowd_df["crowding_score"])) if not crowd_df.empty else {}
        quality_map = dict(zip(quality_df["sector"], quality_df["quality_score"])) if not quality_df.empty else {}
        flow_map = dict(zip(flow_df["sector"], flow_df["flow_score"])) if not flow_df.empty else {}

        w_trend, w_crowd, w_quality, w_flow = REGIME_WEIGHTS.get(regime, REGIME_WEIGHTS["ranging"])

        scores = []
        for name, hist in sectors_data.items():
            life = lifecycles.get(name, Lifecycle.AVOID)
            mult = LIFECYCLE_MULT.get(life, 0.0)

            t = trend_map.get(name, 0.5)
            c = 1.0 - crowd_map.get(name, 0.5)  # invert crowding
            q = quality_map.get(name, 0.5)
            f = flow_map.get(name, 0.5)

            composite = (w_trend * t + w_crowd * c +
                         w_quality * q + w_flow * f) * mult

            # Position: composite mapped to [0, 1], with lifecycle adjustment
            position = max(0.0, min(composite, 1.0))

            # Signal
            if life == Lifecycle.AVOID or composite < 0.15:
                signal = "SELL"
                position = 0.0
            elif life == Lifecycle.BUILDUP and composite > 0.4:
                signal = "BUY"
            elif life == Lifecycle.SUSTAIN and composite > 0.3:
                signal = "BUY"
            elif life == Lifecycle.EXHAUST:
                signal = "HOLD"
                position *= 0.5  # reduce position for exhaust
            else:
                signal = "HOLD"

            scores.append(SectorScore(
                sector=name,
                composite=composite,
                signal=signal,
                position=position,
                factors={
                    "trend": round(t, 3),
                    "crowding": round(crowd_map.get(name, 0.5), 3),
                    "quality": round(q, 3),
                    "flow": round(f, 3),
                    "lifecycle": life.value,
                    "lifecycle_mult": mult,
                },
                regime=regime,
            ))

        scores.sort(key=lambda x: x.composite, reverse=True)
        return scores
