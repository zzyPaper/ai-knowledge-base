"""V4 Hybrid Strategy: V2 stock-picking + V3 risk-control overlay.

Design:
  - Stock selection: V2's 7-factor scoring (trend + momentum + volume + capital +
    sentiment + fundamental + valuation). V2 is more robust at picking sectors
    because it evaluates each sector individually with a rich factor model.
  - Risk overlay: V3's lifecycle detection (Buildup/Sustain/Exhaust/Avoid) applied
    as a position multiplier on top of V2's position.
  - Timing decoupling: market timing only affects max position cap, never forces
    full exit (so we don't miss structural opportunities).

Why this works:
  - V2's per-sector scoring avoids V3's cross-section ranking trap (buying obscure
    high-volatility sectors just because they rank high in a thin cross-section).
  - V3's lifecycle prevents buying exhausted momentum and boosts early-stage
    buildup sectors, improving entry/exit timing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from src.strategy.base import BaseStrategy, SectorScore
from src.strategy.v2_three_factor import V2ThreeFactor
from src.strategy.path_momentum import compute_trend_dimension
from src.strategy.crowding_pro import score_crowding_pro
from src.strategy.quality import compute_quality_dimension
from src.strategy.flow import compute_flow_dimension
from src.strategy.lifecycle import compute_all_lifecycles, Lifecycle
from src.strategy.professional import (
    detect_market_regime_pro,
    volatility_scale,
    index_trend_filter,
)

# Lifecycle multipliers applied ON TOP OF V2 position
LIFECYCLE_MULT = {
    Lifecycle.BUILDUP: 1.25,
    Lifecycle.SUSTAIN: 1.00,
    Lifecycle.EXHAUST: 0.55,
    Lifecycle.AVOID: 0.00,
}

# Timing caps max position based on market condition
TIMING_CAPS = {
    "strong_trend": 1.00,   # trending, in market
    "weak_trend":   0.70,   # trending but timing filter off
    "ranging_ok":   0.85,   # ranging but in market
    "ranging_bad":  0.50,   # ranging and timing filter off
}


class V4HybridStrategy(BaseStrategy):
    """V4: V2 stock-picking + V3 lifecycle/timing risk overlay."""

    name = "v4_hybrid"
    version = "4.0"
    description = "V2选股 + V3生命周期/择时风控叠加"

    def __init__(self):
        self.v2 = V2ThreeFactor()

    def detect_regime(self, index_hist: pd.DataFrame) -> str:
        return detect_market_regime_pro(index_hist)

    def score_sector(
        self,
        sector_hist: pd.DataFrame,
        index_hist: Optional[pd.DataFrame] = None,
        regime: Optional[str] = None,
    ) -> SectorScore:
        """Fallback single-sector scoring (used by engine if score_all_sectors fails)."""
        if regime is None:
            regime = self.detect_regime(index_hist) if index_hist is not None else "ranging"
        return self.v2.score_sector(sector_hist, index_hist, regime)

    def score_all_sectors(
        self,
        sectors_data: dict[str, pd.DataFrame],
        index_hist: Optional[pd.DataFrame] = None,
    ) -> list[SectorScore]:
        """Batch score: V2 per-sector scoring + V3 lifecycle overlay."""
        regime = self.detect_regime(index_hist) if index_hist is not None else "ranging"

        # ── V3 dimension signals (for lifecycle detection only) ──
        trend_df = compute_trend_dimension(sectors_data, index_hist)
        crowd_df = score_crowding_pro(sectors_data, index_hist)
        quality_df = compute_quality_dimension(sectors_data)
        flow_df = compute_flow_dimension(sectors_data)
        lifecycles = compute_all_lifecycles(trend_df, crowd_df, flow_df, quality_df)

        # ── Market timing cap (V3: never full exit) ──
        timing_cap = 1.0
        if index_hist is not None and not index_hist.empty:
            latest_date = pd.to_datetime(index_hist["date"].iloc[-1])
            in_market, trend_strength = index_trend_filter(index_hist, latest_date)
            if regime == "trending":
                timing_cap = TIMING_CAPS["strong_trend"] if in_market else TIMING_CAPS["weak_trend"]
            else:
                timing_cap = TIMING_CAPS["ranging_ok"] if in_market else TIMING_CAPS["ranging_bad"]

            # Volatility scaling
            vol_scale = volatility_scale(index_hist, latest_date)
            timing_cap = min(timing_cap, vol_scale)

        # ── V2 per-sector scoring + lifecycle overlay ──
        scores: list[SectorScore] = []
        for name, hist in sectors_data.items():
            life = lifecycles.get(name, Lifecycle.SUSTAIN)
            life_mult = LIFECYCLE_MULT.get(life, 1.0)

            # V2 base scoring
            v2_score = self.v2.score_sector(hist, index_hist, regime)
            v2_score.sector = name

            # Skip lifecycle-rejected sectors early
            if life == Lifecycle.AVOID:
                v2_score.composite = max(v2_score.composite * 0.1, -1.0)
                v2_score.position = 0.0
                v2_score.signal = "SELL"
                v2_score.factors["lifecycle"] = life.value
                v2_score.factors["lifecycle_mult"] = life_mult
                v2_score.factors["timing_cap"] = timing_cap
                scores.append(v2_score)
                continue

            # Lifecycle multiplier on position
            base_position = v2_score.position
            adjusted_position = base_position * life_mult

            # Apply timing cap (never forces full exit if V2 says BUY)
            # But if timing is bad, we cap the position
            final_position = min(adjusted_position, timing_cap)
            # Ensure we never go below 5% if V2 wants in and lifecycle allows
            if v2_score.signal == "BUY" and life != Lifecycle.AVOID:
                final_position = max(final_position, 0.05)

            # Exhaust penalty: also reduce composite score for ranking
            composite = v2_score.composite
            if life == Lifecycle.EXHAUST:
                composite *= 0.6

            # Build factors
            factors = dict(v2_score.factors)
            factors["lifecycle"] = life.value
            factors["lifecycle_mult"] = life_mult
            factors["timing_cap"] = round(timing_cap, 3)
            factors["base_position"] = round(base_position, 3)
            factors["v2_composite"] = round(v2_score.composite, 3)

            scores.append(SectorScore(
                sector=name,
                composite=float(composite),
                signal=v2_score.signal if final_position > 0.05 else "HOLD",
                position=float(np.clip(final_position, 0.0, 1.0)),
                factors=factors,
                regime=regime,
            ))

        scores.sort(key=lambda x: x.composite, reverse=True)
        return scores
