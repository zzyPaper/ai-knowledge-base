"""Momentum lifecycle detection: Buildup / Sustain / Exhaust / Avoid.

Key V3 innovation: instead of treating all positive-momentum sectors equally,
we identify which phase of the momentum cycle each sector is in.

Lifecycle rules:
  Buildup: momentum turning positive, volume-price confirming, low crowding
  Sustain: momentum positive, stable vol, moderate crowding
  Exhaust: momentum positive but VP diverging, high crowding, vol spiking
  Avoid: momentum negative or crash stop triggered
"""

import numpy as np
from enum import Enum
from typing import Optional


class Lifecycle(Enum):
    BUILDUP = "buildup"
    SUSTAIN = "sustain"
    EXHAUST = "exhaust"
    AVOID = "avoid"

    # Weight multipliers for composite scoring
    MULTIPLIERS = {
        BUILDUP: 1.25,
        SUSTAIN: 1.00,
        EXHAUST: 0.55,
        AVOID: 0.00,
    }

    @classmethod
    def multiplier(cls, phase) -> float:
        return cls.MULTIPLIERS.get(phase, 1.0)


def detect_lifecycle(
    trend_score: float,       # [0, 1] from path_momentum
    crowding_score: float,    # [0, 1] from crowding_pro (higher = more crowded)
    flow_score: float,        # [0, 1] from flow
    quality_score: float,     # [0, 1] from quality
    vp_coord: float,          # [-1, 1] volume-price correlation
    recent_drawdown: float,   # [0, 1] from quality (1 = at high)
    crash_today: bool = False,
) -> Lifecycle:
    """Detect which momentum lifecycle phase a sector is in.

    Parameters are all pre-computed signals for a single sector.
    """
    if crash_today or trend_score < 0.15:
        return Lifecycle.AVOID

    # Derived indicators
    vp_strong = vp_coord > 0.25
    vp_weak = vp_coord < -0.15
    low_crowd = crowding_score < 0.45
    high_crowd = crowding_score > 0.70
    med_crowd = not low_crowd and not high_crowd
    high_quality = quality_score > 0.60
    vol_stable = quality_score > 0.40  # proxy: good quality implies stable vol

    # Buildup: early stage momentum with confirmation
    if trend_score >= 0.15 and trend_score < 0.65:
        if vp_strong and low_crowd and high_quality:
            return Lifecycle.BUILDUP
        if vp_strong and med_crowd:
            return Lifecycle.BUILDUP

    # Exhaust: late stage with divergence
    if trend_score >= 0.50:
        if vp_weak and high_crowd:
            return Lifecycle.EXHAUST
        if high_crowd and not high_quality:
            return Lifecycle.EXHAUST
        if vp_weak and not vol_stable:
            return Lifecycle.EXHAUST

    # Sustain: healthy middle stage
    if trend_score >= 0.30:
        if med_crowd and vol_stable:
            return Lifecycle.SUSTAIN
        if low_crowd:
            return Lifecycle.SUSTAIN

    # Default: if momentum weak but not avoid, treat as sustain if quality ok
    if quality_score > 0.50 and low_crowd:
        return Lifecycle.SUSTAIN

    return Lifecycle.AVOID


def compute_all_lifecycles(
    trend_df,
    crowding_df,
    flow_df,
    quality_df,
) -> dict[str, Lifecycle]:
    """Compute lifecycle phase for each sector from the four dimension DataFrames.

    Returns dict: sector_name -> Lifecycle.
    """
    # Build lookup maps
    trend_map = dict(zip(trend_df["sector"], trend_df["trend_score"])) if not trend_df.empty else {}
    crowd_map = dict(zip(crowding_df["sector"], crowding_df["crowding_score"])) if not crowding_df.empty else {}
    flow_map = dict(zip(flow_df["sector"], flow_df["flow_score"])) if not flow_df.empty else {}
    quality_map = dict(zip(quality_df["sector"], quality_df["quality_score"])) if not quality_df.empty else {}
    vp_map = dict(zip(flow_df["sector"], flow_df["vp_coord"])) if not flow_df.empty else {}
    recovery_map = dict(zip(quality_df["sector"], quality_df["recovery"])) if not quality_df.empty else {}

    all_sectors = set(trend_map.keys()) | set(crowd_map.keys()) | set(flow_map.keys()) | set(quality_map.keys())

    result = {}
    for sector in all_sectors:
        result[sector] = detect_lifecycle(
            trend_score=trend_map.get(sector, 0.5),
            crowding_score=crowd_map.get(sector, 0.5),
            flow_score=flow_map.get(sector, 0.5),
            quality_score=quality_map.get(sector, 0.5),
            vp_coord=vp_map.get(sector, 0.0),
            recent_drawdown=recovery_map.get(sector, 0.5),
        )
    return result
