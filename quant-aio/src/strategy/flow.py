"""Flow dimension: volume-price coordination, capital inflow, turnover trend.

New dimension for V3 to capture money flow characteristics.
"""

import numpy as np
import pandas as pd
from typing import Optional


def calc_volume_price_coordination(
    close: np.ndarray, volume: np.ndarray, period: int = 20
) -> float:
    """Correlation between price change and volume change.

    Positive = price rises on high volume (healthy).
    Negative = price rises on low volume (weak) or falls on high volume (distribution).
    """
    if len(close) < period + 1 or len(volume) < period + 1:
        return 0.0
    price_rets = np.diff(close[-(period + 1):]) / close[-(period + 1):-1]
    vol_rets = np.diff(volume[-(period + 1):]) / (volume[-(period + 1):-1] + 1e-10)
    if len(price_rets) < 2 or len(vol_rets) < 2:
        return 0.0
    # Winsorize extreme volume returns
    vol_rets = np.clip(vol_rets, -0.5, 0.5)
    corr = np.corrcoef(price_rets, vol_rets)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0


def calc_capital_inflow_strength(
    amount: np.ndarray, period: int = 20
) -> float:
    """Amount change rate vs 20-day average. Positive = capital flowing in."""
    if len(amount) < period + 1:
        return 0.0
    recent = float(np.mean(amount[-5:]))  # last 5 days
    baseline = float(np.mean(amount[-(period + 1):-1]))  # previous 20 days
    if baseline < 1e-10:
        return 0.0
    change_rate = (recent - baseline) / baseline
    # Cap at +/- 50% for stability
    return float(np.clip(change_rate, -0.5, 0.5))


def calc_turnover_trend(
    turnover_rate: np.ndarray, short_period: int = 5, long_period: int = 20
) -> float:
    """Turnover rate short-term trend vs long-term. Positive = heating up."""
    if len(turnover_rate) < long_period + 1:
        return 0.0
    short_avg = float(np.mean(turnover_rate[-short_period:]))
    long_avg = float(np.mean(turnover_rate[-long_period:]))
    if long_avg < 1e-10:
        return 0.0
    return float(np.clip((short_avg - long_avg) / long_avg, -0.5, 0.5))


def compute_flow_dimension(
    sectors_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Compute flow dimension scores for all sectors.

    Returns DataFrame with columns: sector, vp_coord, inflow, turnover_trend, flow_score.
    """
    records = []
    for name, df in sectors_data.items():
        close = df["close"].values
        n = len(close)
        if n < 21:
            records.append({
                "sector": name,
                "vp_coord": 0.0,
                "inflow": 0.0,
                "turnover_trend": 0.0,
                "flow_raw": 0.0,
            })
            continue

        volume = df["volume"].values if "volume" in df.columns else np.zeros(n)
        amount = df["amount"].values if "amount" in df.columns else np.zeros(n)
        turnover = df["turnover_rate"].values if "turnover_rate" in df.columns else np.zeros(n)

        vp = calc_volume_price_coordination(close, volume, 20)
        inflow = calc_capital_inflow_strength(amount, 20)
        to_trend = calc_turnover_trend(turnover, 5, 20)

        # Composite flow: all signals in [-0.5, 0.5], map to [0, 1]
        # vp_coord in [-1, 1], inflow in [-0.5, 0.5], turnover_trend in [-0.5, 0.5]
        vp_score = (np.clip(vp, -1.0, 1.0) + 1.0) / 2.0
        inflow_score = (np.clip(inflow, -0.5, 0.5) + 0.5)  # map to [0, 1]
        to_score = (np.clip(to_trend, -0.5, 0.5) + 0.5)  # map to [0, 1]

        flow_raw = vp_score * 0.40 + inflow_score * 0.35 + to_score * 0.25

        records.append({
            "sector": name,
            "vp_coord": vp,
            "inflow": inflow,
            "turnover_trend": to_trend,
            "flow_raw": flow_raw,
        })

    df = pd.DataFrame(records)
    if df.empty:
        return df

    sorted_items = sorted(zip(df["sector"], df["flow_raw"]), key=lambda x: x[1])
    n = len(sorted_items)
    flow_norm = {name: i / (n - 1) if n > 1 else 0.5 for i, (name, _) in enumerate(sorted_items)}
    df["flow_score"] = df["sector"].map(flow_norm)
    return df.sort_values("flow_score", ascending=False).reset_index(drop=True)
