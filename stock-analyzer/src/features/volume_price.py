"""Volume-price pattern detection (8 classic patterns)."""

from typing import Optional, Tuple
import pandas as pd
import numpy as np


def _range_ratio(row, avg_range: float) -> float:
    """Volatility ratio: (high - low) / close, relative to average."""
    if avg_range == 0:
        return 0.0
    return (row["high"] - row["low"]) / row["close"] / avg_range


def _volume_ratio(row, avg_volume: float) -> float:
    """Volume ratio relative to average."""
    if avg_volume == 0:
        return 0.0
    return row["volume"] / avg_volume


PATTERNS = {
    "bullish_surge": {"condition": lambda vr, rr: vr > 1.5 and rr > 1.5, "score": 1.0},
    "bullish_consolidation": {"condition": lambda vr, rr: vr < 0.7 and rr > 1.5, "score": 0.5},
    "bearish_dump": {"condition": lambda vr, rr: vr > 1.5 and rr < -1.5, "score": -1.0},
    "bearish_drift": {"condition": lambda vr, rr: vr < 0.7 and rr < -1.5, "score": -0.5},
    "divergence": {"condition": lambda vr, rr: vr > 1.5 and -1.5 <= rr <= 1.5, "score": -0.3},
    "stabilization": {"condition": lambda vr, rr: vr < 0.7 and -1.5 <= rr <= 1.5, "score": 0.3},
    "climax": {"condition": lambda vr, rr: vr > 3.0 and rr > 3.0, "score": 0.2},
    "exhaustion": {"condition": lambda vr, rr: vr < 0.3 and rr < -3.0, "score": 0.1},
}


def detect_pattern(row, avg_volume: float, avg_high_low: float) -> Tuple[Optional[str], float]:
    """Detect volume-price pattern for a single day row.

    Returns (pattern_name, score).
    """
    vr = _volume_ratio(row, avg_volume) if avg_volume > 0 else 0.0
    close_range = row["high"] - row["low"]
    rr = (close_range / row["close"] / avg_high_low) if avg_high_low > 0 and row["close"] > 0 else 0.0

    roc = 0.0
    if "pre_close" in row:
        roc = (row["close"] - row["pre_close"]) / row["pre_close"]
    elif "pct_chg" in row:
        roc = row["pct_chg"] / 100.0

    _rr = roc / (avg_high_low if avg_high_low > 0 else 0.01)

    for name, pat in PATTERNS.items():
        if pat["condition"](vr, _rr):
            return name, pat["score"]
    return None, 0.0


def score_volume_price(df: pd.DataFrame, lookback: int = 5) -> float:
    """Score volume-price patterns for a sector over the recent `lookback` days.

    Accumulates pattern scores, clips to [-100, 100].
    """
    if len(df) < lookback + 1:
        return 0.0

    df = df.iloc[-(lookback + 1):].reset_index(drop=True)
    avg_volume = df["volume"].iloc[:-1].mean()
    avg_high_low = ((df["high"] - df["low"]) / df["close"].replace(0, np.nan)).iloc[:-1].mean()

    total = 0.0
    for i in range(lookback):
        row = df.iloc[-(lookback - i)] if lookback - i > 0 else df.iloc[-1]
        row = df.iloc[-(lookback - i)]
        close_range = row["high"] - row["low"]
        vr = _volume_ratio(row, avg_volume) if avg_volume > 0 else 0.0
        roc = 0.0
        pre_idx = len(df) - (lookback - i) - 1
        if pre_idx >= 0:
            roc = (row["close"] - df.iloc[pre_idx]["close"]) / df.iloc[pre_idx]["close"]
        rr = roc / (avg_high_low if avg_high_low > 0 else 0.01)

        for name, pat in PATTERNS.items():
            if pat["condition"](vr, rr):
                total += pat["score"]
                break

    return max(-100.0, min(100.0, total * 10.0))
