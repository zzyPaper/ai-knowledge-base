"""Tests for congestion signal."""

import pandas as pd
from src.features.congestion import calc_turnover_ratio, score_congestion


def _make_sector_df(turnover: list[float], amount: list[float]) -> pd.DataFrame:
    n = len(turnover)
    return pd.DataFrame({
        "close": [10.0] * n,
        "open": [10.0] * n,
        "high": [10.5] * n,
        "low": [9.5] * n,
        "volume": [1000] * n,
        "amount": amount,
        "turnover_rate": turnover,
        "date": pd.date_range("2025-01-01", periods=n),
    })


def test_turnover_ratio_high():
    """When current turnover is much higher than average, ratio > 1."""
    df = _make_sector_df(turnover=[1.0, 1.0, 1.0, 1.0, 0.5, 5.0], amount=[1e8] * 6)
    ratio = calc_turnover_ratio(df)
    assert ratio > 1.0, f"Expected ratio > 1.0, got {ratio}"


def test_turnover_ratio_low():
    """When current turnover is much lower than average, ratio < 1."""
    df = _make_sector_df(turnover=[5.0, 5.0, 5.0, 5.0, 5.0, 1.0], amount=[1e8] * 6)
    ratio = calc_turnover_ratio(df)
    assert ratio < 1.0, f"Expected ratio < 1.0, got {ratio}"


def test_score_congestion_in_range():
    """Congestion scores should be in [0, 1]."""
    data = {
        "sector_a": _make_sector_df(turnover=[1.0] * 6, amount=[1e8] * 6),
        "sector_b": _make_sector_df(turnover=[2.0] * 6, amount=[2e8] * 6),
    }
    scores = score_congestion(data)
    for v in scores.values():
        assert 0 <= v <= 1, f"Score {v} outside [0, 1]"


def test_score_congestion_ranks():
    """Higher turnover and volume sectors should score higher."""
    data = {
        "hot": _make_sector_df(turnover=[5.0] * 6, amount=[5e8] * 6),
        "cold": _make_sector_df(turnover=[0.5] * 6, amount=[1e7] * 6),
    }
    scores = score_congestion(data)
    assert scores["hot"] > scores["cold"]
