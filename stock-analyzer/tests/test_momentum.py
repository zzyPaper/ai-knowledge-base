"""Tests for momentum signal calculation."""

import pandas as pd
from src.features.momentum import calc_ma_ratio, calc_roc, rank_momentum


def _synthetic_close(values: list[float]) -> pd.Series:
    return pd.Series(values, dtype=float)


def test_ma_ratio_above_ma():
    """When close > MA, ratio should be positive."""
    series = _synthetic_close([10.0] * 10 + [12.0])
    ratio = calc_ma_ratio(series, period=10)
    assert ratio > 0, f"Expected positive ratio, got {ratio}"


def test_ma_ratio_below_ma():
    """When close < MA, ratio should be negative."""
    series = _synthetic_close([10.0] * 10 + [8.0])
    ratio = calc_ma_ratio(series, period=10)
    assert ratio < 0, f"Expected negative ratio, got {ratio}"


def test_ma_ratio_not_enough_data():
    """When series is shorter than period, return 0."""
    series = _synthetic_close([1, 2, 3])
    assert calc_ma_ratio(series, period=10) == 0.0


def test_roc_positive():
    """When price increased, ROC should be positive."""
    series = _synthetic_close([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    roc = calc_roc(series, period=5)
    assert roc > 0, f"Expected positive ROC, got {roc}"
    assert abs(roc - 0.5) < 1e-6


def test_roc_negative():
    """When price decreased, ROC should be negative."""
    series = _synthetic_close([10.0, 9.5, 9.0, 8.5, 8.0, 7.5])
    roc = calc_roc(series, period=5)
    assert roc < 0, f"Expected negative ROC, got {roc}"


def test_roc_not_enough_data():
    """When series is too short, return 0."""
    series = _synthetic_close([1, 2, 3])
    assert calc_roc(series, period=10) == 0.0


def _make_sector_df(prices: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"close": prices, "date": pd.date_range("2025-01-01", periods=len(prices))})


def test_rank_momentum_ranks():
    """Rank momentum should give higher scores to stronger sectors."""
    winners = _make_sector_df([10 + i * 0.5 for i in range(20)])
    losers = _make_sector_df([10 - i * 0.3 for i in range(20)])
    flat = _make_sector_df([10.0] * 20)

    scores = rank_momentum({"winner": winners, "loser": losers, "flat": flat})
    assert scores["winner"] > scores["flat"], "Winner should beat flat"
    assert scores["flat"] > scores["loser"], "Flat should beat loser"
    assert all(0 <= v <= 1 for v in scores.values()), "All scores in [0, 1]"
