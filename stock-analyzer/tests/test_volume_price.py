"""Tests for volume-price pattern detection."""

import pandas as pd
from src.features.volume_price import score_volume_price


def _make_df(prices: list[float], volumes: list[float], highs: list[float] | None = None) -> pd.DataFrame:
    n = len(prices)
    if highs is None:
        highs = [p * 1.02 for p in prices]
    lows = [p * 0.98 for p in prices]
    return pd.DataFrame({
        "close": prices,
        "open": prices,
        "high": highs,
        "low": lows,
        "volume": volumes,
        "amount": [v * p for v, p in zip(volumes, prices)],
        "date": pd.date_range("2025-01-01", periods=n),
    })


def test_bullish_surge_highest():
    """放量上涨 should score highest among patterns."""
    surge = _make_df(
        prices=[10.0, 10.1, 10.2, 10.3, 10.4, 11.0],
        volumes=[100, 100, 100, 100, 100, 500],
    )
    drift = _make_df(
        prices=[10.0, 9.9, 9.8, 9.7, 9.6, 9.3],
        volumes=[100, 100, 100, 100, 100, 80],
    )
    surge_score = score_volume_price(surge)
    drift_score = score_volume_price(drift)
    assert surge_score > drift_score, f"Surge ({surge_score}) should beat drift ({drift_score})"


def test_score_in_range():
    """Score should be within [-100, 100]."""
    df = _make_df(
        prices=[10.0, 10.1, 10.2, 10.3, 10.4, 10.5],
        volumes=[100, 100, 100, 100, 100, 100],
    )
    score = score_volume_price(df)
    assert -100 <= score <= 100, f"Score {score} outside [-100, 100]"


def test_score_zero_on_short_data():
    """Too little data should return 0."""
    df = _make_df(prices=[10.0, 10.1], volumes=[100, 100])
    assert score_volume_price(df) == 0.0
