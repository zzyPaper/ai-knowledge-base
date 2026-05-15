"""Tests for signal fusion."""

import pandas as pd
import pytest
from src.signals.fusion import detect_market_regime, compute_sector_scores


def _make_sector_df(start_price: float, trend: float = 0, n: int = 20) -> pd.DataFrame:
    prices = [start_price + trend * i for i in range(n)]
    return pd.DataFrame({
        "close": prices,
        "open": prices,
        "high": [p * 1.02 for p in prices],
        "low": [p * 0.98 for p in prices],
        "volume": [1000] * n,
        "amount": [p * 1000 for p in prices],
        "turnover_rate": [1.0] * n,
        "date": pd.date_range("2025-01-01", periods=n),
    })


def test_detect_trending_market():
    """Deviation > 2% should be trending."""
    hist = pd.DataFrame({
        "close": [100 + i for i in range(30)],  # clear uptrend
        "date": pd.date_range("2025-01-01", periods=30),
    })
    assert detect_market_regime(hist) == "trending"


def test_detect_ranging_market():
    """Deviation < 2% should be ranging."""
    hist = pd.DataFrame({
        "close": [100.0] * 30,  # flat
        "date": pd.date_range("2025-01-01", periods=30),
    })
    assert detect_market_regime(hist) == "ranging"


def test_compute_scores_output_shape():
    """Output should have correct columns and be ranked."""
    data = {
        "sector_a": _make_sector_df(100, 1),
        "sector_b": _make_sector_df(100, -0.5),
        "sector_c": _make_sector_df(100, 0),
    }
    result = compute_sector_scores(data)
    assert list(result.columns) == ["sector", "momentum", "vp", "congestion", "composite", "rank"]
    assert len(result) == 3
    assert result["rank"].tolist() == [1, 2, 3]


def test_compute_scores_normalized():
    """All signal columns should be in [0, 1]."""
    data = {
        "a": _make_sector_df(100, 1),
        "b": _make_sector_df(100, 0),
    }
    result = compute_sector_scores(data)
    for col in ["momentum", "vp", "congestion", "composite"]:
        assert result[col].between(0, 1).all(), f"{col} not in [0, 1]"


def test_compute_scores_with_index():
    """Should accept index_hist without error."""
    data = {"a": _make_sector_df(100, 1), "b": _make_sector_df(100, 0)}
    index_hist = pd.DataFrame({
        "close": [100 + i * 0.5 for i in range(20)],
        "date": pd.date_range("2025-01-01", periods=20),
    })
    result = compute_sector_scores(data, index_hist)
    assert len(result) == 2
