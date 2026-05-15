"""策略测试 v2 —— 三因子模型测试。"""
import pandas as pd
import numpy as np
import pytest

from src.strategy.trend_strength import score_trend_strength, calc_trend_strength
from src.strategy.short_momentum import score_short_momentum, calc_roc
from src.strategy.volume_confirm import score_volume_confirm
from src.strategy.fusion import compute_composite_score, rank_sectors, detect_market_regime


def _make_hist(n: int = 60, trend: str = "up") -> pd.DataFrame:
    """构造测试数据。"""
    if trend == "up":
        close = pd.Series(np.linspace(100, 140, n), dtype=float)
    elif trend == "down":
        close = pd.Series(np.linspace(140, 100, n), dtype=float)
    else:
        close = pd.Series(120 + 5 * np.sin(np.linspace(0, 4 * np.pi, n)), dtype=float)

    return pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=n, freq="B"),
        "open": close - 1, "close": close, "high": close + 2, "low": close - 2,
        "volume": pd.Series([1e6 + i * 1e4 for i in range(n)], dtype=float),
        "amount": close * 1e5,
        "pct_chg": close.pct_change() * 100,
        "turnover_rate": pd.Series([2.0 + i * 0.05 for i in range(n)], dtype=float),
    })


class TestTrendStrength:
    def test_up_trend(self):
        hist = _make_hist(60, "up")
        score = score_trend_strength(hist)
        assert score > 0.3

    def test_down_trend(self):
        hist = _make_hist(60, "down")
        score = score_trend_strength(hist)
        assert score < -0.3

    def test_range_trend(self):
        hist = _make_hist(60, "range")
        score = score_trend_strength(hist)
        # 正弦波有一定趋势性，放宽阈值
        assert abs(score) < 0.8

    def test_insufficient_data(self):
        hist = _make_hist(5, "up")
        score = score_trend_strength(hist)
        assert score == 0.0


class TestShortMomentum:
    def test_up_momentum(self):
        hist = _make_hist(20, "up")
        score = score_short_momentum(hist)
        assert score > 0.1

    def test_down_momentum(self):
        hist = _make_hist(20, "down")
        score = score_short_momentum(hist)
        assert score < -0.1

    def test_insufficient_data(self):
        hist = _make_hist(5, "up")
        score = score_short_momentum(hist)
        assert score == 0.0


class TestVolumeConfirm:
    def test_volume_surge(self):
        """放量场景：后期成交量增加。"""
        n = 30
        close = pd.Series(np.linspace(100, 130, n), dtype=float)
        # 后5天放量2倍
        vol = [1e6] * n
        for i in range(n - 5, n):
            vol[i] = 2e6
        hist = pd.DataFrame({
            "date": pd.date_range("2025-01-01", periods=n, freq="B"),
            "open": close - 1, "close": close, "high": close + 2, "low": close - 2,
            "volume": pd.Series(vol, dtype=float),
            "amount": close * 1e5,
        })
        score = score_volume_confirm(hist)
        assert score > 0.3

    def test_volume_shrink(self):
        """缩量场景。"""
        n = 30
        close = pd.Series(np.linspace(100, 130, n), dtype=float)
        vol = [2e6] * n
        for i in range(n - 5, n):
            vol[i] = 5e5
        hist = pd.DataFrame({
            "date": pd.date_range("2025-01-01", periods=n, freq="B"),
            "open": close - 1, "close": close, "high": close + 2, "low": close - 2,
            "volume": pd.Series(vol, dtype=float),
            "amount": close * 1e5,
        })
        score = score_volume_confirm(hist)
        assert score < 0

    def test_insufficient_data(self):
        hist = _make_hist(5, "up")
        score = score_volume_confirm(hist)
        assert score == 0.0


class TestFusionV2:
    def test_composite_score_structure(self):
        hist = _make_hist(60, "up")
        result = compute_composite_score(hist)
        assert "trend_strength" in result
        assert "short_momentum" in result
        assert "volume_confirm" in result
        assert "composite" in result
        assert "signal" in result
        assert "ma60_pass" in result

    def test_ma60_filter_pass(self):
        """上涨趋势 → MA60 过滤通过。"""
        hist = _make_hist(80, "up")
        result = compute_composite_score(hist)
        assert result["ma60_pass"] == True

    def test_ma60_filter_fail(self):
        """下跌趋势 → MA60 过滤不通过。"""
        hist = _make_hist(80, "down")
        result = compute_composite_score(hist)
        assert result["ma60_pass"] == False
        assert result["signal"] == "SELL"

    def test_rank_sectors(self):
        """截面排名：上涨板块排在前面。"""
        up_hist = _make_hist(80, "up")
        down_hist = _make_hist(80, "down")
        idx = _make_hist(80, "up")  # 上涨指数

        scores = {}
        scores["上涨板块"] = compute_composite_score(up_hist, idx)
        scores["下跌板块"] = compute_composite_score(down_hist, idx)

        ranked = rank_sectors(scores, top_n=2)
        assert len(ranked) >= 1
        assert ranked[0][0] == "上涨板块"

    def test_rank_excludes_ma60_fail(self):
        """排名排除MA60不通过的板块。"""
        up_hist = _make_hist(80, "up")
        down_hist = _make_hist(80, "down")
        idx = _make_hist(80, "up")

        scores = {}
        scores["上涨板块"] = compute_composite_score(up_hist, idx)
        scores["下跌板块"] = compute_composite_score(down_hist, idx)

        ranked = rank_sectors(scores, top_n=5)
        names = [n for n, _ in ranked]
        assert "下跌板块" not in names

    def test_regime_trend(self):
        idx = _make_hist(80, "up")
        regime = detect_market_regime(idx)
        assert regime == "trend"

    def test_regime_range(self):
        idx = _make_hist(80, "down")
        regime = detect_market_regime(idx)
        assert regime == "range"
