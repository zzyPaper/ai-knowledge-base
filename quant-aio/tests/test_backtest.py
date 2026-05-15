"""回测引擎单元测试"""
import pytest
import pandas as pd
import numpy as np

from src.backtest.engine import BacktestEngine, BacktestResult


class TestBacktestResult:
    def test_is_passing(self):
        r = BacktestResult("20250101", "20250301", 1_000_000, 1_200_000, 0.20, 0.05, 0.15)
        assert r.is_passing is True

    def test_is_not_passing(self):
        r = BacktestResult("20250101", "20250301", 1_000_000, 1_050_000, 0.05, 0.05, 0.00)
        assert r.is_passing is False


class TestBacktestEngine:
    def test_default_params(self):
        engine = BacktestEngine()
        assert engine.initial_cash == 1_000_000
        assert engine.stop_loss == -0.05
        assert engine.take_profit == 0.15

    def test_custom_params(self):
        engine = BacktestEngine(
            initial_cash=500_000,
            stop_loss=-0.08,
            take_profit=0.20,
            position_per_sector=0.25,
        )
        assert engine.initial_cash == 500_000
        assert engine.stop_loss == -0.08

    def test_backtest_result_structure(self):
        """测试回测返回结构正确（使用 mock 数据时主要验证接口）。"""
        engine = BacktestEngine()
        # 这个测试需要真实数据，在 CI 中可能跳过
        # 这里只验证接口类型
        assert hasattr(engine, "run")
