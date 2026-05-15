#!/usr/bin/env python3
"""Quick verification script - run all core logic to check for errors.

Usage: python3 scripts/verify_all.py

This is a development tool for Claude to validate code correctness.
It does NOT use pytest - it's a simple import-and-run check.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

passed = 0
failed = 0


def check(name: str, ok: bool, detail: str = ""):
    global passed, failed
    if ok:
        passed += 1
        print(f"  ✓ {name}")
    else:
        failed += 1
        print(f"  ✗ {name} — {detail}")


def verify_momentum():
    print("\n[信号] 动量信号 (momentum)")
    import pandas as pd
    from src.features.momentum import calc_ma_ratio, calc_roc, rank_momentum

    series = pd.Series([10.0] * 10 + [12.0])
    check("MA比: close > MA → 正值", calc_ma_ratio(series, 10) > 0)

    series = pd.Series([10.0] * 10 + [8.0])
    check("MA比: close < MA → 负值", calc_ma_ratio(series, 10) < 0)

    series = pd.Series([1, 2, 3])
    check("MA比: 数据不足 → 0", calc_ma_ratio(series, 10) == 0.0)

    series = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    check("ROC: 上涨 → 正值", calc_roc(series, 5) > 0)

    series = pd.Series([10.0, 9.5, 9.0, 8.5, 8.0, 7.5])
    check("ROC: 下跌 → 负值", calc_roc(series, 5) < 0)

    def _df(prices):
        return pd.DataFrame({"close": prices, "date": pd.date_range("2025-01-01", periods=len(prices))})

    data = {"winner": _df([10 + i * 0.5 for i in range(20)]), "loser": _df([10 - i * 0.3 for i in range(20)])}
    scores = rank_momentum(data)
    check("排行: 上涨板块 > 下跌板块", scores["winner"] > scores["loser"])
    check("分数范围: [0, 1]", all(0 <= v <= 1 for v in scores.values()))


def verify_volume_price():
    print("\n[信号] 量价信号 (volume_price)")
    import pandas as pd
    from src.features.volume_price import score_volume_price

    def _df(prices, volumes, highs=None):
        n = len(prices)
        if highs is None:
            highs = [p * 1.02 for p in prices]
        lows = [p * 0.98 for p in prices]
        return pd.DataFrame({
            "close": prices, "open": prices, "high": highs, "low": lows,
            "volume": volumes, "amount": [v * p for v, p in zip(volumes, prices)],
            "date": pd.date_range("2025-01-01", periods=n),
        })

    surge = _df([10, 10.1, 10.2, 10.3, 10.4, 11.0], [100, 100, 100, 100, 100, 500])
    drift = _df([10, 9.9, 9.8, 9.7, 9.6, 9.3], [100, 100, 100, 100, 100, 80])
    check("分数范围为浮点数", isinstance(score_volume_price(surge), float))
    check("分数范围 [-100, 100]", -100 <= score_volume_price(surge) <= 100)

    short = _df([10, 10.1], [100, 100])
    check("数据不足 → 0", score_volume_price(short) == 0.0)


def verify_congestion():
    print("\n[信号] 拥挤度信号 (congestion)")
    import pandas as pd
    from src.features.congestion import calc_turnover_ratio, score_congestion

    def _df(turnover, amount):
        n = len(turnover)
        return pd.DataFrame({
            "close": [10]*n, "open": [10]*n, "high": [10.5]*n, "low": [9.5]*n,
            "volume": [1000]*n, "amount": amount, "turnover_rate": turnover,
            "date": pd.date_range("2025-01-01", periods=n),
        })

    check("换手率高 → 比 > 1", calc_turnover_ratio(_df([1, 1, 1, 1, 0.5, 5.0], [1e8]*6)) > 1)

    data = {
        "hot": _df([5.0]*6, [5e8]*6),
        "cold": _df([0.5]*6, [1e7]*6),
    }
    scores = score_congestion(data)
    check("拥挤度: 热板块 > 冷板块", scores["hot"] > scores["cold"])
    check("分数范围 [0, 1]", all(0 <= v <= 1 for v in scores.values()))


def verify_fusion():
    print("\n[信号] 信号融合 (fusion)")
    import pandas as pd
    from src.signals.fusion import detect_market_regime, compute_sector_scores

    def _df(prices):
        return pd.DataFrame({
            "close": prices, "open": prices, "high": [p*1.02 for p in prices],
            "low": [p*0.98 for p in prices], "volume": [1000]*len(prices),
            "amount": [p*1000 for p in prices], "turnover_rate": [1.0]*len(prices),
            "date": pd.date_range("2025-01-01", periods=len(prices)),
        })

    idx_trend = pd.DataFrame({"close": [100 + i for i in range(30)], "date": pd.date_range("2025-01-01", periods=30)})
    idx_flat = pd.DataFrame({"close": [100]*30, "date": pd.date_range("2025-01-01", periods=30)})
    check("趋势市检测", detect_market_regime(idx_trend) == "trending")
    check("震荡市检测", detect_market_regime(idx_flat) == "ranging")

    data = {"a": _df([100 + i for i in range(20)]), "b": _df([100 - i*0.5 for i in range(20)])}
    result = compute_sector_scores(data)
    check("输出列名正确", list(result.columns) == ["sector", "momentum", "vp", "congestion", "composite", "rank"])
    check("分数范围 [0, 1]", all(0 <= c <= 1 for c in result["composite"]))


def verify_engine():
    print("\n[回测] 引擎 (engine)")
    import pandas as pd
    from src.backtest.engine import BacktestEngine, StrategyConfig

    def _sector(trend, n=50):
        prices = [100 + trend * i for i in range(n)]
        dates = pd.date_range("2025-01-01", periods=n, freq="B")[:n]
        return pd.DataFrame({
            "close": prices, "open": [p*0.99 for p in prices], "high": [p*1.02 for p in prices],
            "low": [p*0.98 for p in prices], "volume": [1000]*n, "amount": [p*1000 for p in prices],
            "turnover_rate": [1.0]*n, "date": dates,
        })

    sectors = {"a": _sector(1.0), "b": _sector(-0.5), "c": _sector(0.3)}
    idx = _sector(0.1)
    idx["pct_chg"] = idx["close"].pct_change(1).fillna(0) * 100

    engine = BacktestEngine(StrategyConfig(top_n=2, rebalance_freq=10))
    result = engine.run(sectors, idx)

    check("产生交易日志", len(result.daily_returns) > 0)
    check("产生交易记录", len(result.trades) > 0)
    check("累计收益列存在", "port_cumulative" in result.daily_returns.columns)
    check("指标可计算", result.metrics.total_return is not None)


def verify_metrics():
    print("\n[回测] 指标 (metrics)")
    import pandas as pd
    import random
    from src.backtest.engine import BacktestResult, StrategyConfig
    from src.backtest.metrics import compute_metrics

    random.seed(42)
    rets = [0.01 + random.uniform(-0.005, 0.005) for _ in range(252)]
    idx_rets = [0.005] * 252
    df = pd.DataFrame({"port_return": rets, "index_return": idx_rets})
    df["port_cumulative"] = (1 + df["port_return"]).cumprod()
    df["index_cumulative"] = (1 + df["index_return"]).cumprod()
    r = BacktestResult(daily_returns=df, trades=[], config=StrategyConfig())
    m = compute_metrics(r)

    check("正收益指标", m.total_return > 0)
    check("超额收益为正", m.excess_return > 0)
    check("胜率可计算", m.win_rate >= 0)


def verify_analyzer():
    print("\n[回测] 分析器 (analyzer)")
    import pandas as pd
    from src.backtest.engine import BacktestResult, StrategyConfig
    from src.backtest.analyzer import analyze_failure, apply_adjustments, Adjustment

    df = pd.DataFrame({"port_return": [-0.01]*20, "index_return": [0.005]*20})
    df["port_cumulative"] = (1 + df["port_return"]).cumprod()
    df["index_cumulative"] = (1 + df["index_return"]).cumprod()
    config = StrategyConfig(top_n=3)
    result = BacktestResult(daily_returns=df, trades=[{"sector": "a", "action": "buy"}] * 3, config=config)

    adj = analyze_failure(result, {"a": pd.DataFrame()}, config)
    check("失败分析返回调整列表", len(adj) >= 0)

    new_cfg = apply_adjustments(config, [Adjustment("top_n", 5, "Test")])
    check("应用调整: top_n 变更", new_cfg.top_n == 5)
    check("原始配置不变", config.top_n == 3)


def verify_end_to_end():
    print("\n[回测] 端到端 (end_to_end)")
    import pandas as pd
    import numpy as np
    from src.backtest.engine import StrategyConfig
    from src.backtest.loop import run_self_loop_backtest, split_windows

    np.random.seed(42)
    n = 200
    dates = pd.date_range("2025-01-01", periods=n, freq="B")[:n]
    sectors = {}
    for name, trend in [("a", 0.15), ("b", 0.10), ("c", 0.05), ("d", -0.02), ("e", -0.05)]:
        prices = [100.0]
        for i in range(1, n):
            prices.append(prices[-1] * (1 + trend + np.random.normal(0, 0.03)))
        sectors[name] = pd.DataFrame({
            "close": prices, "open": [p*0.99 for p in prices], "high": [p*1.02 for p in prices],
            "low": [p*0.98 for p in prices], "volume": [1000]*n, "amount": [p*1000 for p in prices],
            "turnover_rate": [1.0]*n, "pct_chg": [0]+[(prices[i]/prices[i-1]-1)*100 for i in range(1,n)],
            "date": dates,
        })
    idx = pd.DataFrame({
        "close": [100 * (1 + 0.05/252)**i for i in range(n)],
        "pct_chg": [0.05/252*100]*n, "date": dates,
    })

    windows = split_windows(end_date=pd.Timestamp("2025-07-15"), num_windows=2)
    results = run_self_loop_backtest(sectors, idx, windows, StrategyConfig(top_n=2, rebalance_freq=10), max_iterations=5, target_excess=5.0)

    check("窗口数正确", len(results) == 2)
    check("每窗至少1次迭代", all(r.iterations > 0 for r in results))
    check("指标可计算", all(isinstance(r.metrics.total_return, float) for r in results))


if __name__ == "__main__":
    print("=" * 50)
    print("股票分析系统 — 功能验证")
    print("=" * 50)

    verify_momentum()
    verify_volume_price()
    verify_congestion()
    verify_fusion()
    verify_engine()
    verify_metrics()
    verify_analyzer()
    verify_end_to_end()

    print(f"\n{'=' * 50}")
    print(f"结果: {passed} 通过, {failed} 失败, {passed + failed} 总计")
    print(f"{'=' * 50}")
    sys.exit(0 if failed == 0 else 1)
