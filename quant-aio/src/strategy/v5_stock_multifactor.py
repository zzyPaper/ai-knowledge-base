"""
多因子选股策略 V5 —— Barra CNE5 风格。

因子体系（7 类）：
1. 动量因子（Momentum）：1M/3M/6M 收益率
2. 价值因子（Value）：PE/PB 相对行业均值偏离
3. 质量因子（Quality）：ROE、毛利率稳定性
4. 成长因子（Growth）：营收增速、利润增速
5. 波动因子（Volatility）：历史波动率（低波优先）
6. 流动性因子（Liquidity）：换手率、成交额
7. 技术因子（Technical）：均线趋势、RSI

加权方式：等权（ICIR 加权需要长历史数据，当前数据量不足）

与板块轮动的本质区别：
- 选股范围：全 A（经筛选后 ~500-1000 只） vs 30 个板块
- Alpha 来源：个股特质收益 vs 板块共性收益
- 风险控制：Barra 风格因子暴露控制 vs 简单仓位控制
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.stock_engine import StockScore


# ── 因子计算 ──────────────────────────────────────────

def _calc_momentum(close: pd.Series) -> dict:
    """动量因子：1M/3M/6M 收益率。"""
    result = {}
    for period, label in [(20, "mom_1m"), (60, "mom_3m"), (120, "mom_6m")]:
        if len(close) >= period + 1:
            ret = (close.iloc[-1] / close.iloc[-(period + 1)] - 1) * 100
            result[label] = float(ret) if not np.isnan(ret) else 0.0
        else:
            result[label] = 0.0
    return result


def _calc_volatility(close: pd.Series, window: int = 20) -> float:
    """波动率因子（低波优先）。"""
    if len(close) < window + 1:
        return 999.0  # 数据不足，惩罚
    daily_ret = close.pct_change().dropna().tail(window)
    return float(daily_ret.std() * np.sqrt(252) * 100) if daily_ret.std() > 0 else 0.0


def _calc_technical(close: pd.Series, volume: pd.Series | None = None) -> dict:
    """技术因子：MA 趋势 + RSI。"""
    result = {}

    # MA 趋势：价格与 MA20/MA60 的关系
    if len(close) >= 20:
        ma20 = close.rolling(20).mean().iloc[-1]
        result["ma20_ratio"] = float(close.iloc[-1] / ma20 - 1) if ma20 > 0 else 0.0
    else:
        result["ma20_ratio"] = 0.0

    if len(close) >= 60:
        ma60 = close.rolling(60).mean().iloc[-1]
        result["ma60_ratio"] = float(close.iloc[-1] / ma60 - 1) if ma60 > 0 else 0.0
    else:
        result["ma60_ratio"] = 0.0

    # RSI(14)
    if len(close) >= 15:
        delta = close.diff().dropna().tail(14)
        gain = delta.where(delta > 0, 0).mean()
        loss = (-delta.where(delta < 0, 0)).mean()
        if loss > 0:
            rs = gain / loss
            result["rsi14"] = float(100 - 100 / (1 + rs))
        else:
            result["rsi14"] = 100.0
    else:
        result["rsi14"] = 50.0

    # 量价配合
    if volume is not None and len(volume) >= 20:
        vol_ma5 = volume.rolling(5).mean().iloc[-1]
        vol_ma20 = volume.rolling(20).mean().iloc[-1]
        if vol_ma20 > 0:
            result["vol_ratio"] = float(vol_ma5 / vol_ma20)
        else:
            result["vol_ratio"] = 1.0
    else:
        result["vol_ratio"] = 1.0

    return result


def _calc_liquidity(amount: pd.Series | None, turnover_rate: pd.Series | None) -> dict:
    """流动性因子。"""
    result = {}
    if amount is not None and len(amount) >= 20:
        result["avg_amount_20d"] = float(amount.tail(20).mean())
    else:
        result["avg_amount_20d"] = 0.0

    if turnover_rate is not None and len(turnover_rate) >= 20:
        result["avg_turnover_20d"] = float(turnover_rate.tail(20).mean())
    else:
        result["avg_turnover_20d"] = 0.0

    return result


# ── 因子标准化（截面排名归一化到 [0, 1]）──

def _rank_normalize(series: pd.Series) -> pd.Series:
    """排名归一化。"""
    ranked = series.rank(pct=True)
    return ranked


# ── 主评分函数 ──────────────────────────────────────────

def score_stock(
    code: str,
    name: str,
    industry: str,
    stock_hist: pd.DataFrame,
    index_hist: pd.DataFrame,
) -> StockScore | None:
    """对单只股票打分。

    综合评分 = w_momentum * 动量 + w_volatility * 低波 + w_technical * 技术
    （价值/质量/成长需要财务数据，暂用行情数据替代）

    Returns: StockScore 或 None（不满足买入条件时）
    """
    close = stock_hist["close"]
    volume = stock_hist.get("volume")
    amount = stock_hist.get("amount")
    turnover_rate = stock_hist.get("turnover_rate")
    pct_chg = stock_hist.get("pct_chg")

    if len(close) < 20:
        return None

    # ── 硬性过滤 ──
    # 1. 最近20天有成交
    if amount is not None and amount.tail(20).mean() < 5e7:  # 日均成交 < 5000万
        return None

    # 2. 价格不能太低（< 3元 排除）
    if close.iloc[-1] < 3.0:
        return None

    # 3. 最近5日不能有涨跌停（数据质量问题）
    if pct_chg is not None and len(pct_chg) >= 5:
        recent_pcts = pct_chg.tail(5)
        if (recent_pcts.abs() >= 9.8).any():
            return None  # 排除近期涨跌停

    # ── 因子计算 ──
    factors = {}
    factors.update(_calc_momentum(close))
    factors["volatility"] = _calc_volatility(close)
    factors.update(_calc_technical(close, volume))
    factors.update(_calc_liquidity(amount, turnover_rate))

    # ── 因子评分 ──
    score_parts = {}

    # 1. 动量得分（3M动量为主，1M辅助，6M参考）
    mom_1m = factors.get("mom_1m", 0)
    mom_3m = factors.get("mom_3m", 0)
    mom_6m = factors.get("mom_6m", 0)
    momentum_score = 0.4 * _clip_score(mom_3m, 30) + 0.3 * _clip_score(mom_1m, 20) + 0.3 * _clip_score(mom_6m, 50)
    score_parts["momentum"] = momentum_score

    # 2. 低波得分（波动率越低越好）
    vol = factors.get("volatility", 50)
    volatility_score = max(0, 1 - vol / 60)  # 年化60%以上为0分
    score_parts["volatility"] = volatility_score

    # 3. 技术得分
    ma20_r = factors.get("ma20_ratio", 0)
    ma60_r = factors.get("ma60_ratio", 0)
    rsi = factors.get("rsi14", 50)

    trend_score = 0.0
    if ma20_r > 0:
        trend_score += 0.3
    if ma60_r > 0:
        trend_score += 0.3
    # RSI 在 40-70 之间较好（不超买不超卖）
    if 40 <= rsi <= 70:
        trend_score += 0.4
    elif 30 <= rsi <= 80:
        trend_score += 0.2
    score_parts["technical"] = trend_score

    # 4. 流动性得分（成交额越大越好）
    avg_amt = factors.get("avg_amount_20d", 0)
    if avg_amt >= 1e9:      # 日均 10亿以上
        liquidity_score = 1.0
    elif avg_amt >= 5e8:    # 5亿
        liquidity_score = 0.8
    elif avg_amt >= 2e8:    # 2亿
        liquidity_score = 0.6
    elif avg_amt >= 1e8:    # 1亿
        liquidity_score = 0.4
    elif avg_amt >= 5e7:    # 5000万
        liquidity_score = 0.2
    else:
        liquidity_score = 0.0
    score_parts["liquidity"] = liquidity_score

    # ── 综合评分 ──
    # 权重：动量0.30 + 低波0.20 + 技术0.25 + 流动性0.25
    weights = {
        "momentum": 0.30,
        "volatility": 0.20,
        "technical": 0.25,
        "liquidity": 0.25,
    }
    composite = sum(weights[k] * v for k, v in score_parts.items())
    composite = float(np.clip(composite, -1, 1))

    # 信号与仓位
    if composite > 0.4:
        signal = "BUY"
        position = float(np.clip(0.3 + composite * 0.5, 0.3, 1.0))
    elif composite > 0.2:
        signal = "HOLD"
        position = 0.2
    else:
        signal = "SELL"
        position = 0.0

    # 追高惩罚：1M动量 > 30% → 打折
    if mom_1m > 30:
        composite *= 0.5
        factors["overbought"] = True

    # MA 均线下方 → 仓位减半
    if ma20_r < 0:
        position *= 0.5

    return StockScore(
        code=code,
        name=name,
        industry=industry,
        composite=composite,
        signal=signal,
        position=position,
        factors=factors,
    )


def _clip_score(value: float, max_val: float) -> float:
    """将值归一化到 [0, 1]，正值加分，负值减分。"""
    return float(np.clip(value / max_val, -1, 1))
