"""
策略V2 —— 七因子评分 + 市势自适应权重 + 截面排名轮动。

v3.0 根本性重构：
─────────────────────────────────────────
1. 历史因子缓存：外部因子（资金/情绪/估值）从预下载的parquet读取，
   不再依赖实时API → 回测中7因子真正可用
2. Bear市场状态：指数<MA60且近20日跌幅>8% → bear → 空仓
3. ATR动态止损：止损=买入价-2×ATR，追踪止盈=最高价回撤2×ATR
4. 景气度否决弱化：一票否决(×0.3) → 软扣分(×0.8)

因子体系（7因子专业模型）：
─────────────────────────────────────────
核心因子（决定评分）：
1. 趋势强度: 对数价格线性回归斜率 × R²
2. 多窗口动量: 1月/3月/6月/12月 ROC 共振
3. 量能确认: ln(5日均量 / 20日均量)
4. 资金因子: 北向资金 + 主力资金流（从历史缓存读取）
5. 情绪因子: 市场宽度（从历史缓存读取）
6. 景气度: 行业PE+资金流推算（弱化否决权）
7. 估值因子: PE分位仓位调节（从历史缓存读取）
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yaml
from pathlib import Path

from src.strategy.base import BaseStrategy, SectorScore
from src.strategy.trend_strength import score_trend_strength
from src.strategy.short_momentum import score_short_momentum
from src.strategy.volume_confirm import score_volume_confirm
from src.strategy.multi_window_momentum import score_multi_window_momentum, get_momentum_detail


def _load_params() -> dict:
    params_path = Path(__file__).resolve().parent.parent.parent / "config" / "strategy_params.yaml"
    if params_path.exists():
        with open(params_path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


class V2ThreeFactor(BaseStrategy):
    """策略V2 v3.0：七因子+历史缓存+Bear识别+ATR止损。"""

    name = "three_factor"
    version = "3.0"
    description = "7因子+历史因子缓存+Bear空仓+ATR动态止损"

    def __init__(self):
        self._params = _load_params()
        # 回测日期范围（由BacktestEngine设置）
        self._backtest_start = None
        self._backtest_end = None

    @property
    def params(self) -> dict:
        return self._params

    def set_backtest_range(self, start: str, end: str):
        """设置回测日期范围，用于历史因子缓存。"""
        self._backtest_start = start
        self._backtest_end = end

    def detect_regime(self, index_hist: pd.DataFrame) -> str:
        """检测市场状态：trend / range / bear。

        v3.0 新增 bear 状态：
        - 指数低于MA60且近20日跌幅>8% → bear（系统性风险，空仓）
        - 指数在MA60上方且近期上涨 → trend（持仓不动，放大动量）
        - 其他 → range（正常轮动）
        """
        if index_hist is None or len(index_hist) < 5:
            return "range"
        close = index_hist["close"]
        n = len(close)

        # MA60（数据不足时用全部数据）
        ma60_period = min(60, n)
        ma60 = close.rolling(ma60_period).mean()
        ma60_val = ma60.iloc[-1] if not np.isnan(ma60.iloc[-1]) else 0

        # MA20
        ma20_period = min(20, n)
        ma20 = close.rolling(ma20_period).mean()
        ma20_val = ma20.iloc[-1] if not np.isnan(ma20.iloc[-1]) else 0

        current = close.iloc[-1]

        # 近20日跌幅
        lookback = min(20, n)
        ret_20d = (current / close.iloc[-lookback] - 1) if n >= lookback else 0

        # 近5日动量
        lookback5 = min(5, n)
        ret_5d = (current / close.iloc[-lookback5] - 1) if n >= lookback5 else 0

        # ── Bear判定：指数<MA60 且 近20日跌幅>5% ──
        if ma60_val > 0 and current < ma60_val and ret_20d < -0.05:
            return "bear"

        # ── Bear预警：指数<MA20 且 近5日跌幅>2% ──
        if ma20_val > 0 and current < ma20_val and ret_5d < -0.02:
            return "bear_warning"

        # ── Trend判定：指数>MA60 且 近期有上涨 ──
        if ma60_val > 0:
            deviation = (current - ma60_val) / ma60_val
        else:
            deviation = 0

        if deviation > -0.01 and (ret_20d > 0.01 or ret_5d > 0.005):
            return "trend"

        return "range"

    def _get_historical_factor_scores(
        self,
        sector_name: str,
        date_str: str,
        sector_hist: pd.DataFrame = None,
        index_hist: pd.DataFrame = None,
    ) -> dict:
        """计算外部因子得分（回测用）。

        v3.0 策略：优先从历史缓存读取，缓存不可用时用K线衍生指标替代。

        替代逻辑（数据源不可用时的降级方案）：
        - 资金因子 → 缓存优先；不可用时用"指数涨跌幅×成交量"替代
          （指数上涨+放量 = 资金流入信号）
        - 情绪因子 → 缓存优先；不可用时用"指数涨跌家数推算"替代
          （用指数涨跌幅推算市场宽度）
        - 估值PE → 缓存优先；不可用时返回0.5（中性）

        Returns
        -------
        dict: {capital_flow, sentiment, valuation_pe_pct}
        """
        result = {
            "capital_flow": 0.0,
            "sentiment": 0.0,
            "valuation_pe_pct": 0.5,
        }

        # ── 尝试从历史缓存读取 ──
        capital_cache_hit = False
        sentiment_cache_hit = False
        valuation_cache_hit = False

        try:
            from src.data.factor_cache import (
                get_market_fund_flow_cached,
                get_market_breadth_cached,
                get_index_pe_percentile_cached,
            )

            start = self._backtest_start
            end = self._backtest_end

            # 市场主力资金因子
            main_score = get_market_fund_flow_cached(date_str, lookback=5, start_date=start, end_date=end)
            if abs(main_score) > 0.001:
                result["capital_flow"] = main_score
                capital_cache_hit = True

            # 市场宽度
            breadth_score = get_market_breadth_cached(date_str, start_date=start, end_date=end)
            if abs(breadth_score) > 0.001:
                result["sentiment"] = breadth_score
                sentiment_cache_hit = True

            # 估值PE分位
            pe_pct = get_index_pe_percentile_cached(date_str, start_date=start, end_date=end)
            if abs(pe_pct - 0.5) > 0.01:
                result["valuation_pe_pct"] = pe_pct
                valuation_cache_hit = True

        except Exception:
            pass

        # ── 缓存不可用的因子：用K线衍生指标替代 ──
        if (not capital_cache_hit or not sentiment_cache_hit) and index_hist is not None and not index_hist.empty:
            close = index_hist["close"]
            n = len(close)

            if n >= 5:
                ret_5d = (close.iloc[-1] / close.iloc[-5] - 1)

                # 资金因子替代：指数涨跌幅 × 放量程度
                if not capital_cache_hit:
                    vol_5d = 0.0
                    if "volume" in index_hist.columns:
                        vol_recent = index_hist["volume"].iloc[-5:].mean()
                        vol_earlier = index_hist["volume"].iloc[-20:-5].mean() if n >= 20 else vol_recent
                        vol_5d = (vol_recent / vol_earlier - 1) if vol_earlier > 0 else 0
                    capital_proxy = np.clip(ret_5d * 10 + vol_5d * 5, -1, 1)
                    result["capital_flow"] = float(capital_proxy)

                # 情绪因子替代：指数涨跌幅推算市场宽度
                if not sentiment_cache_hit:
                    advance_ratio = 0.5 + ret_5d * 10
                    advance_ratio = np.clip(advance_ratio, 0, 1)
                    sentiment_proxy = (advance_ratio - 0.5) * 4.0
                    result["sentiment"] = float(np.clip(sentiment_proxy, -1, 1))

        # ── 实盘运行时fallback到实时API ──
        if self._backtest_start is None:
            try:
                from src.strategy.capital_flow import score_capital_factor
                result["capital_flow"] = score_capital_factor(sector_name) if sector_name else 0.0
            except Exception:
                pass

            try:
                from src.strategy.sentiment import score_sentiment_factor
                result["sentiment"] = score_sentiment_factor()
            except Exception:
                pass

        return result

    def score_sector(
        self,
        sector_hist: pd.DataFrame,
        index_hist: pd.DataFrame | None = None,
        regime: str | None = None,
    ) -> SectorScore:
        """对单个板块评分（7因子模型 v3.0）。"""
        params = self._params.get("v2", self._params)

        if regime is None:
            regime = self.detect_regime(index_hist) if index_hist is not None else "range"

        # ── Bear状态：直接返回极低分 ──
        if regime == "bear":
            return SectorScore(
                sector="",
                composite=-1.0,
                signal="SELL",
                position=0.0,
                factors={"regime": "bear", "reason": "系统性风险，空仓"},
                regime="bear",
            )

        # ── 读取基础权重 ──
        base_weights = params.get("weights", {
            "trend_strength": 0.25,
            "multi_window_momentum": 0.25,
            "volume_confirm": 0.15,
            "capital_flow": 0.15,
            "sentiment": 0.10,
            "fundamental": 0.05,
            "valuation": 0.05,
        })
        if not base_weights or abs(sum(base_weights.values()) - 1.0) > 0.05:
            base_weights = {
                "trend_strength": 0.25,
                "multi_window_momentum": 0.25,
                "volume_confirm": 0.15,
                "capital_flow": 0.15,
                "sentiment": 0.10,
                "fundamental": 0.05,
                "valuation": 0.05,
            }

        # ── 计算核心因子（纯K线） ──
        trend_score = score_trend_strength(sector_hist)
        momentum_score = score_multi_window_momentum(sector_hist)
        momentum_detail = get_momentum_detail(sector_hist)
        volume_score = score_volume_confirm(sector_hist)
        short_mom_score = score_short_momentum(sector_hist)

        # ── 计算外部因子（从历史缓存读取） ──
        sector_name = getattr(sector_hist, '_sector_name', '')
        current_date = ""
        if sector_hist is not None and not sector_hist.empty and "date" in sector_hist.columns:
            current_date = str(sector_hist["date"].iloc[-1])[:10].replace("-", "")

        hist_factors = self._get_historical_factor_scores(sector_name, current_date, sector_hist, index_hist)
        capital_score = hist_factors["capital_flow"]
        sentiment_score = hist_factors["sentiment"]
        valuation_pe_pct = hist_factors["valuation_pe_pct"]

        # 景气度：简化为基于资金流的判断（不再依赖不可靠的行业PE API）
        fundamental_score = 0.0
        fundamental_pass = True
        if capital_score > 0.2:
            fundamental_score = 0.3  # 资金流入→景气改善
            fundamental_pass = True
        elif capital_score < -0.3:
            fundamental_score = -0.3  # 资金流出→景气恶化
            fundamental_pass = True  # 不再一票否决，只扣分

        # 估值因子得分（基于PE分位）
        if valuation_pe_pct < 0.2:
            valuation_score = 0.8
        elif valuation_pe_pct < 0.4:
            valuation_score = 0.4
        elif valuation_pe_pct < 0.6:
            valuation_score = 0.0
        elif valuation_pe_pct < 0.8:
            valuation_score = -0.3
        else:
            valuation_score = -0.7

        all_scores = {
            "trend_strength": trend_score,
            "multi_window_momentum": momentum_score,
            "volume_confirm": volume_score,
            "capital_flow": capital_score,
            "sentiment": sentiment_score,
            "fundamental": fundamental_score,
            "valuation": valuation_score,
            "short_momentum": short_mom_score,
        }

        # ── 市势自适应权重 ──
        weights = dict(base_weights)
        if regime == "trend":
            weights["multi_window_momentum"] = weights.get("multi_window_momentum", 0.25) + 0.08
            weights["trend_strength"] = weights.get("trend_strength", 0.25) + 0.05
            weights["volume_confirm"] = max(weights.get("volume_confirm", 0.15) - 0.08, 0.03)
            weights["valuation"] = max(weights.get("valuation", 0.05) - 0.03, 0.02)
        elif regime == "bear_warning":
            # Bear预警：放大风控，缩小动量
            weights["volume_confirm"] = weights.get("volume_confirm", 0.15) + 0.05
            weights["valuation"] = weights.get("valuation", 0.05) + 0.05
            weights["capital_flow"] = weights.get("capital_flow", 0.15) + 0.05
            weights["multi_window_momentum"] = max(weights.get("multi_window_momentum", 0.25) - 0.08, 0.10)
        else:  # range
            weights["volume_confirm"] = weights.get("volume_confirm", 0.15) + 0.03
            weights["valuation"] = weights.get("valuation", 0.05) + 0.03

        # ── 外部因子缺失自动重分配 ──
        external_factors = ["capital_flow", "sentiment", "fundamental"]
        kline_factors = ["trend_strength", "multi_window_momentum", "volume_confirm"]
        missing_weight = 0.0
        for f in external_factors:
            if abs(all_scores.get(f, 0.0)) < 0.01:
                missing_weight += weights.get(f, 0.0)
                weights[f] = 0.0

        if missing_weight > 0:
            kline_total = sum(weights.get(f, 0.0) for f in kline_factors)
            if kline_total > 0:
                for f in kline_factors:
                    ratio = weights.get(f, 0.0) / kline_total
                    weights[f] = weights.get(f, 0.0) + missing_weight * ratio

        # 归一化
        wsum = sum(weights.values())
        if wsum > 0:
            weights = {k: v / wsum for k, v in weights.items()}

        # ── 估值因子特殊处理：仓位调节器 ──
        valuation_weight = weights.pop("valuation", 0.0)
        if valuation_weight > 0:
            remaining = [k for k in weights.keys()]
            rem_total = sum(weights.get(k, 0.0) for k in remaining)
            if rem_total > 0:
                for k in remaining:
                    weights[k] = weights.get(k, 0.0) + valuation_weight * (weights.get(k, 0.0) / rem_total)
            wsum2 = sum(weights.values())
            if wsum2 > 0:
                weights = {k: v / wsum2 for k, v in weights.items()}

        # ── 加权综合评分 ──
        scoring_factors = {k: v for k, v in all_scores.items() if k in weights and k != "short_momentum"}
        composite = sum(scoring_factors[k] * weights.get(k, 0) for k in scoring_factors)
        composite = float(np.clip(composite, -1, 1))

        # ── 趋势-动量一致性过滤 ──
        consistency_penalty = 0.3 if regime != "trend" else 0.6
        if trend_score < -0.3 and momentum_score > 0:
            scoring_factors["multi_window_momentum"] *= consistency_penalty
            composite = sum(scoring_factors[k] * weights.get(k, 0) for k in scoring_factors)
            composite = float(np.clip(composite, -1, 1))
        elif trend_score > 0.3 and momentum_score < 0:
            scoring_factors["multi_window_momentum"] *= min(consistency_penalty * 2, 0.8)
            composite = sum(scoring_factors[k] * weights.get(k, 0) for k in scoring_factors)
            composite = float(np.clip(composite, -1, 1))

        # ── 景气度扣分（弱化：×0.8而非×0.3一票否决） ──
        if fundamental_score < -0.2:
            composite *= 0.8  # 软扣分，不再一票否决

        # ── 反转检测（v3.0新增）──
        # 前期涨很多但近5日开始回落 → 动量衰竭 → 大幅扣分
        reversal_score = 0.0
        if len(sector_hist) >= 10:
            close = sector_hist["close"]
            # 近3月涨幅（中短期动量）
            lookback_3m = min(60, len(close))
            ret_3m = (close.iloc[-1] / close.iloc[-lookback_3m] - 1)
            # 近5日涨幅（短期动量）
            ret_5d = (close.iloc[-1] / close.iloc[-5] - 1) if len(close) >= 5 else 0

            # 反转信号：中短期涨>15%但近5日跌>3% → 动量衰竭
            if ret_3m > 0.15 and ret_5d < -0.03:
                reversal_score = -0.5  # 强反转信号
            elif ret_3m > 0.10 and ret_5d < -0.02:
                reversal_score = -0.3  # 中等反转信号
            elif ret_3m > 0.05 and ret_5d < -0.01:
                reversal_score = -0.1  # 弱反转信号

            # 反转扣分
            if reversal_score < 0:
                composite += reversal_score * 0.5  # 反转信号权重50%
                composite = float(np.clip(composite, -1, 1))

        # ── 相对强度检测（v3.0新增）──
        # 板块近20日涨幅 vs 指数近20日涨幅
        # 如果板块严重落后于指数 → 不值得持有
        relative_strength = 0.0
        if index_hist is not None and len(index_hist) >= 20 and len(sector_hist) >= 20:
            idx_close = index_hist["close"]
            sec_close = sector_hist["close"]
            lookback = min(20, len(idx_close), len(sec_close))
            idx_ret = (idx_close.iloc[-1] / idx_close.iloc[-lookback] - 1)
            sec_ret = (sec_close.iloc[-1] / sec_close.iloc[-lookback] - 1)
            relative_strength = sec_ret - idx_ret  # 正值=跑赢大盘

            # 相对强度扣分：落后大盘>5% → 大幅扣分
            if relative_strength < -0.05:
                composite += relative_strength * 2.0  # -5%落后 → -0.1扣分
                composite = float(np.clip(composite, -1, 1))
            # 相对强度加分：跑赢大盘>3% → 小幅加分
            elif relative_strength > 0.03:
                composite += 0.05  # 跑赢大盘加分
                composite = float(np.clip(composite, -1, 1))

        # ── MA60 检查 ──
        ma60_period = params.get("ma60_period", 60)
        ma60_pass = self._check_ma60(sector_hist, ma60_period)

        # ── ATR 波动率 ──
        atr_pct = self._calc_atr_pct(sector_hist, period=20)
        atr_abs = self._calc_atr(sector_hist, period=20)

        # ── 信号判定 ──
        buy_threshold = params.get("buy_threshold", 0.10)
        sell_threshold = params.get("sell_threshold", -0.10)
        if regime == "trend":
            buy_threshold = max(buy_threshold * 0.5, 0.03)
            sell_threshold = sell_threshold * 1.5
        elif regime == "bear_warning":
            buy_threshold = buy_threshold * 1.5  # 提高买入门槛
            sell_threshold = sell_threshold * 0.5  # 降低卖出门槛

        if composite > buy_threshold:
            signal = "BUY"
            position = float(np.clip(0.3 + composite * 0.7, 0.3, 1.0))
        elif composite < sell_threshold:
            signal = "SELL"
            position = 0.0
        else:
            signal = "HOLD"
            position = float(np.clip(0.1 + composite * 0.3, 0.0, 0.3))

        # ── 仓位调节（风控软着陆 v3.0） ──
        position_base = position
        risk_discounts = []

        # Bear预警：仓位上限30%
        if regime == "bear_warning":
            position = min(position, 0.3)
            risk_discounts.append(("bear_warning", 0.3))

        # MA60 软调节
        if not ma60_pass:
            risk_discounts.append(("ma60", 0.7 if regime != "trend" else 0.85))

        # ATR 波动率仓位调节
        atr_baseline = params.get("atr_baseline", 0.03)
        if atr_pct > 0:
            vol_multiplier = min(atr_baseline / atr_pct, 1.0)
            vol_multiplier = max(vol_multiplier, 0.3)
            risk_discounts.append(("atr", vol_multiplier))

        # 尖峰检测
        tr_atr_ratio = 0.0
        if atr_abs > 0 and len(sector_hist) >= 2:
            latest_high = sector_hist["high"].iloc[-1]
            latest_low = sector_hist["low"].iloc[-1]
            prev_close = sector_hist["close"].iloc[-2]
            latest_tr = max(
                latest_high - latest_low,
                abs(latest_high - prev_close),
                abs(latest_low - prev_close),
            )
            tr_atr_ratio = latest_tr / atr_abs
            if tr_atr_ratio > 2.0:
                spike_discount = min(1.0 / tr_atr_ratio, 0.5)
                risk_discounts.append(("spike", spike_discount))

        # 估值仓位调节
        if valuation_score != 0:
            val_multiplier = 1.0 + valuation_score * 0.5
            val_multiplier = float(np.clip(val_multiplier, 0.5, 1.5))
            if regime == "trend" and val_multiplier < 1.0:
                val_multiplier = 1.0 + (val_multiplier - 1.0) * 0.5
            risk_discounts.append(("valuation", val_multiplier))

        # 应用风控折扣
        total_discount = 1.0
        for name, disc in risk_discounts:
            total_discount *= disc
        total_discount = max(total_discount, 0.3)  # 最多降仓70%（从0.5放宽到0.3）
        position = position_base * total_discount

        position = float(np.clip(position, 0.0, 1.0))

        # ── ATR数据（传递给回测引擎做动态止损） ──
        current_price = sector_hist["close"].iloc[-1] if not sector_hist.empty else 0

        # ── 构建因子明细 ──
        factors = {
            "trend_strength": trend_score,
            "short_momentum": short_mom_score,
            "multi_window_momentum": momentum_score,
            "volume_confirm": volume_score,
            "capital_flow": capital_score,
            "sentiment": sentiment_score,
            "fundamental": fundamental_score,
            "fundamental_pass": fundamental_pass,
            "valuation": valuation_score,
            "valuation_pe_pct": valuation_pe_pct,
            "ma60_pass": ma60_pass,
            "atr_pct": round(atr_pct, 4),
            "atr_abs": round(atr_abs, 4),
            "tr_atr_ratio": round(tr_atr_ratio, 2),
            "weights": weights,
            "regime": regime,
            "risk_discounts": {n: round(d, 3) for n, d in risk_discounts},
            "reversal_score": round(reversal_score, 2),
            "relative_strength": round(relative_strength, 4),
            # ATR动态止损用
            "current_price": current_price,
        }
        factors.update({f"mom_{k}": v for k, v in momentum_detail.items()})

        return SectorScore(
            sector="",
            composite=composite,
            signal=signal,
            position=position,
            factors=factors,
            regime=regime,
        )

    @staticmethod
    def _calc_atr(hist: pd.DataFrame, period: int = 20) -> float:
        """计算 ATR。"""
        if hist is None or len(hist) < period + 1:
            return 0.0
        high = hist["high"].values
        low = hist["low"].values
        close = hist["close"].values
        tr_list = []
        for i in range(1, len(high)):
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            tr_list.append(tr)
        if len(tr_list) < period:
            return float(np.mean(tr_list)) if tr_list else 0.0
        return float(np.mean(tr_list[-period:]))

    @staticmethod
    def _calc_atr_pct(hist: pd.DataFrame, period: int = 20) -> float:
        """计算 ATR%。"""
        atr = V2ThreeFactor._calc_atr(hist, period)
        if atr <= 0 or hist is None or len(hist) < 1:
            return 0.0
        close = hist["close"].iloc[-1]
        if close <= 0:
            return 0.0
        return atr / close

    @staticmethod
    def _check_ma60(hist: pd.DataFrame, period: int = 60) -> bool:
        """MA60 过滤。"""
        if hist is None or len(hist) < period:
            return False
        close = hist["close"]
        ma = close.rolling(period).mean()
        if np.isnan(ma.iloc[-1]) or ma.iloc[-1] == 0:
            return False
        return close.iloc[-1] >= ma.iloc[-1]
