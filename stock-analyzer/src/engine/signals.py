"""Multi-dimension signal pipeline.

Computes sector scores across two independent dimensions and combines
them with regime-adaptive weights.

Dimensions:
  1. Trend: multi-timeframe momentum (1M/3M/6M rank-normalized)
  2. Crowding: turnover/volatility/Beta historical percentiles

Both dimensions independently produce [0,1] scores, then combined with
weights that depend on market regime (trend gets more weight in bull,
crowding gets more weight in correction).

Reference:
  - 国盛证券 "趋势-拥挤度" 二维框架
  - 中银证券 S7: rank等权优于zscore等权
"""

from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np

from src.engine.config import SignalConfig
from src.engine.regime import Regime


def _rank_normalize(scores: dict[str, float]) -> dict[str, float]:
    """Rank-based normalization: each sector gets [0,1] by rank order.

    中银 S7: rank等权优于zscore等权 — rank normalization is more robust
    to outliers and produces better top/bottom spread.
    """
    if len(scores) <= 1:
        return {k: 0.5 for k in scores}
    sorted_items = sorted(scores.items(), key=lambda x: x[1])
    n = len(sorted_items)
    result = {}
    for rank, (name, _) in enumerate(sorted_items):
        result[name] = rank / (n - 1)
    return result


@dataclass
class SignalResult:
    """Output of signal computation."""
    scores_df: pd.DataFrame  # columns: sector, trend, crowding, composite, rank
    top_sectors: list[str]
    regime: str
    n_qualified: int


class SignalPipeline:
    """Multi-dimension sector scoring pipeline."""

    def __init__(self, config: SignalConfig = SignalConfig()):
        self.config = config

    def compute(self, sectors_data: dict[str, pd.DataFrame],
                regime: str = Regime.NEUTRAL) -> SignalResult:
        """Compute multi-dimension scores for all sectors.

        Returns SignalResult with ranked sectors and composite scores.
        """
        if len(sectors_data) < 2:
            return SignalResult(
                scores_df=pd.DataFrame(columns=["sector", "trend", "crowding", "composite", "rank"]),
                top_sectors=[], regime=regime, n_qualified=0,
            )

        # Dimension 1: Trend (multi-timeframe momentum)
        trend_raw = self._compute_trend(sectors_data)
        trend_norm = _rank_normalize(trend_raw)

        # Dimension 2: Crowding risk (higher = more crowded)
        crowding_raw = self._compute_crowding(sectors_data)
        # Invert: higher score = LESS crowded (better)
        crowding_inv = {k: 1.0 - v for k, v in crowding_raw.items()}
        crowding_norm = _rank_normalize(crowding_inv)

        # Composite with regime-adaptive weights
        w_trend, w_crowd = Regime.signal_weights(regime)
        composites = {}
        for name in sectors_data:
            t = trend_norm.get(name, 0.5)
            c = crowding_norm.get(name, 0.5)
            composites[name] = w_trend * t + w_crowd * c

        # Build ranked DataFrame
        result = pd.DataFrame({
            "sector": list(composites.keys()),
            "trend": [trend_norm.get(s, 0.5) for s in composites],
            "crowding": [crowding_norm.get(s, 0.5) for s in composites],
            "composite": list(composites.values()),
        }).sort_values("composite", ascending=False).reset_index(drop=True)
        result["rank"] = range(1, len(result) + 1)

        return SignalResult(
            scores_df=result,
            top_sectors=result["sector"].head(5).tolist(),
            regime=regime,
            n_qualified=len(composites),
        )

    def _compute_trend(self, sectors_data: dict[str, pd.DataFrame]) -> dict[str, float]:
        """Multi-timeframe momentum composite.

        Each timeframe produces a ROC, then weighted and rank-normalized.
        """
        windows = self.config.momentum_windows
        weights = self.config.momentum_weights
        max_window = max(windows)

        raw = {}
        for name, df in sectors_data.items():
            close = df["close"].values
            if len(close) < max_window + 1:
                raw[name] = 0.0
                continue
            score = 0.0
            for w, period in zip(weights, windows):
                if len(close) > period:
                    roc = float(close[-1] / close[-(period + 1)] - 1)
                    score += w * roc
            raw[name] = score
        return raw

    def _compute_crowding(self, sectors_data: dict[str, pd.DataFrame]) -> dict[str, float]:
        """Crowding indicators: turnover/volatility/Beta percentiles.

        国盛证券 拥挤度 三维度:
        1. 换手率分位数
        2. 波动率分位数
        3. Beta分位数

        Higher value = more crowded = more risk.
        """
        scores = {}
        for name, df in sectors_data.items():
            close = df["close"].values
            n = len(close)

            if n < self.config.crowding_short + 1:
                scores[name] = 0.5
                continue

            # 1. Turnover percentile
            turnover_pct = 0.5
            if "turnover_rate" in df.columns:
                tr = df["turnover_rate"].values
                recent_tr = float(np.mean(tr[-self.config.crowding_short:]))
                hist_tr = tr[-min(n, self.config.crowding_history):]
                turnover_pct = float((hist_tr < recent_tr).sum()) / max(len(hist_tr), 1)

            # 2. Volatility percentile
            rets = np.diff(close) / (close[:-1] + 1e-10)
            vol_pct = 0.5
            if len(rets) >= self.config.crowding_short:
                recent_vol = float(np.std(rets[-self.config.crowding_short:]))
                # Compute rolling volatilities for percentile
                rolling_vols = np.array([
                    float(np.std(rets[max(0, i - self.config.crowding_short):i + 1]))
                    for i in range(self.config.crowding_short, len(rets))
                ])
                if len(rolling_vols) > 0:
                    vol_pct = float((rolling_vols < recent_vol).sum()) / len(rolling_vols)

            # 3. Beta percentile (approximated as vol_pct when Beta unavailable)
            beta_pct = vol_pct

            # Equal-weighted crowding
            scores[name] = (turnover_pct + vol_pct + beta_pct) / 3.0

        return scores
