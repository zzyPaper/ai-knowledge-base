"""Professional crowding detection with correct Beta and correlation crowding.

Fixes the V2 bug where Beta was approximated as volatility.

References:
  - 国盛证券 "趋势-拥挤度" 二维框架 (2024-2025)
  - Barroso & Santa-Clara (2015) crowding and momentum crashes
"""

import numpy as np
import pandas as pd
from typing import Optional

# Crowding lookback windows — shortened for short-history data
CROWD_SHORT = 20   # 1-month for recent indicators
CROWD_LONG = 60    # ~3-month for historical percentiles (was 500, too long for our data)


def calc_beta(sector_rets: np.ndarray, market_rets: np.ndarray, window: int = 60) -> float:
    """Calculate true regression Beta = Cov(sec, mkt) / Var(mkt).

    Uses rolling window regression. Returns 1.0 if insufficient data.
    """
    n = min(len(sector_rets), len(market_rets))
    if n < window + 1:
        return 1.0

    y = sector_rets[-window:]
    x = market_rets[-window:]

    x_mean, y_mean = np.mean(x), np.mean(y)
    cov = np.mean((x - x_mean) * (y - y_mean))
    var = np.mean((x - x_mean) ** 2)

    if var < 1e-10:
        return 1.0
    return float(cov / var)


def rolling_percentile(value: float, history: np.ndarray) -> float:
    """What percentile is `value` within `history`? Returns [0, 1]."""
    if len(history) == 0:
        return 0.5
    return float((history < value).sum()) / len(history)


def score_crowding_pro(
    sectors_data: dict[str, pd.DataFrame],
    index_hist: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Compute professional crowding scores for all sectors.

    Dimensions:
      1. Turnover percentile (vs 2-year history)
      2. Volatility percentile (vs 2-year history)
      3. Beta percentile (TRUE regression beta vs history)
      4. Correlation crowding (sector-sector correlation elevated?)

    Higher score = MORE crowded = MORE risk.
    """
    # Pre-compute market returns for Beta calculation
    market_rets = None
    if index_hist is not None and not index_hist.empty and "close" in index_hist.columns:
        mkt_close = index_hist["close"].values
        if len(mkt_close) > 1:
            market_rets = np.diff(mkt_close) / mkt_close[:-1]

    # Pre-compute all sector returns for correlation crowding
    sector_rets_map = {}
    for name, df in sectors_data.items():
        close = df["close"].values
        if len(close) > 1:
            sector_rets_map[name] = np.diff(close) / close[:-1]

    # Correlation crowding: are sectors moving together abnormally?
    corr_crowding_map = {}
    if len(sector_rets_map) >= 3:
        # Sample up to 10 sectors for efficiency
        sample_names = list(sector_rets_map.keys())[:10]
        aligned_rets = []
        min_len = min(len(sector_rets_map[n]) for n in sample_names)
        for n in sample_names:
            aligned_rets.append(sector_rets_map[n][-min_len:])
        if aligned_rets:
            ret_matrix = np.column_stack(aligned_rets)
            if ret_matrix.shape[0] >= CROWD_SHORT and ret_matrix.shape[1] >= 2:
                # Recent correlation matrix
                recent = ret_matrix[-CROWD_SHORT:]
                # Pairwise correlations
                corrs = []
                n_cols = recent.shape[1]
                for i in range(n_cols):
                    for j in range(i + 1, n_cols):
                        c = np.corrcoef(recent[:, i], recent[:, j])[0, 1]
                        if not np.isnan(c):
                            corrs.append(abs(c))
                avg_corr = np.mean(corrs) if corrs else 0.5
                # Historical avg correlation (longer window)
                hist_corrs = []
                if ret_matrix.shape[0] >= CROWD_LONG:
                    hist = ret_matrix[-CROWD_LONG:]
                else:
                    hist = ret_matrix
                step = max(1, len(hist) // 10)
                for start in range(0, len(hist) - CROWD_SHORT, step):
                    window = hist[start:start + CROWD_SHORT]
                    for i in range(n_cols):
                        for j in range(i + 1, n_cols):
                            c = np.corrcoef(window[:, i], window[:, j])[0, 1]
                            if not np.isnan(c):
                                hist_corrs.append(abs(c))
                avg_hist_corr = np.mean(hist_corrs) if hist_corrs else 0.5
                # Correlation crowding = recent / historical
                corr_ratio = avg_corr / (avg_hist_corr + 1e-10)
                # Map to [0, 1]: ratio > 1.5 means high crowding
                corr_crowding_global = min(max((corr_ratio - 0.8) / 1.2, 0.0), 1.0)
            else:
                corr_crowding_global = 0.5
        else:
            corr_crowding_global = 0.5
    else:
        corr_crowding_global = 0.5

    records = []
    for name, df in sectors_data.items():
        close = df["close"].values
        n = len(close)
        if n < CROWD_SHORT + 1:
            records.append({
                "sector": name,
                "turnover_pct": 0.5,
                "volatility_pct": 0.5,
                "beta_pct": 0.5,
                "correlation_pct": corr_crowding_global,
                "crowding_score": 0.5,
            })
            continue

        rets = np.diff(close) / close[:-1]

        # 1. Turnover percentile
        turnover_pct = 0.5
        if "turnover_rate" in df.columns:
            tr = df["turnover_rate"].values
            if len(tr) >= CROWD_SHORT:
                recent_tr = float(np.mean(tr[-CROWD_SHORT:]))
                hist_tr = tr[-min(len(tr), CROWD_LONG):]
                turnover_pct = rolling_percentile(recent_tr, hist_tr)

        # 2. Volatility percentile
        vol_pct = 0.5
        if len(rets) >= CROWD_SHORT:
            recent_vol = float(np.std(rets[-CROWD_SHORT:]))
            # Rolling volatilities for historical distribution
            rolling_vols = []
            for i in range(CROWD_SHORT, len(rets)):
                vol = float(np.std(rets[max(0, i - CROWD_SHORT):i + 1]))
                rolling_vols.append(vol)
            if rolling_vols:
                vol_pct = rolling_percentile(recent_vol, np.array(rolling_vols))

        # 3. TRUE Beta percentile (FIXED: no longer using vol as proxy)
        beta_pct = 0.5
        if market_rets is not None and name in sector_rets_map:
            sec_r = sector_rets_map[name]
            # Recent Beta
            recent_beta = calc_beta(sec_r, market_rets, window=min(60, len(sec_r)))
            # Historical Betas
            hist_betas = []
            for start in range(0, max(1, len(sec_r) - 60), 20):
                end = start + 60
                if end > len(sec_r):
                    break
                b = calc_beta(sec_r[start:end], market_rets[start:end], window=min(60, end - start))
                hist_betas.append(b)
            if hist_betas:
                beta_pct = rolling_percentile(recent_beta, np.array(hist_betas))

        # 4. Correlation crowding (global factor applied to all)
        corr_pct = corr_crowding_global

        # Composite crowding: equal-weighted
        crowding = (turnover_pct + vol_pct + beta_pct + corr_pct) / 4.0
        records.append({
            "sector": name,
            "turnover_pct": turnover_pct,
            "volatility_pct": vol_pct,
            "beta_pct": beta_pct,
            "correlation_pct": corr_pct,
            "crowding_score": crowding,
        })

    return pd.DataFrame(records)
