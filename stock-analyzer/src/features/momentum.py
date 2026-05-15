"""Momentum signal: MA ratio and Rate of Change."""

import pandas as pd


def calc_ma_ratio(series: pd.Series, period: int = 20) -> float:
    """Calculate (close / MA - 1). Positive means price above moving average."""
    if len(series) < period:
        return 0.0
    ma = series.iloc[-period:].mean()
    if ma == 0:
        return 0.0
    return series.iloc[-1] / ma - 1.0


def calc_roc(series: pd.Series, period: int = 20) -> float:
    """Calculate Rate of Change: (close_t - close_{t-period}) / close_{t-period}."""
    if len(series) <= period:
        return 0.0
    prev = series.iloc[-(period + 1)]
    if prev == 0:
        return 0.0
    return (series.iloc[-1] - prev) / prev


def rank_momentum(sectors_data: dict[str, pd.DataFrame], ma_period: int = 20, roc_period: int = 20) -> dict[str, float]:
    """Compute momentum score for each sector: 0.5 * norm(MA_ratio) + 0.5 * norm(ROC).

    Returns dict[sector_name -> score in [0, 1]].
    """
    scores = {}
    for name, df in sectors_data.items():
        close = df["close"].values
        series = pd.Series(close)
        ma_r = calc_ma_ratio(series, ma_period)
        roc = calc_roc(series, roc_period)
        scores[name] = {"ma_ratio": ma_r, "roc": roc}

    raw_ma = {s: v["ma_ratio"] for s, v in scores.items()}
    raw_roc = {s: v["roc"] for s, v in scores.items()}

    norm_ma = _min_max(raw_ma)
    norm_roc = _min_max(raw_roc)

    return {s: 0.5 * norm_ma[s] + 0.5 * norm_roc[s] for s in scores}


def _min_max(scores: dict[str, float]) -> dict[str, float]:
    values = list(scores.values())
    mn, mx = min(values), max(values)
    if mx - mn < 1e-10:
        return {k: 0.5 for k in scores}
    return {k: (v - mn) / (mx - mn) for k, v in scores.items()}
