"""Congestion signal: turnover ratio and volume share."""

import pandas as pd


def calc_turnover_ratio(df: pd.DataFrame, lookback: int = 5) -> float:
    """Current turnover_rate / avg turnover_rate over lookback.

    > 1.5 = abnormally active.
    """
    if "turnover_rate" not in df.columns or len(df) < lookback + 1:
        return 0.0
    recent = df["turnover_rate"].iloc[-(lookback + 1):-1]
    avg = recent.mean()
    if avg == 0:
        return 0.0
    return df["turnover_rate"].iloc[-1] / avg


def calc_volume_share(df: pd.DataFrame, all_amounts: dict[str, float]) -> float:
    """Sector avg daily amount / total market avg daily amount.

    all_amounts maps sector_name -> avg daily amount over same window.
    """
    sector_avg = df["amount"].tail(5).mean()
    total_avg = sum(all_amounts.values())
    if total_avg == 0:
        return 0.0
    return sector_avg / total_avg


def score_congestion(
    sectors_data: dict[str, pd.DataFrame],
    lookback: int = 5,
) -> dict[str, float]:
    """Compute congestion score for each sector.

    = 0.5 * min(turnover_ratio / 3.0, 1.0) + 0.5 * volume_share

    Returns dict[sector_name -> score in [0, 1]].
    """
    turnover_ratios = {}
    all_amounts = {}
    for name, df in sectors_data.items():
        turnover_ratios[name] = calc_turnover_ratio(df, lookback)
        all_amounts[name] = df["amount"].tail(5).mean()

    scores = {}
    for name, df in sectors_data.items():
        tr_score = min(turnover_ratios[name] / 3.0, 1.0)
        vs = calc_volume_share(df, all_amounts)
        scores[name] = 0.5 * tr_score + 0.5 * vs

    return scores
