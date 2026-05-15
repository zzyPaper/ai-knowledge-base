"""Backtest engine: simulate sector rotation trading on historical data."""

from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
from src.signals.fusion import compute_sector_scores, detect_market_regime
from src.signals.position_sizing import compute_position_pct
from config.sector_etf_map import get_etf_code


@dataclass
class StrategyConfig:
    top_n: int = 3
    top_n_trending: int = 2
    rebalance_freq: int = 5
    ma_period: int = 20
    roc_period: int = 20
    lookback: int = 10
    trending_weights: tuple = (0.80, 0.20)
    ranging_weights: tuple = (0.60, 0.40)
    regime_adaptive: bool = True


@dataclass
class BacktestResult:
    daily_returns: pd.DataFrame
    trades: list[dict]
    config: StrategyConfig
    sector_snapshots: list[dict] = field(default_factory=list)

    @property
    def metrics(self):
        from src.backtest.metrics import compute_metrics
        return compute_metrics(self)


class BacktestEngine:
    """Simulates sector rotation trading on historical data."""

    def __init__(self, config: StrategyConfig = None):
        self.config = config or StrategyConfig()

    def run(
        self,
        sectors_data: dict[str, pd.DataFrame],
        index_data: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> BacktestResult:
        trading_days = sorted(index_data["date"].unique())
        if start_date:
            trading_days = [d for d in trading_days if d >= pd.Timestamp(start_date)]
        if end_date:
            trading_days = [d for d in trading_days if d <= pd.Timestamp(end_date)]

        rebalance_days = trading_days[:: self.config.rebalance_freq]

        positions: dict[str, float] = {}
        daily_log = []
        trades = []
        sector_snapshots = []

        for day in trading_days:
            day_str = day.strftime("%Y-%m-%d") if hasattr(day, "strftime") else str(day)

            if day in rebalance_days:
                current_data = {}
                for name, df in sectors_data.items():
                    subset = df[df["date"] <= day].copy()
                    if len(subset) >= 10:
                        current_data[name] = subset

                current_index = index_data[index_data["date"] <= day].copy() if index_data is not None else None

                if len(current_data) > 0:
                    # Regime-adaptive top_n
                    if self.config.regime_adaptive and current_index is not None and len(current_index) >= 20:
                        regime = detect_market_regime(current_index)
                        effective_top_n = self.config.top_n_trending if regime == "trending" else self.config.top_n
                    else:
                        effective_top_n = self.config.top_n

                    scores_df = compute_sector_scores(
                        current_data, current_index,
                        ma_period=self.config.ma_period,
                        roc_period=self.config.roc_period,
                        lookback=self.config.lookback,
                        trending_weights=self.config.trending_weights,
                        ranging_weights=self.config.ranging_weights,
                    )
                    top_sectors = scores_df.head(effective_top_n)
                    top_names = top_sectors["sector"].tolist()
                    # ETF-safe filtering: skip sectors without ETF mapping
                    top_names = [s for s in top_names if get_etf_code(s) != "—"]

                    # Position sizing based on market absolute momentum
                    pos_pct = compute_position_pct(current_index)
                    pos_weight = (pos_pct / 100.0) / len(top_names) if top_names else 0

                    for s in list(positions.keys()):
                        if s not in top_names:
                            try:
                                sdf = sectors_data[s]
                                row = sdf[sdf["date"] == day]
                                if not row.empty:
                                    exit_price = float(row["close"].iloc[0])
                                    trades.append({"date": day_str, "sector": s, "action": "sell", "price": exit_price, "weight": positions[s]})
                                del positions[s]
                            except Exception:
                                del positions[s]

                    weight = pos_weight
                    for s in top_names:
                        if s not in positions:
                            try:
                                sdf = sectors_data[s]
                                row = sdf[sdf["date"] == day]
                                if not row.empty:
                                    entry_price = float(row["close"].iloc[0])
                                    trades.append({"date": day_str, "sector": s, "action": "buy", "price": entry_price, "weight": weight})
                                positions[s] = weight
                            except Exception:
                                positions[s] = weight

                    sector_snapshots.append({"date": day_str, "top_sectors": top_names[:3]})

            port_return = 0.0
            if positions:
                returns = []
                for s, w in positions.items():
                    df = sectors_data.get(s)
                    if df is None or df.empty:
                        continue
                    row = df[df["date"] == day]
                    if row.empty:
                        continue
                    idx = df[df["date"] < day].index
                    if len(idx) == 0:
                        continue
                    prev_close = float(df.loc[idx[-1], "close"])
                    today_close = float(row["close"].iloc[0])
                    returns.append(w * (today_close / prev_close - 1))
                port_return = sum(returns) if returns else 0.0

            idx_row = index_data[index_data["date"] == day]
            idx_return = float(idx_row["pct_chg"].iloc[0] / 100.0) if not idx_row.empty else 0.0

            daily_log.append({"date": day_str, "port_return": port_return, "index_return": idx_return})

        df_log = pd.DataFrame(daily_log)
        if not df_log.empty:
            df_log["port_cumulative"] = (1 + df_log["port_return"]).cumprod()
            df_log["index_cumulative"] = (1 + df_log["index_return"]).cumprod()
        else:
            df_log["port_cumulative"] = 1.0
            df_log["index_cumulative"] = 1.0

        return BacktestResult(daily_returns=df_log, trades=trades, config=self.config, sector_snapshots=sector_snapshots)
