#!/usr/bin/env python3
"""Forward test v7 — Round 6: Classic Dual Momentum (Faber 2007).

Back to basics. Faber's original approach:
  1. Absolute momentum: index 10d return > 0 → invest, else → all cash
  2. Relative momentum: pick top N sectors by (MA + ROC) composite
  3. No crash stops, no trailing stops, no vol targeting, no regime caps
  4. Invest 80% when in market, 0% when out

The simplicity is the point — every layer of "risk management" added
in Rounds 1-5 made things worse by causing whipsaw.
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import pandas as pd
import numpy as np
from src.signals.fusion import compute_sector_scores, detect_market_regime
from config.sector_etf_map import get_etf_code, get_etf

INITIAL_CAPITAL = 5000
REBALANCE_FREQ = 5

MA_PERIOD = 20
ROC_PERIOD = 20
ABS_LOOKBACK = 10
INVEST_PCT = 80  # position when in market
MIN_SECTORS = 3


def load_data():
    sectors = pd.read_pickle(BASE_DIR / "data" / "sectors_full.pkl")
    index_data = pd.read_pickle(BASE_DIR / "data" / "index_full.pkl")
    for df in sectors.values():
        df["date"] = pd.to_datetime(df["date"])
    index_data["date"] = pd.to_datetime(index_data["date"])
    for k, df in sectors.items():
        if "pct_chg" not in df.columns:
            df["pct_chg"] = df["close"].pct_change(1).fillna(0) * 100
    if "pct_chg" not in index_data.columns:
        index_data["pct_chg"] = index_data["close"].pct_change(1).fillna(0) * 100
    return sectors, index_data


def _sector_pct(sectors, name, date):
    df = sectors.get(name)
    if df is None:
        return 0.0
    row = df[df["date"] == date]
    return float(row["pct_chg"].iloc[0]) if not row.empty else 0.0


def index_n_day_return(index, date, n=10):
    """Index N-day return for absolute momentum filter."""
    hist = index[(index["date"] <= date)]
    if len(hist) < n + 1:
        return 0.0
    return float(hist["close"].iloc[-1] / hist["close"].iloc[-(n + 1)] - 1) * 100


class Round6Engine:
    """Classic Dual Momentum — Faber 2007."""

    def __init__(self, capital: int = INITIAL_CAPITAL):
        self.cash = float(capital)
        self.positions: dict[str, float] = {}
        self.daily_log: list[dict] = []
        self.last_rebalance_day = -REBALANCE_FREQ

    def run(self, sectors: dict, index: pd.DataFrame,
            start_date: str, end_date: str):
        trading_days = sorted(index["date"].unique())
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)
        trading_days = [d for d in trading_days if start_dt <= d <= end_dt]

        market_states = []

        for day_idx, day in enumerate(trading_days):
            day_str = day.strftime("%Y-%m-%d")

            nav_before = self.cash + sum(self.positions.values())

            # ---- P&L ----
            daily_pnl = 0.0
            for sector_name, value in list(self.positions.items()):
                pct = _sector_pct(sectors, sector_name, day)
                pnl = value * pct / 100.0
                daily_pnl += pnl
                self.positions[sector_name] = value + pnl

            trades_today = []
            idx_row = index[index["date"] == day]
            idx_pct = float(idx_row["pct_chg"].iloc[0]) if not idx_row.empty else 0.0

            # ---- Absolute momentum: index 10d return ----
            idx_10d = index_n_day_return(index, day, ABS_LOOKBACK)
            in_market = idx_10d > 0

            prev_in_market = getattr(self, "last_in_market", None)
            if in_market != prev_in_market:
                market_states.append(
                    f"  {day_str}: {'IN' if in_market else 'OUT'} "
                    f"(沪深300 {ABS_LOOKBACK}d={idx_10d:+.2f}%)")
                self.last_in_market = in_market

            pos_cap = INVEST_PCT if in_market else 0

            # ---- Rebalance ----
            is_rebalance = (day_idx - self.last_rebalance_day) >= REBALANCE_FREQ

            if is_rebalance:
                for sector_name, value in list(self.positions.items()):
                    trades_today.append({
                        "date": day_str, "action": "sell", "reason": "调仓",
                        "sector": sector_name, "etf": get_etf_code(sector_name),
                        "amount": round(value),
                    })
                    self.cash += value
                    del self.positions[sector_name]

                if in_market:
                    targets = self._compute_targets(sectors, index, day, pos_cap)
                    total_buy = sum(t["amount"] for t in targets)
                    for t in targets:
                        self.positions[t["sector"]] = float(t["amount"])
                        trades_today.append({
                            "date": day_str, "action": "buy", "reason": "调仓",
                            "sector": t["sector"], "etf": t["etf"],
                            "amount": t["amount"],
                        })
                    self.cash -= total_buy
                self.last_rebalance_day = day_idx

            # ---- Record ----
            nav = self.cash + sum(self.positions.values())
            self.daily_log.append({
                "date": day_str, "nav": nav, "cash": self.cash,
                "invested": sum(self.positions.values()),
                "daily_pnl": daily_pnl,
                "daily_return": (nav / nav_before - 1) * 100 if nav_before > 0 else 0,
                "index_return": idx_pct,
                "positions": {s: round(v) for s, v in self.positions.items()},
                "is_rebalance": is_rebalance,
                "trades": trades_today,
                "in_market": in_market,
                "pos_cap": pos_cap,
            })

        if market_states:
            print(f"\n{'='*60}")
            print(f"市场进出信号 ({len(market_states)} 次)")
            print(f"{'='*60}")
            for ms in market_states:
                print(ms)

        return self._summary(start_date, end_date, index)

    def _compute_targets(self, sectors, index, date, pos_cap):
        start_180d = (date - timedelta(days=180)).strftime("%Y-%m-%d")

        sectors_data = {}
        for name in sectors:
            df = sectors[name]
            subset = df[(df["date"] >= pd.Timestamp(start_180d)) & (df["date"] <= date)].copy()
            if len(subset) >= 10:
                sectors_data[name] = subset

        idx_subset = index[(index["date"] >= pd.Timestamp(start_180d)) & (index["date"] <= date)].copy()

        if len(sectors_data) == 0 or idx_subset.empty:
            return []

        scores_df = compute_sector_scores(
            sectors_data, idx_subset,
            ma_period=MA_PERIOD, roc_period=ROC_PERIOD, lookback=ABS_LOOKBACK,
        )

        if scores_df.empty:
            return []

        qualified = []
        for _, row in scores_df.iterrows():
            code = get_etf_code(row["sector"])
            if code != "—":
                qualified.append({"sector": row["sector"], "code": code,
                                  "rank": len(qualified) + 1})
            if len(qualified) >= MIN_SECTORS:
                break

        if not qualified:
            return []

        n = len(qualified)
        total_nav = self.cash + sum(self.positions.values())
        invest_amount = round(total_nav * pos_cap / 100.0)
        if invest_amount <= 0:
            return []

        targets = []
        rank_sum = sum(range(1, n + 1))
        for q in qualified:
            rel_weight = (n - q["rank"] + 1) / rank_sum
            amount = round(invest_amount * rel_weight)
            if amount > 0:
                targets.append({"sector": q["sector"], "etf": q["code"], "amount": amount})
        return targets

    def _summary(self, start_date, end_date, index):
        df = pd.DataFrame(self.daily_log)
        if df.empty:
            return df

        start_nav = INITIAL_CAPITAL
        idx_start = index[index["date"] == pd.Timestamp(start_date)]
        idx_end = index[index["date"] == pd.Timestamp(end_date)]
        idx_return = 0.0
        if not idx_start.empty and not idx_end.empty:
            idx_return = (float(idx_end["close"].iloc[0]) / float(idx_start["close"].iloc[0]) - 1) * 100

        port_return = (df["nav"].iloc[-1] / start_nav - 1) * 100
        excess = port_return - idx_return

        cumulative = df["nav"].values / start_nav
        peak = pd.Series(cumulative).cummax().values
        drawdowns = (cumulative - peak) / peak
        max_dd = float(drawdowns.min()) * 100 if len(drawdowns) > 0 else 0

        days_in = (df["in_market"] == True).sum()
        days_out = (df["in_market"] == False).sum()

        print(f"\n{'='*80}")
        print(f"Round 6: Classic Dual Momentum (Faber 2007) — {start_date} → {end_date}")
        print(f"Config: MA={MA_PERIOD} ROC={ROC_PERIOD} Abs={ABS_LOOKBACK}d Invest={INVEST_PCT}%")
        print(f"Initial: {INITIAL_CAPITAL}元 | Final NAV: {df['nav'].iloc[-1]:.2f}元")
        print(f"Portfolio: {port_return:+.2f}% | 沪深300: {idx_return:+.2f}% | Excess: {excess:+.2f}%")
        print(f"Max drawdown: {max_dd:.2f}%")
        print(f"Days in market: {days_in} | Days in cash: {days_out}")
        print(f"{'='*80}")
        return df


def print_daily_report(df):
    print(f"\n{'日期':<12} {'NAV':>8} {'日收益':>8} {'仓位':>7} {'市场':>6} {'操作':<45} {'沪深300':>8}")
    print("-" * 104)

    for _, row in df.iterrows():
        date = row["date"]
        nav = row["nav"]
        invested = row["invested"]
        pct = invested / nav * 100 if nav > 0 else 0
        idx_ret = row["index_return"]
        in_mkt = "IN" if row.get("in_market") else "OUT"

        trade_str = ""
        if row["trades"]:
            parts = []
            for t in row["trades"]:
                a = "B" if t["action"] == "buy" else "S"
                r = t.get("reason", "")[:3]
                parts.append(f"{a}({r}){t['etf']}({t['amount']}元)")
            trade_str = " | ".join(parts)
        if len(trade_str) > 45:
            trade_str = trade_str[:42] + "..."

        print(f"{date:<12} {nav:>8.0f} {row['daily_return']:>+7.2f}% {pct:>6.0f}% "
              f" {in_mkt:<6} {trade_str:<45} {idx_ret:>+7.2f}%")
    print("-" * 104)


def main():
    parser = argparse.ArgumentParser(description="Round 6: Classic Dual Momentum (Faber 2007)")
    parser.add_argument("--start", type=str, default="2026-02-24")
    parser.add_argument("--end", type=str, default="2026-03-31")
    parser.add_argument("--capital", type=int, default=INITIAL_CAPITAL)
    args = parser.parse_args()

    print(f"Loading data...")
    sectors, index_data = load_data()
    print(f"\n>>> Round 6: Classic Dual Momentum (Faber 2007) <<<")
    print(f"  Abs momentum: 沪深300 {ABS_LOOKBACK}d return > 0 → IN ({INVEST_PCT}%), else CASH")
    print(f"  Rel momentum: MA{MA_PERIOD} + ROC{ROC_PERIOD}, top {MIN_SECTORS} sectors")

    engine = Round6Engine(capital=args.capital)
    df = engine.run(sectors, index_data, args.start, args.end)
    print_daily_report(df)

    from config.settings import RESULTS_DIR
    out_path = RESULTS_DIR / f"forward_{args.start}_{args.end}_round6.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
