#!/usr/bin/env python3
"""Forward test v8 — Round 7: Slow Entry + Fast Exit Dual Momentum.

Key insight from Rounds 1-6: entering too early in a fragile bounce is the #1
cause of losses. The Mar 3 crash was avoidable — on Feb 24, index 20d return
was -0.57% (still in downtrend on longer timeframe).

Strategy:
  1. ENTRY: index 20d return > 0 (slow, conservative — avoid false bounces)
  2. EXIT:  index 10d return < 0 (fast, protective — don't wait)
  3. Invest 80% when in market, 0% when out
  4. Select top 3 sectors by Dual Momentum (MA20 + ROC20 + abs10d)
  5. Simple crash stop: sector single-day -5% → sell that sector only
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
ABS_ENTRY_LOOKBACK = 20  # slower for entry
ABS_EXIT_LOOKBACK = 10   # faster for exit
INVEST_PCT = 80
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


def index_n_day_return(index, date, n):
    hist = index[(index["date"] <= date)]
    if len(hist) < n + 1:
        return 0.0
    return float(hist["close"].iloc[-1] / hist["close"].iloc[-(n + 1)] - 1) * 100


class Round7Engine:
    """Slow entry (20d) + Fast exit (10d) Dual Momentum."""

    def __init__(self, capital: int = INITIAL_CAPITAL):
        self.cash = float(capital)
        self.positions: dict[str, float] = {}
        self.daily_log: list[dict] = []
        self.last_rebalance_day = -REBALANCE_FREQ
        self.in_market = False  # start out

    def run(self, sectors: dict, index: pd.DataFrame,
            start_date: str, end_date: str):
        trading_days = sorted(index["date"].unique())
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)
        trading_days = [d for d in trading_days if start_dt <= d <= end_dt]

        market_signals = []

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

            # ---- Crash stop: individual sector only ----
            for sector_name, value in list(self.positions.items()):
                pct = _sector_pct(sectors, sector_name, day)
                if pct < -5.0:
                    self.cash += value
                    trades_today.append({
                        "date": day_str, "action": "sell",
                        "reason": f"跌停({pct:+.1f}%)",
                        "sector": sector_name, "etf": get_etf_code(sector_name),
                        "amount": round(value),
                    })
                    del self.positions[sector_name]

            # ---- Market timing: slow entry (20d), fast exit (10d) ----
            ret_entry = index_n_day_return(index, day, ABS_ENTRY_LOOKBACK)
            ret_exit = index_n_day_return(index, day, ABS_EXIT_LOOKBACK)

            if not self.in_market:
                # Slow entry: need 20d return > 0 to enter
                should_enter = ret_entry > 0
                if should_enter:
                    self.in_market = True
                    market_signals.append(
                        f"  {day_str}: ENTER (20d={ret_entry:+.2f}%)")
            else:
                # Fast exit: 10d return < 0 → exit
                should_exit = ret_exit < 0
                if should_exit:
                    self.in_market = False
                    market_signals.append(
                        f"  {day_str}: EXIT (10d={ret_exit:+.2f}%)")

            # ---- Rebalance ----
            is_rebalance = (day_idx - self.last_rebalance_day) >= REBALANCE_FREQ

            # Force rebalance on market state change
            prev_in = getattr(self, "last_in_market_signal", None)
            if prev_in is not None and prev_in != self.in_market:
                is_rebalance = True
            self.last_in_market_signal = self.in_market

            if is_rebalance:
                for sector_name, value in list(self.positions.items()):
                    trades_today.append({
                        "date": day_str, "action": "sell", "reason": "调仓",
                        "sector": sector_name, "etf": get_etf_code(sector_name),
                        "amount": round(value),
                    })
                    self.cash += value
                    del self.positions[sector_name]

                if self.in_market:
                    targets = self._compute_targets(sectors, index, day, INVEST_PCT)
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
                "in_market": self.in_market,
                "pos_cap": INVEST_PCT if self.in_market else 0,
            })

        if market_signals:
            print(f"\n{'='*60}")
            print(f"市场进出信号 ({len(market_signals)} 次)")
            print(f"{'='*60}")
            for ms in market_signals:
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
        print(f"Round 7: Slow Entry ({ABS_ENTRY_LOOKBACK}d) + Fast Exit ({ABS_EXIT_LOOKBACK}d)")
        print(f"  {start_date} → {end_date}")
        print(f"Config: MA={MA_PERIOD} ROC={ROC_PERIOD} Invest={INVEST_PCT}%")
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
    parser = argparse.ArgumentParser(description="Round 7: Slow entry + fast exit")
    parser.add_argument("--start", type=str, default="2026-02-24")
    parser.add_argument("--end", type=str, default="2026-03-31")
    parser.add_argument("--capital", type=int, default=INITIAL_CAPITAL)
    args = parser.parse_args()

    print(f"Loading data...")
    sectors, index_data = load_data()
    print(f"\n>>> Round 7: Slow Entry ({ABS_ENTRY_LOOKBACK}d) + Fast Exit ({ABS_EXIT_LOOKBACK}d) <<<")
    print(f"  ENTRY: 沪深300 {ABS_ENTRY_LOOKBACK}d > 0")
    print(f"  EXIT:  沪深300 {ABS_EXIT_LOOKBACK}d < 0")

    engine = Round7Engine(capital=args.capital)
    df = engine.run(sectors, index_data, args.start, args.end)
    print_daily_report(df)

    from config.settings import RESULTS_DIR
    out_path = RESULTS_DIR / f"forward_{args.start}_{args.end}_round7.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
