#!/usr/bin/env python3
"""Forward test with risk management improvements (Round 1).

Improvements over baseline Dual Momentum:
  1. Daily absolute momentum re-check — sell if 10d return turns negative
  2. Single-day crash stop — sector drop >5% → immediate exit
  3. Market circuit breaker — 沪深300 drop >3% → halve positions
  4. Trailing stop — 8% from peak since purchase
  5. Volatility-adjusted position sizing — cap exposure in high-vol regimes
"""

import argparse
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import pandas as pd
import numpy as np
from src.signals.fusion import compute_sector_scores, detect_market_regime
from src.signals.position_sizing import compute_position_pct
from config.sector_etf_map import get_etf_code, get_etf

INITIAL_CAPITAL = 5000
REBALANCE_FREQ = 5

# ---- Risk management parameters ----
CRASH_STOP_SECTOR = -5.0      # single-day sector drop threshold (%)
CRASH_STOP_INDEX = -3.0       # single-day index drop threshold (%)
TRAILING_STOP_PCT = 0.08      # 8% trailing stop from peak
DAILY_ABS_MOM_THRESHOLD = 0.0 # 10d return must stay positive
VOL_HIGH_THRESHOLD = 2.0      # daily vol >2% → high vol regime
VOL_CAP_HIGH = 0.40           # max 40% position in high vol
VOL_CAP_NORMAL = 0.80         # max 80% position in normal vol


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


def _sector_val(sectors, name, date, field="close"):
    df = sectors.get(name)
    if df is None:
        return None
    row = df[df["date"] == date]
    return float(row[field].iloc[0]) if not row.empty else None


def _sector_pct(sectors, name, date):
    return _sector_val(sectors, name, date, "pct_chg") or 0.0


def _sector_n_day_return(sectors, name, date, n):
    """Return over last N trading days (close[-1] / close[-n-1] - 1)."""
    df = sectors.get(name)
    if df is None:
        return None
    hist = df[df["date"] <= date]
    if len(hist) < n + 1:
        return None
    return float(hist["close"].iloc[-1] / hist["close"].iloc[-(n + 1)] - 1) * 100


def _index_volatility(index, date, lookback=20):
    """Estimate daily return std over lookback days (%)."""
    hist = index[(index["date"] <= date)].tail(lookback + 1)
    if len(hist) < 10:
        return 1.5  # default moderate
    rets = hist["close"].pct_change().dropna()
    return float(rets.std()) * 100  # daily std in %


class RiskManagedEngine:
    """Forward test engine with multi-layer risk management."""

    def __init__(self, capital: int = INITIAL_CAPITAL):
        self.cash = float(capital)
        self.positions: dict[str, float] = {}    # sector -> dollar value
        self.peak_values: dict[str, float] = {}  # sector -> peak value since purchase
        self.trades_log: list[dict] = []
        self.daily_log: list[dict] = []
        self.last_rebalance_day = -REBALANCE_FREQ

    def run(self, sectors: dict, index: pd.DataFrame,
            start_date: str, end_date: str):
        trading_days = sorted(index["date"].unique())
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)
        trading_days = [d for d in trading_days if start_dt <= d <= end_dt]

        # Track stop-loss events for reporting
        stop_events = []

        for day_idx, day in enumerate(trading_days):
            day_str = day.strftime("%Y-%m-%d")

            # ---- Step 1: NAV before P&L ----
            nav_before = self.cash + sum(self.positions.values())

            # ---- Step 2: P&L for current positions ----
            daily_pnl = 0.0
            for sector_name, value in list(self.positions.items()):
                pct = _sector_pct(sectors, sector_name, day)
                pnl = value * pct / 100.0
                daily_pnl += pnl
                self.positions[sector_name] = value + pnl
                # Update peak
                if sector_name in self.peak_values:
                    self.peak_values[sector_name] = max(
                        self.peak_values[sector_name], self.positions[sector_name])

            # ---- Step 3: Risk management checks (fire ANY day) ----
            trades_today = []
            index_pct = _sector_pct(index, "dummy", day)  # won't work, fix below
            idx_row = index[index["date"] == day]
            idx_pct = float(idx_row["pct_chg"].iloc[0]) if not idx_row.empty else 0.0

            # 3a. Market circuit breaker: index drops >3%
            market_crash = idx_pct < CRASH_STOP_INDEX
            if market_crash and self.positions:
                # Halve all positions
                for sector_name, value in list(self.positions.items()):
                    sell_amount = value / 2.0
                    self.cash += sell_amount
                    self.positions[sector_name] = value - sell_amount
                    trades_today.append({
                        "date": day_str, "action": "sell", "reason": "market_crash",
                        "sector": sector_name,
                        "etf": get_etf_code(sector_name),
                        "amount": round(sell_amount),
                    })
                stop_events.append(f"{day_str}: 市场熔断(沪深300 {idx_pct:+.2f}%) → 仓位减半")

            # 3b. Individual position risk checks
            for sector_name, value in list(self.positions.items()):
                if value <= 0:
                    continue
                close_p = _sector_val(sectors, sector_name, day, "close")
                pct = _sector_pct(sectors, sector_name, day)

                should_sell = False
                reason = ""

                # Crash stop: single-day >5% drop
                if pct < CRASH_STOP_SECTOR:
                    should_sell = True
                    reason = f"crash_stop({pct:+.1f}%)"

                # Daily absolute momentum: 10d return negative?
                elif not should_sell:
                    ret_10d = _sector_n_day_return(sectors, sector_name, day, 10)
                    if ret_10d is not None and ret_10d < DAILY_ABS_MOM_THRESHOLD:
                        should_sell = True
                        reason = f"abs_mom_break({ret_10d:+.1f}%)"

                # Trailing stop: 8% from peak
                elif not should_sell and close_p and sector_name in self.peak_values:
                    peak_v = self.peak_values[sector_name]
                    if value < peak_v * (1 - TRAILING_STOP_PCT):
                        should_sell = True
                        drawdown = (value / peak_v - 1) * 100
                        reason = f"trailing_stop({drawdown:+.1f}%)"

                if should_sell:
                    self.cash += value
                    trades_today.append({
                        "date": day_str, "action": "sell", "reason": reason,
                        "sector": sector_name,
                        "etf": get_etf_code(sector_name),
                        "amount": round(value),
                    })
                    stop_events.append(f"  {day_str}: {reason} → 卖出 {sector_name}({get_etf_code(sector_name)}) {round(value)}元")
                    del self.positions[sector_name]
                    if sector_name in self.peak_values:
                        del self.peak_values[sector_name]

            # ---- Step 4: Rebalance ----
            is_rebalance = (day_idx - self.last_rebalance_day) >= REBALANCE_FREQ

            if is_rebalance:
                # Sell remaining positions
                for sector_name, value in list(self.positions.items()):
                    trades_today.append({
                        "date": day_str, "action": "sell", "reason": "rebalance",
                        "sector": sector_name,
                        "etf": get_etf_code(sector_name),
                        "amount": round(value),
                    })
                    self.cash += value
                    del self.positions[sector_name]
                    if sector_name in self.peak_values:
                        del self.peak_values[sector_name]

                # Compute new targets with vol-adjusted position sizing
                targets = self._compute_targets(sectors, index, day)
                total_buy = sum(t["amount"] for t in targets)

                for t in targets:
                    self.positions[t["sector"]] = float(t["amount"])
                    self.peak_values[t["sector"]] = float(t["amount"])
                    trades_today.append({
                        "date": day_str, "action": "buy", "reason": "rebalance",
                        "sector": t["sector"],
                        "etf": t["etf"],
                        "amount": t["amount"],
                    })
                self.cash -= total_buy
                self.last_rebalance_day = day_idx

            # ---- Step 5: Record ----
            nav = self.cash + sum(self.positions.values())

            self.daily_log.append({
                "date": day_str,
                "nav": nav,
                "cash": self.cash,
                "invested": sum(self.positions.values()),
                "daily_pnl": daily_pnl,
                "daily_return": (nav / nav_before - 1) * 100 if nav_before > 0 else 0,
                "index_return": idx_pct,
                "positions": {s: round(v) for s, v in self.positions.items()},
                "is_rebalance": is_rebalance,
                "trades": trades_today,
            })

        # Print stop events
        if stop_events:
            print(f"\n{'='*80}")
            print(f"风控事件记录 ({len(stop_events)} 次)")
            print(f"{'='*80}")
            for ev in stop_events:
                print(ev)

        return self._summary(start_date, end_date, index)

    def _compute_targets(self, sectors, index, date):
        start_180d = (date - timedelta(days=180)).strftime("%Y-%m-%d")

        sectors_data = {}
        for name in sectors:
            df = sectors[name]
            subset = df[(df["date"] >= pd.Timestamp(start_180d)) & (df["date"] <= date)].copy()
            if len(subset) >= 10:
                sectors_data[name] = subset

        idx_subset = index[(index["date"] >= pd.Timestamp(start_180d)) & (df["date"] <= date)].copy() if False else \
                     index[(index["date"] >= pd.Timestamp(start_180d)) & (index["date"] <= date)].copy()

        if len(sectors_data) == 0 or idx_subset.empty:
            return []

        scores_df = compute_sector_scores(sectors_data, idx_subset)
        regime = detect_market_regime(idx_subset)
        pos_pct = compute_position_pct(idx_subset)

        # ---- Volatility-adjusted position cap ----
        vol = _index_volatility(index, date)
        vol_cap = VOL_CAP_HIGH if vol > VOL_HIGH_THRESHOLD else VOL_CAP_NORMAL
        pos_pct = min(pos_pct, int(vol_cap * 100))

        if scores_df.empty:
            return []

        effective_top_n = 2 if regime == "trending" else 3
        top_sectors = scores_df.head(effective_top_n)

        qualified = []
        for _, row in top_sectors.iterrows():
            code = get_etf_code(row["sector"])
            if code != "—":
                qualified.append({"sector": row["sector"], "code": code,
                                  "rank": len(qualified) + 1})

        if not qualified:
            return []

        n = len(qualified)
        total_nav = self.cash + sum(self.positions.values())
        invest_amount = round(total_nav * pos_pct / 100.0)
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
        df["port_cumulative"] = df["nav"] / start_nav

        idx_start = index[index["date"] == pd.Timestamp(start_date)]
        idx_end = index[index["date"] == pd.Timestamp(end_date)]
        if not idx_start.empty and not idx_end.empty:
            idx_return = (float(idx_end["close"].iloc[0]) / float(idx_start["close"].iloc[0]) - 1) * 100
        else:
            idx_return = 0.0

        port_return = (df["nav"].iloc[-1] / start_nav - 1) * 100
        excess = port_return - idx_return

        cumulative = df["port_cumulative"].values
        peak = pd.Series(cumulative).cummax().values
        drawdowns = (cumulative - peak) / peak
        max_dd = float(drawdowns.min()) * 100 if len(drawdowns) > 0 else 0

        print(f"\n{'='*80}")
        print(f"Risk-Managed Forward Test: {start_date} → {end_date}")
        print(f"Initial: {INITIAL_CAPITAL}元  |  Final NAV: {df['nav'].iloc[-1]:.2f}元")
        print(f"Portfolio: {port_return:+.2f}%  |  沪深300: {idx_return:+.2f}%  |  Excess: {excess:+.2f}%")
        print(f"Max drawdown: {max_dd:.2f}%")
        print(f"{'='*80}")
        return df


def print_daily_report(df):
    print(f"\n{'日期':<12} {'NAV':>8} {'日收益':>8} {'仓位':>8} {'操作':<55} {'沪深300':>8}")
    print("-" * 110)

    for _, row in df.iterrows():
        date = row["date"]
        nav = row["nav"]
        invested = row["invested"]
        pct = invested / nav * 100 if nav > 0 else 0
        idx_ret = row["index_return"]

        trade_str = ""
        if row["trades"]:
            parts = []
            for t in row["trades"]:
                a = "B" if t["action"] == "buy" else "S"
                r = t.get("reason", "")[:6]
                parts.append(f"{a}({r}){t['etf']}({t['amount']}元)")
            trade_str = " | ".join(parts)
        if len(trade_str) > 55:
            trade_str = trade_str[:52] + "..."

        print(f"{date:<12} {nav:>8.0f} {row['daily_return']:>+7.2f}% {pct:>6.0f}%  {trade_str:<55} {idx_ret:>+7.2f}%")
    print("-" * 110)


def main():
    parser = argparse.ArgumentParser(description="Risk-managed forward test")
    parser.add_argument("--start", type=str, default="2026-02-24")
    parser.add_argument("--end", type=str, default="2026-04-07")
    parser.add_argument("--capital", type=int, default=INITIAL_CAPITAL)
    parser.add_argument("--baseline", action="store_true", help="Run without risk management (original)")
    args = parser.parse_args()

    print(f"Loading data...")
    sectors, index_data = load_data()
    print(f"Loaded {len(sectors)} sectors, {len(index_data)} index days")

    if args.baseline:
        # Fallback to original engine
        from scripts.forward_test import ForwardTestEngine
        engine = ForwardTestEngine(capital=args.capital)
        print("\n>>> BASELINE MODE (no risk management) <<<")
    else:
        engine = RiskManagedEngine(capital=args.capital)
        print(f"\n>>> RISK-MANAGED MODE <<<")
        print(f"  Crash stop: sector {CRASH_STOP_SECTOR}% / index {CRASH_STOP_INDEX}%")
        print(f"  Trailing stop: {TRAILING_STOP_PCT*100:.0f}% from peak")
        print(f"  Daily abs momentum re-check: 10d return < {DAILY_ABS_MOM_THRESHOLD}%")
        print(f"  Vol cap: {VOL_CAP_NORMAL*100:.0f}% normal / {VOL_CAP_HIGH*100:.0f}% high vol")

    df = engine.run(sectors, index_data, args.start, args.end)
    print_daily_report(df)

    from config.settings import RESULTS_DIR
    tag = "baseline" if args.baseline else "risk_managed"
    out_path = RESULTS_DIR / f"forward_{args.start}_{args.end}_{tag}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
