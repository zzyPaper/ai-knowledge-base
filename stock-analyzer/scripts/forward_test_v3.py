#!/usr/bin/env python3
"""Forward test v3 — Round 2: Volatility Targeting + Shorter A-share Lookbacks.

Improvements (from research):
  1. Volatility targeting (Barroso & Santa-Clara 2015):
     position_pct = base_pct × (target_vol / realized_vol), clamped to [0.2, 1.2]
  2. Shorter momentum windows (华福证券 2025):
     MA 20→10, ROC 20→10, absolute momentum 10→5
     (A股轮动加速, 热点持续性<20天, MA10分年度胜率优于MA20)
  3. Keeps daily crash stop (-5% sector), drops market circuit breaker (whipsaw)
  4. Keeps daily abs momentum re-check (5d window)
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
from config.sector_etf_map import get_etf_code, get_etf

INITIAL_CAPITAL = 5000
REBALANCE_FREQ = 5

# ---- Risk parameters ----
CRASH_STOP_SECTOR = -5.0       # single-day sector drop → immediate exit
DAILY_ABS_MOM_THRESHOLD = -2.0 # 5d return below this → exit (was 0%, now -2% to reduce whipsaw)

# ---- Volatility targeting (Barroso & Santa-Clara 2015) ----
TARGET_ANNUAL_VOL = 0.12       # 12% annual target vol
VOL_LOOKBACK = 20              # trading days for realized vol estimation
VOL_SCALE_MIN = 0.25           # minimum scale (never go below 25% of base)
VOL_SCALE_MAX = 1.20           # maximum scale (cap leverage)

# ---- Shorter A-share lookbacks (华福证券 2025) ----
MA_PERIOD = 10                 # was 20
ROC_PERIOD = 10                # was 20
ABS_LOOKBACK = 5               # was 10


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
    df = sectors.get(name)
    if df is None:
        return None
    hist = df[df["date"] <= date]
    if len(hist) < n + 1:
        return None
    return float(hist["close"].iloc[-1] / hist["close"].iloc[-(n + 1)] - 1) * 100


def _vol_scale(index, date):
    """Barroso & Santa-Clara (2015) volatility scaling factor.

    scale = target_vol / realized_vol
    realized_vol = std(daily_rets, 20d) * sqrt(252)
    """
    hist = index[(index["date"] <= date)].tail(VOL_LOOKBACK + 1)
    if len(hist) < 10:
        return 1.0
    rets = hist["close"].pct_change().dropna()
    daily_vol = float(rets.std())
    annual_vol = daily_vol * np.sqrt(252)
    if annual_vol < 0.01:
        return VOL_SCALE_MAX
    scale = TARGET_ANNUAL_VOL / annual_vol
    return max(VOL_SCALE_MIN, min(scale, VOL_SCALE_MAX))


class Round2Engine:
    """Forward test with volatility targeting + short A-share lookbacks."""

    def __init__(self, capital: int = INITIAL_CAPITAL):
        self.cash = float(capital)
        self.positions: dict[str, float] = {}
        self.trades_log: list[dict] = []
        self.daily_log: list[dict] = []
        self.last_rebalance_day = -REBALANCE_FREQ

    def run(self, sectors: dict, index: pd.DataFrame,
            start_date: str, end_date: str):
        trading_days = sorted(index["date"].unique())
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)
        trading_days = [d for d in trading_days if start_dt <= d <= end_dt]

        stop_events = []

        for day_idx, day in enumerate(trading_days):
            day_str = day.strftime("%Y-%m-%d")

            # ---- Step 1: NAV before P&L ----
            nav_before = self.cash + sum(self.positions.values())

            # ---- Step 2: P&L ----
            daily_pnl = 0.0
            for sector_name, value in list(self.positions.items()):
                pct = _sector_pct(sectors, sector_name, day)
                pnl = value * pct / 100.0
                daily_pnl += pnl
                self.positions[sector_name] = value + pnl

            # ---- Step 3: Risk checks (daily crash stop + abs momentum) ----
            trades_today = []
            idx_row = index[index["date"] == day]
            idx_pct = float(idx_row["pct_chg"].iloc[0]) if not idx_row.empty else 0.0

            for sector_name, value in list(self.positions.items()):
                if value <= 0:
                    continue
                pct = _sector_pct(sectors, sector_name, day)
                should_sell = False
                reason = ""

                # Crash stop: single-day >5% drop
                if pct < CRASH_STOP_SECTOR:
                    should_sell = True
                    reason = f"跌停({pct:+.1f}%)"
                else:
                    # Daily abs momentum: 5d return < threshold
                    ret_5d = _sector_n_day_return(sectors, sector_name, day, ABS_LOOKBACK)
                    if ret_5d is not None and ret_5d < DAILY_ABS_MOM_THRESHOLD:
                        should_sell = True
                        reason = f"动量为负({ret_5d:+.1f}%)"

                if should_sell:
                    self.cash += value
                    trades_today.append({
                        "date": day_str, "action": "sell", "reason": reason,
                        "sector": sector_name, "etf": get_etf_code(sector_name),
                        "amount": round(value),
                    })
                    stop_events.append(
                        f"  {day_str}: {reason} → 卖出 {sector_name}({get_etf_code(sector_name)}) {round(value)}元")
                    del self.positions[sector_name]

            # ---- Step 4: Rebalance ----
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

                targets = self._compute_targets(sectors, index, day)
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

        if stop_events:
            print(f"\n{'='*60}")
            print(f"风控事件 ({len(stop_events)} 次)")
            print(f"{'='*60}")
            for ev in stop_events:
                print(ev)

        return self._summary(start_date, end_date, index)

    def _compute_targets(self, sectors, index, date):
        """Compute target positions with vol targeting + short lookbacks."""
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

        # ---- Use shorter A-share lookbacks ----
        scores_df = compute_sector_scores(
            sectors_data, idx_subset,
            ma_period=MA_PERIOD,
            roc_period=ROC_PERIOD,
            lookback=ABS_LOOKBACK,
        )
        regime = detect_market_regime(idx_subset)

        # ---- Volatility targeting (Barroso & Santa-Clara 2015) ----
        base_pct = _compute_base_position_pct(idx_subset)
        vol_scale = _vol_scale(index, date)
        pos_pct = int(base_pct * vol_scale)
        pos_pct = max(10, min(pos_pct, 90))  # floor 10%, ceiling 90%

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

        # Log vol adjustment
        if abs(pos_pct - base_pct) > 5:
            pass  # will be shown in daily report

        return targets

    def _summary(self, start_date, end_date, index):
        df = pd.DataFrame(self.daily_log)
        if df.empty:
            return df

        start_nav = INITIAL_CAPITAL
        df["port_cumulative"] = df["nav"] / start_nav

        idx_start = index[index["date"] == pd.Timestamp(start_date)]
        idx_end = index[index["date"] == pd.Timestamp(end_date)]
        idx_return = 0.0
        if not idx_start.empty and not idx_end.empty:
            idx_return = (float(idx_end["close"].iloc[0]) / float(idx_start["close"].iloc[0]) - 1) * 100

        port_return = (df["nav"].iloc[-1] / start_nav - 1) * 100
        excess = port_return - idx_return

        cumulative = df["port_cumulative"].values
        peak = pd.Series(cumulative).cummax().values
        drawdowns = (cumulative - peak) / peak
        max_dd = float(drawdowns.min()) * 100 if len(drawdowns) > 0 else 0

        df["vol_scale"] = [_vol_scale(index, pd.Timestamp(d)) for d in df["date"]]

        print(f"\n{'='*80}")
        print(f"Round 2: Vol Targeting + Short Lookbacks — {start_date} → {end_date}")
        print(f"Config: MA={MA_PERIOD} ROC={ROC_PERIOD} AbsLookback={ABS_LOOKBACK}")
        print(f"Vol target: {TARGET_ANNUAL_VOL*100:.0f}% annual | Scale: [{VOL_SCALE_MIN},{VOL_SCALE_MAX}]")
        print(f"Initial: {INITIAL_CAPITAL}元 | Final NAV: {df['nav'].iloc[-1]:.2f}元")
        print(f"Portfolio: {port_return:+.2f}% | 沪深300: {idx_return:+.2f}% | Excess: {excess:+.2f}%")
        print(f"Max drawdown: {max_dd:.2f}%")
        print(f"{'='*80}")
        return df


def _compute_base_position_pct(index_hist) -> int:
    """Original position sizing without vol targeting (Faber 2007 approach)."""
    if index_hist is None or len(index_hist) < 10:
        return 60
    close = float(index_hist["close"].iloc[-1])
    prev = float(index_hist["close"].iloc[-11]) if len(index_hist) >= 11 else float(index_hist["close"].iloc[0])
    if prev == 0:
        return 60
    ret_pct = (close / prev - 1) * 100
    # Same mapping as compute_position_pct but 10d instead of 20d
    if ret_pct >= 10:
        return 100
    elif ret_pct >= 5:
        return int(80 + (ret_pct - 5) * 4)
    elif ret_pct >= 0:
        return int(60 + ret_pct * 4)
    elif ret_pct >= -5:
        return int(35 + (ret_pct + 5) * 5)
    elif ret_pct >= -10:
        return int(10 + (ret_pct + 10) * 5)
    else:
        return 10


def print_daily_report(df):
    print(f"\n{'日期':<12} {'NAV':>8} {'日收益':>8} {'仓位':>7} {'波动率':>7} {'操作':<50} {'沪深300':>8}")
    print("-" * 112)

    for _, row in df.iterrows():
        date = row["date"]
        nav = row["nav"]
        invested = row["invested"]
        pct = invested / nav * 100 if nav > 0 else 0
        idx_ret = row["index_return"]
        vol_s = row.get("vol_scale", 1.0)

        trade_str = ""
        if row["trades"]:
            parts = []
            for t in row["trades"]:
                a = "B" if t["action"] == "buy" else "S"
                r = t.get("reason", "")[:4]
                parts.append(f"{a}({r}){t['etf']}({t['amount']}元)")
            trade_str = " | ".join(parts)
        if len(trade_str) > 50:
            trade_str = trade_str[:47] + "..."

        print(f"{date:<12} {nav:>8.0f} {row['daily_return']:>+7.2f}% {pct:>6.0f}% "
              f" {vol_s:>5.2f}x {trade_str:<50} {idx_ret:>+7.2f}%")
    print("-" * 112)


def main():
    parser = argparse.ArgumentParser(description="Round 2: Vol targeting + short lookbacks")
    parser.add_argument("--start", type=str, default="2026-02-24")
    parser.add_argument("--end", type=str, default="2026-04-07")
    parser.add_argument("--capital", type=int, default=INITIAL_CAPITAL)
    args = parser.parse_args()

    print(f"Loading data...")
    sectors, index_data = load_data()
    print(f"Loaded {len(sectors)} sectors, {len(index_data)} index days")
    print(f"\n>>> Round 2: Vol Targeting + Short A-share Lookbacks <<<")
    print(f"  MA={MA_PERIOD} ROC={ROC_PERIOD} AbsLookback={ABS_LOOKBACK}")
    print(f"  Vol target={TARGET_ANNUAL_VOL*100:.0f}% | Crash stop={CRASH_STOP_SECTOR}% | "
          f"Abs mom threshold={DAILY_ABS_MOM_THRESHOLD}%")

    engine = Round2Engine(capital=args.capital)
    df = engine.run(sectors, index_data, args.start, args.end)
    print_daily_report(df)

    from config.settings import RESULTS_DIR
    out_path = RESULTS_DIR / f"forward_{args.start}_{args.end}_round2.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
