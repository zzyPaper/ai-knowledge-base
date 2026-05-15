#!/usr/bin/env python3
"""Forward test v5 — Round 4: Faster Re-entry + Reversal Day Detection.

Target: beat 沪深300 by >= 5 percentage points.

Changes from Round 3:
  1. MA10 for regime detection (replaces MA20) — faster trend response
  2. Reversal day boost: index daily return > 2% → bump regime by 1 level
  3. Higher aggressive cap: 90% (was 80%)
  4. Immediate rebalance on reversal signal (don't wait for regular schedule)
  5. Cash regime: 0% (was 10%) — stop the slow bleed
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

# ---- Standard Dual Momentum ----
MA_PERIOD = 20
ROC_PERIOD = 20
ABS_LOOKBACK = 10

# ---- Regime caps ----
REGIME_CAPS = {
    "aggressive": 90,
    "moderate": 60,
    "defensive": 30,
    "cash": 0,
}

# ---- Re-entry acceleration ----
REVERSAL_DAY_THRESHOLD = 2.0   # index daily return >2% → reversal day
REGIME_MA_PERIOD = 10           # shorter MA for faster regime detection


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


def market_regime(index, date) -> str:
    """Classify market regime using MA10 (faster than MA20).

    Returns: aggressive / moderate / defensive / cash
    """
    hist = index[(index["date"] <= date)].tail(40)
    if len(hist) < 15:
        return "moderate"

    close = float(hist["close"].iloc[-1])
    ma = float(hist["close"].rolling(REGIME_MA_PERIOD).mean().iloc[-1])
    deviation = (close / ma - 1) * 100

    # MA slope (5d change)
    ma_series = hist["close"].rolling(REGIME_MA_PERIOD).mean()
    ma_5d_ago = float(ma_series.iloc[-6]) if len(ma_series) >= 6 else ma
    ma_slope = (ma - ma_5d_ago) / ma_5d_ago * 100 if ma_5d_ago > 0 else 0

    if deviation > 2:
        return "aggressive"
    elif deviation > 0:
        if ma_slope > 0:
            return "moderate"  # price > MA10 AND MA10 rising
        return "defensive"     # price > MA10 BUT MA10 falling
    elif deviation > -2:
        return "defensive"
    else:
        return "cash"


def bump_regime(regime: str, levels: int = 1) -> str:
    """Boost regime by N levels (for reversal day)."""
    order = ["cash", "defensive", "moderate", "aggressive"]
    idx = order.index(regime)
    new_idx = min(idx + levels, len(order) - 1)
    return order[new_idx]


class Round4Engine:
    """Forward test with fast re-entry and reversal detection."""

    def __init__(self, capital: int = INITIAL_CAPITAL):
        self.cash = float(capital)
        self.positions: dict[str, float] = {}
        self.daily_log: list[dict] = []
        self.last_rebalance_day = -REBALANCE_FREQ
        self.force_rebalance = False

    def run(self, sectors: dict, index: pd.DataFrame,
            start_date: str, end_date: str):
        trading_days = sorted(index["date"].unique())
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)
        trading_days = [d for d in trading_days if start_dt <= d <= end_dt]

        regime_changes = []
        reversal_events = []

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

            # ---- Step 3: Crash stop ----
            trades_today = []
            idx_row = index[index["date"] == day]
            idx_pct = float(idx_row["pct_chg"].iloc[0]) if not idx_row.empty else 0.0

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

            # ---- Step 4: Regime detection + reversal check ----
            regime = market_regime(index, day)

            # Reversal day: index > 2% → force rebalance at higher regime
            is_reversal = idx_pct > REVERSAL_DAY_THRESHOLD
            if is_reversal and regime in ("cash", "defensive"):
                old_regime = regime
                regime = bump_regime(regime, 2 if regime == "cash" else 1)
                self.force_rebalance = True
                reversal_events.append(
                    f"  {day_str}: 反弹日(沪深300 {idx_pct:+.2f}%) → {old_regime}→{regime}, 强制调仓")

            is_rebalance = ((day_idx - self.last_rebalance_day) >= REBALANCE_FREQ
                            or self.force_rebalance)

            prev_regime = getattr(self, "last_regime", None)
            if regime != prev_regime:
                regime_changes.append(
                    f"  {day_str}: {prev_regime} → {regime} "
                    f"(cap={REGIME_CAPS[regime]}%)"
                    f"{' [反弹]' if is_reversal else ''}")
                self.last_regime = regime

            # ---- Step 5: Rebalance ----
            if is_rebalance:
                for sector_name, value in list(self.positions.items()):
                    trades_today.append({
                        "date": day_str, "action": "sell", "reason": "调仓",
                        "sector": sector_name, "etf": get_etf_code(sector_name),
                        "amount": round(value),
                    })
                    self.cash += value
                    del self.positions[sector_name]

                pos_cap = REGIME_CAPS[regime]
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
                self.force_rebalance = False

            # ---- Step 6: Record ----
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
                "regime": regime,
                "pos_cap": REGIME_CAPS.get(regime, 50),
            })

        if reversal_events:
            print(f"\n{'='*60}")
            print(f"反弹日检测 ({len(reversal_events)} 次)")
            print(f"{'='*60}")
            for ev in reversal_events:
                print(ev)

        if regime_changes:
            print(f"\n{'='*60}")
            print(f"市场状态切换 ({len(regime_changes)} 次)")
            print(f"{'='*60}")
            for rc in regime_changes:
                print(rc)

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
        dm_regime = detect_market_regime(idx_subset)

        if scores_df.empty:
            return []

        effective_top_n = 2 if dm_regime == "trending" else 3
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

        print(f"\n{'='*80}")
        print(f"Round 4: Fast Re-entry + Reversal Detection — {start_date} → {end_date}")
        print(f"Regime MA: {REGIME_MA_PERIOD}d | Reversal threshold: {REVERSAL_DAY_THRESHOLD}%")
        print(f"Caps: {REGIME_CAPS}")
        print(f"Initial: {INITIAL_CAPITAL}元 | Final NAV: {df['nav'].iloc[-1]:.2f}元")
        print(f"Portfolio: {port_return:+.2f}% | 沪深300: {idx_return:+.2f}% | Excess: {excess:+.2f}%")
        print(f"Max drawdown: {max_dd:.2f}%")
        print(f"{'='*80}")
        return df


def print_daily_report(df):
    print(f"\n{'日期':<12} {'NAV':>8} {'日收益':>8} {'仓位':>7} {'状态':>10} {'操作':<45} {'沪深300':>8}")
    print("-" * 108)

    for _, row in df.iterrows():
        date = row["date"]
        nav = row["nav"]
        invested = row["invested"]
        pct = invested / nav * 100 if nav > 0 else 0
        idx_ret = row["index_return"]
        regime = row.get("regime", "?")

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
              f" {regime:<10} {trade_str:<45} {idx_ret:>+7.2f}%")
    print("-" * 108)


def main():
    parser = argparse.ArgumentParser(description="Round 4: Fast re-entry + reversal detection")
    parser.add_argument("--start", type=str, default="2026-02-02")
    parser.add_argument("--end", type=str, default="2026-04-30")
    parser.add_argument("--capital", type=int, default=INITIAL_CAPITAL)
    args = parser.parse_args()

    print(f"Loading data...")
    sectors, index_data = load_data()
    print(f"\n>>> Round 4: MA{REGIME_MA_PERIOD} Regime + "
          f"Reversal>{REVERSAL_DAY_THRESHOLD}% + Caps={REGIME_CAPS} <<<")

    engine = Round4Engine(capital=args.capital)
    df = engine.run(sectors, index_data, args.start, args.end)
    print_daily_report(df)

    from config.settings import RESULTS_DIR
    out_path = RESULTS_DIR / f"forward_{args.start}_{args.end}_round4.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
