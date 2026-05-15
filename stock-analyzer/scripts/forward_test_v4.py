#!/usr/bin/env python3
"""Forward test v4 — Round 3: Market Regime Position Filter.

Simple but robust: use 沪深300 trend (MA20 position + MA crossover) to
determine max position size. Only deploy capital when the market is trending.

Why this should work:
  - Feb-Apr was a downtrend → should have been mostly cash
  - Complex stops/vol targeting caused whipsaw in R1/R2
  - Dual Momentum sector selection works — problem was position sizing
    in a falling market

Regime logic:
  - close > MA20 + 2% → AGGRESSIVE: up to 80% position
  - close > MA20 (within 2%) → MODERATE: up to 50%
  - close < MA20 (within 2%) → DEFENSIVE: up to 30%
  - close < MA20 - 2% → CASH: 0% (or minimum 10%)
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

# ---- Standard Dual Momentum parameters (keep original, they work) ----
MA_PERIOD = 20
ROC_PERIOD = 20
ABS_LOOKBACK = 10

# ---- Regime-based position caps ----
REGIME_CAPS = {
    "aggressive": 80,   # close > MA20 + 2%
    "moderate": 50,     # close > MA20, within 2%
    "defensive": 30,    # close < MA20, within 2%
    "cash": 10,         # close < MA20 - 2%
}


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
    """Classify market as aggressive/moderate/defensive/cash based on MA20 position.

    Also checks the MA direction (slope) to avoid buying into declining MAs.
    """
    hist = index[(index["date"] <= date)].tail(40)
    if len(hist) < 25:
        return "moderate"

    close = float(hist["close"].iloc[-1])
    ma20 = float(hist["close"].rolling(20).mean().iloc[-1])
    deviation = (close / ma20 - 1) * 100

    # MA slope (5d change in MA20)
    ma20_series = hist["close"].rolling(20).mean()
    ma20_5d_ago = float(ma20_series.iloc[-6]) if len(ma20_series) >= 6 else ma20
    ma_slope = (ma20 - ma20_5d_ago) / ma20_5d_ago * 100 if ma20_5d_ago > 0 else 0

    if deviation > 2:
        return "aggressive"
    elif deviation > 0:
        return "moderate" if ma_slope > -0.3 else "defensive"  # falling MA → downgrade
    elif deviation > -2:
        return "defensive"
    else:
        return "cash"


class Round3Engine:
    """Forward test with market regime position filter."""

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

        regime_changes = []

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

            # ---- Step 3: Simple crash stop only (keep 1 layer) ----
            trades_today = []
            idx_row = index[index["date"] == day]
            idx_pct = float(idx_row["pct_chg"].iloc[0]) if not idx_row.empty else 0.0

            for sector_name, value in list(self.positions.items()):
                pct = _sector_pct(sectors, sector_name, day)
                if pct < -5.0:  # single-day crash → sell
                    self.cash += value
                    trades_today.append({
                        "date": day_str, "action": "sell", "reason": f"跌停({pct:+.1f}%)",
                        "sector": sector_name, "etf": get_etf_code(sector_name),
                        "amount": round(value),
                    })
                    del self.positions[sector_name]

            # ---- Step 4: Rebalance with regime-based position cap ----
            is_rebalance = (day_idx - self.last_rebalance_day) >= REBALANCE_FREQ

            if is_rebalance:
                regime = market_regime(index, day)
                pos_cap = REGIME_CAPS[regime]
                prev_regime = getattr(self, "last_regime", None)
                if regime != prev_regime:
                    regime_changes.append(f"  {day_str}: {prev_regime} → {regime} (cap={pos_cap}%)")
                    self.last_regime = regime

                # Sell all
                for sector_name, value in list(self.positions.items()):
                    trades_today.append({
                        "date": day_str, "action": "sell", "reason": "调仓",
                        "sector": sector_name, "etf": get_etf_code(sector_name),
                        "amount": round(value),
                    })
                    self.cash += value
                    del self.positions[sector_name]

                # Buy with regime cap
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

            # ---- Step 5: Record ----
            nav = self.cash + sum(self.positions.values())
            regime = market_regime(index, day)
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
        regime = detect_market_regime(idx_subset)

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
        print(f"Round 3: Market Regime Filter — {start_date} → {end_date}")
        print(f"Config: MA={MA_PERIOD} ROC={ROC_PERIOD} AbsLookback={ABS_LOOKBACK}")
        print(f"Regime caps: {REGIME_CAPS}")
        print(f"Initial: {INITIAL_CAPITAL}元 | Final NAV: {df['nav'].iloc[-1]:.2f}元")
        print(f"Portfolio: {port_return:+.2f}% | 沪深300: {idx_return:+.2f}% | Excess: {excess:+.2f}%")
        print(f"Max drawdown: {max_dd:.2f}%")
        print(f"{'='*80}")
        return df


def print_daily_report(df):
    print(f"\n{'日期':<12} {'NAV':>8} {'日收益':>8} {'仓位':>7} {'状态':>10} {'操作':<50} {'沪深300':>8}")
    print("-" * 112)

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
        if len(trade_str) > 50:
            trade_str = trade_str[:47] + "..."

        print(f"{date:<12} {nav:>8.0f} {row['daily_return']:>+7.2f}% {pct:>6.0f}% "
              f" {regime:<10} {trade_str:<50} {idx_ret:>+7.2f}%")
    print("-" * 112)


def main():
    parser = argparse.ArgumentParser(description="Round 3: Market regime filter")
    parser.add_argument("--start", type=str, default="2026-02-24")
    parser.add_argument("--end", type=str, default="2026-04-07")
    parser.add_argument("--capital", type=int, default=INITIAL_CAPITAL)
    args = parser.parse_args()

    print(f"Loading data...")
    sectors, index_data = load_data()
    print(f"Loaded {len(sectors)} sectors, {len(index_data)} index days")
    print(f"\n>>> Round 3: Market Regime Position Filter <<<")
    print(f"  Regime caps: aggressive={REGIME_CAPS['aggressive']}% moderate={REGIME_CAPS['moderate']}% "
          f"defensive={REGIME_CAPS['defensive']}% cash={REGIME_CAPS['cash']}%")

    engine = Round3Engine(capital=args.capital)
    df = engine.run(sectors, index_data, args.start, args.end)
    print_daily_report(df)

    from config.settings import RESULTS_DIR
    out_path = RESULTS_DIR / f"forward_{args.start}_{args.end}_round3.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
