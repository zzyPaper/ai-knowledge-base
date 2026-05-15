"""Forward test executor — daily trade simulation engine.

Executes the professional sector rotation strategy day-by-day with:
  1. Proper P&L timing (nav captured BEFORE P&L, new positions earn from next day)
  2. Market regime detection (multi-timeframe MA structure)
  3. Signal computation (trend + crowding)
  4. Risk management (vol targeting + crash stop)
  5. Portfolio construction (rank-weighted + constraints)

The executor is strategy-agnostic — it calls into the layers but doesn't
contain strategy logic itself.
"""

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Optional
import pandas as pd
import numpy as np

from src.engine.config import StrategyConfig, MarketTimingConfig
from src.engine.regime import RegimeDetector, RegimeResult, Regime
from src.engine.signals import SignalPipeline, SignalResult
from src.engine.risk import RiskManager, RiskAssessment
from src.engine.portfolio import PortfolioBuilder, PortfolioPlan, TargetAllocation


def _sector_pct(sectors: dict, name: str, date: pd.Timestamp) -> float:
    df = sectors.get(name)
    if df is None:
        return 0.0
    row = df[df["date"] == date]
    return float(row["pct_chg"].iloc[0]) if not row.empty else 0.0


def _index_n_day_return(index: pd.DataFrame, date: pd.Timestamp, n: int) -> float:
    """Index N-trading-day return in pct."""
    hist = index[(index["date"] <= date)]
    if len(hist) < n + 1:
        return 0.0
    return float(hist["close"].iloc[-1] / hist["close"].iloc[-(n + 1)] - 1) * 100


def _market_breadth(sectors: dict, date: pd.Timestamp, lookback: int = 5) -> float:
    """Fraction of sectors with negative N-day returns.

    High values (>50%) signal broad market deterioration — used
    to reduce position cap and protect against correlated drawdowns.
    """
    declining = 0
    total = 0
    for df in sectors.values():
        hist = df[df["date"] <= date]
        if len(hist) < lookback + 1:
            continue
        ret = float(hist["close"].iloc[-1] / hist["close"].iloc[-(lookback + 1)] - 1)
        total += 1
        if ret < 0:
            declining += 1
    return declining / total if total > 0 else 0.0


@dataclass
class DailyRecord:
    """One day's portfolio snapshot."""
    date: str
    nav: float
    cash: float
    invested: float
    daily_pnl: float
    daily_return_pct: float
    index_return_pct: float
    in_market: bool
    regime: str
    position_cap: int
    vol_scale: float
    is_rebalance: bool
    trades: list[dict] = field(default_factory=list)
    positions: dict = field(default_factory=dict)


class ForwardExecutor:
    """Professional forward-test execution engine."""

    def __init__(self, config: StrategyConfig = StrategyConfig(), verbose: bool = True,
                 reset_nav: Optional[float] = None, warmup_only: bool = False):
        """reset_nav: if provided, start with this NAV instead of initial_capital
        warmup_only: if True, skip actual trading during warmup period (only compute signals)"""
        self.config = config
        self.verbose = verbose
        self._reset_nav = reset_nav
        self._warmup_only = warmup_only
        self.cash = reset_nav if reset_nav is not None else float(config.portfolio.initial_capital)
        self.positions: dict[str, float] = {}
        self.daily_records: list[DailyRecord] = []
        self.last_rebalance_day = -config.portfolio.rebalance_freq
        self.in_market = False

        # Sub-components
        self.regime_detector = RegimeDetector(config.regime)
        self.signal_pipeline = SignalPipeline(config.signals)
        self.risk_manager = RiskManager(config.risk)
        self.portfolio_builder = PortfolioBuilder(config.portfolio)

        # State tracking
        self.events: list[str] = []
        self.regime_changes: list[str] = []
        self.reversal_lock_until: int = -1  # day_idx until which exit is blocked
        self.reversal_grace_until: int = -1  # day_idx until which cap gradually decays
        self.reversal_peak_cap: int = 0  # cap used during lock (decay starting point)
        self.last_reversal_entry: str = ""  # date of last reversal-triggered entry
        self.last_entry_day: int = -1  # day_idx of last market entry (anti-whipsaw)
        self.last_breadth_warning: int = -999  # day_idx of last breadth deterioration

    def run(self, sectors: dict, index: pd.DataFrame,
            start_date: str, end_date: str,
            warmup_days: int = 0) -> pd.DataFrame:
        """Execute forward test day by day.
        
        warmup_days: skip this many days of actual trading (only compute signals).
        The account starts with reset_nav during warmup, but positions/cash
        only change after warmup_days.
        """
        trading_days = self._trading_days(index, start_date, end_date)

        # Initialize risk peak (use reset_nav if provided)
        init_nav = self._reset_nav if self._reset_nav is not None else self.config.portfolio.initial_capital
        self.risk_manager.update_peak(init_nav)
        
        # Warmup tracking
        warmup_nav_snapshot = init_nav  # track NAV during warmup
        is_in_warmup = warmup_days > 0

        for day_idx, day in enumerate(trading_days):
            day_str = day.strftime("%Y-%m-%d")
            
            # Check if warmup ended this iteration
            if is_in_warmup and day_idx == warmup_days:
                # End of warmup: reset cash to initial, clear positions
                self.cash = init_nav
                self.positions = {}
                self.last_rebalance_day = day_idx - self.config.portfolio.rebalance_freq
                is_in_warmup = False

            # ---- Step 1: Snapshot NAV before P&L ----
            nav_before = self.cash + sum(self.positions.values())

            # ---- Step 2: Apply P&L to existing positions ----
            daily_pnl = 0.0
            for sector_name, value in list(self.positions.items()):
                pct = _sector_pct(sectors, sector_name, day)
                pnl = value * pct / 100.0
                daily_pnl += pnl
                self.positions[sector_name] = value + pnl

            # ---- Step 3: Crash stop (sector-level) ----
            idx_row = index[index["date"] == day]
            idx_pct = float(idx_row["pct_chg"].iloc[0]) if not idx_row.empty else 0.0
            trades_today = []
            had_crash = False

            for sector_name, value in list(self.positions.items()):
                pct = _sector_pct(sectors, sector_name, day)
                if self.risk_manager.check_crash_stop(sector_name, pct):
                    etf_code = self._etf_for(sector_name)
                    trades_today.append({
                        "date": day_str, "action": "sell",
                        "reason": f"跌停({pct:+.1f}%)",
                        "sector": sector_name, "etf": etf_code,
                        "amount": round(value),
                    })
                    self.cash += value
                    del self.positions[sector_name]
                    had_crash = True
                    self.events.append(f"{day_str}: 跌停止损 {sector_name} {pct:+.1f}%")

            # ---- Step 4: Market timing + Regime detection ----
            in_market, _, is_reversal = self._market_timing(index, day, day_idx)
            regime_result = self.regime_detector.detect(index, day)

            # Track regime changes
            prev_regime = getattr(self, "last_regime", None)
            if prev_regime and prev_regime != regime_result.regime:
                self.regime_changes.append(
                    f"  {day_str}: {prev_regime} → {regime_result.regime} "
                    f"(dev={regime_result.deviation_pct:+.1f}%)")
            self.last_regime = regime_result.regime

            # Market entry/exit logic
            prev_in = getattr(self, "last_in_market_signal", None)
            if prev_in is not None and prev_in != in_market:
                self.events.append(
                    f"  {day_str}: {'ENTER' if in_market else 'EXIT'} market")
                if not in_market:
                    self.in_market = False
            self.last_in_market_signal = in_market

            # ---- Step 5: Rebalance ----
            is_rebalance = (day_idx - self.last_rebalance_day) >= self.config.portfolio.rebalance_freq

            # Enter on next rebalance after signal
            if is_rebalance and in_market and not self.in_market:
                self.in_market = True
                self.last_entry_day = day_idx
            # Exit immediately on signal
            if not in_market and self.in_market:
                is_rebalance = True
            # Force immediate rebalance on reversal entry (don't wait)
            if is_reversal and not self.in_market:
                is_rebalance = True
                self.in_market = True
                self.last_entry_day = day_idx
            # Crash forces rebalance
            if had_crash:
                is_rebalance = True

            base_cap = regime_result.position_cap if self.in_market else 0

            # Reversal lock: maintain RECOVERY cap throughout lock period
            if day_idx <= self.reversal_lock_until and self.in_market:
                base_cap = max(base_cap, Regime.CAPS.get(Regime.RECOVERY, 70))

            # Grace period: linearly decay cap from lock-era peak to current regime cap
            if self.reversal_lock_until < day_idx <= self.reversal_grace_until:
                grace_total = self.reversal_grace_until - self.reversal_lock_until
                grace_elapsed = day_idx - self.reversal_lock_until
                decay = grace_elapsed / max(grace_total, 1)
                grace_cap = round(
                    self.reversal_peak_cap +
                    (regime_result.position_cap - self.reversal_peak_cap) * decay
                )
                base_cap = max(base_cap, grace_cap)

            # Market breadth filter: regime-calibrated thresholds.
            # BULL uses highest bar (80%+ declining = genuine panic), BEAR already at 0%.
            _BREADTH_DANGER = {
                Regime.BULL: 0.80,
                Regime.RECOVERY: 0.70,
                Regime.NEUTRAL: 0.60,
                Regime.CORRECTION: 0.50,
            }
            if self.in_market and base_cap > 0:
                danger = _BREADTH_DANGER.get(regime_result.regime)
                if danger is not None:
                    pct_declining = _market_breadth(sectors, day, self.config.timing.breadth_lookback)
                    if pct_declining > danger:
                        self.last_breadth_warning = day_idx
                        breadth_cap = max(int(base_cap * 0.5), self.config.risk.position_cap_min)
                        if breadth_cap < base_cap:
                            self.events.append(
                                f"  {day_str}: 市场广度恶化 "
                                f"({pct_declining:.0%}板块下跌) → 仓位 {base_cap}%→{breadth_cap}%")
                            base_cap = breadth_cap

            # Breadth-regime convergence: when breadth recently warned,
            # don't let regime push cap above NEUTRAL. This prevents the
            # lagging MA-based regime from overruling the faster breadth signal
            # in choppy/transitioning markets (e.g., September 2025).
            if self.in_market and base_cap > Regime.CAPS.get(Regime.NEUTRAL, 50):
                if day_idx - self.last_breadth_warning <= 5:
                    base_cap = Regime.CAPS.get(Regime.NEUTRAL, 50)

            if is_rebalance:
                # Sell all existing positions
                for sector_name, value in list(self.positions.items()):
                    etf_code = self._etf_for(sector_name)
                    trades_today.append({
                        "date": day_str, "action": "sell", "reason": "调仓",
                        "sector": sector_name, "etf": etf_code,
                        "amount": round(value),
                    })
                    self.cash += value
                    del self.positions[sector_name]

                if self.in_market and base_cap > 0:
                    # Risk assessment
                    nav = self.cash + sum(self.positions.values())
                    risk = self.risk_manager.assess(index, day, nav, base_cap)
                    effective_cap = risk.position_cap
                    vol_scale = risk.vol_scale

                    # Compute signals and build portfolio
                    signals = self._compute_signals(sectors, index, day, regime_result.regime)
                    plan = self.portfolio_builder.build(
                        signals.scores_df, nav, effective_cap)

                    for t in plan.targets:
                        self.positions[t.sector] = t.amount
                        trades_today.append({
                            "date": day_str, "action": "buy", "reason": "调仓",
                            "sector": t.sector, "etf": t.etf,
                            "amount": t.amount,
                        })
                    self.cash -= plan.total_invest
                else:
                    effective_cap = 0
                    vol_scale = 1.0

                self.last_rebalance_day = day_idx
            else:
                effective_cap = base_cap if self.in_market else 0
                vol_scale = 1.0

            # ---- Step 6: Record ----
            nav = self.cash + sum(self.positions.values())
            self.risk_manager.update_peak(nav)

            self.daily_records.append(DailyRecord(
                date=day_str,
                nav=nav,
                cash=self.cash,
                invested=sum(self.positions.values()),
                daily_pnl=daily_pnl,
                daily_return_pct=(nav / nav_before - 1) * 100 if nav_before > 0 else 0,
                index_return_pct=idx_pct,
                in_market=self.in_market,
                regime=regime_result.regime,
                position_cap=effective_cap,
                vol_scale=vol_scale,
                is_rebalance=is_rebalance,
                trades=trades_today,
                positions={s: round(v) for s, v in self.positions.items()},
            ))

        if self.verbose:
            self._print_events()
        return self._summary(start_date, end_date, index, verbose=self.verbose)

    def _market_timing(self, index: pd.DataFrame, date: pd.Timestamp,
                        day_idx: int) -> tuple[bool, float, bool]:
        """Slow entry + fast exit market timing with reversal detection.

        Returns (in_market, trend_strength, is_reversal).
        """
        cfg = self.config.timing
        ret_entry = _index_n_day_return(index, date, cfg.entry_lookback)
        ret_exit = _index_n_day_return(index, date, cfg.exit_lookback)

        # During reversal lock, force stay in market if already in
        if day_idx <= self.reversal_lock_until and self.in_market:
            return True, ret_entry, False

        # Minimum holding period: force stay in market (anti-whipsaw)
        if self.in_market and self.last_entry_day >= 0:
            if day_idx - self.last_entry_day < cfg.min_hold_days:
                return True, ret_entry, False

        # Entry: long-term trend positive + above threshold
        can_enter = ret_entry > cfg.entry_threshold
        is_reversal = False

        # Adaptive exit threshold based on 60d trend strength.
        # Stronger long-term trend → more tolerant of short-term pullbacks.
        ret_60d = _index_n_day_return(index, date, 60)
        if ret_60d > 5.0:
            adaptive_exit = -3.0
        elif ret_60d > 3.0:
            adaptive_exit = -2.0
        elif ret_60d > 1.0:
            adaptive_exit = -1.0
        elif ret_60d > 0.0:
            adaptive_exit = -0.8
        else:
            adaptive_exit = cfg.exit_threshold

        # Reversal detection: override slow entry
        if cfg.reversal_enabled and not can_enter:
            idx_row = index[index["date"] == date]
            idx_pct = float(idx_row["pct_chg"].iloc[0]) if not idx_row.empty else 0.0
            if idx_pct > cfg.reversal_single_day:
                can_enter = True
                is_reversal = True
                # Lock exit for 5 days after reversal entry
                self.reversal_lock_until = day_idx + 5
                # Grace period: cap decays from RECOVERY to regime cap over N days
                self.reversal_grace_until = day_idx + 5 + cfg.reversal_grace_days
                self.reversal_peak_cap = Regime.CAPS.get(Regime.RECOVERY, 70)
                self.last_reversal_entry = date.strftime('%Y-%m-%d')
                self.events.append(
                    f"  {date.strftime('%Y-%m-%d')}: 反弹日 "
                    f"(单日{idx_pct:+.2f}%) → 强制入场, 锁仓至"
                    f" D+5, 仓位衰减至 D+{5 + cfg.reversal_grace_days}")


        should_exit = ret_exit < adaptive_exit

        # Hard override: don't exit if 20d is exceptionally strong
        if should_exit and ret_entry > cfg.exit_override_20d_threshold:
            should_exit = False

        # Reversal lock: block exit during lock period
        if should_exit and day_idx <= self.reversal_lock_until:
            should_exit = False

        # Minimum holding period: block exit for N days after entry (anti-whipsaw)
        if should_exit and self.in_market and self.last_entry_day >= 0:
            if day_idx - self.last_entry_day < cfg.min_hold_days:
                should_exit = False

        return can_enter and not should_exit, ret_entry, is_reversal

    def _compute_signals(self, sectors: dict, index: pd.DataFrame,
                         date: pd.Timestamp, regime: str) -> SignalResult:
        """Prepare sector data and compute signals."""
        start_dt = date - timedelta(days=self.config.signals.data_lookback_days)

        sectors_data = {}
        for name, df in sectors.items():
            subset = df[(df["date"] >= start_dt) & (df["date"] <= date)].copy()
            if len(subset) >= self.config.signals.min_data_points:
                sectors_data[name] = subset

        return self.signal_pipeline.compute(sectors_data, regime)

    def _trading_days(self, index: pd.DataFrame,
                      start_date: str, end_date: str) -> list:
        days = sorted(index["date"].unique())
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)
        return [d for d in days if start_dt <= d <= end_dt]

    @staticmethod
    def _etf_for(sector_name: str) -> str:
        from config.sector_etf_map import get_etf_code
        return get_etf_code(sector_name)

    def _print_events(self):
        if self.events:
            print(f"\n{'='*60}")
            print(f"策略事件 ({len(self.events)} 次)")
            print(f"{'='*60}")
            for ev in self.events:
                print(ev)
        if self.regime_changes:
            print(f"\n{'='*60}")
            print(f"市场状态切换 ({len(self.regime_changes)} 次)")
            print(f"{'='*60}")
            for rc in self.regime_changes:
                print(rc)

    def _summary(self, start_date: str, end_date: str,
                 index: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        df = pd.DataFrame([r.__dict__ for r in self.daily_records])
        if df.empty:
            return df

        start_nav = self.config.portfolio.initial_capital
        idx_start = index[index["date"] == pd.Timestamp(start_date)]
        idx_end = index[index["date"] == pd.Timestamp(end_date)]
        idx_return = 0.0
        if not idx_start.empty and not idx_end.empty:
            idx_return = (float(idx_end["close"].iloc[0]) /
                          float(idx_start["close"].iloc[0]) - 1) * 100

        port_return = (df["nav"].iloc[-1] / start_nav - 1) * 100
        excess = port_return - idx_return

        cumulative = df["nav"].values / start_nav
        peak_series = pd.Series(cumulative).cummax().values
        drawdowns = (cumulative - peak_series) / peak_series
        max_dd = float(drawdowns.min()) * 100 if len(drawdowns) > 0 else 0

        days_in = (df["in_market"] == True).sum()
        avg_vol_scale = df[df["vol_scale"] > 0]["vol_scale"].mean()

        if verbose:
            print(f"\n{'='*80}")
            print(f"Professional Sector Rotation — {start_date} → {end_date}")
            print(f"Strategy: {self.config.name} v{self.config.version}")
            print(f"Initial: {start_nav:.0f}元 | Final NAV: {df['nav'].iloc[-1]:.2f}元")
            print(f"Portfolio: {port_return:+.2f}% | 沪深300: {idx_return:+.2f}% | Excess: {excess:+.2f}%")
            print(f"Max DD: {max_dd:.2f}% | Days in market: {days_in}/{len(df)} "
                  f"| Avg vol scale: {avg_vol_scale:.2f}")
            print(f"Final positions: {df['positions'].iloc[-1]}")
            print(f"{'='*80}")
        return df
