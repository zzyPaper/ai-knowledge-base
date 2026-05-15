"""Portfolio construction layer.

Takes sector scores and builds target allocation with constraints:
  - Rank-weighted allocation (中银 S7 methodology)
  - Sector concentration limits
  - ETF availability filtering
  - Minimum/maximum holdings
  - Turnover control

Reference:
  - 中银证券 S7: rank等权分配
  - 国盛证券 ETF配置模型
"""

from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

from src.engine.config import PortfolioConfig
from config.sector_etf_map import get_etf_code


@dataclass
class TargetAllocation:
    """A single target position."""
    sector: str
    etf: str
    amount: float
    weight: float  # pct of total portfolio


@dataclass
class PortfolioPlan:
    """Complete portfolio plan for a rebalance event."""
    targets: list[TargetAllocation] = field(default_factory=list)
    total_invest: float = 0.0
    cash_remain: float = 0.0
    n_holdings: int = 0
    reason: str = ""


class PortfolioBuilder:
    """Builds target portfolio from sector scores with constraints."""

    def __init__(self, config: PortfolioConfig = PortfolioConfig()):
        self.config = config

    def build(self, scores_df: pd.DataFrame, total_nav: float,
              position_cap: int) -> PortfolioPlan:
        """Build target allocation from ranked sector scores.

        Args:
            scores_df: Ranked sector scores (from SignalPipeline)
            total_nav: Current total NAV (cash + positions)
            position_cap: Max position as pct of NAV (0-100)

        Returns PortfolioPlan with target allocations.
        """
        if scores_df.empty or position_cap <= 0:
            return PortfolioPlan(reason="no signals or zero cap")

        invest_total = round(total_nav * position_cap / 100.0)
        if invest_total <= 0:
            return PortfolioPlan(reason="invest amount too small")

        # Filter: ETF availability + crowding + ETF dedup
        qualified = []
        seen_etfs: set = set()
        for _, row in scores_df.iterrows():
            code = get_etf_code(row["sector"])
            if code == "—":
                continue
            crowding = row.get("crowding", 0.5)
            # Skip extremely crowded sectors (top 10% most crowded)
            if crowding < 0.10:
                continue
            # ETF dedup: skip duplicate ETF (keep first/highest-ranked)
            if code in seen_etfs:
                continue
            seen_etfs.add(code)
            qualified.append({
                "sector": row["sector"],
                "code": code,
                "composite": row["composite"],
            })
            if len(qualified) >= self.config.max_sectors:
                break

        if len(qualified) < self.config.min_sectors:
            if not qualified:
                return PortfolioPlan(reason="no qualified sectors after filtering")

        n = len(qualified)

        # Rank-weighted allocation
        rank_sum = sum(range(1, n + 1))
        targets = []
        for i, q in enumerate(qualified):
            rel_weight = (n - i) / rank_sum  # rank 1 gets highest weight
            raw_amount = round(invest_total * rel_weight)

            # Sector concentration cap
            max_per_sector = round(invest_total * self.config.sector_concentration_cap)
            amount = min(raw_amount, max_per_sector)
            weight = amount / total_nav if total_nav > 0 else 0

            if amount > 0:
                targets.append(TargetAllocation(
                    sector=q["sector"],
                    etf=q["code"],
                    amount=float(amount),
                    weight=round(weight, 4),
                ))

        total_invest = sum(t.amount for t in targets)

        return PortfolioPlan(
            targets=targets,
            total_invest=total_invest,
            cash_remain=total_nav - total_invest,
            n_holdings=len(targets),
            reason="ok",
        )
