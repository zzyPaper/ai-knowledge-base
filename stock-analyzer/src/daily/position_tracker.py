"""Track current portfolio holdings and rebalance schedule."""
import json
from pathlib import Path
from typing import Optional

POSITION_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "current_position.json"
MIN_REBALANCE_INTERVAL = 5  # trading days between rebalances


def load_position() -> Optional[dict]:
    """Load last saved position."""
    if POSITION_FILE.exists():
        return json.loads(POSITION_FILE.read_text(encoding="utf-8"))
    return None


def should_rebalance(trading_day_count: int, last_rebalance_date: Optional[str], current_date: str) -> bool:
    """Check if enough trading days have passed since last rebalance."""
    if last_rebalance_date is None:
        return True
    return trading_day_count >= MIN_REBALANCE_INTERVAL


def save_position(date_str: str, buys: list[dict], cash: int, total: int, rebalanced: bool = False):
    """Save current position so next run can compute diff."""
    holdings = {}
    for b in buys:
        holdings[b["code"]] = {
            "name": b["name"],
            "amount": b["amount"],
            "sector": b.get("sector", ""),
        }
    data = {
        "date": date_str,
        "holdings": holdings,
        "cash": cash,
        "total": total,
        "last_rebalance_date": date_str if rebalanced else None,
    }

    # Preserve previous rebalance date if we're not rebalancing today
    old = load_position()
    if old and not rebalanced:
        data["last_rebalance_date"] = old.get("last_rebalance_date")

    POSITION_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def compute_trades(
    old: Optional[dict],
    new_buys: list[dict],
    new_cash: int,
    total: int,
) -> list[dict]:
    """Compare old holdings with new target, return trade instructions.

    Each trade: {"action": "buy"/"sell", "code": str, "name": str, "amount": int}
    """
    trades = []

    if old is None:
        for b in new_buys:
            trades.append({"action": "buy", "code": b["code"], "name": b["name"], "amount": b["amount"]})
        return trades

    old_holdings = old.get("holdings", {})
    new_holdings_map = {b["code"]: b for b in new_buys}

    for code, info in old_holdings.items():
        if code not in new_holdings_map:
            trades.append({"action": "sell", "code": code, "name": info["name"], "amount": info["amount"]})

    for b in new_buys:
        old_info = old_holdings.get(b["code"])
        if old_info is None:
            trades.append({"action": "buy", "code": b["code"], "name": b["name"], "amount": b["amount"]})
        else:
            diff = b["amount"] - old_info["amount"]
            if diff > 50:
                trades.append({"action": "buy", "code": b["code"], "name": b["name"], "amount": diff})
            elif diff < -50:
                trades.append({"action": "sell", "code": b["code"], "name": b["name"], "amount": -diff})

    return trades
