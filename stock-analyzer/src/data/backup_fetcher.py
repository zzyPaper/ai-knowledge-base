"""Backup data source: mootdx (通达信) when AKShare is rate-limited or unavailable.

Data path is completely independent of AKShare (同花顺/新浪):
  - Sector lists via TDX stocks(market=1) → 880xxx board codes + names
  - Sector/index K-lines via TDX index_bars()
  - No registration, no API key, direct TCP to TDX servers

API notes (verified 2026-05-12):
  - client.index_bars(symbol, frequency=9, offset=N) → daily K-line
  - frequency: 9=daily, 4=60min, 7=5min, 8=1min, 10=weekly, 11=monthly
  - index_bars works for BOTH index codes (000300) and board codes (880491)
  - client.stocks(market=1) returns ~27048 entries including 652 boards (880xxx)
  - Columns from index_bars: open, close, high, low, vol, amount, datetime, volume
"""

import time
import logging
from typing import Optional, Dict

import pandas as pd

from config.settings import CACHE_DIR

logger = logging.getLogger(__name__)

# ---------- TDX client (lazy singleton) ----------

_client = None
_client_fail_count = 0
_MAX_RETRIES = 3


def _get_client():
    """Lazily create and cache the mootdx Quotes client with retry."""
    global _client, _client_fail_count

    if _client is not None:
        try:
            _client.quotes(symbol=["600519"])
            _client_fail_count = 0
            return _client
        except Exception:
            _client = None

    # Retry up to _MAX_RETRIES times across calls
    if _client_fail_count >= _MAX_RETRIES:
        logger.warning("[mootdx] Max retries reached, giving up for this session")
        return None

    try:
        from mootdx.quotes import Quotes
        _client = Quotes.factory(market="std", bestip=True, timeout=15)
        _client_fail_count = 0
        logger.info("[mootdx] Client connected")
        return _client
    except Exception as e:
        _client_fail_count += 1
        logger.warning("[mootdx] Failed to connect (attempt %d/%d): %s",
                       _client_fail_count, _MAX_RETRIES, e)
        return None


# ---------- Constants ----------

TDX_INDEX_MAP: Dict[str, str] = {
    "沪深300": "000300",
    "上证指数": "000001",
    "深证成指": "399001",
}

# Cache for sector name → TDX code mapping
_name_to_code: Dict[str, str] = {}
_code_to_name: Dict[str, str] = {}
_name_map_loaded = False


def _load_sector_name_map():
    """Build sector name↔code mapping from TDX stocks(market=1).

    Industry boards: 880301~880497
    Concept/other boards: 880498~880999
    """
    global _name_map_loaded
    if _name_map_loaded:
        return

    client = _get_client()
    if client is None:
        return

    try:
        df = client.stocks(market=1)
        if df is None or df.empty:
            return

        boards = df[df["code"].str.startswith("880")]
        for _, row in boards.iterrows():
            code = str(row["code"]).strip().strip("\x00")
            name = str(row["name"]).strip().strip("\x00")
            if code and name:
                _name_to_code[name] = code
                _code_to_name[code] = name

        # Only mark as loaded if we actually got data
        if _name_to_code:
            _name_map_loaded = True
            logger.info("[mootdx] Loaded %d board name mappings", len(_name_to_code))

    except Exception as e:
        logger.warning("[mootdx] Failed to load sector name map: %s", e)


# ---------- Column normalization ----------

COLUMN_MAP = {
    "datetime": "date",
    "open": "open",
    "close": "close",
    "high": "high",
    "low": "low",
    "vol": "volume",
    "amount": "amount",
}


def _normalize_bars(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """Normalize index_bars output to match fetcher.py's data shape."""
    df = df.rename(columns=COLUMN_MAP)

    keep = ["date", "open", "close", "high", "low", "volume", "amount"]
    df = df[[c for c in keep if c in df.columns]].copy()

    df["date"] = pd.to_datetime(df["date"])
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)
    df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]
    df = df.sort_values("date").reset_index(drop=True)

    if not df.empty:
        df["pct_chg"] = df["close"].pct_change(1).fillna(0) * 100
        if "volume" in df.columns and "amount" not in df.columns:
            df["amount"] = df["volume"] * df["close"]
        for col in ["open", "close", "high", "low", "volume", "amount", "pct_chg", "turnover_rate"]:
            if col not in df.columns:
                df[col] = 0.0

    return df


# ========== Sector List ==========

def get_sectors_list() -> Optional[pd.DataFrame]:
    """Get industry board list from TDX (~130 sectors).

    Returns DataFrame with columns: 板块名称, 板块代码
    """
    _load_sector_name_map()

    if not _name_to_code:
        return None

    # Filter to industry board code range only
    results = []
    for name, code in _name_to_code.items():
        if "880301" <= code <= "880497":
            results.append({"板块名称": name, "板块代码": code})

    if not results:
        return None

    return pd.DataFrame(results)


def get_concept_boards_list() -> Optional[pd.DataFrame]:
    """Get concept/extended board list from TDX.

    Returns DataFrame with columns: 板块名称, 板块代码
    """
    _load_sector_name_map()

    if not _name_to_code:
        return None

    results = []
    for name, code in _name_to_code.items():
        # Include boards outside industry range that look like thematic boards
        if code.startswith("880") and not ("880301" <= code <= "880497"):
            results.append({"板块名称": name, "板块代码": code})

    if not results:
        return None

    return pd.DataFrame(results)


# ========== Sector Historical K-Lines ==========

def _find_sector_code(sector_name: str) -> Optional[str]:
    """Find TDX board code for a sector name.

    Tries exact match first, then substring match (e.g. "半导体及元件" -> "半导体").
    """
    _load_sector_name_map()

    # Exact match
    if sector_name in _name_to_code:
        return _name_to_code[sector_name]

    # Substring match: TDX name is a substring of the query, or vice versa
    for tdx_name, code in _name_to_code.items():
        if tdx_name in sector_name or sector_name in tdx_name:
            return code

    return None


def _reconnect():
    """Force reconnect: close old client and reset fail count."""
    global _client, _client_fail_count
    if _client is not None:
        try:
            _client.close()
        except Exception:
            pass
    _client = None
    _client_fail_count = 0


def get_sector_history(
    sector: str,
    start_date: str,
    end_date: str,
    board_type: str = "industry",
) -> Optional[pd.DataFrame]:
    """Get sector daily K-line from TDX via index_bars.

    Args:
        sector: Sector name (e.g. "半导体"). Note: TDX naming may differ
                slightly from THS (e.g. "半导体" in TDX vs "半导体及元件" in THS).
        start_date: YYYY-MM-DD.
        end_date: YYYY-MM-DD.
        board_type: ignored (auto-detected from code range).
    """
    for attempt in range(2):
        client = _get_client()
        if client is None:
            if attempt == 0:
                _reconnect()
                continue
            return None

        code = _find_sector_code(sector)
        if code is None:
            logger.warning("[mootdx] Sector '%s' not found in TDX board list", sector)
            return None

        try:
            df = client.index_bars(symbol=code, frequency=9, offset=200)
            if df is None or df.empty:
                return None
            return _normalize_bars(df, start_date, end_date)
        except Exception as e:
            logger.warning("[mootdx] get_sector_history('%s') attempt %d failed: %s", sector, attempt + 1, e)
            if attempt == 0:
                _reconnect()
                continue
            return None

    return None


# ========== Index Historical K-Lines ==========

def get_index_history(
    index_name: str,
    start_date: str,
    end_date: str,
) -> Optional[pd.DataFrame]:
    """Get index daily K-line from TDX via index_bars.

    Args:
        index_name: e.g. "沪深300" or raw index code like "000300".
        start_date: YYYY-MM-DD.
        end_date: YYYY-MM-DD.
    """
    code = TDX_INDEX_MAP.get(index_name, index_name)

    for attempt in range(2):
        client = _get_client()
        if client is None:
            if attempt == 0:
                _reconnect()
                continue
            return None

        try:
            df = client.index_bars(symbol=code, frequency=9, offset=200)
            if df is None or df.empty:
                return None
            return _normalize_bars(df, start_date, end_date)
        except Exception as e:
            logger.warning("[mootdx] get_index_history('%s') attempt %d failed: %s", index_name, attempt + 1, e)
            if attempt == 0:
                _reconnect()
                continue
            return None

    return None
