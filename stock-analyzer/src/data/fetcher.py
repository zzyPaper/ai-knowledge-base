"""Unified data fetcher: THS (Tonghuashun) via AKShare + Sina for index data.

Data source strategy:
  1. THS Industry Boards (90) + Concept Boards (372) via AKShare
  2. THS Board K-lines (daily) via AKShare
  3. Sina Finance for CSI 300 index daily data
"""

import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List

import pandas as pd
import akshare as ak

from config.settings import CACHE_DIR

COLUMN_MAP = {
    "date": "date", "日期": "date",
    "open": "open", "开盘价": "open",
    "close": "close", "收盘价": "close",
    "high": "high", "最高价": "high",
    "low": "low", "最低价": "low",
    "volume": "volume", "成交量": "volume",
    "amount": "amount", "成交额": "amount",
    "pct_chg": "pct_chg",
    "换手率": "turnover_rate",
}


def _cache_path(name: str) -> str:
    safe = name.replace("/", "_").replace(" ", "_").replace("（", "(").replace("）", ")").replace("'", "").replace('"', "")
    return f"{CACHE_DIR}/{safe}.pkl"


def _read_cache(path: str, max_age: int = 86400) -> Optional[pd.DataFrame]:
    from pathlib import Path
    p = Path(path)
    if p.exists() and (time.time() - p.stat().st_mtime) < max_age:
        return pd.read_pickle(path)
    return None


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename Chinese columns to English."""
    rename = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    if rename:
        df = df.rename(columns=rename)
    return df


# ========== Sector List ==========

def get_sectors_list() -> Optional[pd.DataFrame]:
    """Get all industry boards from THS (~90 sectors)."""
    try:
        df = ak.stock_board_industry_name_ths()
        if df is not None and not df.empty:
            df.columns = ["板块名称", "板块代码"]
            return df
    except Exception:
        pass
    return None


def get_concept_boards_list() -> Optional[pd.DataFrame]:
    """Get all concept boards from THS (~370)."""
    try:
        df = ak.stock_board_concept_name_ths()
        if df is not None and not df.empty:
            df.columns = ["板块名称", "板块代码"]
            return df
    except Exception:
        pass
    return None


# ========== Sector Historical K-Lines ==========

def get_sector_history(sector: str, start_date: str, end_date: str,
                       use_cache: bool = True, board_type: str = "industry") -> Optional[pd.DataFrame]:
    """Get sector daily K-line from THS via AKShare.

    Args:
        sector: Sector name (e.g. "半导体").
        start_date: YYYY-MM-DD or YYYYMMDD.
        end_date: YYYY-MM-DD or YYYYMMDD.
        use_cache: Use parquet cache if available.
        board_type: "industry" or "concept".
    """
    path = _cache_path(sector)
    if use_cache:
        cached = _read_cache(path)
        if cached is not None:
            return cached

    start = start_date.replace("-", "")
    end = end_date.replace("-", "")

    try:
        if board_type == "concept":
            df = ak.stock_board_concept_index_ths(symbol=sector, start_date=start, end_date=end)
        else:
            df = ak.stock_board_industry_index_ths(symbol=sector, start_date=start, end_date=end)
    except Exception:
        # Fallback: try the other board type
        try:
            if board_type == "concept":
                df = ak.stock_board_industry_index_ths(symbol=sector, start_date=start, end_date=end)
            else:
                df = ak.stock_board_concept_index_ths(symbol=sector, start_date=start, end_date=end)
        except Exception:
            return None

    if df is None or df.empty:
        return None

    df = _normalize_columns(df)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)
        df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]
        df = df.sort_values("date").reset_index(drop=True)

    if not df.empty:
        # Add computed fields
        if "close" in df.columns:
            df["pct_chg"] = df["close"].pct_change(1).fillna(0) * 100
        if "volume" in df.columns and "amount" not in df.columns:
            df["amount"] = df["volume"] * df["close"]
        # Ensure standard columns exist
        for col in ["open", "close", "high", "low", "volume", "amount", "pct_chg", "turnover_rate"]:
            if col not in df.columns:
                df[col] = 0.0
        df.to_pickle(path)
    return df


# ========== Index Data (Sina) ==========

def get_index_history(index_name: str, start_date: str, end_date: str,
                      use_cache: bool = True) -> Optional[pd.DataFrame]:
    """Get index daily K-line from Sina (works for CSI 300 etc.)."""
    path = _cache_path(index_name)
    if use_cache:
        cached = _read_cache(path)
        if cached is not None:
            return cached

    try:
        df = ak.stock_zh_index_daily(symbol=_index_code_map().get(index_name, index_name))
        if df is None or df.empty:
            return None

        df = _normalize_columns(df)
        df["date"] = pd.to_datetime(df["date"])
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)
        df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]
        df = df.sort_values("date").reset_index(drop=True)

        if not df.empty:
            df["pct_chg"] = df["close"].pct_change(1).fillna(0) * 100
            df.to_pickle(path)
        return df
    except Exception:
        return None


def _index_code_map() -> Dict[str, str]:
    return {
        "沪深300": "sh000300",
        "上证指数": "sh000001",
        "深证成指": "sz399001",
    }


# ========== Tencent Real-time (fallback) ==========

def get_tencent_quote(codes: List[str]) -> Dict[str, dict]:
    """Fetch real-time quotes from Tencent."""
    import requests
    all_codes = []
    for code in codes:
        if not code.startswith(("sh", "sz", "SH", "SZ")):
            code = f"sh{code}" if code.startswith(("6", "9")) else f"sz{code}"
        all_codes.append(code.lower())

    results = {}
    for i in range(0, len(all_codes), 50):
        batch = all_codes[i:i + 50]
        url = f"http://qt.gtimg.cn/q={','.join(batch)}"
        try:
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            text = resp.text
        except Exception:
            continue
        for line in text.strip().split(";"):
            if not line.strip():
                continue
            try:
                parts = line.split("~")
                if len(parts) < 40:
                    continue
                code = parts[0].split("=")[0].replace("v_", "").strip()
                results[code] = {
                    "code": code,
                    "name": parts[1],
                    "price": float(parts[3]) if parts[3] else 0.0,
                    "change_pct": float(parts[32]) if parts[32] else 0.0,
                    "amount": float(parts[37]) if parts[37] else 0.0,
                    "turnover": float(parts[38]) if parts[38] else 0.0,
                }
            except (ValueError, IndexError):
                continue
    return results


# ========== Sina Sector Spot (real-time stats fallback) ==========

def get_sina_sector_spot() -> Optional[pd.DataFrame]:
    """Get real-time sector stats from Sina (49 sectors)."""
    try:
        df = ak.stock_sector_spot()
        if df is not None and not df.empty:
            return df
    except Exception:
        pass
    return None


# ========== Unified Safe API (with mootdx fallback) ==========

import logging

logger = logging.getLogger(__name__)


def _try_backup(func_name: str, *args, **kwargs) -> Optional[pd.DataFrame]:
    """Call the corresponding backup_fetcher function as fallback."""
    try:
        from src.data import backup_fetcher as bf
        fn = getattr(bf, func_name, None)
        if fn is None:
            return None
        result = fn(*args, **kwargs)
        if result is not None and not result.empty:
            logger.info("[FALLBACK] %s succeeded via mootdx", func_name)
            return result
    except Exception as e:
        logger.warning("[FALLBACK] %s also failed: %s", func_name, e)
    return None


def get_sectors_list_safe() -> Optional[pd.DataFrame]:
    df = get_sectors_list()
    if df is not None and not df.empty:
        return df
    logger.info("[FALLBACK] get_sectors_list -> mootdx")
    return _try_backup("get_sectors_list")


def get_sector_history_safe(sector: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    df = get_sector_history(sector, start_date, end_date)
    if df is not None and not df.empty:
        return df
    logger.info("[FALLBACK] get_sector_history('%s') -> mootdx", sector)
    return _try_backup("get_sector_history", sector, start_date, end_date)


def get_index_history_safe(index_name: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    df = get_index_history(index_name, start_date, end_date)
    if df is not None and not df.empty:
        return df
    logger.info("[FALLBACK] get_index_history('%s') -> mootdx", index_name)
    return _try_backup("get_index_history", index_name, start_date, end_date)
