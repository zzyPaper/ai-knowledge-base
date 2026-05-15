#!/usr/bin/env python3
"""Update cached market data to latest available via AKShare.

Fetches THS industry board K-line + CSI 300 index data and updates pickle files.
Runs incrementally: only fetches missing dates since last update.

Usage:
  python scripts/update_data.py               # update all data
  python scripts/update_data.py --quick        # only index + top 20 sectors
"""

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import pandas as pd
import akshare as ak

from src.data.fetcher import _normalize_columns, _cache_path

SECTORS_PKL = BASE_DIR / "data" / "sectors_full.pkl"
INDEX_PKL = BASE_DIR / "data" / "index_full.pkl"


def _latest_date_in_sectors(sectors: dict) -> str:
    """Find the latest date across all sector DataFrames."""
    latest = None
    for df in sectors.values():
        if "date" in df.columns and not df.empty:
            max_d = df["date"].max()
            if latest is None or max_d > latest:
                latest = max_d
    return latest.strftime("%Y-%m-%d") if latest else "2023-01-01"


def _latest_date_in_index(index_df: pd.DataFrame) -> str:
    if index_df.empty or "date" not in index_df.columns:
        return "2023-01-01"
    return index_df["date"].max().strftime("%Y-%m-%d")


def fetch_index(index_name: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch CSI 300 index data from Sina via AKShare."""
    code_map = {"沪深300": "sh000300"}
    code = code_map.get(index_name, index_name)

    try:
        df = ak.stock_zh_index_daily(symbol=code)
        if df is None or df.empty:
            return pd.DataFrame()

        df = _normalize_columns(df)
        df["date"] = pd.to_datetime(df["date"])
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)
        df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]
        df = df.sort_values("date").reset_index(drop=True)

        if not df.empty and "pct_chg" not in df.columns:
            df["pct_chg"] = df["close"].pct_change(1).fillna(0) * 100
        return df
    except Exception as e:
        print(f"  [ERROR] 沪深300: {e}")
        return pd.DataFrame()


def fetch_sector(sector_name: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch one sector's K-line from THS via AKShare."""
    start = start_date.replace("-", "")
    end = end_date.replace("-", "")

    try:
        df = ak.stock_board_industry_index_ths(symbol=sector_name, start_date=start, end_date=end)
    except Exception:
        try:
            df = ak.stock_board_concept_index_ths(symbol=sector_name, start_date=start, end_date=end)
        except Exception:
            return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    df = _normalize_columns(df)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)
        df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]
        df = df.sort_values("date").reset_index(drop=True)

    if not df.empty:
        if "pct_chg" not in df.columns and "close" in df.columns:
            df["pct_chg"] = df["close"].pct_change(1).fillna(0) * 100
        for col in ["open", "close", "high", "low", "volume", "amount", "pct_chg", "turnover_rate"]:
            if col not in df.columns:
                df[col] = 0.0
    return df


def get_sector_list() -> list[str]:
    """Get all THS industry board names."""
    try:
        df = ak.stock_board_industry_name_ths()
        return df["name"].tolist()
    except Exception as e:
        print(f"[ERROR] Cannot get sector list: {e}")
        return []


def update_index(existing: pd.DataFrame) -> pd.DataFrame:
    """Update index data with latest from AKShare."""
    last_date = _latest_date_in_index(existing) if not existing.empty else "2025-01-01"
    today = datetime.now().strftime("%Y-%m-%d")

    if last_date >= today:
        print(f"[索引] 已是最新 ({last_date})")
        return existing

    print(f"[索引] 更新 {last_date} → {today}")
    new_data = fetch_index("沪深300", last_date, today)

    if new_data.empty:
        print("[索引] 无新数据")
        return existing

    if not existing.empty:
        existing["date"] = pd.to_datetime(existing["date"])
        combined = pd.concat([existing, new_data], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    else:
        combined = new_data

    if "pct_chg" not in combined.columns:
        combined["pct_chg"] = combined["close"].pct_change(1).fillna(0) * 100

    print(f"[索引] 更新完成, {len(combined)} 行, 最新 {combined['date'].max().strftime('%Y-%m-%d')}")
    return combined


def update_sectors(existing: dict, quick: bool = False) -> dict:
    """Update all sector DataFrames with latest data."""
    all_sectors = get_sector_list()
    if not all_sectors:
        print("[板块] 无法获取板块列表")
        return existing

    if quick:
        # Only update top 20 sectors by recent volume
        top = all_sectors[:20]
        print(f"[板块] Quick模式: 仅更新 {len(top)} 个板块")
        all_sectors = top

    last_date = _latest_date_in_sectors(existing) if existing else "2025-01-01"
    today = datetime.now().strftime("%Y-%m-%d")

    if last_date >= today:
        print(f"[板块] 已是最新 ({last_date})")
        return existing

    print(f"[板块] 更新 {len(all_sectors)} 个板块 {last_date} → {today}")

    updated = dict(existing) if existing else {}
    new_count = 0

    for i, name in enumerate(all_sectors):
        # Check if we already have data through yesterday
        if name in updated and not updated[name].empty:
            df_last = updated[name]["date"].max().strftime("%Y-%m-%d") if "date" in updated[name].columns else "2000-01-01"
            if df_last >= today:
                continue

        try:
            new_df = fetch_sector(name, last_date, today)
            if new_df.empty:
                continue

            if name in updated and not updated[name].empty:
                existing_df = updated[name].copy()
                existing_df["date"] = pd.to_datetime(existing_df["date"])
                combined = pd.concat([existing_df, new_df], ignore_index=True)
                combined = combined.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
                updated[name] = combined
            else:
                updated[name] = new_df
            new_count += 1

        except Exception as e:
            print(f"  [WARN] {name}: {e}")
            continue

        # Rate limiting
        if i % 10 == 9:
            time.sleep(0.3)

        if (i + 1) % 20 == 0:
            print(f"  ... {i + 1}/{len(all_sectors)}")

    print(f"[板块] 更新完成, {new_count} 个板块有新数据, 共 {len(updated)} 个板块")
    return updated


def main():
    parser = argparse.ArgumentParser(description="更新缓存行情数据")
    parser.add_argument("--quick", action="store_true", help="仅更新指数+前20板块")
    args = parser.parse_args()

    t0 = time.time()

    # Load existing
    existing_sectors = None
    existing_index = None

    if SECTORS_PKL.exists():
        existing_sectors = pd.read_pickle(SECTORS_PKL)
        print(f"[加载] 现有板块数据: {len(existing_sectors)} 个板块")

    if INDEX_PKL.exists():
        existing_index = pd.read_pickle(INDEX_PKL)
        print(f"[加载] 现有指数数据: {len(existing_index)} 行")

    # Update index
    updated_index = update_index(existing_index if existing_index is not None else pd.DataFrame())
    # Update sectors
    updated_sectors = update_sectors(existing_sectors if existing_sectors is not None else {}, quick=args.quick)

    # Save
    if not updated_index.empty:
        updated_index.to_pickle(INDEX_PKL)
        print(f"[保存] {INDEX_PKL} ({len(updated_index)} 行)")

    if updated_sectors:
        pd.to_pickle(updated_sectors, SECTORS_PKL)
        print(f"[保存] {SECTORS_PKL} ({len(updated_sectors)} 个板块)")

    elapsed = time.time() - t0
    print(f"\n[完成] 耗时 {elapsed:.1f}s")


if __name__ == "__main__":
    main()
