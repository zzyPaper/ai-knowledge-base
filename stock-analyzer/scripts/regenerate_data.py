#!/usr/bin/env python3
"""Regenerate pickle data files using AKShare.
Run once after environment changes to rebuild compatible pickle files."""
import sys, time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import pandas as pd
import akshare as ak

DATA_DIR = BASE / "data"
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _normalize(df):
    rename = {}
    for src, dst in [("日期", "date"), ("开盘价", "open"), ("收盘价", "close"),
                     ("最高价", "high"), ("最低价", "low"), ("成交量", "volume"),
                     ("成交额", "amount"), ("换手率", "turnover_rate")]:
        if src in df.columns:
            rename[src] = dst
    if rename:
        df = df.rename(columns=rename)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "close", "high", "low", "volume", "amount", "pct_chg", "turnover_rate"]:
        if col not in df.columns:
            df[col] = 0.0
    return df


def load_sectors():
    print("Fetching sector list...")
    boards = ak.stock_board_industry_name_ths()
    names = boards["板块名称"].tolist()
    print(f"Found {len(names)} industry boards")

    sectors = {}
    for i, name in enumerate(names):
        print(f"  [{i+1}/{len(names)}] {name}...")
        try:
            df = ak.stock_board_industry_index_ths(symbol=name, start_date="20230101", end_date="20260511")
            if df is not None and not df.empty:
                df = _normalize(df)
                # Save per-sector cache
                safe = name.replace("/", "_").replace(" ", "")
                df.to_pickle(CACHE_DIR / f"{safe}.pkl")
                sectors[name] = df
            time.sleep(0.3)
        except Exception as e:
            print(f"    SKIP: {e}")
    return sectors


def load_index():
    print("\nFetching 沪深300 index...")
    df = ak.stock_zh_index_daily(symbol="sh000300")
    if df is None or df.empty:
        print("ERROR: Failed to fetch index")
        return None
    df = _normalize(df)
    df["pct_chg"] = df["close"].pct_change(1).fillna(0) * 100
    return df


if __name__ == "__main__":
    print("Regenerating data files...")
    sectors = load_sectors()
    index = load_index()

    sectors_pkl = DATA_DIR / "sectors_full.pkl"
    index_pkl = DATA_DIR / "index_full.pkl"

    pd.to_pickle(sectors, sectors_pkl)
    pd.to_pickle(index, index_pkl)
    print(f"\nSaved {len(sectors)} sectors to {sectors_pkl}")
    print(f"Saved index ({len(index)} rows) to {index_pkl}")
    print("Done!")
