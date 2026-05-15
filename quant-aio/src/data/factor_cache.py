"""
历史因子数据预下载与缓存系统。

核心问题：回测时外部因子（资金/情绪/景气度/估值）依赖实时API，
历史日期的数据拿不到 → 因子全返回0 → 7因子退化为3因子。

解决方案：回测前一次性预下载所有历史因子数据，存为parquet，
回测时从本地文件读取，与K线数据一样确定性可用。

数据源：
- 北向资金：ak.stock_hsgt_hist_em（历史序列，可靠）
- 市场资金流：ak.stock_market_fund_flow（历史序列，可靠）
- 指数PE：ak.stock_index_pe_lg（历史序列，可靠）
- A股PB：ak.stock_a_all_pb（历史序列，可靠）
- 行业PE：ak.stock_industry_pe_ratio_cninfo（快照，按日缓存）
- 行业资金流：ak.stock_fund_flow_industry（快照，按日缓存）
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import CACHE_DIR, RATE_LIMIT_DELAY_MIN, RATE_LIMIT_DELAY_MAX

logger = logging.getLogger(__name__)

FACTOR_CACHE_DIR = CACHE_DIR / "factor_history"
FACTOR_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _rate_limit():
    import random
    time.sleep(random.uniform(RATE_LIMIT_DELAY_MIN, RATE_LIMIT_DELAY_MAX))


# ════════════════════════════════════════════════════════
#  北向资金历史
# ════════════════════════════════════════════════════════

def preload_north_flow(start_date: str, end_date: str) -> pd.DataFrame:
    """预下载北向资金历史，存为parquet。

    Returns
    -------
    pd.DataFrame: columns=[date, net_flow_hu, net_flow_shen]
    """
    cache_file = FACTOR_CACHE_DIR / f"north_flow_{start_date}_{end_date}.parquet"
    if cache_file.exists():
        try:
            return pd.read_parquet(cache_file)
        except Exception:
            pass

    rows = []
    for symbol, col_prefix in [("沪股通", "hu"), ("深股通", "shen")]:
        try:
            import akshare as ak
            _rate_limit()
            df = ak.stock_hsgt_hist_em(symbol=symbol)
            if df is None or df.empty:
                continue

            date_col = None
            flow_col = None
            for c in df.columns:
                if "日期" in str(c):
                    date_col = c
                elif "成交净买额" in str(c) or "净买额" in str(c):
                    flow_col = c

            if date_col and flow_col:
                df = df[[date_col, flow_col]].copy()
                df.columns = ["date", f"net_flow_{col_prefix}"]
                df["date"] = pd.to_datetime(df["date"])
                df[f"net_flow_{col_prefix}"] = pd.to_numeric(df[f"net_flow_{col_prefix}"], errors="coerce")
                # 过滤NaN（2024年8月后的数据经常NaN）
                df = df.dropna(subset=[f"net_flow_{col_prefix}"])

                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]

                if not rows:
                    rows.append(df)
                else:
                    rows[0] = rows[0].merge(df, on="date", how="outer")
        except Exception as e:
            logger.warning(f"预下载北向资金({symbol})失败: {e}")

    if not rows:
        return pd.DataFrame()

    result = rows[0].sort_values("date").reset_index(drop=True)
    # 合计北向 = 沪股通 + 深股通
    if "net_flow_hu" in result.columns and "net_flow_shen" in result.columns:
        result["net_flow_total"] = result["net_flow_hu"].fillna(0) + result["net_flow_shen"].fillna(0)
    elif "net_flow_hu" in result.columns:
        result["net_flow_total"] = result["net_flow_hu"].fillna(0)
    elif "net_flow_shen" in result.columns:
        result["net_flow_total"] = result["net_flow_shen"].fillna(0)

    try:
        result.to_parquet(cache_file, index=False)
    except Exception:
        pass
    return result


def get_north_flow_cached(date: str, lookback: int = 20, start_date: str = None, end_date: str = None) -> float:
    """从缓存获取北向资金因子得分（用于回测）。

    Parameters
    ----------
    date : 当前回测日期 (YYYYMMDD or YYYY-MM-DD)
    lookback : 回看天数
    start_date, end_date : 预下载的日期范围
    """
    date = date.replace("-", "")
    if start_date is None:
        start_date = (datetime.strptime(date, "%Y%m%d") - timedelta(days=365)).strftime("%Y%m%d")
    if end_date is None:
        end_date = date

    cache_file = _find_cache_file("north_flow", start_date, end_date)
    if cache_file is None:
        return 0.0

    try:
        df = pd.read_parquet(cache_file)
    except Exception:
        return 0.0

    if df.empty or "net_flow_total" not in df.columns:
        return 0.0

    target_date = pd.to_datetime(date)
    df["date"] = pd.to_datetime(df["date"])
    recent = df[df["date"] <= target_date].tail(lookback)

    if recent.empty:
        return 0.0

    flows = recent["net_flow_total"].dropna()
    if flows.empty:
        return 0.0

    total_flow = flows.sum()
    positive_days = (flows > 0).sum()
    positive_ratio = positive_days / len(flows) if len(flows) > 0 else 0.5

    flow_score = np.clip(total_flow / 50e8, -1, 1)
    ratio_score = np.clip((positive_ratio - 0.5) * 4, -1, 1)
    composite = 0.6 * flow_score + 0.4 * ratio_score
    return float(np.clip(composite, -1, 1))


# ════════════════════════════════════════════════════════
#  市场资金流历史
# ════════════════════════════════════════════════════════

def preload_market_fund_flow(start_date: str, end_date: str) -> pd.DataFrame:
    """预下载市场整体资金流历史。"""
    cache_file = FACTOR_CACHE_DIR / f"market_fund_flow_{start_date}_{end_date}.parquet"
    if cache_file.exists():
        try:
            return pd.read_parquet(cache_file)
        except Exception:
            pass

    try:
        import akshare as ak
        _rate_limit()
        df = ak.stock_market_fund_flow()
        if df is None or df.empty:
            return pd.DataFrame()

        # 统一列名
        rename_map = {}
        for col in df.columns:
            if "日期" in str(col):
                rename_map[col] = "date"
            elif "主力净流入-净额" in str(col):
                rename_map[col] = "main_net_inflow"
            elif "主力净流入-净占比" in str(col):
                rename_map[col] = "main_net_pct"
            elif "超大单净流入-净额" in str(col):
                rename_map[col] = "super_large_net"
            elif "大单净流入-净额" in str(col):
                rename_map[col] = "large_net"
            elif "小单净流入-净占比" in str(col):
                rename_map[col] = "small_net_pct"

        df = df.rename(columns=rename_map)

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        for col in ["main_net_inflow", "main_net_pct", "super_large_net", "large_net", "small_net_pct"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        if "date" in df.columns:
            df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]

        try:
            df.to_parquet(cache_file, index=False)
        except Exception:
            pass
        return df
    except Exception as e:
        logger.warning(f"预下载市场资金流失败: {e}")
        return pd.DataFrame()


def _find_cache_file(prefix: str, start_date: str, end_date: str) -> Path | None:
    """查找包含目标日期范围的缓存文件。

    不再要求文件名完全匹配，而是找到 end_date 匹配的文件即可。
    """
    # 精确匹配
    exact = FACTOR_CACHE_DIR / f"{prefix}_{start_date}_{end_date}.parquet"
    if exact.exists():
        return exact

    # 模糊匹配：找 end_date 匹配的文件
    import re
    pattern = re.compile(rf"^{prefix}_(\d{{8}})_{end_date}\.parquet$")
    for f in FACTOR_CACHE_DIR.iterdir():
        if pattern.match(f.name):
            return f

    # 更宽松匹配：找任何 end_date >= 目标end_date 的文件
    pattern2 = re.compile(rf"^{prefix}_(\d{{8}})_(\d{{8}})\.parquet$")
    candidates = []
    for f in FACTOR_CACHE_DIR.iterdir():
        m = pattern2.match(f.name)
        if m:
            file_end = m.group(2)
            if file_end >= end_date:
                candidates.append((f, file_end))
    if candidates:
        # 优先选end_date最近的
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]

    return None


def get_market_fund_flow_cached(date: str, lookback: int = 5, start_date: str = None, end_date: str = None) -> float:
    """从缓存获取市场主力资金因子得分。"""
    date = date.replace("-", "")
    if start_date is None:
        start_date = (datetime.strptime(date, "%Y%m%d") - timedelta(days=365)).strftime("%Y%m%d")
    if end_date is None:
        end_date = date

    cache_file = _find_cache_file("market_fund_flow", start_date, end_date)
    if cache_file is None:
        return 0.0

    try:
        df = pd.read_parquet(cache_file)
    except Exception:
        return 0.0

    if df.empty:
        return 0.0

    target_date = pd.to_datetime(date)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        recent = df[df["date"] <= target_date].tail(lookback)
    else:
        recent = df.tail(lookback)

    if "main_net_inflow" in recent.columns:
        recent_vals = pd.to_numeric(recent["main_net_inflow"], errors="coerce").dropna()
        if not recent_vals.empty:
            total = recent_vals.sum()
            return float(np.clip(total / 50e8, -1, 1))

    return 0.0


def get_market_breadth_cached(date: str, start_date: str = None, end_date: str = None) -> float:
    """从缓存获取市场宽度因子得分（基于主力资金净占比推算）。"""
    date = date.replace("-", "")
    if start_date is None:
        start_date = (datetime.strptime(date, "%Y%m%d") - timedelta(days=365)).strftime("%Y%m%d")
    if end_date is None:
        end_date = date

    cache_file = _find_cache_file("market_fund_flow", start_date, end_date)
    if cache_file is None:
        return 0.0

    try:
        df = pd.read_parquet(cache_file)
    except Exception:
        return 0.0

    if df.empty:
        return 0.0

    target_date = pd.to_datetime(date)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        on_date = df[df["date"] <= target_date].tail(1)
    else:
        on_date = df.tail(1)

    if on_date.empty:
        return 0.0

    if "main_net_pct" in on_date.columns:
        main_pct = float(pd.to_numeric(on_date.iloc[0]["main_net_pct"], errors="coerce") or 0)
        advance_ratio = 0.5 + main_pct / 10.0
        advance_ratio = np.clip(advance_ratio, 0, 1)
        score = (advance_ratio - 0.5) * 4.0
        return float(np.clip(score, -1, 1))

    return 0.0


# ════════════════════════════════════════════════════════
#  指数PE历史
# ════════════════════════════════════════════════════════

def preload_index_pe(start_date: str, end_date: str, symbol: str = "沪深300") -> pd.DataFrame:
    """预下载指数PE历史。"""
    cache_file = FACTOR_CACHE_DIR / f"index_pe_{symbol}_{start_date}_{end_date}.parquet"
    if cache_file.exists():
        try:
            return pd.read_parquet(cache_file)
        except Exception:
            pass

    try:
        import akshare as ak
        _rate_limit()
        df = ak.stock_index_pe_lg(symbol=symbol)
        if df is None or df.empty:
            return pd.DataFrame()

        rename_map = {}
        for col in df.columns:
            if "日期" in str(col):
                rename_map[col] = "date"
            elif "滚动市盈率" == str(col) or "滚动市盈率" in str(col):
                if "等权" not in str(col) and "中位" not in str(col):
                    rename_map[col] = "pe_ttm"
            elif "静态市盈率" == str(col) or "静态市盈率" in str(col):
                if "等权" not in str(col) and "中位" not in str(col):
                    rename_map[col] = "pe_static"

        df = df.rename(columns=rename_map)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]

        try:
            df.to_parquet(cache_file, index=False)
        except Exception:
            pass
        return df
    except Exception as e:
        logger.warning(f"预下载指数PE失败: {e}")
        return pd.DataFrame()


def get_index_pe_percentile_cached(date: str, start_date: str = None, end_date: str = None, symbol: str = "沪深300") -> float:
    """从缓存获取指数PE分位（用于估值因子）。"""
    date = date.replace("-", "")
    if start_date is None:
        start_date = (datetime.strptime(date, "%Y%m%d") - timedelta(days=365 * 3)).strftime("%Y%m%d")
    if end_date is None:
        end_date = date

    cache_file = _find_cache_file(f"index_pe_{symbol}", start_date, end_date)
    if cache_file is None:
        return 0.5  # 无数据返回中性

    try:
        df = pd.read_parquet(cache_file)
    except Exception:
        return 0.5

    if df.empty:
        return 0.5

    target_date = pd.to_datetime(date)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        on_date = df[df["date"] <= target_date]

    if on_date.empty or "pe_ttm" not in on_date.columns:
        return 0.5

    pe_series = pd.to_numeric(on_date["pe_ttm"], errors="coerce").dropna()
    if len(pe_series) < 20:
        return 0.5

    current = pe_series.iloc[-1]
    lookback = pe_series.tail(min(len(pe_series), 750))
    pct = (lookback < current).sum() / len(lookback)
    return float(pct)


# ════════════════════════════════════════════════════════
#  A股PB历史
# ════════════════════════════════════════════════════════

def preload_a_share_pb(start_date: str, end_date: str) -> pd.DataFrame:
    """预下载A股整体PB历史。"""
    cache_file = FACTOR_CACHE_DIR / f"a_share_pb_{start_date}_{end_date}.parquet"
    if cache_file.exists():
        try:
            return pd.read_parquet(cache_file)
        except Exception:
            pass

    try:
        import akshare as ak
        _rate_limit()
        df = ak.stock_a_all_pb()
        if df is None or df.empty:
            return pd.DataFrame()

        try:
            df.to_parquet(cache_file, index=False)
        except Exception:
            pass
        return df
    except Exception as e:
        logger.warning(f"预下载A股PB失败: {e}")
        return pd.DataFrame()


def get_a_share_pb_percentile_cached(date: str, start_date: str = None, end_date: str = None) -> float:
    """从缓存获取A股PB分位。"""
    date = date.replace("-", "")
    if start_date is None:
        start_date = (datetime.strptime(date, "%Y%m%d") - timedelta(days=365 * 3)).strftime("%Y%m%d")
    if end_date is None:
        end_date = date

    cache_file = _find_cache_file("a_share_pb", start_date, end_date)
    if cache_file is None:
        return 0.5

    try:
        df = pd.read_parquet(cache_file)
    except Exception:
        return 0.5

    if df.empty:
        return 0.5

    for col in df.columns:
        if "quantileInRecent10YearsMiddlePB" in str(col):
            pb_pct = float(pd.to_numeric(df.iloc[-1][col], errors="coerce") or 0.5)
            return pb_pct

    return 0.5


# ════════════════════════════════════════════════════════
#  一键预下载所有因子数据
# ════════════════════════════════════════════════════════

def preload_all_factors(start_date: str, end_date: str) -> dict[str, bool]:
    """预下载所有历史因子数据。

    Parameters
    ----------
    start_date : YYYYMMDD
    end_date : YYYYMMDD

    Returns
    -------
    dict : 各数据源下载是否成功
    """
    results = {}
    logger.info(f"开始预下载因子数据 {start_date} ~ {end_date}")

    # 扩展日期范围，确保有足够的历史数据
    extended_start = (datetime.strptime(start_date, "%Y%m%d") - timedelta(days=365)).strftime("%Y%m%d")

    tasks = [
        ("北向资金", lambda: preload_north_flow(extended_start, end_date)),
        ("市场资金流", lambda: preload_market_fund_flow(extended_start, end_date)),
        ("指数PE", lambda: preload_index_pe(extended_start, end_date)),
        ("A股PB", lambda: preload_a_share_pb(extended_start, end_date)),
    ]

    for name, func in tasks:
        try:
            df = func()
            results[name] = not df.empty
            logger.info(f"  {name}: {'✓' if results[name] else '✗'} ({len(df)} rows)")
        except Exception as e:
            results[name] = False
            logger.warning(f"  {name}: ✗ ({e})")

    return results
