"""
估值因子 —— 行业PE分位 + 仓位调节。

逻辑：
- PE分位低 → 估值便宜 → 仓位可放大
- PE分位高 → 估值偏贵 → 仓位应缩减
- 本因子不直接改变评分，而是作为仓位调节器

数据源（akshare 1.18+）：
- 行业PE：ak.stock_industry_pe_ratio_cninfo(date='20260508')
- 指数PE：ak.stock_index_pe_lg(symbol='沪深300')
- A股整体PB：ak.stock_a_all_pb()
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from config.settings import CACHE_DIR, RATE_LIMIT_DELAY_MIN, RATE_LIMIT_DELAY_MAX

logger = logging.getLogger(__name__)


# ── 缓存 ──
def _cache_path(name: str):
    return CACHE_DIR / f"{name}.parquet"


def _read_cache(name: str, max_age: int):
    p = _cache_path(name)
    if not p.exists():
        return None
    if (time.time() - p.stat().st_mtime) > max_age:
        return None
    try:
        return pd.read_parquet(p)
    except Exception:
        return None


def _write_cache(name: str, df: pd.DataFrame):
    try:
        df.to_parquet(_cache_path(name), index=False)
    except Exception:
        pass


def _rate_limit():
    import random
    time.sleep(random.uniform(RATE_LIMIT_DELAY_MIN, RATE_LIMIT_DELAY_MAX))


# ── 行业PE ──
def get_industry_pe(date: str = None) -> pd.DataFrame:
    """获取全行业PE数据。

    Parameters
    ----------
    date : 日期字符串，如 '20260508'，默认最近交易日

    Returns
    -------
    pd.DataFrame: 行业PE数据
    """
    if date is None:
        date = datetime.now().strftime("%Y%m%d")

    cache_key = f"industry_pe_{date}"
    cached = _read_cache(cache_key, 86400 * 3)  # 3天缓存
    if cached is not None:
        return cached

    try:
        import akshare as ak
        _rate_limit()

        df = ak.stock_industry_pe_ratio_cninfo(date=date)
        if df is None or df.empty:
            return pd.DataFrame()

        _write_cache(cache_key, df)
        return df
    except Exception as e:
        logger.debug(f"行业PE获取失败: {e}")
        return pd.DataFrame()


# ── 指数PE ──
def get_index_pe(symbol: str = "沪深300", days: int = 750) -> pd.DataFrame:
    """获取指数PE历史。

    Parameters
    ----------
    symbol : 指数名称，如 '沪深300'
    days : 回看天数

    Returns
    -------
    pd.DataFrame: 指数PE历史
    """
    cache_key = f"index_pe_{symbol}"
    cached = _read_cache(cache_key, 86400)
    if cached is not None:
        return cached

    try:
        import akshare as ak
        _rate_limit()

        df = ak.stock_index_pe_lg(symbol=symbol)
        if df is None or df.empty:
            return pd.DataFrame()

        # 统一列名
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

        _write_cache(cache_key, df)
        return df
    except Exception as e:
        logger.debug(f"指数PE获取失败: {e}")
        return pd.DataFrame()


# ── A股整体PB分位 ──
def get_a_share_pb() -> pd.DataFrame:
    """获取A股整体PB及分位。

    Returns
    -------
    pd.DataFrame: A股PB历史 + 分位
    """
    cache_key = "a_share_pb"
    cached = _read_cache(cache_key, 86400)
    if cached is not None:
        return cached

    try:
        import akshare as ak
        _rate_limit()

        df = ak.stock_a_all_pb()
        if df is None or df.empty:
            return pd.DataFrame()

        _write_cache(cache_key, df)
        return df
    except Exception as e:
        logger.debug(f"A股PB获取失败: {e}")
        return pd.DataFrame()


def get_sector_pe(sector: str) -> dict:
    """获取板块估值数据。

    优先从行业PE数据中查找，其次用指数PE推算。

    Returns
    -------
    dict: {pe_ttm, pe_percentile, pb_percentile}
    """
    cache_key = f"sector_pe_{sector}"
    cached = _read_cache(cache_key, 86400)
    if cached is not None:
        return cached.to_dict("records")[0] if not cached.empty else {}

    result = {}

    # 1) 从行业PE数据查找
    try:
        industry_pe = get_industry_pe()
        if not industry_pe.empty:
            # 查找对应行业
            name_col = None
            pe_col = None
            for col in industry_pe.columns:
                if "行业名称" in str(col):
                    name_col = col
                elif "静态市盈率-加权平均" in str(col):
                    pe_col = col

            if name_col and pe_col:
                # 模糊匹配行业名
                match = industry_pe[industry_pe[name_col].astype(str).str.contains(sector, na=False)]
                if match.empty:
                    # 反向匹配
                    for _, row in industry_pe.iterrows():
                        if sector in str(row[name_col]) or str(row[name_col]) in sector:
                            match = industry_pe.iloc[[_] if isinstance(_, int) else [0]]
                            break

                if not match.empty:
                    pe_val = pd.to_numeric(match.iloc[0][pe_col], errors="coerce")
                    if not pd.isna(pe_val):
                        result["pe_ttm"] = float(pe_val)

                        # 计算PE分位（相对于全行业）
                        all_pe = pd.to_numeric(industry_pe[pe_col], errors="coerce").dropna()
                        if len(all_pe) > 10:
                            percentile = (all_pe < pe_val).sum() / len(all_pe)
                            result["pe_percentile"] = round(float(percentile), 4)
    except Exception as e:
        logger.debug(f"行业PE查找失败 {sector}: {e}")

    # 2) 从指数PE获取整体分位
    try:
        index_pe = get_index_pe("沪深300")
        if not index_pe.empty and "pe_ttm" in index_pe.columns:
            pe_series = pd.to_numeric(index_pe["pe_ttm"], errors="coerce").dropna()
            if len(pe_series) >= 60:
                current = pe_series.iloc[-1]
                lookback = pe_series.tail(min(len(pe_series), 750))
                pct = (lookback < current).sum() / len(lookback)
                result["index_pe_percentile"] = round(float(pct), 4)
                # 如果没有行业分位，用指数分位代替
                if "pe_percentile" not in result:
                    result["pe_percentile"] = result["index_pe_percentile"]
    except Exception as e:
        logger.debug(f"指数PE查找失败: {e}")

    # 3) 从A股PB获取整体估值分位
    try:
        pb_df = get_a_share_pb()
        if not pb_df.empty:
            # 查找分位列
            for col in pb_df.columns:
                if "quantileInRecent10YearsMiddlePB" in str(col):
                    latest = pb_df.iloc[-1]
                    pb_pct = pd.to_numeric(latest[col], errors="coerce")
                    if not pd.isna(pb_pct):
                        result["pb_percentile"] = round(float(pb_pct), 4)
                        # 用PB分位作为估值参考
                        if "pe_percentile" not in result:
                            result["pe_percentile"] = result["pb_percentile"]
                    break
    except Exception as e:
        logger.debug(f"A股PB查找失败: {e}")

    if result:
        _write_cache(cache_key, pd.DataFrame([result]))
    return result


def score_valuation(sector: str = None) -> float:
    """估值因子得分（用于仓位调节）。

    逻辑：
    - PE分位 < 20% → 极度低估 → 正分0.8（加仓信号）
    - PE分位 20%-40% → 低估 → 正分0.4
    - PE分位 40%-60% → 合理 → 零分
    - PE分位 60%-80% → 偏高 → 负分-0.3
    - PE分位 > 80% → 极度偏高 → 负分-0.7（减仓信号）

    Returns
    -------
    float : [-1, 1] 区间，正值=低估可加仓，负值=高估应减仓
    """
    data = get_sector_pe(sector) if sector else {}
    if not data:
        # 无板块数据时，用A股整体PB分位
        pb_df = get_a_share_pb()
        if not pb_df.empty:
            for col in pb_df.columns:
                if "quantileInRecent10YearsMiddlePB" in str(col):
                    pb_pct = float(pd.to_numeric(pb_df.iloc[-1][col], errors="coerce") or 0.5)
                    data = {"pe_percentile": pb_pct}
                    break

    if not data or "pe_percentile" not in data:
        return 0.0

    pct = data["pe_percentile"]

    if pct < 0.2:
        return 0.8
    elif pct < 0.4:
        return 0.4
    elif pct < 0.6:
        return 0.0
    elif pct < 0.8:
        return -0.3
    else:
        return -0.7


def calc_position_multiplier(sector: str = None) -> float:
    """根据估值因子计算仓位调节系数。

    Returns
    -------
    float : 仓位乘数 [0.5, 1.5]
    """
    score = score_valuation(sector)
    multiplier = 1.0 + score * 0.5
    return float(np.clip(multiplier, 0.5, 1.5))
