"""
资金因子 —— 北向资金 + 主力资金流。

逻辑：
- 北向资金（沪深港通）：外资持续净流入 → 机构看多 → 信号强
- 主力资金：大单净流入 → 主力在吸筹 → 信号强
- 两者共振 → 资金面得分更高

数据源（akshare 1.18+）：
- 北向资金：ak.stock_hsgt_hist_em(symbol='沪股通')
- 行业资金流向：ak.stock_fund_flow_industry()
- 市场主力资金：ak.stock_market_fund_flow()
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
    from pathlib import Path
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


# ── 北向资金 ──
def get_north_net_flow(days: int = 20) -> pd.DataFrame:
    """获取北向资金近N日净流入。

    数据源优先级：
    1. ak.stock_hsgt_hist_em（历史数据，但2024年8月后净买额为NaN）
    2. ak.stock_hsgt_fund_flow_summary_em（实时汇总，仅当日）

    Returns
    -------
    pd.DataFrame: columns=[date, net_flow]
    """
    cache_key = f"north_flow_{days}"
    cached = _read_cache(cache_key, 3600)
    if cached is not None:
        return cached

    rows = []

    # 1) 尝试实时汇总接口
    try:
        import akshare as ak
        _rate_limit()

        df = ak.stock_hsgt_fund_flow_summary_em()
        if df is not None and not df.empty:
            # 筛选北向资金
            north = df[df["资金方向"] == "北向"]
            for _, row in north.iterrows():
                trade_date = pd.to_datetime(row.get("交易日", ""), errors="coerce")
                net_buy = pd.to_numeric(row.get("成交净买额", None), errors="coerce")
                net_inflow = pd.to_numeric(row.get("资金净流入", None), errors="coerce")

                flow = net_buy if pd.notna(net_buy) else (net_inflow if pd.notna(net_inflow) else None)
                if pd.notna(trade_date) and flow is not None:
                    rows.append({"date": trade_date, "net_flow": flow})
    except Exception as e:
        logger.debug(f"北向资金实时汇总失败: {e}")

    # 2) 补充历史数据
    try:
        import akshare as ak
        _rate_limit()

        df = ak.stock_hsgt_hist_em(symbol="沪股通")
        if df is not None and not df.empty:
            for col_name in df.columns:
                if "日期" in str(col_name):
                    df = df.rename(columns={col_name: "date"})
                elif "成交净买额" in str(col_name) or "净买额" in str(col_name):
                    df = df.rename(columns={col_name: "net_flow"})

            if "date" in df.columns and "net_flow" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df["net_flow"] = pd.to_numeric(df["net_flow"], errors="coerce")
                # 过滤有效数据（非NaN）
                valid = df[df["net_flow"].notna()].tail(days)
                for _, row in valid.iterrows():
                    # 避免与实时数据重复
                    if not any(r["date"] == row["date"] for r in rows):
                        rows.append({"date": row["date"], "net_flow": row["net_flow"]})
    except Exception as e:
        logger.debug(f"北向资金历史获取失败: {e}")

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows).sort_values("date").tail(days).reset_index(drop=True)
    _write_cache(cache_key, result)
    return result


def score_north_flow(days: int = 20) -> float:
    """计算北向资金因子得分。

    逻辑：
    - 近N日净流入累计 > 0 → 正分
    - 近N日持续净流入天数占比 > 60% → 加分
    - 归一化到 [-1, 1]

    Returns
    -------
    float : [-1, 1] 区间
    """
    df = get_north_net_flow(days)
    if df.empty or "net_flow" not in df.columns:
        return 0.0

    flows = df["net_flow"].dropna()
    if flows.empty:
        return 0.0

    # 累计净流入（单位：亿元）
    total_flow = flows.sum()

    # 净流入天数占比
    positive_days = (flows > 0).sum()
    positive_ratio = positive_days / len(flows) if len(flows) > 0 else 0.5

    # 综合评分
    # 累计流入归一化：50亿为满分
    flow_score = np.clip(total_flow / 50e8, -1, 1)
    # 正流入占比归一化
    ratio_score = np.clip((positive_ratio - 0.5) * 4, -1, 1)

    composite = 0.6 * flow_score + 0.4 * ratio_score
    return float(np.clip(composite, -1, 1))


# ── 主力资金（行业级别）──
def get_industry_fund_flow() -> pd.DataFrame:
    """获取行业资金流向（东方财富行业资金流）。

    Returns
    -------
    pd.DataFrame: columns=[行业, 净额, 流入资金, 流出资金, ...]
    """
    cache_key = "industry_fund_flow"
    cached = _read_cache(cache_key, 3600)
    if cached is not None:
        return cached

    try:
        import akshare as ak
        _rate_limit()

        df = ak.stock_fund_flow_industry()
        if df is None or df.empty:
            return pd.DataFrame()

        _write_cache(cache_key, df)
        return df
    except Exception as e:
        logger.debug(f"行业资金流向获取失败: {e}")
        return pd.DataFrame()


def get_market_fund_flow(days: int = 20) -> pd.DataFrame:
    """获取市场整体主力资金流数据。

    Returns
    -------
    pd.DataFrame: columns=[date, main_net_inflow, super_large_net, large_net, ...]
    """
    cache_key = f"market_fund_flow_{days}"
    cached = _read_cache(cache_key, 3600)
    if cached is not None:
        return cached

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

        df = df.rename(columns=rename_map)

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").tail(days).reset_index(drop=True)

        # 转数值
        for col in ["main_net_inflow", "main_net_pct", "super_large_net", "large_net"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        _write_cache(cache_key, df)
        return df
    except Exception as e:
        logger.debug(f"市场资金流向获取失败: {e}")
        return pd.DataFrame()


def score_main_fund_flow(sector: str = None, days: int = 5) -> float:
    """计算主力资金因子得分。

    逻辑：
    - 优先用行业资金流向的板块净额
    - 其次用市场整体主力资金流向
    - 归一化到 [-1, 1]

    Parameters
    ----------
    sector : 板块名称（可选）
    days : 回看天数

    Returns
    -------
    float : [-1, 1] 区间
    """
    # 1) 优先尝试行业资金流向
    if sector:
        industry_flow = get_industry_fund_flow()
        if not industry_flow.empty:
            # 找对应行业
            for col in industry_flow.columns:
                if "行业" in str(col):
                    match = industry_flow[industry_flow[col].astype(str).str.contains(sector, na=False)]
                    if not match.empty:
                        net_col = None
                        for c in match.columns:
                            if "净额" in str(c):
                                net_col = c
                                break
                        if net_col:
                            net_val = pd.to_numeric(match.iloc[0][net_col], errors="coerce")
                            if not pd.isna(net_val):
                                # 归一化：10亿为满分
                                return float(np.clip(net_val / 10e8, -1, 1))

    # 2) 用市场整体主力资金
    market_flow = get_market_fund_flow(days=days)
    if market_flow.empty:
        return 0.0

    if "main_net_inflow" in market_flow.columns:
        recent = market_flow["main_net_inflow"].dropna().tail(days)
        if not recent.empty:
            total = recent.sum()
            # 归一化：50亿为满分
            return float(np.clip(total / 50e8, -1, 1))

    return 0.0


def score_capital_factor(sector: str = None, days: int = 20) -> float:
    """资金因子综合得分：北向资金55% + 主力资金45%。

    Parameters
    ----------
    sector : 板块名称
    days : 北向资金回看天数

    Returns
    -------
    float : [-1, 1] 区间
    """
    north = score_north_flow(days)
    main = score_main_fund_flow(sector, days=min(5, days))

    composite = 0.55 * north + 0.45 * main
    return float(np.clip(composite, -1, 1))
