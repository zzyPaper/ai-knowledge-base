"""
情绪因子 —— 涨停占比 + 市场宽度。

逻辑：
- 涨停占比：市场极端情绪指标，涨停家数占比高 → 市场过热或强势
- 市场宽度（涨跌家数）：上涨家数占比 → 市场健康度
- 两者结合判断市场情绪状态

数据源（akshare 1.18+）：
- 涨停统计：ak.stock_zt_pool_em(date='20260508')
- 市场资金流向（间接推算市场宽度）：ak.stock_market_fund_flow()
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


# ── 涨停统计 ──
def get_zt_stats(date: str = None) -> dict:
    """获取涨停统计。

    Parameters
    ----------
    date : 日期字符串，如 '20260508'，默认今天

    Returns
    -------
    dict: {zt_count, total_count, zt_ratio, date}
    """
    if date is None:
        date = datetime.now().strftime("%Y%m%d")

    cache_key = f"zt_stats_{date}"
    cached = _read_cache(cache_key, 86400)  # 涨停数据按日缓存
    if cached is not None:
        return cached.to_dict("records")[0] if not cached.empty else {}

    try:
        import akshare as ak
        _rate_limit()

        df = ak.stock_zt_pool_em(date=date)
        if df is None or df.empty:
            return {}

        zt_count = len(df)
        # A股约5000只
        total_count = 5000
        zt_ratio = zt_count / total_count

        result = {
            "zt_count": zt_count,
            "total_count": total_count,
            "zt_ratio": round(zt_ratio, 4),
            "date": date,
        }

        _write_cache(cache_key, pd.DataFrame([result]))
        return result
    except Exception as e:
        logger.debug(f"涨停统计获取失败: {e}")
        return {}


def score_zt_ratio(date: str = None) -> float:
    """涨停占比因子得分。

    逻辑：
    - 涨停占比 > 3% → 市场极度活跃 → 正分（但过热要警惕）
    - 涨停占比 1%-3% → 市场正常偏强 → 正分
    - 涨停占比 < 1% → 市场冷清 → 零或负分
    - 涨停占比 > 5% → 市场过热 → 打折（可能见顶）

    Returns
    -------
    float : [-1, 1] 区间
    """
    stats = get_zt_stats(date)
    if not stats:
        return 0.0

    zt_ratio = stats.get("zt_ratio", 0)

    if zt_ratio <= 0:
        return -0.5
    elif zt_ratio < 0.01:
        return -0.2
    elif zt_ratio < 0.02:
        return 0.2
    elif zt_ratio < 0.03:
        return 0.5
    elif zt_ratio < 0.05:
        return 0.8
    else:
        return 0.3  # 过热打折


# ── 市场宽度 ──
def get_market_breadth() -> dict:
    """获取市场宽度（通过市场资金流向推算）。

    使用 ak.stock_market_fund_flow() 的 上证-涨跌幅 + 深证-涨跌幅
    以及主力/小单资金流向比例来估算市场情绪。

    Returns
    -------
    dict: {sh_pct, sz_pct, main_net_pct, small_net_pct, advance_ratio}
    """
    cache_key = "market_breadth"
    cached = _read_cache(cache_key, 1800)
    if cached is not None:
        return cached.to_dict("records")[0] if not cached.empty else {}

    try:
        import akshare as ak
        _rate_limit()

        df = ak.stock_market_fund_flow()
        if df is None or df.empty:
            return {}

        latest = df.iloc[-1]
        result = {}

        for col in df.columns:
            col_str = str(col)
            if "上证-涨跌幅" in col_str:
                result["sh_pct"] = float(pd.to_numeric(latest[col], errors="coerce") or 0)
            elif "深证-涨跌幅" in col_str:
                result["sz_pct"] = float(pd.to_numeric(latest[col], errors="coerce") or 0)
            elif "主力净流入-净占比" in col_str:
                result["main_net_pct"] = float(pd.to_numeric(latest[col], errors="coerce") or 0)
            elif "小单净流入-净占比" in col_str:
                result["small_net_pct"] = float(pd.to_numeric(latest[col], errors="coerce") or 0)

        # 用主力净占比推算市场宽度
        # 主力净流入占比 > 0 → 大资金看多 → 市场偏强
        # 主力净流入占比 < 0 → 大资金看空 → 市场偏弱
        main_pct = result.get("main_net_pct", 0)
        # 归一化到 [0, 1] 的上涨比例
        # main_pct 范围大约 [-5, 5]，映射到 [0.2, 0.8]
        advance_ratio = 0.5 + main_pct / 10.0
        advance_ratio = np.clip(advance_ratio, 0, 1)
        result["advance_ratio"] = round(float(advance_ratio), 4)

        if result:
            _write_cache(cache_key, pd.DataFrame([result]))
        return result
    except Exception as e:
        logger.debug(f"市场宽度获取失败: {e}")
        return {}


def score_market_breadth() -> float:
    """市场宽度因子得分。

    逻辑：
    - 上涨家数占比 > 60% → 市场健康 → 正分
    - 上涨家数占比 40%-60% → 市场中性 → 零分附近
    - 上涨家数占比 < 40% → 市场偏弱 → 负分

    Returns
    -------
    float : [-1, 1] 区间
    """
    stats = get_market_breadth()
    if not stats or "advance_ratio" not in stats:
        return 0.0

    ratio = stats["advance_ratio"]
    score = (ratio - 0.5) * 4.0
    return float(np.clip(score, -1, 1))


def score_sentiment_factor(date: str = None) -> float:
    """情绪因子综合得分：涨停占比40% + 市场宽度60%。

    Returns
    -------
    float : [-1, 1] 区间
    """
    zt = score_zt_ratio(date)
    breadth = score_market_breadth()

    composite = 0.4 * zt + 0.6 * breadth
    return float(np.clip(composite, -1, 1))
