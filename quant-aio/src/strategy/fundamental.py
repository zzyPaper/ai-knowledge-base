"""
景气度因子 —— 行业营收/利润增速初筛。

逻辑：
- 景气度高的行业（营收+利润双增）更容易有持续性行情
- 用作初筛过滤：景气度不达标 → 一票否决，不给买入
- 数据来自最新财报，更新频率低（季度），适合做中长期的行业筛选

数据源（akshare 1.18+）：
- 行业PE（间接推算景气度）：ak.stock_industry_pe_ratio_cninfo
- 行业资金流向（间接推算）：ak.stock_fund_flow_industry

注：直接获取行业营收/利润增速的API在当前akshare版本不可靠，
    改用行业PE变化率 + 资金净流入来间接衡量景气度。
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


def get_sector_fundamental(sector: str) -> dict:
    """获取板块景气度指标。

    综合使用行业PE和资金流向来推算景气度：
    - PE环比变化：PE上升 → 市场给更高估值 → 景气预期改善
    - 资金净流入：主力持续流入 → 机构看好 → 景气确认

    Returns
    -------
    dict: {pe_change, fund_flow_score, composite}
    """
    cache_key = f"sector_fundamental_{sector}"
    cached = _read_cache(cache_key, 86400 * 3)  # 3天缓存
    if cached is not None:
        return cached.to_dict("records")[0] if not cached.empty else {}

    result = {}

    # 1) 行业PE变化率（间接衡量景气度改善）
    try:
        from src.strategy.valuation import get_industry_pe
        pe_df = get_industry_pe()
        if not pe_df.empty:
            name_col = None
            pe_col = None
            for col in pe_df.columns:
                if "行业名称" in str(col):
                    name_col = col
                elif "静态市盈率-加权平均" in str(col):
                    pe_col = col

            if name_col and pe_col:
                match = pe_df[pe_df[name_col].astype(str).str.contains(sector, na=False)]
                if not match.empty:
                    pe_val = pd.to_numeric(match.iloc[0][pe_col], errors="coerce")
                    if not pd.isna(pe_val):
                        # 全行业PE中位数
                        all_pe = pd.to_numeric(pe_df[pe_col], errors="coerce").dropna()
                        median_pe = all_pe.median()
                        # PE相对位置
                        relative_pe = (pe_val - median_pe) / median_pe if median_pe > 0 else 0
                        result["pe_relative"] = round(float(relative_pe), 4)
                        # PE水平本身反映估值
                        # 中低PE + 上升 → 景气改善初期
                        # 高PE → 可能过热或高景气已反映
                        if pe_val < median_pe * 0.8:
                            result["pe_signal"] = 0.5   # 低估值有空间
                        elif pe_val < median_pe * 1.2:
                            result["pe_signal"] = 0.0   # 中性
                        else:
                            result["pe_signal"] = -0.3  # 高估值需谨慎
    except Exception as e:
        logger.debug(f"行业PE获取失败 {sector}: {e}")

    # 2) 行业资金流向（确认景气）
    try:
        from src.strategy.capital_flow import get_industry_fund_flow
        flow_df = get_industry_fund_flow()
        if not flow_df.empty:
            for col in flow_df.columns:
                if "行业" in str(col):
                    match = flow_df[flow_df[col].astype(str).str.contains(sector, na=False)]
                    if not match.empty:
                        net_col = None
                        for c in match.columns:
                            if "净额" in str(c):
                                net_col = c
                                break
                        if net_col:
                            net_val = pd.to_numeric(match.iloc[0][net_col], errors="coerce")
                            if not pd.isna(net_val):
                                result["fund_net"] = float(net_val)
                                # 资金净流入>0 → 机构看好 → 景气确认
                                result["fund_signal"] = float(np.clip(net_val / 10e8, -1, 1))
    except Exception as e:
        logger.debug(f"行业资金流向获取失败 {sector}: {e}")

    if result:
        _write_cache(cache_key, pd.DataFrame([result]))
    return result


def score_fundamental(sector: str) -> float:
    """景气度因子得分。

    逻辑：
    - PE信号 + 资金信号 综合评估
    - PE低位+资金流入 → 强景气初期 → 0.8
    - PE中性+资金流入 → 景气确认 → 0.4
    - PE高位+资金流出 → 景气衰退 → -0.6
    - 无数据 → 中性0

    Returns
    -------
    float : [-1, 1] 区间
    """
    data = get_sector_fundamental(sector)
    if not data:
        return 0.0

    pe_signal = data.get("pe_signal", 0.0)
    fund_signal = data.get("fund_signal", 0.0)

    if pe_signal == 0.0 and fund_signal == 0.0:
        return 0.0

    # PE信号40% + 资金信号60%（资金是更直接的景气确认）
    composite = 0.4 * pe_signal + 0.6 * fund_signal
    return float(np.clip(composite, -1, 1))


def check_fundamental_pass(sector: str, min_score: float = -0.2) -> bool:
    """景气度初筛：是否通过。

    Parameters
    ----------
    sector : 板块名称
    min_score : 最低景气度得分阈值

    Returns
    -------
    bool : True=通过（可以买入），False=不通过（一票否决）
    """
    score = score_fundamental(sector)
    return score >= min_score
