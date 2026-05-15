"""
个股数据获取层 —— 基于 baostock + akshare 的多因子选股数据支持。

支持的数据：
1. 全 A 股票列表（含市值、行业等）
2. 个股日 K 线（OHLCV + 换手率 + 涨跌幅）
3. 个股财务数据（营收、利润、ROE、EPS 等）
4. 个股估值数据（PE、PB、PS 等）
5. 申万行业分类映射

所有接口均有 parquet 缓存 + 限频保护。
"""
from __future__ import annotations

import time
import random
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from config.settings import CACHE_DIR, RATE_LIMIT_PER_SEC, RATE_LIMIT_DELAY_MIN, RATE_LIMIT_DELAY_MAX

logger = logging.getLogger(__name__)

# ── 限频（复用全局） ──
_last_call_times: list[float] = []


def _rate_limit() -> None:
    global _last_call_times
    now = time.monotonic()
    _last_call_times = [t for t in _last_call_times if now - t < 1.0]
    if len(_last_call_times) >= RATE_LIMIT_PER_SEC:
        wait = 1.0 - (now - _last_call_times[0]) + 0.05
        if wait > 0:
            time.sleep(wait)
    _last_call_times.append(time.monotonic())
    time.sleep(random.uniform(RATE_LIMIT_DELAY_MIN, RATE_LIMIT_DELAY_MAX))


def _rate_limit_heavy() -> None:
    time.sleep(random.uniform(2.0, 4.0))


# ── 缓存 ──
def _cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.parquet"


def _read_cache(name: str, max_age: int) -> pd.DataFrame | None:
    p = _cache_path(name)
    if not p.exists():
        return None
    if (time.time() - p.stat().st_mtime) > max_age:
        return None
    try:
        return pd.read_parquet(p)
    except Exception:
        return None


def _write_cache(name: str, df: pd.DataFrame) -> None:
    try:
        df.to_parquet(_cache_path(name), index=False)
    except Exception:
        pass


# ── baostock 工具 ──
_bs_logged_in = False


def _bs_login() -> bool:
    global _bs_logged_in
    if _bs_logged_in:
        return True
    try:
        import baostock as bs
        lg = bs.login()
        if lg.error_code == '0':
            _bs_logged_in = True
            return True
        return False
    except ImportError:
        return False


def _normalize_date(d: str) -> str:
    d = d.strip()
    if "-" in d:
        return d
    if len(d) == 8:
        return f"{d[:4]}-{d[4:6]}-{d[6:8]}"
    return d


def _bs_query_history(
    code: str,
    start: str,
    end: str,
    fields: str = "date,open,high,low,close,volume,amount,turn,pctChg",
    frequency: str = "d",
) -> pd.DataFrame:
    """baostock 查询 K 线。"""
    if not _bs_login():
        return pd.DataFrame()
    start = _normalize_date(start)
    end = _normalize_date(end)
    import baostock as bs
    rs = bs.query_history_k_data_plus(
        code, fields,
        start_date=start, end_date=end, frequency=frequency,
    )
    rows = []
    while rs.error_code == '0' and rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=rs.fields.split(",") if isinstance(rs.fields, str) else rs.fields)
    for col in df.columns:
        if col == "date":
            continue
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        except Exception:
            pass
    return df


# ════════════════════════════════════════════════════════
#  公开 API —— 个股级别
# ════════════════════════════════════════════════════════

def get_stock_list(date: str | None = None) -> pd.DataFrame:
    """获取全 A 股票列表，含市值、行业分类等。

    返回列: code, name, industry, list_date, type(1股票/2指数/3基金)
    """
    if date is None:
        date = datetime.now().strftime("%Y%m%d")

    cache_key = f"stock_list_{date}"
    cached = _read_cache(cache_key, 86400)
    if cached is not None:
        return cached

    # 1) akshare 实时行情（含市值）
    try:
        import akshare as ak
        _rate_limit_heavy()
        df = ak.stock_zh_a_spot_em()
        if df is not None and not df.empty:
            # 标准化列名
            col_map = {
                "序号": "seq", "代码": "code", "名称": "name",
                "最新价": "close", "涨跌幅": "pct_chg", "涨跌额": "change",
                "成交量": "volume", "成交额": "amount", "振幅": "amplitude",
                "最高": "high", "最低": "low", "今开": "open", "昨收": "pre_close",
                "量比": "volume_ratio", "换手率": "turnover_rate",
                "市盈率-动态": "pe_ttm", "市净率": "pb",
                "总市值": "market_cap", "流通市值": "float_market_cap",
                "涨速": "pct_speed", "5分钟涨跌": "pct_5min",
                "60日涨跌幅": "pct_60d", "年初至今涨跌幅": "pct_ytd",
            }
            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

            # 转换 baostock 代码格式: 600519 → sh.600519
            def _to_bs_code(row_code: str) -> str:
                c = str(row_code).strip()
                if c.startswith(("6", "5")):
                    return f"sh.{c}"
                elif c.startswith(("0", "3")):
                    return f"sz.{c}"
                elif c.startswith("4") or c.startswith("8"):
                    return f"bj.{c}"  # 北交所
                return f"sz.{c}"

            if "code" in df.columns:
                df["bs_code"] = df["code"].apply(_to_bs_code)

            # 过滤非主板/创业板/科创板
            if "code" in df.columns:
                df = df[~df["code"].astype(str).str.startswith(("4", "8"))]  # 排除北交所
                df = df[df["code"].astype(str).str.match(r"^\d{6}$")]  # 只要6位数字代码

            _write_cache(cache_key, df)
            return df
    except Exception as e:
        logger.warning(f"akshare 股票列表获取失败: {e}")

    # 2) baostock fallback
    try:
        if not _bs_login():
            return pd.DataFrame()
        import baostock as bs
        rs = bs.query_stock_basic(code_name="")
        rows = []
        while rs.error_code == '0' and rs.next():
            row = rs.get_row_data()
            if len(row) >= 5 and row[3] == "1":  # type=1 股票
                rows.append({
                    "code": row[1],
                    "name": row[2],
                    "type": row[3],
                    "list_date": row[4] if len(row) > 4 else "",
                    "industry": row[5] if len(row) > 5 else "",
                })
        if rows:
            df = pd.DataFrame(rows)
            _write_cache(cache_key, df)
            return df
    except Exception as e:
        logger.warning(f"baostock 股票列表获取失败: {e}")

    return pd.DataFrame()


def get_stock_history(
    code: str,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """获取个股日 K 线。

    Parameters
    ----------
    code : baostock 格式代码, 如 'sh.600519', 'sz.000858'
    start : 开始日期 YYYYMMDD 或 YYYY-MM-DD
    end : 结束日期
    """
    if end is None:
        end = datetime.now().strftime("%Y%m%d")
    if start is None:
        start = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")

    cache_key = f"stock_hist_{code}_{start}_{end}"
    cached = _read_cache(cache_key, 86400)
    if cached is not None:
        return cached

    # baostock 主力
    df = _bs_query_history(code, start, end)
    if not df.empty:
        rename = {"turn": "turnover_rate", "pctChg": "pct_chg"}
        df = df.rename(columns=rename)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        _write_cache(cache_key, df)
        return df

    return pd.DataFrame()


def get_stock_financial(
    code: str,
    year: int = 0,
    quarter: int = 0,
) -> pd.DataFrame:
    """获取个股财务指标（盈利能力）。

    Parameters
    ----------
    code : baostock 格式代码
    year : 年份（0 = 最近5年）
    quarter : 季度 1-4（0 = 全年）
    """
    if not _bs_login():
        return pd.DataFrame()

    cache_key = f"stock_fin_profit_{code}_{year}_{quarter}"
    cached = _read_cache(cache_key, 86400 * 7)  # 财务数据缓存7天
    if cached is not None:
        return cached

    import baostock as bs

    try:
        if year > 0:
            rs = bs.query_profit_data(code=code, year=year, quarter=quarter)
        else:
            # 最近5年
            all_rows = []
            cur_year = datetime.now().year
            for y in range(cur_year - 4, cur_year + 1):
                for q in [4, 3, 2, 1]:
                    if y == cur_year and q > (datetime.now().month // 3 + 1):
                        continue
                    rs = bs.query_profit_data(code=code, year=y, quarter=q)
                    while rs.error_code == '0' and rs.next():
                        all_rows.append(rs.get_row_data())
            if all_rows:
                df = pd.DataFrame(all_rows, columns=rs.fields.split(",") if isinstance(rs.fields, str) else rs.fields)
                for col in df.columns:
                    if col in ("code", "statDate"):
                        continue
                    try:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    except Exception:
                        pass
                _write_cache(cache_key, df)
                return df
            return pd.DataFrame()

        rows = []
        while rs.error_code == '0' and rs.next():
            rows.append(rs.get_row_data())
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=rs.fields.split(",") if isinstance(rs.fields, str) else rs.fields)
        for col in df.columns:
            if col in ("code", "statDate"):
                continue
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            except Exception:
                pass
        _write_cache(cache_key, df)
        return df
    except Exception as e:
        logger.debug(f"财务数据获取失败 {code}: {e}")
        return pd.DataFrame()


def get_stock_growth(
    code: str,
    year: int = 0,
    quarter: int = 0,
) -> pd.DataFrame:
    """获取个股成长性指标（营收增长率、利润增长率等）。"""
    if not _bs_login():
        return pd.DataFrame()

    cache_key = f"stock_fin_growth_{code}_{year}_{quarter}"
    cached = _read_cache(cache_key, 86400 * 7)
    if cached is not None:
        return cached

    import baostock as bs

    try:
        if year > 0:
            rs = bs.query_growth_data(code=code, year=year, quarter=quarter)
        else:
            all_rows = []
            cur_year = datetime.now().year
            for y in range(cur_year - 4, cur_year + 1):
                for q in [4, 3, 2, 1]:
                    if y == cur_year and q > (datetime.now().month // 3 + 1):
                        continue
                    rs = bs.query_growth_data(code=code, year=y, quarter=q)
                    while rs.error_code == '0' and rs.next():
                        all_rows.append(rs.get_row_data())
            if all_rows:
                df = pd.DataFrame(all_rows, columns=rs.fields.split(",") if isinstance(rs.fields, str) else rs.fields)
                for col in df.columns:
                    if col in ("code", "statDate"):
                        continue
                    try:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    except Exception:
                        pass
                _write_cache(cache_key, df)
                return df
            return pd.DataFrame()

        rows = []
        while rs.error_code == '0' and rs.next():
            rows.append(rs.get_row_data())
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=rs.fields.split(",") if isinstance(rs.fields, str) else rs.fields)
        for col in df.columns:
            if col in ("code", "statDate"):
                continue
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            except Exception:
                pass
        _write_cache(cache_key, df)
        return df
    except Exception as e:
        logger.debug(f"成长数据获取失败 {code}: {e}")
        return pd.DataFrame()


def get_stock_industry_map() -> dict[str, str]:
    """获取 {baostock代码: 申万行业名} 映射。"""
    cache_key = "stock_industry_map_v2"
    cached = _read_cache(cache_key, 86400 * 7)
    if cached is not None and not cached.empty:
        return dict(zip(cached["code"], cached["industry"]))

    if not _bs_login():
        return {}

    import baostock as bs

    try:
        rs = bs.query_stock_industry()
        rows = []
        while rs.error_code == '0' and rs.next():
            row = rs.get_row_data()
            if len(row) >= 4:
                rows.append({"code": row[1], "name": row[2], "industry": row[3]})

        if rows:
            df = pd.DataFrame(rows)
            _write_cache(cache_key, df)
            return dict(zip(df["code"], df["industry"]))
    except Exception as e:
        logger.warning(f"行业映射获取失败: {e}")

    return {}


def get_stock_valuation_batch(codes: list[str]) -> pd.DataFrame:
    """批量获取个股估值数据（PE/PB/市值）。

    通过 akshare 的实时行情接口获取。
    """
    cache_key = f"stock_valuation_batch_{datetime.now().strftime('%Y%m%d')}"
    cached = _read_cache(cache_key, 86400)
    if cached is not None:
        return cached[cached["code"].isin(codes)] if not cached.empty else cached

    try:
        import akshare as ak
        _rate_limit_heavy()
        df = ak.stock_zh_a_spot_em()
        if df is None or df.empty:
            return pd.DataFrame()

        # 只保留需要的列
        keep_cols = {}
        for cn, en in [("代码", "code"), ("名称", "name"), ("最新价", "close"),
                       ("涨跌幅", "pct_chg"), ("换手率", "turnover_rate"),
                       ("市盈率-动态", "pe_ttm"), ("市净率", "pb"),
                       ("总市值", "market_cap"), ("流通市值", "float_market_cap"),
                       ("成交额", "amount")]:
            if cn in df.columns:
                keep_cols[cn] = en

        df = df.rename(columns=keep_cols)
        keep = [c for c in keep_cols.values() if c in df.columns]
        df = df[keep]

        # 转换 baostock 代码格式
        def _to_bs_code(row_code: str) -> str:
            c = str(row_code).strip()
            if c.startswith(("6", "5")):
                return f"sh.{c}"
            elif c.startswith(("0", "3")):
                return f"sz.{c}"
            return f"sz.{c}"

        if "code" in df.columns:
            df["bs_code"] = df["code"].apply(_to_bs_code)

        for col in ["close", "pct_chg", "turnover_rate", "pe_ttm", "pb", "market_cap", "float_market_cap", "amount"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        _write_cache(cache_key, df)
        return df[df["code"].isin(codes)] if "code" in df.columns else df
    except Exception as e:
        logger.warning(f"批量估值获取失败: {e}")
        return pd.DataFrame()


def batch_get_stock_history(
    codes: list[str],
    start: str,
    end: str,
    show_progress: bool = True,
) -> dict[str, pd.DataFrame]:
    """批量获取多只股票的历史 K 线。

    Returns: {code: DataFrame}
    """
    result = {}
    total = len(codes)

    for i, code in enumerate(codes):
        if show_progress and (i + 1) % 50 == 0:
            logger.info(f"  下载进度: {i+1}/{total}")
        df = get_stock_history(code, start, end)
        if not df.empty and len(df) >= 20:  # 至少20个交易日
            result[code] = df

    return result
