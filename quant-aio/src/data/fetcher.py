"""
数据获取层 —— 多数据源 fallback（baostock → 东方财富 → 腾讯）。

支持的数据：
1. 指数日 K 线（东方财富优先）
2. 行业板块列表 & 聚合行情
3. 板块日 K 线历史（baostock 代表个股聚合优先）
4. 指数实时行情
5. V2因子数据：北向资金、行业资金流向、市场资金流向、涨停池、行业PE、指数PE

变更记录：
- 2026-05-13: 板块K线改为baostock优先（东方财富接口不稳定）
- 2026-05-13: 新增V2因子数据获取方法
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

from config.settings import (
    CACHE_DIR,
    RATE_LIMIT_PER_SEC,
    RATE_LIMIT_DELAY_MIN,
    RATE_LIMIT_DELAY_MAX,
    CACHE_TTL_SECTOR_LIST,
    CACHE_TTL_SECTOR_HIST,
    CACHE_TTL_INDEX_HIST,
    CACHE_TTL_LIVE_QUOTE,
    HOT_SECTOR_TOP_N,
    KLINE_LOOKBACK_DAYS,
)

logger = logging.getLogger(__name__)

# ── 限频 ──────────────────────────────────────────────
_last_call_times: list[float] = []


def _rate_limit() -> None:
    """全局 API 限频。"""
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
    """重度限频（东方财富等易被封的源）。"""
    time.sleep(random.uniform(2.0, 4.0))


# ── 重试装饰器 ──
def _retry(func, max_retries=2, delay=3.0, *args, **kwargs):
    """带重试的函数调用。"""
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt < max_retries:
                wait = delay * (attempt + 1) + random.uniform(0, 1)
                logger.debug(f"重试 {func.__name__} (第{attempt+1}次)，等待{wait:.1f}s: {e}")
                time.sleep(wait)
            else:
                raise


# ── 缓存 ──────────────────────────────────────────────
_COL_MAP_KLINE = {
    "日期": "date", "开盘": "open", "收盘": "close",
    "最高": "high", "最低": "low", "成交量": "volume",
    "成交额": "amount", "振幅": "amplitude",
    "涨跌幅": "pct_chg", "涨跌额": "change", "换手率": "turnover_rate",
}


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


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """中文字段 → 英文。"""
    return df.rename(columns=_COL_MAP_KLINE)


# ── baostock 工具 ─────────────────────────────────────
_baostock_logged_in = False


def _normalize_date(d: str) -> str:
    """日期格式统一为 YYYY-MM-DD（baostock 要求）。"""
    d = d.strip()
    if "-" in d:
        return d
    if len(d) == 8:
        return f"{d[:4]}-{d[4:6]}-{d[6:8]}"
    return d


def _bs_login() -> bool:
    """baostock 登录（单例）。"""
    global _baostock_logged_in
    if _baostock_logged_in:
        return True
    try:
        import baostock as bs
        lg = bs.login()
        if lg.error_code == '0':
            _baostock_logged_in = True
            return True
        logger.warning(f"baostock login failed: {lg.error_msg}")
        return False
    except ImportError:
        logger.warning("baostock not installed")
        return False


def _bs_logout() -> None:
    global _baostock_logged_in
    if _baostock_logged_in:
        try:
            import baostock as bs
            bs.logout()
        except Exception:
            pass
        _baostock_logged_in = False


def _bs_query_history(
    code: str,
    start: str,
    end: str,
    fields: str = "date,open,high,low,close,volume,amount,turn,pctChg",
    frequency: str = "d",
) -> pd.DataFrame:
    """baostock 查询 K 线，返回 DataFrame。日期格式 YYYY-MM-DD。"""
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


# ── 申万行业映射 ──────────────────────────────────────
_SW_INDUSTRY_MAP: dict[str, dict] | None = None

# 申万行业 → 代表股票映射（用于baostock板块K线聚合）
_SECTOR_REPRESENTATIVE_STOCKS: dict[str, list[str]] = {
    # ── 金融 ──
    "银行": ["sh.600000", "sh.601398", "sh.600036", "sh.601288"],
    "证券": ["sh.601211", "sh.600030", "sz.000776", "sh.601688"],
    "保险": ["sh.601318", "sh.601601", "sh.601336", "sz.002141"],
    "多元金融": ["sh.600823", "sz.000562", "sh.600643", "sz.002673"],
    # ── 科技 ──
    "软件开发": ["sh.600588", "sz.002230", "sz.300033", "sz.000977"],
    "半导体": ["sh.601012", "sz.002049", "sh.688981", "sz.300782"],
    "计算机": ["sz.000977", "sh.600845", "sz.300449", "sz.002405"],
    "通信设备": ["sz.000063", "sh.600050", "sz.002415", "sz.300136"],
    "电子": ["sz.002475", "sh.601138", "sz.300408", "sz.002371"],
    "消费电子": ["sz.002475", "sz.000725", "sh.601138", "sz.002236"],
    "光学光电子": ["sz.000725", "sh.600703", "sz.002049", "sz.300136"],
    "通信服务": ["sh.600050", "sz.000063", "sh.600941", "sz.300310"],
    "互联网服务": ["sz.300059", "sz.002602", "sh.601928", "sz.300418"],
    "人工智能": ["sz.002230", "sz.300033", "sz.002415", "sh.688787"],
    "数字芯片设计": ["sh.688981", "sz.300782", "sz.002049", "sh.688256"],
    "集成电路": ["sh.688981", "sz.300782", "sz.002049", "sh.601012"],
    # ── 医药 ──
    "医药制造": ["sh.600276", "sz.000538", "sz.300760", "sh.600196"],
    "医疗器械": ["sz.300760", "sh.688111", "sz.002432", "sh.603259"],
    "医疗服务": ["sz.300015", "sh.601607", "sz.002432", "sh.688111"],
    "中药": ["sz.000538", "sh.600085", "sz.000963", "sh.600436"],
    "生物制品": ["sh.600196", "sz.300122", "sz.300601", "sh.688111"],
    # ── 消费 ──
    "白酒": ["sh.600519", "sz.000858", "sz.000568", "sh.600809"],
    "食品饮料": ["sh.600887", "sz.000895", "sh.603288", "sz.002304"],
    "家用电器": ["sz.000651", "sz.000333", "sh.600690", "sz.002032"],
    "汽车制造": ["sh.600104", "sz.000625", "sh.601238", "sz.002594"],
    "汽车零部件": ["sz.002594", "sh.600741", "sz.002048", "sz.002284"],
    "汽车整车": ["sh.600104", "sz.000625", "sh.601238", "sz.002594"],
    "零售": ["sh.600655", "sz.002024", "sh.600827", "sz.000413"],
    "纺织服装": ["sh.600177", "sz.002003", "sz.000726", "sh.600555"],
    "美容护理": ["sh.603983", "sz.300957", "sz.002612", "sz.300896"],
    "宠物经济": ["sz.002891", "sz.300673", "sh.603566", "sz.002124"],
    "旅游酒店": ["sh.600054", "sz.000524", "sh.601888", "sz.002707"],
    # ── 周期/资源 ──
    "有色金属": ["sh.601899", "sh.600489", "sz.000831", "sh.603993"],
    "化工": ["sh.600309", "sz.002493", "sh.600426", "sz.002092"],
    "钢铁": ["sh.600019", "sh.600782", "sz.000708", "sh.601003"],
    "煤炭": ["sh.601088", "sh.601898", "sz.000983", "sh.600188"],
    "石油": ["sh.601857", "sh.600028", "sz.000554", "sh.601808"],
    "黄金": ["sh.600489", "sh.600547", "sz.002155", "sh.600988"],
    "铜": ["sh.601899", "sh.600362", "sz.000831", "sh.603993"],
    "锂": ["sz.002460", "sz.002466", "sz.300073", "sh.603993"],
    "稀土": ["sh.600111", "sz.000831", "sh.600259", "sz.002497"],
    "钨": ["sh.600549", "sz.002378", "sz.000657", "sh.603993"],
    "钼": ["sh.601958", "sh.603993", "sz.000657", "sh.600549"],
    # ── 制造/基建 ──
    "电力设备": ["sh.600886", "sz.300750", "sh.601012", "sz.002129"],
    "房地产": ["sh.600048", "sz.000002", "sh.600340", "sz.001979"],
    "国防军工": ["sh.600893", "sh.600760", "sz.002179", "sh.600862"],
    "建筑": ["sh.601668", "sh.601390", "sh.600585", "sz.000066"],
    "机械设备": ["sz.002008", "sh.600150", "sz.000528", "sz.300124"],
    "电力": ["sh.600886", "sh.600795", "sh.600023", "sz.000539"],
    "光伏": ["sh.601012", "sz.002459", "sz.300274", "sh.600438"],
    "风电": ["sh.600905", "sz.002202", "sh.601615", "sz.300564"],
    "储能": ["sz.300750", "sz.002074", "sz.300014", "sz.002129"],
    "输变电设备": ["sh.600089", "sz.002169", "sh.600550", "sz.002070"],
    "火力发电": ["sh.600886", "sh.600795", "sh.600023", "sz.000539"],
    "水力发电": ["sh.600886", "sh.600900", "sz.000539", "sh.600674"],
    # ── 其他 ──
    "交通运输": ["sh.601111", "sh.601006", "sh.600029", "sh.601872"],
    "传媒": ["sz.300059", "sz.002602", "sh.601928", "sz.300418"],
    "教育": ["sz.000526", "sz.002308", "sh.600661", "sz.300010"],
    "环保": ["sz.000826", "sh.601200", "sz.300070", "sz.002573"],
    "农业": ["sz.000998", "sh.600598", "sz.002714", "sh.600467"],
    "造纸": ["sz.002078", "sh.600567", "sz.000488", "sh.600797"],
}

# 板块 → ETF代码映射（优先使用ETF数据，波动率更真实）
_SECTOR_ETF_MAP: dict[str, str] = {
    # 金融
    "银行": "512800",       # 银行ETF
    "证券": "512880",       # 证券ETF
    "保险": "512870",       # 保险ETF
    # 科技
    "半导体": "512480",     # 半导体ETF
    "软件开发": "515230",   # 软件ETF
    "电子": "159996",       # 电子ETF
    "通信设备": "515880",   # 通信ETF
    "人工智能": "159819",   # AI龙头ETF
    "集成电路": "159546",   # 集成电路ETF
    # 医药
    "医药制造": "512010",   # 医药ETF
    "医疗器械": "159898",   # 医疗器械ETF
    "中药": "159883",       # 中药ETF
    "生物制品": "159881",   # 生物医药ETF
    # 消费
    "白酒": "512690",       # 白酒ETF
    "食品饮料": "515170",   # 食品饮料ETF
    "家用电器": "159996",   # 家电ETF(借用)
    "汽车整车": "516110",   # 汽车ETF
    "汽车制造": "516110",   # 汽车ETF
    "美容护理": "159901",   # 消费ETF(借用)
    "旅游酒店": "159766",   # 旅游ETF
    # 周期/资源
    "有色金属": "512400",   # 有色金属ETF
    "化工": "159870",       # 化工ETF
    "钢铁": "515210",       # 钢铁ETF
    "煤炭": "515220",       # 煤炭ETF
    "石油": "516260",       # 石油ETF
    "黄金": "518880",       # 黄金ETF
    # 制造/基建
    "电力设备": "159611",   # 电力设备ETF
    "房地产": "512200",     # 房地产ETF
    "国防军工": "512660",   # 军工ETF
    "建筑": "159745",       # 基建ETF
    "电力": "159611",       # 电力ETF(借用)
    "光伏": "515790",       # 光伏ETF
    "风电": "159611",       # 风电ETF(借用)
    # 其他
    "交通运输": "516160",   # 新能源车ETF(借用)
    "传媒": "159805",       # 传媒ETF
    "环保": "159861",       # 环保ETF
    "农业": "159825",       # 农业ETF
}


def _get_sw_industry_map() -> dict[str, dict]:
    """获取申万行业映射 {行业名: {code, name, stocks: [...]}}."""
    global _SW_INDUSTRY_MAP
    if _SW_INDUSTRY_MAP is not None:
        return _SW_INDUSTRY_MAP

    cached = _read_cache("sw_industry_map", CACHE_TTL_SECTOR_LIST)
    if cached is not None and not cached.empty:
        result = {}
        for _, row in cached.iterrows():
            ind = row["industry"]
            if ind not in result:
                result[ind] = {"industry": ind, "stocks": []}
            result[ind]["stocks"].append({"code": row["code"], "name": row["name"]})
        _SW_INDUSTRY_MAP = result
        return _SW_INDUSTRY_MAP

    if not _bs_login():
        _SW_INDUSTRY_MAP = _fallback_industry_map()
        return _SW_INDUSTRY_MAP

    import baostock as bs
    rs = bs.query_stock_industry()
    rows = []
    while rs.error_code == '0' and rs.next():
        rows.append(rs.get_row_data())

    result: dict[str, dict] = {}
    cache_rows = []
    for row in rows:
        if len(row) < 4 or not row[3]:
            continue
        ind = row[3]
        code = row[1]
        name = row[2]
        if ind not in result:
            result[ind] = {"industry": ind, "stocks": []}
        result[ind]["stocks"].append({"code": code, "name": name})
        cache_rows.append({"industry": ind, "code": code, "name": name})

    if cache_rows:
        _write_cache("sw_industry_map", pd.DataFrame(cache_rows))

    _SW_INDUSTRY_MAP = result
    return _SW_INDUSTRY_MAP


def _fallback_industry_map() -> dict[str, dict]:
    """网络不可用时的行业 fallback。"""
    fb = {
        "银行": ["sh.600000", "sh.601398", "sh.600036"],
        "软件开发": ["sh.600588", "sz.002230", "sz.300033"],
        "半导体": ["sh.601012", "sz.002049", "sh.688981"],
        "医药制造": ["sh.600276", "sz.000538", "sz.300760"],
        "证券": ["sh.601211", "sh.600030", "sz.000776"],
        "白酒": ["sh.600519", "sz.000858", "sz.000568"],
        "汽车制造": ["sh.600104", "sz.000625", "sh.601238"],
        "电力设备": ["sh.600886", "sz.300750", "sh.601012"],
        "食品饮料": ["sh.600887", "sz.000895", "sh.603288"],
        "通信设备": ["sz.000063", "sh.600050", "sz.002415"],
        "房地产": ["sh.600048", "sz.000002", "sh.600340"],
        "国防军工": ["sh.600893", "sh.600760", "sz.002179"],
        "计算机": ["sz.000977", "sh.600845", "sz.300449"],
        "传媒": ["sz.300059", "sz.002602", "sh.601928"],
        "电子": ["sz.002475", "sh.601138", "sz.300408"],
    }
    result = {}
    for ind, codes in fb.items():
        result[ind] = {
            "industry": ind,
            "stocks": [{"code": c, "name": c} for c in codes],
        }
    return result


# ── 行业简称映射 ──────────────────────────────────────
_INDUSTRY_SHORT_NAMES = {
    "C39计算机、通信和其他电子设备制造业": "电子",
    "I65软件和信息技术服务业": "软件开发",
    "C27医药制造业": "医药",
    "C36汽车制造业": "汽车",
    "J66货币金融服务": "银行",
    "K70房地产业": "房地产",
    "D44电力、热力生产和供应业": "电力",
    "C38电气机械和器材制造业": "电气设备",
    "C35专用设备制造业": "专用设备",
    "C26化学原料和化学制品制造业": "化工",
    "C34通用设备制造业": "通用设备",
    "C32有色金属冶炼和压延加工业": "有色金属",
    "C37铁路、船舶、航空航天和其他运输设备制造业": "军工",
    "G54道路运输业": "交通运输",
    "F52零售业": "零售",
    "G56航空运输业": "航空",
    "C29橡胶和塑料制品业": "橡胶塑料",
    "C30非金属矿物制品业": "建材",
    "C40仪器仪表制造业": "仪器仪表",
    "L72商务服务业": "商务服务",
    "I64互联网和相关服务": "互联网",
    "C33金属制品业": "金属制品",
    "F51批发业": "批发",
    "E47房屋建筑业": "建筑",
    "M74专业技术服务业": "专业服务",
    "R88体育": "体育",
    "R89体育": "体育",
    "N78公共设施管理业": "公共设施",
}


def _short_industry_name(full_name: str) -> str:
    """申万行业全名 → 干净简称。"""
    import re
    clean = re.sub(r'^[A-Z]\d+', '', full_name)
    if len(clean) <= 2:
        clean = full_name
    _SHORTCUT = {
        "水的生产和供应业": "水务",
        "黑色金属冶炼和压延加工业": "钢铁",
        "水上运输业": "航运",
        "石油和天然气开采业": "石油开采",
        "资本市场服务": "证券",
        "电信、广播电视和卫星传输服务": "通信",
        "土木工程建筑业": "基建",
        "酒、饮料和精制茶制造业": "食品饮料",
        "农副食品加工业": "农产品",
        "广播、电视、电影和录音制作业": "传媒",
        "纺织服装、服饰业": "纺织服装",
        "其他金融业": "多元金融",
        "煤炭开采和洗选业": "煤炭",
        "铁路运输业": "铁路",
        "生态保护和环境治理业": "环保",
        "计算机、通信和其他电子设备制造业": "电子",
        "软件和信息技术服务业": "软件开发",
        "医药制造业": "医药",
        "汽车制造业": "汽车",
        "货币金融服务": "银行",
        "房地产业": "房地产",
        "电力、热力生产和供应业": "电力",
        "电气机械和器材制造业": "电气设备",
        "化学原料和化学制品制造业": "化工",
        "有色金属冶炼和压延加工业": "有色金属",
        "铁路、船舶、航空航天和其他运输设备制造业": "军工",
    }
    return _SHORTCUT.get(full_name, clean)


# ════════════════════════════════════════════════════════
#  公开 API
# ════════════════════════════════════════════════════════

# ── 指数数据 ──────────────────────────────────────────
_INDEX_CODE_MAP_TX = {
    "中证A500": "sh000510",
    "上证指数": "sh000001",
    "深证成指": "sz399001",
    "创业板指": "sz399006",
    "沪深300": "sh000300",
}

_INDEX_CODE_MAP_BS = {
    "中证A500": "sh.000510",
    "上证指数": "sh.000001",
    "深证成指": "sz.399001",
    "创业板指": "sz.399006",
    "沪深300": "sh.000300",
}


def get_index_history(
    index_name: str = "中证A500",
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """获取指数日 K 线。数据源：东方财富 → baostock → 腾讯。"""
    if end is None:
        end = datetime.now().strftime("%Y%m%d")
    if start is None:
        start = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")

    cache_key = f"index_hist_{index_name}_{start}_{end}"
    cached = _read_cache(cache_key, CACHE_TTL_INDEX_HIST)
    if cached is not None:
        return cached

    # 1) 东方财富
    df = _get_index_em(index_name, start, end)
    if df is not None and not df.empty:
        _write_cache(cache_key, df)
        return df

    # 2) baostock
    df = _get_index_bs(index_name, start, end)
    if df is not None and not df.empty:
        _write_cache(cache_key, df)
        return df

    # 3) 腾讯
    df = _get_index_tx(index_name, start, end)
    if df is not None and not df.empty:
        _write_cache(cache_key, df)
        return df

    logger.warning(f"所有数据源均无法获取指数 {index_name}")
    return pd.DataFrame()


def _get_index_em(index_name: str, start: str, end: str) -> pd.DataFrame | None:
    """东方财富源获取指数日线。"""
    try:
        import akshare as ak
        code = _INDEX_CODE_MAP_TX.get(index_name, "sh000300")
        _rate_limit()
        df = ak.stock_zh_index_daily_em(symbol=code, start_date=start, end_date=end)
        if df is None or df.empty:
            return None
        for col in ["open", "close", "high", "low", "volume", "amount"]:
            if col not in df.columns:
                df[col] = np.nan
        if "pct_chg" not in df.columns:
            df["pct_chg"] = df["close"].pct_change() * 100
        df["date"] = pd.to_datetime(df["date"])
        return df.reset_index(drop=True)
    except Exception as e:
        logger.debug(f"东方财富指数日线失败: {e}")
        return None


def _get_index_bs(index_name: str, start: str, end: str) -> pd.DataFrame | None:
    """baostock 获取指数日线。"""
    try:
        code = _INDEX_CODE_MAP_BS.get(index_name, "sh.000300")
        df = _bs_query_history(code, start, end)
        if df.empty:
            return None
        rename = {"turn": "turnover_rate", "pctChg": "pct_chg"}
        df = df.rename(columns=rename)
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception as e:
        logger.debug(f"baostock获取指数失败: {e}")
        return None


def _get_index_tx(index_name: str, start: str, end: str) -> pd.DataFrame | None:
    """腾讯源获取指数日线。"""
    try:
        import akshare as ak
        code = _INDEX_CODE_MAP_TX.get(index_name, "sh000510")
        _rate_limit()
        df = ak.stock_zh_index_daily_tx(symbol=code)
        if df is None or df.empty:
            return None
        df["date"] = pd.to_datetime(df["date"])
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)].copy()
        for col in ["open", "close", "high", "low", "volume", "amount"]:
            if col not in df.columns:
                df[col] = np.nan
        if "pct_chg" not in df.columns:
            df["pct_chg"] = df["close"].pct_change() * 100
        return df.reset_index(drop=True)
    except Exception as e:
        logger.debug(f"腾讯源获取指数失败: {e}")
        return None


# ── 板块数据 ──────────────────────────────────────────
def get_sectors_list() -> pd.DataFrame:
    """获取所有行业板块列表 + 聚合行情。"""
    cached = _read_cache("sector_list", CACHE_TTL_SECTOR_LIST)
    if cached is not None:
        return cached

    # 1) 东方财富
    df = _get_sectors_em()
    if df is not None and not df.empty:
        _write_cache("sector_list", df)
        return df

    # 2) baostock 行业聚合
    df = _get_sectors_bs()
    if df is not None and not df.empty:
        _write_cache("sector_list", df)
        return df

    # 3) 硬编码 fallback
    fb = _fallback_industry_map()
    return pd.DataFrame({
        "板块名称": list(fb.keys()),
        "涨跌幅": [0.0] * len(fb),
        "成交额": [0.0] * len(fb),
    })


def _get_sectors_em() -> pd.DataFrame | None:
    """东方财富获取板块列表。"""
    try:
        import akshare as ak
        _rate_limit_heavy()
        df = ak.stock_board_industry_name_em()
        return df
    except Exception as e:
        logger.debug(f"东方财富板块列表失败: {e}")
        return None


def _get_sectors_bs() -> pd.DataFrame | None:
    """baostock 行业分类 + 聚合最近涨跌幅。"""
    try:
        industry_map = _get_sw_industry_map()
        if not industry_map:
            return None

        rows = []
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        for ind_name, ind_data in industry_map.items():
            short = _short_industry_name(ind_name)
            import re
            if re.match(r'^[A-Z]\d+', short):
                continue
            stocks = ind_data.get("stocks", [])
            if not stocks:
                continue

            pcts = []
            vol_total = 0.0
            for s in stocks[:4]:
                code = s["code"]
                df = _bs_query_history(code, start, end, fields="date,close,volume,amount,pctChg")
                if df.empty:
                    continue
                last_pct = df["pctChg"].iloc[-1] if "pctChg" in df.columns else 0.0
                pcts.append(float(last_pct) if pd.notna(last_pct) else 0.0)
                if "amount" in df.columns:
                    amt = df["amount"].iloc[-1]
                    vol_total += float(amt) if pd.notna(amt) else 0.0

            avg_pct = sum(pcts) / len(pcts) if pcts else 0.0
            rows.append({
                "板块名称": short,
                "行业全名": ind_name,
                "涨跌幅": avg_pct,
                "成交额": vol_total,
            })

        if not rows:
            return None
        return pd.DataFrame(rows)
    except Exception as e:
        logger.debug(f"baostock行业聚合失败: {e}")
        return None


def get_sector_history(
    sector: str,
    start: str | None = None,
    end: str | None = None,
    period: str = "daily",
) -> pd.DataFrame:
    """获取板块日 K 线历史。

    数据源优先级：ETF基金K线 → baostock代表个股聚合 → 东方财富板块K线。
    v3.0: ETF优先，因为ETF波动率更真实，ATR止损更准确。
    """
    if end is None:
        end = datetime.now().strftime("%Y%m%d")
    if start is None:
        start = (datetime.now() - timedelta(days=KLINE_LOOKBACK_DAYS * 2)).strftime("%Y%m%d")

    cache_key = f"sector_hist_{sector}_{period}_{start}_{end}"
    cached = _read_cache(cache_key, CACHE_TTL_SECTOR_HIST)
    if cached is not None:
        return cached

    # 0) ETF基金K线（优先，波动率最真实）
    etf_code = _SECTOR_ETF_MAP.get(sector)
    if etf_code:
        df = _get_sector_hist_etf(etf_code, start, end)
        if df is not None and not df.empty:
            _write_cache(cache_key, df)
            return df

    # 1) baostock 代表个股聚合
    df = _get_sector_hist_bs(sector, start, end)
    if df is not None and not df.empty:
        _write_cache(cache_key, df)
        return df

    # 2) 东方财富
    df = _get_sector_hist_em(sector, start, end, period)
    if df is not None and not df.empty:
        _write_cache(cache_key, df)
        return df

    logger.warning(f"所有数据源均无法获取板块 {sector} 历史")
    return pd.DataFrame()


def _get_sector_hist_etf(etf_code: str, start: str, end: str) -> pd.DataFrame | None:
    """获取ETF基金K线（akshare）。"""
    try:
        import akshare as ak
        _rate_limit()
        # fund_etf_hist_em 获取ETF历史K线
        df = ak.fund_etf_hist_em(symbol=etf_code, start_date=start, end_date=end, period="daily")
        if df is None or df.empty:
            return None

        df = _rename_columns(df)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        if "pct_chg" not in df.columns and "close" in df.columns:
            df["pct_chg"] = df["close"].pct_change() * 100
        if "turnover_rate" not in df.columns:
            df["turnover_rate"] = 0.0
        return df.reset_index(drop=True)
    except Exception as e:
        logger.debug(f"ETF K线获取失败({etf_code}): {e}")
        return None


def _get_sector_hist_em(sector: str, start: str, end: str, period: str) -> pd.DataFrame | None:
    """东方财富获取板块K线。"""
    try:
        import akshare as ak
        _rate_limit_heavy()
        df = ak.stock_board_industry_hist_em(
            symbol=sector, start_date=start, end_date=end, period=period
        )
        df = _rename_columns(df)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception as e:
        logger.debug(f"东方财富板块K线失败: {e}")
        return None


def _lookup_full_industry_name(short_name: str) -> str | None:
    """从板块列表缓存中查找简称对应的行业全名。"""
    cached = _read_cache("sector_list", CACHE_TTL_SECTOR_LIST)
    if cached is not None and not cached.empty and "行业全名" in cached.columns:
        match = cached[cached["板块名称"] == short_name]
        if not match.empty:
            return str(match.iloc[0]["行业全名"])
    return None


def _get_sector_hist_bs(sector: str, start: str, end: str) -> pd.DataFrame | None:
    """baostock 用行业代表股聚合板块K线。"""
    try:
        bs_start = _normalize_date(start)
        bs_end = _normalize_date(end)

        # 1) 优先用预定义的代表股票列表
        representative_stocks = _SECTOR_REPRESENTATIVE_STOCKS.get(sector)

        # 2) 其次从行业映射中查找
        if not representative_stocks:
            industry_map = _get_sw_industry_map()
            full_name = _lookup_full_industry_name(sector)
            target_ind = None
            if full_name and full_name in industry_map:
                target_ind = industry_map[full_name]
            if target_ind is None:
                for ind_name, ind_data in industry_map.items():
                    short = _short_industry_name(ind_name)
                    if short == sector or ind_name == sector:
                        target_ind = ind_data
                        break
            if target_ind is None:
                for ind_name, ind_data in industry_map.items():
                    short = _short_industry_name(ind_name)
                    if sector in short or short in sector or sector in ind_name:
                        target_ind = ind_data
                        break

            if target_ind is None:
                return None

            stocks = target_ind.get("stocks", [])
            representative_stocks = [s["code"] for s in stocks[:4]]

        if not representative_stocks:
            return None

        # 聚合代表股K线
        all_dfs = []
        for code in representative_stocks[:4]:
            df = _bs_query_history(code, bs_start, bs_end)
            if df.empty:
                continue
            all_dfs.append(df)

        if not all_dfs:
            return None

        combined = pd.concat(all_dfs, ignore_index=True)
        agg = combined.groupby("date").agg({
            "open": "mean",
            "close": "mean",
            "high": "mean",
            "low": "mean",
            "volume": "sum",
            "amount": "sum",
        }).reset_index()

        agg["pct_chg"] = agg["close"].pct_change() * 100
        agg["turnover_rate"] = 0.0
        agg["date"] = pd.to_datetime(agg["date"])
        return agg
    except Exception as e:
        logger.debug(f"baostock板块聚合失败: {e}")
        return None


def get_hot_sectors(top_n: int = HOT_SECTOR_TOP_N) -> pd.DataFrame:
    """获取当日热门板块（按成交额降序取 top_n）。"""
    df = get_sectors_list()
    amt_col = None
    for c in ("总成交额", "成交额"):
        if c in df.columns:
            amt_col = c
            break
    if amt_col:
        df = df.sort_values(amt_col, ascending=False)
    return df.head(top_n)


def get_hot_sectors_detail(top_n: int = HOT_SECTOR_TOP_N) -> dict[str, pd.DataFrame]:
    """获取热门板块 + 每个板块最近 N 天日 K 线。"""
    hot = get_hot_sectors(top_n)
    result: dict[str, pd.DataFrame] = {}
    for _, row in hot.iterrows():
        name = row.get("板块名称", row.iloc[0])
        hist = get_sector_history(name)
        if hist is not None and not hist.empty:
            hist = hist.tail(KLINE_LOOKBACK_DAYS)
            result[name] = hist
    return result


# ── 指数实时行情 ──────────────────────────────────────
def get_index_live(keyword: str = "中证A500") -> dict:
    """获取指数实时行情快照。"""
    cache_key = f"index_live_{keyword}"
    cached = _read_cache(cache_key, CACHE_TTL_LIVE_QUOTE)
    if cached is not None:
        return cached.to_dict("records")[0] if not cached.empty else {}

    try:
        import akshare as ak
        _rate_limit()
        df = ak.stock_zh_index_spot_em()
        match = df[df["名称"].str.contains(keyword, na=False)]
        _write_cache(cache_key, match)
        if match.empty:
            return {}
        return match.iloc[0].to_dict()
    except Exception:
        try:
            import akshare as ak
            code = _INDEX_CODE_MAP_TX.get(keyword, "sh000510")
            _rate_limit()
            df = ak.stock_zh_index_daily_tx(symbol=code)
            if df is not None and not df.empty:
                latest = df.iloc[-1]
                return {
                    "名称": keyword,
                    "最新价": latest.get("close", 0),
                    "涨跌幅": latest.get("pct_chg", 0),
                }
        except Exception:
            pass
        return {}


# ── 板块涨跌归因 ──────────────────────────────────────
def get_sector_attribution(sector: str) -> pd.DataFrame:
    """获取板块涨跌归因。"""
    cache_key = f"sector_attr_{sector}"
    cached = _read_cache(cache_key, 3600)
    if cached is not None:
        return cached

    try:
        import akshare as ak
        _rate_limit_heavy()
        df = ak.stock_board_industry_cons_em(symbol=sector)
        _write_cache(cache_key, df)
        return df
    except Exception:
        try:
            industry_map = _get_sw_industry_map()
            for ind_name, ind_data in industry_map.items():
                short = _short_industry_name(ind_name)
                if short == sector or ind_name == sector:
                    stocks = ind_data.get("stocks", [])[:10]
                    return pd.DataFrame([
                        {"代码": s["code"], "名称": s["name"]}
                        for s in stocks
                    ])
        except Exception:
            pass
        return pd.DataFrame()


# ════════════════════════════════════════════════════════
#  V2 因子数据获取
# ════════════════════════════════════════════════════════

def get_north_flow_history(symbol: str = "沪股通") -> pd.DataFrame:
    """获取北向资金历史数据。

    Parameters
    ----------
    symbol : '沪股通' 或 '深股通'

    Returns
    -------
    pd.DataFrame: 北向资金历史
    """
    cache_key = f"north_flow_hist_{symbol}"
    cached = _read_cache(cache_key, 3600)
    if cached is not None:
        return cached

    try:
        import akshare as ak
        _rate_limit()
        df = ak.stock_hsgt_hist_em(symbol=symbol)
        if df is None or df.empty:
            return pd.DataFrame()

        _write_cache(cache_key, df)
        return df
    except Exception as e:
        logger.debug(f"北向资金历史获取失败: {e}")
        return pd.DataFrame()


def get_industry_fund_flow() -> pd.DataFrame:
    """获取行业资金流向。"""
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


def get_market_fund_flow() -> pd.DataFrame:
    """获取市场整体资金流向（主力/超大单/大单/中单/小单）。"""
    cache_key = "market_fund_flow"
    cached = _read_cache(cache_key, 3600)
    if cached is not None:
        return cached

    try:
        import akshare as ak
        _rate_limit()
        df = ak.stock_market_fund_flow()
        if df is None or df.empty:
            return pd.DataFrame()

        _write_cache(cache_key, df)
        return df
    except Exception as e:
        logger.debug(f"市场资金流向获取失败: {e}")
        return pd.DataFrame()


def get_zt_pool(date: str = None) -> pd.DataFrame:
    """获取涨停池数据。

    Parameters
    ----------
    date : 日期字符串，如 '20260508'，默认今天
    """
    if date is None:
        date = datetime.now().strftime("%Y%m%d")

    cache_key = f"zt_pool_{date}"
    cached = _read_cache(cache_key, 86400)
    if cached is not None:
        return cached

    try:
        import akshare as ak
        _rate_limit()
        df = ak.stock_zt_pool_em(date=date)
        if df is None or df.empty:
            return pd.DataFrame()

        _write_cache(cache_key, df)
        return df
    except Exception as e:
        logger.debug(f"涨停池获取失败: {e}")
        return pd.DataFrame()


def get_industry_pe_data(date: str = None) -> pd.DataFrame:
    """获取行业PE数据。"""
    if date is None:
        date = datetime.now().strftime("%Y%m%d")

    cache_key = f"industry_pe_data_{date}"
    cached = _read_cache(cache_key, 86400 * 3)
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


def get_index_pe_data(symbol: str = "沪深300") -> pd.DataFrame:
    """获取指数PE历史。"""
    cache_key = f"index_pe_data_{symbol}"
    cached = _read_cache(cache_key, 86400)
    if cached is not None:
        return cached

    try:
        import akshare as ak
        _rate_limit()
        df = ak.stock_index_pe_lg(symbol=symbol)
        if df is None or df.empty:
            return pd.DataFrame()

        _write_cache(cache_key, df)
        return df
    except Exception as e:
        logger.debug(f"指数PE获取失败: {e}")
        return pd.DataFrame()


def get_a_share_pb_data() -> pd.DataFrame:
    """获取A股整体PB及分位。"""
    cache_key = "a_share_pb_data"
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


# ── 辅助 ──────────────────────────────────────────────
def sector_summary_for_prompt(sector_name: str, hist: pd.DataFrame) -> str:
    """将板块 K 线摘要为可读文本。"""
    if hist.empty:
        return f"【{sector_name}】无数据"

    latest = hist.iloc[-1]
    lines = [
        f"【{sector_name}】",
        f"  开盘: {latest.get('open', 'N/A')}",
        f"  收盘: {latest.get('close', 'N/A')}",
        f"  最高: {latest.get('high', 'N/A')}",
        f"  最低: {latest.get('low', 'N/A')}",
        f"  涨跌幅: {latest.get('pct_chg', 'N/A')}%",
        f"  成交额: {latest.get('amount', 'N/A')}",
        f"  换手率: {latest.get('turnover_rate', 'N/A')}%",
    ]

    if len(hist) >= 5:
        pct_col = hist.get("pct_chg", pd.Series(dtype=float))
        if not pct_col.empty:
            lines.append(f"  近{len(hist)}日均值涨跌幅: {pct_col.mean():.2f}%")
            lines.append(f"  近{len(hist)}日最大涨幅: {pct_col.max():.2f}%")
            lines.append(f"  近{len(hist)}日最大跌幅: {pct_col.min():.2f}%")

    return "\n".join(lines)
