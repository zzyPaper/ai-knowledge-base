"""交易日历工具 - 借鉴QLib的时间处理思路，针对A股基金简化"""
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import pickle


# A股交易日数据（可通过akshare实时更新）
_TRADING_CALENDAR = None


def load_trading_calendar(cal_path: str = None) -> pd.DatetimeIndex:
    """加载交易日历，优先从本地缓存读取

    借鉴QLib的CalendarProvider思路，但更轻量
    """
    global _TRADING_CALENDAR
    if _TRADING_CALENDAR is not None:
        return _TRADING_CALENDAR

    if cal_path and Path(cal_path).exists():
        with open(cal_path, "rb") as f:
            _TRADING_CALENDAR = pickle.load(f)
        return _TRADING_CALENDAR

    # fallback: 用完整的A股交易日数据
    # QLib中交易日基于历史数据生成，这里简化用akshare获取
    try:
        import akshare as ak
        calendar = ak.tool_trade_date_hist_sina()
        calendar = pd.DatetimeIndex(pd.to_datetime(calendar["trade_date"]).sort_values())
        _TRADING_CALENDAR = calendar
        if cal_path:
            Path(cal_path).parent.mkdir(parents=True, exist_ok=True)
            with open(cal_path, "wb") as f:
                pickle.dump(_TRADING_CALENDAR, f)
        return _TRADING_CALENDAR
    except ImportError:
        # 没有akshare，用pandas生成简单的日期范围（仅用于开发和测试）
        dates = pd.bdate_range(start="2005-01-01", end=datetime.now() + timedelta(days=365),
                                freq="B")  # 仅周末，不含节假日
        _TRADING_CALENDAR = dates
        return _TRADING_CALENDAR


def get_trading_dates(start: str, end: str = None) -> pd.DatetimeIndex:
    """获取[start, end]区间内的交易日（闭区间）"""
    cal = load_trading_calendar()
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")
    mask = (cal >= pd.Timestamp(start)) & (cal <= pd.Timestamp(end))
    return cal[mask]


def get_pre_trade_date(date: str = None) -> pd.Timestamp:
    """获取上一个交易日（QLib相同功能简化版）"""
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    cal = load_trading_calendar()
    idx = cal.get_indexer([pd.Timestamp(date)], method="ffill")
    if idx[0] > 0:
        return cal[idx[0] - 1]
    return cal[0]


def get_next_trade_date(date: str = None) -> pd.Timestamp:
    """获取下一个交易日"""
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    cal = load_trading_calendar()
    idx = cal.get_indexer([pd.Timestamp(date)], method="bfill")
    if idx[0] < len(cal) - 1:
        return cal[idx[0] + 1]
    return cal[-1]


def is_trade_date(date: str = None) -> bool:
    """判断是否为交易日"""
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    cal = load_trading_calendar()
    return pd.Timestamp(date) in cal


def get_last_trade_time() -> pd.Timestamp:
    """返回今天尾盘时间（14:50），用于判断是否接近收盘"""
    now = datetime.now()
    return pd.Timestamp(now.year, now.month, now.day, 14, 50)


def get_holding_days(entry_date: str, current_date: str) -> int:
    """计算两个交易日之间的持仓天数（自然日）"""
    return (pd.Timestamp(current_date) - pd.Timestamp(entry_date)).days


def is_holding_sufficient(entry_date: str, current_date: str, min_days: int = 7) -> bool:
    """检查是否持有足够天数（>=7天），少于7天有1.5%惩罚性赎回费"""
    return get_holding_days(entry_date, current_date) >= min_days


def get_week_ago(date: str = None) -> str:
    """获取7天前的日期"""
    if date is None:
        d = datetime.now()
    else:
        d = pd.Timestamp(date)
    return (d - timedelta(days=7)).strftime("%Y-%m-%d")
