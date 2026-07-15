"""数据获取层 - 从公开API获取基金数据

支持的源:
1. akshare (免费, 推荐)
2. tushare (需要token)
3. 本地CSV缓存
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class FundDataFetcher:
    """基金数据获取器，从akshare等源拉取数据"""

    def __init__(self, source: str = "akshare", data_dir: str = "data/cache"):
        self.source = source
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # ============ 基金列表 ============

    def get_fund_list(self, refresh: bool = False) -> pd.DataFrame:
        """获取公募基金列表（包含ETF、LOF、场外基金）

        Returns:
            DataFrame: [code, name, type, mgmt_fee, cust_fee, ...]
        """
        cache_file = self.data_dir / "fund_list.parquet"
        if cache_file.exists() and not refresh:
            df = pd.read_parquet(cache_file)
            logger.info(f"从缓存加载基金列表: {len(df)} 只")
            return df

        try:
            import akshare as ak
            # 获取所有公募基金基本信息
            df = ak.fund_name_em()  # 东方财富基金列表
            df = df.rename(columns={
                "基金代码": "code",
                "基金简称": "name",
                "基金类型": "fund_type",
                "基金管理人": "manager",
                "成立日期": "establish_date",
                "基金规模": "scale",
            })
            # 过滤出我们关注的类型
            keep_types = ["ETF-场内", "LOF", "股票型", "混合型", "指数型"]
            mask = df["fund_type"].apply(lambda x: any(kt in str(x) for kt in keep_types))

            # 保存缓存
            df.to_parquet(cache_file)
            logger.info(f"从akshare获取基金列表: {len(df)} 只")
            return df
        except ImportError:
            logger.warning("akshare未安装，尝试从本地加载")
            if cache_file.exists():
                return pd.read_parquet(cache_file)
            raise
        except Exception as e:
            logger.error(f"获取基金列表失败: {e}")
            if cache_file.exists():
                return pd.read_parquet(cache_file)
            raise

    # ============ 基金净值数据 ============

    def get_fund_nav(self, code: str, start: str = None, end: str = None,
                     refresh: bool = False) -> pd.DataFrame:
        """获取单只基金的历史净值

        从akshare获取，缓存到本地

        Args:
            code: 基金代码
            start: 起始日期 YYYY-MM-DD
            end: 结束日期 YYYY-MM-DD
            refresh: 是否强制刷新缓存

        Returns:
            DataFrame: [date, nav, acc_nav] 净值/累计净值
        """
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")
        if start is None:
            start = "2020-01-01"

        cache_file = self.data_dir / f"nav_{code}.parquet"

        # 尝试从缓存读取
        if cache_file.exists() and not refresh:
            df = pd.read_parquet(cache_file)
            df["date"] = pd.to_datetime(df["date"])
            # 如果缓存够新且覆盖时间范围
            if df["date"].max() >= pd.Timestamp(end) - timedelta(days=5):
                mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
                return df[mask].reset_index(drop=True)

        try:
            import akshare as ak
            df = ak.fund_open_fund_info_em(symbol=code, indicator="单位净值走势")
            df = df.rename(columns={
                "净值日期": "date",
                "单位净值": "nav",
                "累计净值": "acc_nav",
                "日增长率": "daily_return",
            })
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)

            # 更新本地缓存
            if cache_file.exists():
                old_df = pd.read_parquet(cache_file)
                old_df["date"] = pd.to_datetime(old_df["date"])
                combined = pd.concat([old_df, df]).drop_duplicates(subset=["date"]).sort_values("date")
                combined.to_parquet(cache_file)
            else:
                df.to_parquet(cache_file)

            mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
            result = df[mask].reset_index(drop=True)
            logger.info(f"获取基金 {code} 净值: {len(result)} 条")
            return result

        except ImportError:
            logger.warning("akshare未安装，请 pip install akshare")
            raise
        except Exception as e:
            logger.error(f"获取基金 {code} 净值失败: {e}")
            if cache_file.exists():
                df = pd.read_parquet(cache_file)
                df["date"] = pd.to_datetime(df["date"])
                mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
                return df[mask].reset_index(drop=True)
            raise

    def get_multi_fund_nav(self, codes: List[str], start: str = None,
                           end: str = None) -> Dict[str, pd.DataFrame]:
        """批量获取多只基金净值"""
        result = {}
        for code in codes:
            try:
                result[code] = self.get_fund_nav(code, start, end)
            except Exception as e:
                logger.warning(f"跳过基金 {code}: {e}")
        return result

    # ============ ETF实时行情 ============

    def get_etf_realtime(self) -> pd.DataFrame:
        """获取ETF实时行情（用于尾盘决策）"""
        try:
            import akshare as ak
            df = ak.fund_etf_spot_em()
            df = df.rename(columns={
                "代码": "code",
                "名称": "name",
                "最新价": "price",
                "涨跌幅": "change_pct",
                "成交量": "volume",
                "成交额": "amount",
                "最高": "high",
                "最低": "low",
                "开盘价": "open",
                "昨收": "pre_close",
            })
            return df
        except ImportError:
            raise
        except Exception as e:
            logger.error(f"获取ETF实时行情失败: {e}")
            raise

    # ============ 指数行情 ============

    def get_index_data(self, index_code: str = "sh000300",
                       start: str = None, end: str = None) -> pd.DataFrame:
        """获取指数历史行情（用于基准比较）"""
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")
        if start is None:
            start = "2020-01-01"

        cache_file = self.data_dir / f"index_{index_code}.parquet"
        if cache_file.exists():
            df = pd.read_parquet(cache_file)
            df["date"] = pd.to_datetime(df["date"])
            mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
            return df[mask].reset_index(drop=True)

        try:
            import akshare as ak
            df = ak.stock_zh_index_daily(symbol=index_code)
            df = df.rename(columns={"date": "date", "close": "close"})
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")
            df.to_parquet(cache_file)
            mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
            return df[mask].reset_index(drop=True)
        except Exception as e:
            logger.error(f"获取指数 {index_code} 数据失败: {e}")
            # 尝试备用方案
            try:
                import akshare as ak
                df = ak.index_zh_a_hist(symbol=index_code.replace("sh", "").replace("sz", ""))
                df["date"] = pd.to_datetime(df["日期"])
                df = df.rename(columns={"收盘": "close"})
                df = df.sort_values("date")
                return df
            except:
                raise

    def get_index_realtime(self, index_code: str = "sh000300") -> dict:
        """获取指数实时行情"""
        try:
            import akshare as ak
            df = ak.stock_zh_index_spot_em()
            idx_map = {
                "sh000001": "上证指数",
                "sh000300": "沪深300",
                "sh000016": "上证50",
                "sh000688": "科创50",
                "sz399006": "创业板指",
            }
            name = idx_map.get(index_code, "")
            if name:
                row = df[df["名称"] == name]
                if not row.empty:
                    return {
                        "name": name,
                        "price": row["最新价"].values[0],
                        "change_pct": row["涨跌幅"].values[0],
                    }
            return {}
        except Exception as e:
            logger.error(f"获取指数实时行情失败: {e}")
            return {}
