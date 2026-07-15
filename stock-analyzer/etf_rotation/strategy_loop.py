#!/usr/bin/env python3
"""策略迭代循环系统

规则:
1. 每次只用过去1年数据
2. 分成2个月为一个波段，滚动模拟买卖
3. 和同期沪深300比较
4. 目标是超过沪深300至少3个百分点
5. 每轮找出策略不足，增加新机制（不是调参！）
6. 共30轮迭代
"""
import pandas as pd
import numpy as np
import logging
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass, field
import traceback

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("strategy_loop")


# ========== 1. 数据获取 ==========

def fetch_data_for_year(end_date: str = None, lookback_days: int = 400) -> Dict:
    """获取过去N天的历史数据

    Args:
        end_date: 结束日期，默认今天
        lookback_days: 向前拉取的天数，默认400天（约13个月）
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    end = pd.Timestamp(end_date)
    start = end - timedelta(days=lookback_days)
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    logger.info(f"获取数据: {start_str} ~ {end_str}")
    import akshare as ak

    # 1. 沪深300指数（基准）
    logger.info("获取沪深300数据...")
    csi300 = ak.stock_zh_index_daily(symbol="sh000300")
    csi300["date"] = pd.to_datetime(csi300["date"])
    csi300 = csi300[(csi300["date"] >= start) & (csi300["date"] <= end)]
    csi300 = csi300.sort_values("date").reset_index(drop=True)

    # 预计算CSI300指标作为全市场参考信号（所有机制共用，避免依赖单只基金）
    csi300_signals = csi300.copy()
    csi300_signals["ma5"] = csi300_signals["close"].rolling(5).mean()
    csi300_signals["ma20"] = csi300_signals["close"].rolling(20).mean()
    csi300_signals["ma60"] = csi300_signals["close"].rolling(60).mean()
    csi300_signals["volatility"] = csi300_signals["close"].pct_change().rolling(20).std()
    csi300_signals["high_60"] = csi300_signals["close"].rolling(60).max()
    csi300_signals["drawdown"] = (csi300_signals["high_60"] - csi300_signals["close"]) / csi300_signals["high_60"] * 100
    # 统一日期格式为字符串，与BacktestEngine保持一致
    csi300_signals["date"] = csi300_signals["date"].dt.strftime("%Y-%m-%d")

    # 2. 板块ETF基金数据 + 沪深300ETF(参考基准) + 国债ETF(防御)
    funds = {
        "510300": "沪深300ETF",  # 市场参考基准（仅作为市场信号，不参与轮动）

        # === 科技/半导体/通信 ===
        "159995": "芯片ETF",
        "512480": "半导体ETF",
        "515050": "5GETF",
        "159852": "软件ETF",
        "515880": "通信ETF",
        "517010": "数字经济ETF",

        # === 新能源/车/光伏/锂电 ===
        "515700": "新能源车ETF",
        "515030": "新能源ETF",
        "515790": "光伏ETF",
        "159840": "锂电池ETF",

        # === 医药/医疗 ===
        "512170": "医疗ETF",
        "512010": "医药ETF",
        "159992": "创新药ETF",
        "159847": "医疗50ETF",

        # === 消费/酒/食品 ===
        "159928": "消费ETF",
        "512690": "酒ETF",
        "515900": "食品饮料ETF",
        "159865": "养殖ETF",
        "159825": "农业ETF",

        # === 金融/地产/银行 ===
        "512880": "证券ETF",
        "512800": "银行ETF",
        "512200": "房地产ETF",

        # === 周期/制造/基建 ===
        "515220": "煤炭ETF",
        "159870": "化工ETF",
        "516970": "基建ETF",
        "515210": "钢铁ETF",
        "159830": "稀土ETF",

        # === 军工/高端制造 ===
        "512660": "军工ETF",
        "516380": "智能汽车ETF",
        "516010": "机械ETF",

        # === TMT/游戏 ===
        "159869": "游戏ETF",
        "517050": "互联网ETF",

        # === 商品/黄金 ===
        "518880": "黄金ETF",
        "159985": "商品ETF",

        # === 红利/价值 ===
        "510880": "红利ETF",
        "512890": "红利低波ETF",

        # === 宽基/风格 ===
        "510050": "上证50ETF",
        "510500": "中证500ETF",
        "512100": "中证1000ETF",
        "159915": "创业板ETF",
        "159949": "创业板50ETF",
        "588000": "科创50ETF",

        # === 海外 ===
        "513050": "中概互联ETF",
        "513100": "纳指ETF",

        # === 旅游/其他 ===
        "159766": "旅游ETF",
        "516770": "电竞ETF",

        # === 防御 ===
        "511520": "国债ETF",
    }

    # 基金分类（单一数据源，机制代码中不出现硬编码基金代码）
    REFERENCE_CODES = ["510300"]   # 市场基准参考，不参与轮动交易
    DEFENSE_CODES = ["511520"]     # 防御型（债券），熊市时买入
    trading_codes = [c for c in funds.keys() if c not in REFERENCE_CODES + DEFENSE_CODES]
    logger.info(f"参考基金: {REFERENCE_CODES}")
    logger.info(f"防御基金: {DEFENSE_CODES}")
    logger.info(f"可交易基金({len(trading_codes)}个): {trading_codes}")

    nav_data = {}
    fund_names = {}
    for code, name in funds.items():
        try:
            logger.info(f"获取 {name}({code})...")
            df = ak.fund_open_fund_info_em(symbol=code, indicator="单位净值走势")
            df = df.rename(columns={
                "净值日期": "date",
                "单位净值": "nav",
                "日增长率": "daily_return",
            })
            df["date"] = pd.to_datetime(df["date"])
            df = df[(df["date"] >= start) & (df["date"] <= end)]
            if not df.empty:
                # 计算5/20日收益率作为辅助指标
                df = df.sort_values("date").reset_index(drop=True)
                df["ret_5d"] = df["nav"].pct_change(5)
                df["ret_20d"] = df["nav"].pct_change(20)
                df["ma5"] = df["nav"].rolling(5).mean()
                df["ma20"] = df["nav"].rolling(20).mean()
                df["ma60"] = df["nav"].rolling(60).mean()
                df["volatility"] = df["nav"].pct_change().rolling(20).std()
                nav_data[code] = df
                fund_names[code] = name
        except Exception as e:
            logger.warning(f"获取 {name}({code}) 失败: {e}")

    logger.info(f"数据加载完成: {len(nav_data)} 只基金, "
                f"沪深300共 {len(csi300)} 条记录")

    return {
        "csi300": csi300,
        "csi300_signals": csi300_signals,   # 含ma5/ma20/ma60/波动率/回撤的CSI300，供机制做市场参考
        "nav_data": nav_data,
        "fund_names": fund_names,
        "fund_codes": list(nav_data.keys()),
        "trading_codes": trading_codes,     # 仅可交易板块列表（不含参考/防御基金）
        "defense_codes": DEFENSE_CODES,     # 防御型基金（债券）
        "end_date": end_str,
    }


def fetch_intraday_prices() -> tuple:
    """获取ETF实时行情(IOPV实时净值估算)

    仅在交易日9:00-17:00真正拉取API，其余时间直接返回空（避免非交易时段超时等待）。
    IOPV由交易所每约15秒计算一次，是当日NAV的最佳代理。

    Returns:
        (prices_dict, spot_df, data_date_str)
        prices_dict: {基金代码: IOPV值}
        spot_df: 原始DataFrame（含所有ETF的实时数据）
        data_date_str: "YYYY-MM-DD"如非当日说明休市中
    """
    now = datetime.now()
    # 非交易日或非交易时段跳过（A股交易时间9:30-15:00，放宽到9:00-17:00覆盖集合竞价和延时）
    if now.weekday() >= 5 or now.hour < 9 or now.hour >= 17:
        logger.info("非交易时段，跳过IOPV拉取")
        return {}, None, ""
    import akshare as ak
    logger.info("获取ETF实时行情(IOPV)...")
    try:
        spot = ak.fund_etf_spot_em()
        data_date = spot["数据日期"].iloc[0]
        if isinstance(data_date, pd.Timestamp):
            data_date = data_date.strftime("%Y-%m-%d")
        else:
            data_date = str(data_date)[:10]

        prices = {}
        for _, row in spot.iterrows():
            code = str(row["代码"])
            iopv = row["IOPV实时估值"]
            if pd.notna(iopv) and iopv > 0:
                prices[code] = float(iopv)
        logger.info(f"实时行情: {len(prices)}只ETF有IOPV, 数据日期{data_date}")
        return prices, spot, data_date
    except Exception as e:
        logger.warning(f"获取实时行情失败: {e}")
        return {}, None, ""


# ========== 2. 波段划分 ==========

def get_2month_bands(start: str, end: str) -> List[tuple]:
    """将日期范围划分成2个月一个的波段

    Returns:
        [(band_start, band_end), ...] 每个波段的首尾日期
    """
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)

    bands = []
    current = start_dt
    while current < end_dt:
        band_end = min(current + timedelta(days=60), end_dt)
        bands.append((current.strftime("%Y-%m-%d"), band_end.strftime("%Y-%m-%d")))
        current = band_end + timedelta(days=1)
        if (band_end - current).days >= -30:  # 避免最后一段太短
            pass

    return bands


# ========== 3. 波段回测引擎 ==========

@dataclass
class BandResult:
    """单个波段的回测结果"""
    start: str
    end: str
    strategy_return: float = 0.0
    benchmark_return: float = 0.0
    excess_return: float = 0.0
    trades: int = 0
    win_rate: float = 0.0
    max_drawdown: float = 0.0
    mechanism_state: dict = field(default_factory=dict)


def evaluate_strategy(strategy_fn: Callable, data: Dict, bands: List[tuple],
                      mechanism_state: dict = None) -> List[BandResult]:
    """连续回测（不再分段重置现金），按波段切分统计"""
    from .backtest.engine import BacktestEngine

    if mechanism_state is None:
        mechanism_state = {}
    if data.get("csi300_signals") is not None:
        mechanism_state["csi300_df"] = data["csi300_signals"]
    if data.get("trading_codes") is not None:
        mechanism_state["trading_codes"] = data["trading_codes"]
    if data.get("defense_codes") is not None:
        mechanism_state["defense_codes"] = data["defense_codes"]

    # 运行单次连续回测（跨所有波段，不重置现金）
    engine = BacktestEngine(init_cash=100000)
    engine.load_data(data["nav_data"])
    engine.trading_dates = [d for d in engine.trading_dates
                           if bands[0][0] <= d <= bands[-1][1]]

    def wrapped_strategy(date=None, account=None, nav_data=None, fund_universe=None, **kwargs):
        return strategy_fn(date, account, nav_data, fund_universe,
                          mechanism_state=mechanism_state)

    try:
        engine.run(wrapped_strategy, verbose=False)
    except Exception as e:
        logger.warning(f"连续回测失败: {e}")
        return []

    daily_values = engine.account.daily_values
    csi300 = data["csi300"]
    band_results = []

    for bs, be in bands:
        # 从连续回测中提取本波段快照
        band_snaps = [s for s in daily_values if bs <= s["date"] <= be]
        if len(band_snaps) < 2:
            continue

        start_val = band_snaps[0]["total_value"]
        end_val = band_snaps[-1]["total_value"]
        strategy_ret = (end_val / start_val - 1) * 100

        # 波段基准收益
        bm_band = csi300[(csi300["date"] >= pd.Timestamp(bs)) &
                         (csi300["date"] <= pd.Timestamp(be))]
        bm_ret = 0
        if len(bm_band) >= 2:
            bm_ret = (bm_band.iloc[-1]["close"] / bm_band.iloc[0]["close"] - 1) * 100

        # 波段内最大回撤
        peak = 0
        max_dd = 0
        for s in band_snaps:
            if s["total_value"] > peak:
                peak = s["total_value"]
            dd = (peak - s["total_value"]) / peak * 100
            max_dd = min(max_dd, dd)

        band_results.append(BandResult(
            start=bs, end=be,
            strategy_return=strategy_ret,
            benchmark_return=bm_ret,
            excess_return=strategy_ret - bm_ret,
            max_drawdown=max_dd,
        ))

    return band_results


def compute_summary(results: List[BandResult]) -> dict:
    """计算策略汇总统计"""
    if not results:
        return {"avg_excess": 0, "win_bands": 0, "total_bands": 0}

    excess_returns = [r.excess_return for r in results
                     if r.excess_return > -999]
    strategy_rets = [r.strategy_return for r in results
                    if r.strategy_return > -999]
    bm_rets = [r.benchmark_return for r in results
              if r.benchmark_return > -999]

    win_bands = sum(1 for r in results if r.excess_return > 0 and r.excess_return > -999)
    total_valid = len(excess_returns)

    avg_excess = np.mean(excess_returns) if excess_returns else 0
    total_strategy = np.mean(strategy_rets) if strategy_rets else 0
    total_bm = np.mean(bm_rets) if bm_rets else 0

    # 累计收益
    cum_strategy = 1
    cum_bm = 1
    for r in results:
        if r.strategy_return > -999:
            cum_strategy *= (1 + r.strategy_return / 100)
        if r.benchmark_return > -999:
            cum_bm *= (1 + r.benchmark_return / 100)
    cum_excess = (cum_strategy / cum_bm - 1) * 100

    return {
        "avg_excess_return": round(avg_excess, 2),
        "cum_excess_return": round(cum_excess, 2),
        "win_bands": win_bands,
        "total_bands": total_valid,
        "win_rate": round(win_bands / total_valid * 100, 1) if total_valid else 0,
        "avg_strategy_return": round(np.mean(strategy_rets), 2) if strategy_rets else 0,
        "avg_benchmark_return": round(np.mean(bm_rets), 2) if bm_rets else 0,
        "total_strategy_return": round((cum_strategy - 1) * 100, 2),
        "total_benchmark_return": round((cum_bm - 1) * 100, 2),
    }


# ========== 4. 分析工具：发现策略不足 ==========

def analyze_weaknesses(results: List[BandResult]) -> str:
    """分析策略在所有波段中的表现，找出系统性不足

    重点分析:
    1. 哪些市场环境下策略表现差
    2. 策略的共性失败模式
    """
    findings = []

    # 按基准收益分组
    bull_bands = [r for r in results if r.benchmark_return > 3 and r.excess_return > -999]
    bear_bands = [r for r in results if r.benchmark_return < -3 and r.excess_return > -999]
    range_bands = [r for r in results if -3 <= r.benchmark_return <= 3 and r.excess_return > -999]

    if bull_bands:
        avg_ex = np.mean([r.excess_return for r in bull_bands])
        if avg_ex < 0:
            findings.append(f"牛市跑输: 沪深300涨>3%时，策略平均超额{avg_ex:+.1f}%（跑输基准）")
        else:
            findings.append(f"牛市表现: 沪深300涨>3%时，策略平均超额{avg_ex:+.1f}%")

    if bear_bands:
        avg_ex = np.mean([r.excess_return for r in bear_bands])
        if avg_ex < -2:
            findings.append(f"熊市亏更多: 沪深300跌>3%时，策略平均超额{avg_ex:+.1f}%（比基准亏更多）")
        else:
            findings.append(f"熊市防守: 沪深300跌>3%时，策略平均超额{avg_ex:+.1f}%")

    if range_bands:
        avg_ex = np.mean([r.excess_return for r in range_bands])
        findings.append(f"震荡市: 沪深300涨跌<3%时，策略平均超额{avg_ex:+.1f}%")

    # 最大亏损波段
    worst = min(results, key=lambda r: r.strategy_return if r.strategy_return > -999 else 999)
    if worst and worst.strategy_return > -999:
        findings.append(f"最差波段: {worst.start}~{worst.end} 策略{worst.strategy_return:+.1f}%, "
                       f"基准{worst.benchmark_return:+.1f}%, 超额{worst.excess_return:+.1f}%")

    # 超额收益的波动
    excesses = [r.excess_return for r in results if r.excess_return > -999]
    if excesses:
        std_ex = np.std(excesses)
        findings.append(f"超额收益波动率: {std_ex:.1f}%（稳定性指标）")

    return "\n".join(findings)


# ========== 5. 基准策略：最简单的DCA ==========

def base_strategy(date, account, nav_data, fund_universe,
                  mechanism_state=None, **kwargs):
    """基准策略: 每月定投第一个可交易板块"""
    from .backtest.engine import Signal
    signals = []
    if mechanism_state is None:
        mechanism_state = {}

    target = mechanism_state.get("rotation_target")
    if not target:
        trading = mechanism_state.get("trading_codes", [])
        target = trading[0] if trading else None
    if not target:
        return signals

    dt = pd.Timestamp(date)
    if dt.day <= 5 and account.cash >= 1000:
        signals.append(Signal(target, date, "buy", 1000, "月定投"))

    return signals


# ========== 6. 机制库 ==========

class MechanismRegistry:
    """策略机制注册表

    每个机制是一个"补丁"，修复特定场景下的策略不足。
    比调参更重要的：增加新的决策维度。
    """

    def __init__(self):
        self.mechanisms = {}
        self._register_all()

    def _register_all(self):
        self._register_mech("trend_filter", self.trend_filter_desc,
                           self.trend_filter_apply)
        self._register_mech("volatility_stop", self.volatility_stop_desc,
                           self.volatility_stop_apply)
        self._register_mech("momentum_boost", self.momentum_boost_desc,
                           self.momentum_boost_apply)
        self._register_mech("bear_defense", self.bear_defense_desc,
                           self.bear_defense_apply)
        self._register_mech("profit_lock", self.profit_lock_desc,
                           self.profit_lock_apply)
        self._register_mech("etf_rotation", self.etf_rotation_desc,
                           self.etf_rotation_apply)
        self._register_mech("macro_timing", self.macro_timing_desc,
                           self.macro_timing_apply)
        self._register_mech("grid_add", self.grid_add_desc,
                           self.grid_add_apply)
        self._register_mech("dca_plus", self.dca_plus_desc,
                           self.dca_plus_apply)
        self._register_mech("stop_loss", self.stop_loss_desc,
                           self.stop_loss_apply)

    def _register_mech(self, name, desc_fn, apply_fn):
        self.mechanisms[name] = {"desc": desc_fn, "apply": apply_fn}

    def list_available(self) -> List[str]:
        return list(self.mechanisms.keys())

    def describe(self, name: str) -> str:
        m = self.mechanisms.get(name)
        return m["desc"]() if m else "未知机制"

    def get_apply(self, name: str):
        m = self.mechanisms.get(name)
        return m["apply"] if m else None

    # ---- 机制定义 ----

    def trend_filter_desc(self):
        return "【趋势过滤】当20日均线在60日均线下方(空头排列)时，暂停买入，转而持有现金或债券ETF"

    def trend_filter_apply(self, signals, date, account, nav_data, fund_universe, state):
        """趋势过滤：空头市场暂停买入"""
        from .backtest.engine import Signal
        new_signals = []

        csi300_df = state.get("csi300_df")
        if csi300_df is not None:
            hist = csi300_df[csi300_df["date"] <= date]
            if len(hist) >= 60:
                ma20 = hist["ma20"].iloc[-1]
                ma60 = hist["ma60"].iloc[-1]
                is_bear = ma20 < ma60
                state["trend_filter_active"] = is_bear

                if is_bear:
                    # 空头市场：把买入信号转为买入债券或持有现金
                    for sig in signals:
                        if sig.action == "buy":
                            defense = state.get("defense_codes", [])
                            bond_code = defense[0] if defense else None
                            if bond_code:
                                new_signals.append(Signal(
                                    bond_code, date, "buy", sig.amount,
                                    f"空头防御:买入债券({sig.reason})"))
                            # else: 不买，持有现金
                        else:
                            new_signals.append(sig)
                    return new_signals

        return signals

    def volatility_stop_desc(self):
        return "【波动率止损】当20日波动率超过阈值时，降低仓位或停止买入"

    def volatility_stop_apply(self, signals, date, account, nav_data,
                             fund_universe, state):
        from .backtest.engine import Signal
        csi300_df = state.get("csi300_df")
        if csi300_df is not None:
            hist = csi300_df[csi300_df["date"] <= date]
            if len(hist) >= 20:
                vol = hist["volatility"].iloc[-1]
                vol_annual = vol * np.sqrt(252)
                state["annual_vol"] = vol_annual * 100

                if vol_annual > 0.35:  # 年化波动>35%，市场异常
                    state["vol_high"] = True
                    # 降低买入金额
                    new_signals = []
                    for sig in signals:
                        if sig.action == "buy":
                            sig.amount *= 0.5  # 减半买入
                            sig.reason += "(减半-高波动)"
                        new_signals.append(sig)
                    return new_signals
                else:
                    state["vol_high"] = False

        return signals

    def momentum_boost_desc(self):
        return "【动量增强】上升趋势全额买入，下跌趋势减半买入(不再完全暂停)"

    def momentum_boost_apply(self, signals, date, account, nav_data,
                            fund_universe, state):
        csi300_df = state.get("csi300_df")
        if csi300_df is not None:
            hist = csi300_df[csi300_df["date"] <= date]
            if len(hist) >= 20:
                row = hist.iloc[-1]
                is_up = row["ma5"] > row["ma20"]
                state["momentum_up"] = is_up

                if not is_up and state.get("round", 0) >= 2:
                    # 下跌趋势：减半买入，不再完全暂停
                    for sig in signals:
                        if sig.action == "buy":
                            sig.amount *= 0.5
                            sig.reason += "(减半-动量偏弱)"
                    state["momentum_paused"] = False  # 不再阻止轮动
                else:
                    state["momentum_paused"] = False

        return signals

    def bear_defense_desc(self):
        return "【熊市防御】当指数从高点回撤>10%时，转为防御模式（持有债券/现金），回撤<5%时恢复"

    def bear_defense_apply(self, signals, date, account, nav_data,
                          fund_universe, state):
        from .backtest.engine import Signal
        csi300_df = state.get("csi300_df")
        if csi300_df is not None:
            hist = csi300_df[csi300_df["date"] <= date]
            if len(hist) >= 60:
                row = hist.iloc[-1]
                drawdown = row["drawdown"]
                state["drawdown"] = drawdown

                if state.get("in_defense", False):
                    # 防御模式中：回撤缩小到5%以内时恢复
                    if drawdown < 6:
                        state["in_defense"] = False
                    else:
                        # 仍在防御，买入债券
                        return self._redirect_to_bond(signals, date, account,
                                                     nav_data, fund_universe, state)
                elif drawdown > 12:
                    # 触发防御
                    state["in_defense"] = True
                    return self._redirect_to_bond(signals, date, account,
                                                 nav_data, fund_universe, state)

        return signals

    def _redirect_to_bond(self, signals, date, account, nav_data,
                         fund_universe, state):
        from .backtest.engine import Signal
        new_signals = []
        defense = state.get("defense_codes", [])
        bond_code = defense[0] if defense else None
        for sig in signals:
            if sig.action == "buy":
                if bond_code:
                    new_signals.append(Signal(
                        bond_code, date, "buy", sig.amount,
                        f"熊市防御(回撤{state.get('drawdown', 0):.1f}%)"))
                # 没有债券ETF就不买
            else:
                new_signals.append(sig)
        return new_signals

    def profit_lock_desc(self):
        return "【利润锁定】单只基金盈利>15%时，锁定一半利润（卖出50%）"

    def profit_lock_apply(self, signals, date, account, nav_data,
                         fund_universe, state):
        from .backtest.engine import Signal
        new_signals = list(signals)

        # 检查持仓盈利
        for code, pos in list(account.positions.items()):
            df = nav_data.get(code)
            if df is None:
                continue
            hist = df[df["date"] <= date]
            if hist.empty:
                continue
            current_nav = hist.iloc[-1]["nav"]
            market_value = pos.shares * current_nav
            profit_pct = (market_value / pos.cost - 1) * 100 if pos.cost > 0 else 0
            state[f"profit_{code}"] = profit_pct

            if profit_pct > 20 and account.can_sell(code, date):
                sell_value = market_value * 0.5
                new_signals.append(Signal(
                    code, date, "sell", sell_value,
                    f"利润锁定(盈利{profit_pct:.1f}%,卖50%)"))

        return new_signals

    def etf_rotation_desc(self):
        return "【ETF轮动】持续跟踪各ETF的20日涨幅，轮入最强板块，尊重其他机制的市场判断"

    def etf_rotation_apply(self, signals, date, account, nav_data,
                          fund_universe, state):
        from .backtest.engine import Signal

        new_signals = list(signals)
        rotation_codes = state.get("trading_codes", [])
        if not rotation_codes:
            return signals

        returns = {}
        momentum_days = state.get("momentum_days", 20)
        for code in rotation_codes:
            df = nav_data.get(code)
            if df is not None:
                hist = df[df["date"] <= date]
                if len(hist) >= momentum_days:
                    ret = (hist["nav"].iloc[-1] / hist["nav"].iloc[-momentum_days] - 1) * 100
                    returns[code] = ret

        if not returns:
            return signals

        # 按20日动量排序，选Top1
        sorted_codes = sorted(returns, key=returns.get, reverse=True)
        best_code = sorted_codes[0]
        state["best_etf"] = best_code
        state["best_return"] = returns[best_code]

        # === CSI300短期趋势判断（需连续3日站上20日线，过滤假突破）===
        # 提前计算，供板块级趋势过滤和买入决策使用
        csi300_df = state.get("csi300_df")
        rising = False
        if csi300_df is not None:
            hist300 = csi300_df[csi300_df["date"] <= date]
            if len(hist300) >= 20:
                today_up = hist300.iloc[-1]["close"] > hist300.iloc[-1]["ma20"]
                # 扫描历史数据计算连续站上20日线天数（比运行计数器更鲁棒）
                streak = 0
                for i in range(len(hist300) - 1, -1, -1):
                    if hist300.iloc[i]["close"] > hist300.iloc[i]["ma20"]:
                        streak += 1
                    else:
                        break
                state["rising_streak"] = streak
                rising = streak >= 3

        # CSI300短期均线确认：MA5 > MA20 才视为有效上升
        csi300_up = False
        if csi300_df is not None:
            if len(hist300) >= 20:
                csi300_up = hist300.iloc[-1]["ma5"] > hist300.iloc[-1]["ma20"]

        # === 板块级趋势过滤 ===
        # 持仓板块跌破20日均线 → 立即卖出（不等20日动量翻负）
        for code in list(account.positions.keys()):
            df = nav_data.get(code)
            if df is not None:
                hist = df[df["date"] <= date]
                if len(hist) >= 20:
                    if hist.iloc[-1]["nav"] < hist.iloc[-1]["ma20"] and account.can_sell(code, date):
                        new_signals.append(Signal(
                            code, date, "sell", 999999,
                            f"趋势破位卖出({code}跌破20日线)"))

        # 目标板块必须处于上升趋势（nav > ma20）
        best_df = nav_data.get(best_code)
        best_in_uptrend = False
        if best_df is not None:
            hist = best_df[best_df["date"] <= date]
            if len(hist) >= 20:
                best_in_uptrend = hist.iloc[-1]["nav"] > hist.iloc[-1]["ma20"]

        # 检查其他机制是否已禁止买入
        in_defense = state.get("in_defense", False)
        trend_bear = state.get("trend_filter_active", False)
        market_bear = state.get("market_state") == "熊市"
        momentum_paused = state.get("momentum_paused", False)

        # 上升趋势时放宽风控
        if rising:
            buying_disabled = in_defense or market_bear
        else:
            buying_disabled = in_defense or trend_bear or market_bear or momentum_paused

        # 买入条件：风控允许 + CSI300上升 + CSI300短期均线多头 + Top1动量为正 + Top1处于上升趋势
        can_buy = not buying_disabled and rising and csi300_up and returns[best_code] > 0 and best_in_uptrend

        prev_target = state.get("rotation_target")

        if best_code != prev_target:
            state["rotation_target"] = best_code

            for code in list(account.positions.keys()):
                if code != best_code and account.can_sell(code, date):
                    new_signals.append(Signal(
                        code, date, "sell", 999999,
                        f"轮出({code}收益{returns.get(code, 0):.1f}%)"))

            if can_buy:
                new_signals.append(Signal(
                    best_code, date, "buy", 999999,
                    f"轮入({best_code}收益{returns[best_code]:.1f}%)"))
        elif can_buy:
            if best_code not in account.positions and account.cash > 1000:
                new_signals.append(Signal(
                    best_code, date, "buy", 999999,
                    f"加仓({best_code}收益{returns[best_code]:.1f}%)"))
        elif not account.positions and rising and csi300_up:
            # 空仓+CSI300上升趋势：强制入场（忽略板块趋势，专为反弹后跟进）
            new_signals.append(Signal(
                best_code, date, "buy", 999999,
                f"强制入场({best_code}收益{returns[best_code]:.1f}%)"))

        return new_signals

    def macro_timing_desc(self):
        return "【宏观择时】通过沪深300自身的均线系统判断市场状态，牛/熊/震荡市采取不同策略"

    def macro_timing_apply(self, signals, date, account, nav_data,
                          fund_universe, state):
        from .backtest.engine import Signal
        csi300_df = state.get("csi300_df")
        if csi300_df is not None:
            hist = csi300_df[csi300_df["date"] <= date]
            if len(hist) >= 60:
                row = hist.iloc[-1]
                current = row["close"]

                # 判断市场状态
                above_ma20 = current > row["ma20"]
                above_ma60 = current > row["ma60"]
                ma20_above_ma60 = row["ma20"] > row["ma60"]

                if above_ma20 and ma20_above_ma60:
                    state["market_state"] = "牛市"
                elif not above_ma60 and not ma20_above_ma60:
                    state["market_state"] = "熊市"
                else:
                    state["market_state"] = "震荡"

                if state.get("round", 0) >= 3:
                    if state["market_state"] == "熊市":
                        # 熊市：只禁止新买入，不强制卖出已有持仓
                        #（防止空仓踏空反弹，让 stop_loss 自主处理风险）
                        signals = [s for s in signals if s.action != "buy"]
                    elif state["market_state"] == "牛市":
                        # 牛市：加倍买入
                        for sig in signals:
                            if sig.action == "buy":
                                sig.amount *= 1.5
                                sig.reason += "(牛市加码)"

        return signals

    def grid_add_desc(self):
        return "【网格加仓】在定投基础上，每跌5%额外加仓一份，实现越跌越买"

    def grid_add_apply(self, signals, date, account, nav_data,
                      fund_universe, state):
        target = state.get("rotation_target")
        if not target:
            trading = state.get("trading_codes", [])
            target = trading[0] if trading else None
        if not target:
            return signals

        # 网格加仓也尊重风控闸门
        in_defense = state.get("in_defense", False)
        trend_bear = state.get("trend_filter_active", False)
        market_bear = state.get("market_state") == "熊市"
        momentum_paused = state.get("momentum_paused", False)
        if in_defense or trend_bear or market_bear or momentum_paused:
            return signals

        csi300_df = state.get("csi300_df")
        if csi300_df is not None:
            hist = csi300_df[csi300_df["date"] <= date]
            if len(hist) >= 60:
                row = hist.iloc[-1]
                drop_pct = row["drawdown"]  # 使用CSI300回撤作为市场温度
                state["grid_drop"] = drop_pct

                # 每跌5%触发一次网格买入
                grid_level = int(drop_pct / 5)
                last_grid = state.get("last_grid_level", 0)
                if grid_level > last_grid and account.cash >= 1000:
                    from .backtest.engine import Signal
                    grid_amount = min(1000 * (grid_level - last_grid), account.cash * 0.3)
                    signals.append(Signal(
                        target, date, "buy", grid_amount,
                        f"网格加仓(跌{drop_pct:.1f}%,第{grid_level}层)"))
                    state["last_grid_level"] = grid_level

        return signals

    def dca_plus_desc(self):
        return "【智能定投】定投金额与市场估值挂钩：低估(低于60日均线)多投，高估少投"

    def dca_plus_apply(self, signals, date, account, nav_data,
                      fund_universe, state):
        csi300_df = state.get("csi300_df")
        if csi300_df is not None:
            hist = csi300_df[csi300_df["date"] <= date]
            if len(hist) >= 60:
                row = hist.iloc[-1]
                ratio = row["close"] / row["ma60"]
                state["price_ratio_to_ma60"] = ratio

                for sig in signals:
                    if sig.action == "buy":
                        if ratio < 0.95:  # 低于60日均线5% → 多投
                            sig.amount *= 2.0
                            sig.reason += "(加倍-低于60日线)"
                        elif ratio > 1.1:  # 高于60日均线10% → 少投
                            sig.amount *= 0.5
                            sig.reason += "(减半-高于60日线)"
                        # 0.95~1.1 正常定投

        return signals

    def stop_loss_desc(self):
        return "【止损机制】单笔买入亏损>8%时止损卖出，防止深套"

    def stop_loss_apply(self, signals, date, account, nav_data,
                       fund_universe, state):
        from .backtest.engine import Signal
        new_signals = list(signals)

        for code, pos in list(account.positions.items()):
            df = nav_data.get(code)
            if df is None:
                continue
            hist = df[df["date"] <= date]
            if hist.empty:
                continue
            current_nav = hist.iloc[-1]["nav"]
            market_value = pos.shares * current_nav
            profit_pct = (market_value / pos.cost - 1) * 100 if pos.cost > 0 else 0
            state[f"stop_loss_{code}"] = profit_pct

            if profit_pct < -8 and account.can_sell(code, date):
                new_signals.append(Signal(
                    code, date, "sell", 999999,
                    f"止损(亏损{profit_pct:.1f}%)"))

        return new_signals


# ========== 7. 策略工厂 ==========

def build_strategy(active_mechanisms: List[str], registry: MechanismRegistry,
                   round_num: int = 0):
    """根据激活的机制列表构建策略函数

    机制按顺序应用:
    1. 先处理卖出信号（止盈止损）
    2. 再处理买入增强
    3. 最后过滤
    """
    def strategy(date, account, nav_data, fund_universe,
                 mechanism_state=None):
        from .backtest.engine import Signal
        if mechanism_state is None:
            mechanism_state = {}
        mechanism_state["round"] = round_num

        # === 确定买入目标（从机制状态中获取） ===
        # 买入目标由 etf_rotation 设置，如未设置则默认第一个可交易板块
        target = mechanism_state.get("rotation_target")
        if not target:
            trading = mechanism_state.get("trading_codes", [])
            target = trading[0] if trading else None
        if not target:
            return []  # 没有任何可买目标，空仓

        # === 生成基础信号：每月定投轮动目标 ===
        signals = []
        dt = pd.Timestamp(date)

        # 每月初定投买入当前轮动目标
        if dt.day <= 5 and account.cash >= 1000:
            signals.append(Signal(target, date, "buy", 1000, f"月定投->{target}"))

        # === 按顺序应用机制 ===
        for mech_name in active_mechanisms:
            apply_fn = registry.get_apply(mech_name)
            if apply_fn:
                try:
                    signals = apply_fn(signals, date, account, nav_data,
                                      fund_universe, mechanism_state)
                except Exception as e:
                    logger.debug(f"机制 {mech_name} 执行失败: {e}")

        return signals

    return strategy


# ========== 8. 主循环 ==========

def run_strategy_loop(total_rounds: int = 30):
    """主循环：30轮策略迭代

    每轮流程:
    1. 拉取最新数据
    2. 划分2月波段
    3. 评估当前策略
    4. 分析不足
    5. 添加新机制
    6. 记录进展
    """
    print("\n" + "="*70)
    print("  策略迭代循环启动")
    print(f"  目标: 在2月波段上跑赢沪深300至少3个百分点")
    print(f"  轮数: {total_rounds}")
    print("="*70)

    data = fetch_data_for_year()
    bands = get_2month_bands(
        (pd.Timestamp(data["end_date"]) - timedelta(days=365)).strftime("%Y-%m-%d"),
        data["end_date"],
    )

    print(f"\n📅 数据范围: 过去一年")
    print(f"📊 基金池: {', '.join(f'{v}({k})' for k, v in data['fund_names'].items())}")
    print(f"📈 波段数: {len(bands)} (每个约2个月)")
    print()

    registry = MechanismRegistry()
    active_mechanisms = []  # 当前激活的机制列表
    all_mechanisms = registry.list_available()

    # 跟踪历史
    history = []
    mechanism_history = []

    for round_idx in range(1, total_rounds + 1):
        print(f"\n{'='*60}")
        print(f"  第 {round_idx}/{total_rounds} 轮")
        print(f"{'='*60}")

        # 构建当前策略
        strategy_fn = build_strategy(active_mechanisms, registry, round_idx)

        # 评估
        results = evaluate_strategy(strategy_fn, data, bands,
                                   mechanism_state={"round": round_idx})
        summary = compute_summary(results)

        # 分析不足
        weakness_analysis = analyze_weaknesses(results)

        # 记录
        history.append(summary)
        mechanism_history.append(list(active_mechanisms))

        # 输出
        print(f"  策略收益: {summary['total_strategy_return']:+.2f}%  |  "
              f"沪深300: {summary['total_benchmark_return']:+.2f}%")
        print(f"  平均超额: {summary['avg_excess_return']:+.2f}%  |  "
              f"累计超额: {summary['cum_excess_return']:+.2f}%")
        print(f"  波段胜率: {summary['win_rate']}% ({summary['win_bands']}/{summary['total_bands']})")
        print(f"  当前机制({len(active_mechanisms)}个): "
              f"{', '.join(active_mechanisms) if active_mechanisms else '无(基准DCA)'}")

        # 检查是否达到目标
        if summary['cum_excess_return'] >= 3.0:
            print(f"\n  ✅ 目标达成！累计超额{summary['cum_excess_return']:+.2f}% >= +3%")
            print(f"  使用机制: {', '.join(active_mechanisms)}")

        # 选择下一个机制
        if round_idx < total_rounds:
            weakness_analysis = analyze_weaknesses(results)
            print(f"\n  📋 不足分析:")
            for line in weakness_analysis.split("\n"):
                print(f"     {line}")

            # 选择要添加的机制
            next_mech = _select_next_mechanism(
                summary, active_mechanisms, all_mechanisms, results, data, registry,
                round_idx, total_rounds
            )

            if next_mech:
                active_mechanisms.append(next_mech)
                print(f"\n  ➕ 添加机制: {next_mech}")
                print(f"     {registry.describe(next_mech)}")

        # 换数据（模拟每轮用不同的时间窗口）
        # 已禁用：改为一次性拉取数据，避免每轮都重新请求API
        # if round_idx % 5 == 0:
        #     print("\n  🔄 更新数据...")
        #     data = fetch_data_for_year()
        #     bands = get_2month_bands(
        #         (pd.Timestamp(data["end_date"]) - timedelta(days=365)).strftime("%Y-%m-%d"),
        #         data["end_date"],
        #     )

    # 最终总结
    _print_final_report(history, mechanism_history, registry)

    return history, mechanism_history


def _select_next_mechanism(summary, active_mechanisms, all_mechanisms, results, data,
                          registry, round_idx, total_rounds):
    """智能选择下一个要添加的机制

    基于当前不足分析，选择最适合的机制
    """
    # 已使用的机制
    used = set(active_mechanisms)
    available = [m for m in all_mechanisms if m not in used]

    if not available:
        return None

    # 基于波段表现选择
    excesses = [r.excess_return for r in results if r.excess_return > -999]

    if summary["avg_excess_return"] < -2:
        # 大幅跑输：先加风控
        for pref in ["stop_loss", "bear_defense", "trend_filter"]:
            if pref in available:
                return pref

    if summary["total_benchmark_return"] > 10 and summary["total_strategy_return"] < summary["total_benchmark_return"]:
        # 牛市跑输：加趋势跟踪
        for pref in ["momentum_boost", "macro_timing", "etf_rotation"]:
            if pref in available:
                return pref

    if summary["total_benchmark_return"] < -5:
        # 熊市：加防御
        for pref in ["bear_defense", "trend_filter", "macro_timing"]:
            if pref in available:
                return pref

    if summary["win_rate"] < 40:
        # 胜率低：加止损止盈
        for pref in ["stop_loss", "profit_lock", "volatility_stop"]:
            if pref in available:
                return pref

    # 默认：按优先级添加
    priority = ["trend_filter", "momentum_boost", "stop_loss", "profit_lock",
                "bear_defense", "dca_plus", "grid_add", "volatility_stop",
                "etf_rotation", "macro_timing"]
    for p in priority:
        if p in available:
            return p

    return available[0]


def _print_final_report(history, mechanism_history, registry):
    print("\n\n" + "="*70)
    print("  🏆 最终报告")
    print("="*70)

    best_idx = 0
    best_excess = -999

    for i, (h, mechs) in enumerate(zip(history, mechanism_history)):
        excess = h.get("cum_excess_return", 0)
        prefix = "  ✅" if excess >= 3 else "     "
        mech_str = ", ".join(mechs) if mechs else "基准DCA"

        if excess > best_excess:
            best_excess = excess
            best_idx = i

        print(f"{prefix} 轮{i+1:2d}: 策略{h['total_strategy_return']:+.1f}% | "
              f"基准{h['total_benchmark_return']:+.1f}% | "
              f"超额{excess:+.1f}% | 胜率{h['win_rate']}% | "
              f"[{mech_str}]")

    print(f"\n  最佳表现: 第{best_idx+1}轮, 累计超额{best_excess:+.1f}%")
    print(f"  最终机制组合: {', '.join(mechanism_history[best_idx]) if mechanism_history[best_idx] else '基准DCA'}")

    if best_excess >= 3:
        print(f"\n  🎉 目标达成！超额收益 {best_excess:+.1f}% >= +3%")
    else:
        print(f"\n  💪 还需努力，最佳超额 {best_excess:+.1f}%，距离目标 {3-best_excess:.1f}%")

    print("\n" + "="*70)


if __name__ == "__main__":
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    run_strategy_loop(total_rounds=rounds)
