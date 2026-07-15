"""
每日策略工作流 (14:40执行)
===========================
用法: python daily_run.py
      python daily_run.py --confirm   # 收盘后确认成交并更新持仓

资金: 5,000元
信号基于最新可用NAV数据生成，您在15:00前执行交易，以当日收盘净值结算。
"""
import sys, os, json, argparse
from datetime import datetime
from dataclasses import dataclass

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
from strategy_loop import fetch_data_for_year, fetch_intraday_prices, MechanismRegistry, build_strategy
from .backtest.account import Account, Position
from .backtest.engine import Signal

STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "daily_state.json")
INIT_CASH = 5000.0


def _merge_iopv_to_nav(nav_data: dict, prices: dict, today: str):
    """将盘中IOPV合并到基金NAV数据，追加今日行并重算均线"""
    today_ts = pd.Timestamp(today)
    for code, df in nav_data.items():
        if code not in prices:
            continue
        iopv = prices[code]
        if df.empty:
            continue
        last_date = df["date"].iloc[-1]
        if isinstance(last_date, pd.Timestamp) and last_date >= today_ts:
            continue
        if isinstance(last_date, str) and last_date[:10] >= today:
            continue
        new_row = df.iloc[-1].copy()
        new_row["date"] = today_ts
        new_row["nav"] = iopv
        new_row["daily_return"] = np.nan
        df.loc[len(df)] = new_row
    # 统一重算所有基金的滚动指标
    for df in nav_data.values():
        if len(df) < 2:
            continue
        df["ret_5d"] = df["nav"].pct_change(5)
        df["ret_20d"] = df["nav"].pct_change(20)
        df["ma5"] = df["nav"].rolling(5, min_periods=1).mean()
        df["ma20"] = df["nav"].rolling(20, min_periods=1).mean()
        df["ma60"] = df["nav"].rolling(60, min_periods=1).mean()
        df["volatility"] = df["nav"].pct_change().rolling(20, min_periods=1).std()


def _merge_iopv_to_csi300(csi_df: pd.DataFrame, spot_df: pd.DataFrame, today: str):
    """用510300ETF的IOPV推算今日CSI300水平"""
    s = spot_df[spot_df["代码"].astype(str) == "510300"]
    if s.empty:
        return
    iopv = s.iloc[0]["IOPV实时估值"]
    prev_close = s.iloc[0]["昨收"]
    if pd.isna(iopv) or pd.isna(prev_close) or prev_close <= 0:
        return
    change_ratio = iopv / prev_close
    last_csi_close = csi_df["close"].iloc[-1]
    today_csi_close = last_csi_close * change_ratio
    new_row = csi_df.iloc[-1].copy()
    new_row["date"] = pd.Timestamp(today)
    new_row["close"] = today_csi_close
    csi_df.loc[len(csi_df)] = new_row


def serialize_state(account, ms):
    """保存当前持仓和策略状态到JSON"""
    positions_out = []
    for code, pos in account.positions.items():
        positions_out.append({
            "code": pos.code,
            "name": pos.name,
            "shares": pos.shares,
            "cost": pos.cost,
            "entry_date": pos.entry_date,
            "last_buy_date": pos.last_buy_date,
        })
    # 只保存可序列化的策略状态
    exclude_keys = {"csi300_df", "trading_codes", "defense_codes", "fund_names"}
    ms_out = {k: v for k, v in ms.items() if k not in exclude_keys}
    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "cash": account.cash,
        "positions": positions_out,
        "mechanism_state": ms_out,
    }


def deserialize_state(s, data, trading_codes, defense_codes, csi300_sig):
    """从JSON恢复持仓和策略状态"""
    account = Account(init_cash=INIT_CASH)
    account.cash = s.get("cash", INIT_CASH)

    fund_names = data.get("fund_names", {})
    for pd_ in s.get("positions", []):
        account.positions[pd_["code"]] = Position(
            code=pd_["code"],
            name=fund_names.get(pd_["code"], pd_.get("name", "")),
            shares=pd_["shares"],
            cost=pd_["cost"],
            entry_date=pd_["entry_date"],
            last_buy_date=pd_["last_buy_date"],
        )

    ms = s.get("mechanism_state", {})
    ms["csi300_df"] = csi300_sig
    ms["trading_codes"] = trading_codes
    ms["defense_codes"] = defense_codes
    return account, ms



def main():
    parser = argparse.ArgumentParser(description="每日策略运行")
    parser.add_argument("--confirm", action="store_true",
                        help="收盘后确认成交，更新持仓状态")
    args = parser.parse_args()

    today = datetime.now().strftime("%Y-%m-%d")

    # ─── 1. 获取最新数据 ───
    print("⏳ 获取最新市场数据...")
    data = fetch_data_for_year()
    fund_names = data.get("fund_names", {})

    REFERENCE_CODES = ["510300"]
    DEFENSE_CODES = ["511520"]
    trading_codes = [c for c in data["fund_codes"]
                     if c not in REFERENCE_CODES + DEFENSE_CODES]

    # ─── 1b. 获取当日盘中数据(IOPV实时净值) ───
    print("⏳ 获取ETF实时行情(IOPV)...")
    intraday_prices, spot_df, spot_date = fetch_intraday_prices()
    has_intraday = bool(intraday_prices) and spot_date == today

    if has_intraday:
        print(f"  📡 获取到今日盘中IOPV({len(intraday_prices)}只ETF)")
        _merge_iopv_to_nav(data["nav_data"], intraday_prices, today)
        _merge_iopv_to_csi300(data["csi300"], spot_df, today)

    # ─── 2. CSI300信号数据（从已合并的数据构建） ───
    csi300_raw = data["csi300"].copy()
    csi300_sig = pd.DataFrame({
        "date": csi300_raw["date"].dt.strftime("%Y-%m-%d"),
        "close": csi300_raw["close"],
        "ma5": csi300_raw["close"].rolling(5).mean(),
        "ma20": csi300_raw["close"].rolling(20).mean(),
        "ma60": csi300_raw["close"].rolling(60).mean(),
        "volatility": csi300_raw["close"].pct_change().rolling(20).std(),
        "high_60": csi300_raw["close"].rolling(60).max(),
    })
    csi300_sig["drawdown"] = (csi300_sig["high_60"] - csi300_sig["close"]) / csi300_sig["high_60"] * 100

    # ─── 3. 构建策略函数 ───
    all_10 = [
        "momentum_boost", "stop_loss", "bear_defense", "trend_filter",
        "macro_timing", "etf_rotation", "profit_lock", "dca_plus",
        "grid_add", "volatility_stop"
    ]
    registry = MechanismRegistry()
    strategy_fn = build_strategy(all_10, registry, round_num=30)

    # ─── 4. 恢复或初始化状态 ───
    if os.path.exists(STATE_FILE) and not args.confirm:
        with open(STATE_FILE, encoding="utf-8") as f:
            saved = json.load(f)
        last_date = saved.get("date", "")
        if last_date == today:
            print(f"今日({today})已分析过，如需重新运行请删除 {STATE_FILE}")
            return
        print(f"  上次更新: {last_date}")
        account, ms = deserialize_state(saved, data, trading_codes,
                                         DEFENSE_CODES, csi300_sig)
        if saved["positions"]:
            for p in saved["positions"]:
                name = fund_names.get(p["code"], p.get("name", p["code"]))
                print(f"  持仓: {name}  {p['shares']:.2f}份  成本{p['cost']:.2f}")
        print(f"  现金: {account.cash:.2f}")
    else:
        account = Account(init_cash=INIT_CASH)
        account.cash = INIT_CASH
        ms = {
            "round": 30,
            "csi300_df": csi300_sig,
            "trading_codes": trading_codes,
            "defense_codes": DEFENSE_CODES,
        }
        print(f"  初始化: 空仓, 现金 {INIT_CASH}元")

    # ─── 5. 获取最近的交易日 ───
    if has_intraday:
        latest_data_date = today
        print(f"  数据日期: {latest_data_date} (含今日IOPV盘中数据)")
    else:
        latest_data_date = data["csi300_signals"]["date"].iloc[-1]
        # 交叉验证：统计可交易基金有多少在最新日期有数据
        funds_ok = 0
        funds_total = 0
        for code in trading_codes:
            df = data["nav_data"].get(code)
            if df is not None:
                funds_total += 1
                date_set = set(df["date"].dt.strftime("%Y-%m-%d"))
                if latest_data_date in date_set:
                    funds_ok += 1
        coverage_pct = (funds_ok / funds_total * 100) if funds_total > 0 else 0
        data_age_days = (datetime.now() - pd.Timestamp(latest_data_date)).days
        if coverage_pct < 80:
            print(f"  ⚠ 数据覆盖: 仅{funds_ok}/{funds_total}({coverage_pct:.0f}%)基金在{latest_data_date}有数据")
        else:
            print(f"  数据覆盖: {funds_ok}/{funds_total}({coverage_pct:.0f}%)基金在{latest_data_date}有数据")
        if data_age_days > 7:
            print(f"  ⚠ 最新数据({latest_data_date})距今{data_age_days}天，请检查akshare是否正常")
        elif data_age_days > 3:
            print(f"  ℹ 最新数据{latest_data_date}（{data_age_days}天前），确认是否为假期")

    # ─── 5. 运行策略 ───
    print(f"\n📊 运行策略 (基于{latest_data_date}数据，今日{today}交易)...")
    signals = list(strategy_fn(latest_data_date, account, data["nav_data"],
                                trading_codes, mechanism_state=ms))

    # ─── 6. CSI300行情报告 ───
    csi = csi300_sig[csi300_sig["date"] <= latest_data_date]
    print()
    print("=" * 60)
    print(f"  📈 行情概览 ({latest_data_date})")
    print("=" * 60)
    if not csi.empty:
        r = csi.iloc[-1]
        streak = ms.get("rising_streak", 0)
        csi300_up = r["ma5"] > r["ma20"]
        market_state = ms.get("market_state", "?")
        in_defense = ms.get("in_defense", False)
        print(f"  CSI300: {r['close']:.2f}")
        print(f"  均线:   MA5={r['ma5']:.2f}  MA20={r['ma20']:.2f}  MA60={r['ma60']:.2f}")
        print(f"  回撤:   {r['drawdown']:.1f}%")
        print(f"  市场:   {market_state}  防御模式:{'是' if in_defense else '否'}")
        trend_str = "↑" if csi300_up else "↓"
        print(f"  趋势:   MA5{'↑' if csi300_up else '↓'}MA20  收盘{'↑' if r['close']>r['ma20'] else '↓'}MA20"
              f"  连续{streak}天站上20日线")

        # 20日动量Top5
        print(f"\n  20日动量 Top 5:")
        returns = {}
        for code in trading_codes:
            df = data["nav_data"].get(code)
            if df is not None:
                hist = df[df["date"] <= latest_data_date]
                if len(hist) >= 20:
                    ret = (hist["nav"].iloc[-1] / hist["nav"].iloc[-20] - 1) * 100
                    returns[code] = ret
        top5 = sorted(returns.items(), key=lambda x: x[1], reverse=True)[:5]
        for code, ret in top5:
            name = fund_names.get(code, code)
            flag = " ← 最优" if code == ms.get("best_etf") else ""
            print(f"    {name:<10} {ret:>+7.2f}%{flag}")

    # ─── 7. 账户概况 ───
    print()
    print("=" * 60)
    print("  💰 账户概况")
    print("=" * 60)
    pos_value = 0.0
    if account.positions:
        for code, pos in account.positions.items():
            nv = data["nav_data"][code]
            hist = nv[nv["date"] <= latest_data_date]
            cur_nav = hist.iloc[-1]["nav"] if not hist.empty else 0
            mv = pos.shares * cur_nav if cur_nav > 0 else pos.cost
            pos_value += mv
            profit = mv - pos.cost
            profit_pct = (mv / pos.cost - 1) * 100 if pos.cost > 0 else 0
            name = fund_names.get(code, code)
            hold_days = (pd.Timestamp(latest_data_date) - pd.Timestamp(pos.entry_date)).days
            print(f"  {name:<10} {pos.shares:>8.2f}份  "
                  f"成本{pos.cost:>7.2f}  市值{mv:>7.2f}  "
                  f"盈亏{profit:>+7.2f}({profit_pct:>+.1f}%)  持有{hold_days}天")
    else:
        print(f"  空仓")
    total = account.cash + pos_value
    print(f"  现金: {account.cash:.2f}")
    print(f"  总资产: {total:.2f}")
    total_profit = total - INIT_CASH
    print(f"  累计盈亏: {total_profit:+.2f}  ({(total/INIT_CASH-1)*100:+.2f}%)")

    # ─── 8. 信号输出 ───
    print()
    print("=" * 60)
    print("  🎯 交易信号")
    print("=" * 60)

    buy_signals = [s for s in signals if s.action == "buy"]
    sell_signals = [s for s in signals if s.action == "sell"]

    if not signals:
        print("  无操作")
        print("  建议: 持有不动")

        # 显示当前最优板块
        best_etf = ms.get("best_etf")
        best_ret = ms.get("best_return", 0)
        if best_etf:
            name = fund_names.get(best_etf, best_etf)
            print(f"  当前最优: {name}({best_etf})  {best_ret:+.2f}%")
    else:
        for sig in signals:
            name = fund_names.get(sig.code, sig.code)
            if sig.action == "buy":
                amt = "全部现金" if sig.amount >= 999999 else f"{sig.amount:.0f}元"
                print(f"  ⤴ 买入 {name}({sig.code})")
                print(f"     金额: {amt}")
                print(f"     理由: {sig.reason}")
            elif sig.action == "sell":
                print(f"  ⤵ 卖出 {name}({sig.code})")
                print(f"     理由: {sig.reason}")

    # 检查7天持有期约束
    if sell_signals and account.positions:
        print(f"\n  ⚠ 注意: 以下持仓尚不足7天，卖出会产生1.5%惩罚费:")
        for code, pos in account.positions.items():
            hold_days = (pd.Timestamp(latest_data_date) - pd.Timestamp(pos.entry_date)).days
            if hold_days < 7:
                name = fund_names.get(code, code)
                print(f"    {name} 仅持有{hold_days}天 (需{7-hold_days}天后可免费卖出)")

    # ─── 9. 更新并保存状态 ───
    if args.confirm:
        # 确认模式：交互式更新持仓
        print(f"\n{'=' * 60}")
        print("  收盘确认模式")
        print("=" * 60)
        # 让用户输入实际的成交情况
        print("  请输入今日实际成交 (直接回车跳过):")
        while True:
            inp = input("  操作 (buy/sell/done): ").strip().lower()
            if inp == "done" or inp == "":
                break
            code = input("  基金代码: ").strip()
            amount = float(input("  金额: ").strip() or "0")
            nav = float(input("  成交净值: ").strip() or "0")
            if inp == "buy" and code and amount > 0 and nav > 0:
                account.buy(code, amount, nav, today,
                           name=fund_names.get(code, ""), buy_fee_rate=0.0)
                print(f"  已记录: 买入{fund_names.get(code,code)} {amount}元 @净值{nav}")
            elif inp == "sell" and code and nav > 0:
                account.sell(code, 999999, nav, today)
                print(f"  已记录: 卖出{fund_names.get(code,code)} @净值{nav}")

        state = serialize_state(account, ms)
        state["date"] = today
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False, cls=NpEncoder)
        print(f"\n  状态已保存至 {STATE_FILE}")
    else:
        # 普通模式：保存策略状态（不含成交更新）
        state = serialize_state(account, ms)
        state["date"] = today
        if signals:
            state["todays_signals"] = [
                {"action": s.action, "code": s.code, "amount": s.amount, "reason": s.reason}
                for s in signals
            ]
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False, cls=NpEncoder)
        print(f"\n  策略状态已保存至 {STATE_FILE}")
        print("  ℹ 执行交易后，请在收盘后运行: python daily_run.py --confirm")

    # 输出策略状态关键指标
    print()
    print("=" * 60)
    print("  策略状态")
    print("=" * 60)
    for k in ["rotation_target", "best_etf", "best_return", "in_defense",
              "market_state", "trend_filter_active", "rising_streak"]:
        v = ms.get(k)
        if k == "best_etf" and v in fund_names:
            v = f"{v}({fund_names[v]})"
        if v is not None:
            print(f"  {k} = {v}")


if __name__ == "__main__":
    main()
