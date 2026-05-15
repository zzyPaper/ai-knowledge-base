"""Daily analysis pipeline - runs at 2:30 PM to analyze hot sectors."""

from datetime import datetime, timedelta
from src.data.fetcher import get_sectors_list_safe as get_live_sectors_safe
from src.data.fetcher import get_sector_history_safe, get_index_history_safe
from src.signals.fusion import compute_sector_scores, detect_market_regime
from src.signals.position_sizing import compute_position_pct
from src.daily.reporter import generate_report
from config.settings import RESULTS_DIR


def run_daily_analysis(date_str: str = None, top_k: int = 20):
    """Fetch data, compute scores, generate report.

    Args:
        date_str: YYYY-MM-DD format, defaults to today.
        top_k: Number of top sectors by turnover to analyze.

    Returns:
        Markdown report string.
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    today = datetime.strptime(date_str, "%Y-%m-%d")
    start_60d = (today - timedelta(days=60)).strftime("%Y-%m-%d")

    live = get_live_sectors_safe()
    if live is None or live.empty:
        print("[ERROR] No live sector data available")
        return "# 报告生成失败：无法获取行情数据"

    if "成交额" in live.columns:
        top_sectors = live.nlargest(top_k, "成交额")["板块名称"].tolist()
    else:
        top_sectors = live["板块名称"].head(top_k).tolist()

    sectors_data = {}
    for s in top_sectors:
        df = get_sector_history_safe(s, start_60d, date_str)
        if df is not None and not df.empty:
            sectors_data[s] = df

    index_data = get_index_history_safe("沪深300", start_60d, date_str)
    if index_data is not None and not index_data.empty:
        index_data["pct_chg"] = index_data["close"].pct_change(1).fillna(0) * 100

    regime = detect_market_regime(index_data) if index_data is not None else "unknown"
    position_pct = compute_position_pct(index_data) if index_data is not None else 50
    ma20_level = float(index_data["close"].rolling(20).mean().iloc[-1]) if index_data is not None and len(index_data) >= 20 else 0
    scores_df = compute_sector_scores(sectors_data, index_data)

    today_data = {}
    for s, df in sectors_data.items():
        row = df[df["date"] == date_str] if "date" in df.columns else df.iloc[-1:]
        if not row.empty:
            latest = row.iloc[-1]
            today_data[s] = {
                "close": latest.get("close", 0),
                "pct_chg": latest.get("pct_chg", 0),
                "amount": latest.get("amount", 0),
                "turnover_rate": latest.get("turnover_rate", 0),
            }

    report = generate_report(
        date_str=date_str,
        scores_df=scores_df,
        regime=regime,
        top_sectors=top_sectors[:5],
        today_data=today_data,
        index_recent=index_data.tail(5) if index_data is not None else None,
        position_pct=position_pct,
        ma20_level=ma20_level,
    )

    results_path = RESULTS_DIR / f"{date_str}_report.md"
    results_path.write_text(report, encoding="utf-8")

    return report
