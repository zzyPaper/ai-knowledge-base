"""全局配置"""
from pathlib import Path

# ── 路径 ──
PROJECT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_DIR / "src"
DATA_DIR = PROJECT_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
RESULTS_DIR = DATA_DIR / "results"
BACKTEST_DIR = DATA_DIR / "backtest_results"
LOG_DIR = DATA_DIR / "logs"

for d in (CACHE_DIR, RESULTS_DIR, BACKTEST_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── 限频（AKShare / 东方财富接口）──
RATE_LIMIT_PER_SEC = 3
RATE_LIMIT_DELAY_MIN = 0.2
RATE_LIMIT_DELAY_MAX = 0.5

# ── 缓存 TTL（秒）──
CACHE_TTL_SECTOR_LIST = 86400       # 板块列表 1 天
CACHE_TTL_SECTOR_HIST = 86400       # 板块历史 1 天
CACHE_TTL_INDEX_HIST = 86400        # 指数历史 1 天
CACHE_TTL_LIVE_QUOTE = 300          # 实时行情 5 分钟

# ── 策略默认参数 ──
HOT_SECTOR_TOP_N = 5                # 取成交额前 N 个热门板块
KLINE_LOOKBACK_DAYS = 30            # 板块日 K 线回看天数
BENCHMARK_INDEX = "沪深300"           # 基准指数（中证A500仅有2024年9月后数据）

# ── 回测 ──
BACKTEST_INITIAL_CASH = 1_000_000   # 初始资金 100 万
BACKTEST_BENCHMARK_THRESHOLD = 0.10 # 超额收益阈值（10%）
BACKTEST_WINDOW_MONTHS = 2         # 每轮回测窗口 2 个月
BACKTEST_MAX_ROUNDS = 12           # 最多迭代 12 轮（覆盖 1 年）

# ── 个股选股 ──
STOCK_UNIVERSE_MIN_MARKET_CAP = 5e9    # 最低市值 50 亿
STOCK_UNIVERSE_MIN_AMOUNT = 5e7        # 日均成交额最低 5000 万
STOCK_TOP_N = 30                       # 选股持仓数量
STOCK_MAX_INDUSTRY_WEIGHT = 0.20       # 单行业最大权重 20%

# ── 调度 ──
DAILY_RUN_TIME = "14:30"            # 每日运行时间

# ── 训练 ──
TRAINING_MONTHS = 12                 # 训练覆盖月数
TRAINING_MAX_ROUNDS_PER_WINDOW = 12  # 每窗口最多迭代轮次
