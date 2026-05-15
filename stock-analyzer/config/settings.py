from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_DIR / "src"
DATA_DIR = PROJECT_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
RESULTS_DIR = DATA_DIR / "results"
BACKTEST_RESULTS_DIR = DATA_DIR / "backtest_results"

# API rate limiting
RATE_LIMIT = 3  # max requests per second
DELAY_RANGE = (0.2, 0.5)  # random delay between requests

# Cache TTL (seconds)
CACHE_TTL_SECTOR_LIST = 86400  # 1 day
CACHE_TTL_SECTOR_HISTORY = 86400  # 1 day
CACHE_TTL_INDEX_HISTORY = 86400  # 1 day

for d in [CACHE_DIR, RESULTS_DIR, BACKTEST_RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
