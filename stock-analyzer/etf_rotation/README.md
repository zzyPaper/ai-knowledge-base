# 板块ETF轮动策略系统

多机制协同的板块ETF轮动量化交易策略。48只ETF基金池，10个并行决策机制，以沪深300为市场基准，每日14:40生成交易信号。

## 核心思想

1. **板块轮动** — 48只行业/主题ETF组成交易池，每月按20日动量选出最强板块
2. **多机制协同** — 10个并行机制各司其职：选股、风控、择时、仓位管理
3. **市场择时** — 以CSI300沪深300指数MA5/MA20为短期趋势判断，结合回撤和均线排列判断市场状态
4. **持续运行** — 现金和持仓跨周期累积，非波段重置

## 策略机制

| # | 机制 | 作用 |
|---|------|------|
| 1 | **基础定投** (dca_plus) | 每月初自动定投当前轮动目标 |
| 2 | **止损机制** (stop_loss) | 持仓浮亏>8%时全部卖出 |
| 3 | **利润锁定** (profit_lock) | 持仓浮盈>15%时锁定50%利润 |
| 4 | **ETF轮动** (etf_rotation) | 核心选股 — 20日动量选最强板块 |
| 5 | **宏观择时** (macro_timing) | 基于CSI300均线判断牛/熊/震荡 |
| 6 | **趋势过滤** (trend_filter) | 空头排列时买入转国债 |
| 7 | **动量增强** (momentum_boost) | 下跌趋势中过滤买入信号 |
| 8 | **智能定投** (dca_plus) | 根据CSI300偏离60日线幅度调整金额 |
| 9 | **网格加仓** (grid_add) | 每跌5%追加买入 |
| 10 | **波动率止损** (volatility_stop) | 高波动时减半买入 |
| 11 | **熊市防御** (bear_defense) | 回撤>10%转国债避险 |

### 买入条件

```
csi300_up = MA5 > MA20                           # 短期上升趋势
rising = 连续3天收盘站上20日线                     # 上涨势头
best_return > 0                                   # 最优板块收益为正
best_in_uptrend = best_price > best_ma20          # 最优板块在上升趋势
buying_disabled = in_defense OR trend_bear OR market_bear OR momentum_paused

can_buy = NOT buying_disabled AND rising AND csi300_up
          AND best_return > 0 AND best_in_uptrend
```

## 交易标的

### 基金池（48只）三层架构

| 分类 | 数量 | 说明 |
|------|------|------|
| **REFERENCE_CODES** | 1 (510300) | 沪深300ETF，市场参考基准，不参与轮动 |
| **DEFENSE_CODES** | 1 (511520) | 国债ETF，防御/熊市避风港 |
| **trading_codes** | 46 | 行业/主题ETF，轮动交易标的 |

### 行业覆盖

科技、半导体、5G、软件、数字经济、新能源车、新能源、光伏、锂电池、医疗、医药、创新药、消费、酒、食品饮料、养殖、农业、证券、银行、房地产、煤炭、化工、基建、钢铁、稀土、军工、智能汽车、机械、游戏、互联网、黄金、商品、红利、红利低波、上证50、中证500、中证1000、创业板、创业板50、科创50、中概互联、纳指、旅游、家电、传媒、电力

### 费用规则

| 项目 | 费率 |
|------|------|
| 申购（买入） | 0% |
| 赎回（卖出）<7天 | 1.5%惩罚费 |
| 赎回（卖出）≥7天 | 0% |
| 最短持有期 | 7天 |

## 快速开始

### 安装

```bash
git clone <repo-url>
cd quant
pip install -r requirements.txt
```

### 运行回测

```bash
# 30轮机制迭代回测
python strategy_loop.py 30

# 指定迭代轮数
python strategy_loop.py 10
```

### 每日交易工作流

1. 每天14:40运行分析：
```bash
python daily_run.py
```

2. 查看交易信号，在15:00前执行交易

3. 收盘后确认成交：
```bash
python daily_run.py --confirm
```

### 运行测试

```bash
python test_fix.py
```

## 项目结构

```
quant/
├── strategy_loop.py        # 策略迭代循环系统（核心）
├── daily_run.py            # 每日交易工作流（14:40执行）
├── config.yaml             # 系统配置文件
├── requirements.txt        # Python依赖
├── .gitignore
├── README.md
├── 项目文档.md              # 完整项目架构文档
│
├── backtest/               # 回测引擎
│   ├── engine.py           # BacktestEngine + Signal
│   └── account.py          # Account + Position
│
├── data/                   # 数据模块
│   └── fetcher.py          # 数据获取（akshare）
├── strategy/               # 策略模块（旧版，供参考）
│   ├── base.py
│   └── builtin.py
├── trading/                # 交易模块
│   ├── broker.py
│   └── scheduler.py
├── utils/                  # 工具函数
│   ├── config.py
│   └── time_utils.py
├── report/                 # 报告生成
│   └── dashboard.py
├── examples/               # 示例代码
│   └── custom_strategy.py
│
├── .claude/skills/         # Claude Code技能定义
│   └── daily_trading.md
│
├── data/cache/             # 数据缓存（gitignore）
├── output/                 # 回测输出（gitignore）
└── logs/                   # 日志文件（gitignore）
```

## 30轮迭代方法论

**不调参，加机制**。每轮迭代：

1. 用当前策略回测过去1年（6个波段，每波段2个月）
2. 与CSI300对比，分析超额收益
3. 找出策略的系统性不足
4. 添加一个新机制来弥补该不足
5. 进入下一轮

## 数据源

使用 [akshare](https://github.com/akfamily/akshare) 获取基金净值与指数数据，无需API Key。

## 依赖

- pandas, numpy — 数据处理
- akshare — A股数据获取
- pyyaml — 配置管理
- matplotlib, seaborn — 可视化
- joblib — 并行计算
- tqdm — 进度条

## 许可

MIT
