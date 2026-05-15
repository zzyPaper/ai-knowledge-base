# quant-aio

A股热门板块实时分析 & 量化买卖策略系统（含 AI Agent 驱动的自循环训练）

## 核心设计

- **训练阶段**：AI Agent 驱动自循环 — 逐轮回测→分析结果→调整参数→重新回测，反复迭代直到超额收益超过基准 10%
- **运行阶段**：程序独立运行，读取训练好的参数，不需要 AI 参与

## 策略体系

### V1 简单动量策略 (Dual Momentum)

| 因子 | 权重 | 说明 |
|------|------|------|
| 绝对动量 (ROC) | 100% | 20日涨幅过滤 + MA20均线确认 |

- 参考: Antonacci (2012) Dual Momentum
- 特点: 逻辑简单，信号清晰，适合入门
- 适用: 趋势市

### V2 七因子专业策略

| 因子 | 权重 | 说明 |
|------|------|------|
| 趋势强度 | 20% | 对数价格线性回归斜率 × R² |
| 多窗口动量 | 20% | 1/3/6/12月 ROC 共振确认 |
| 量能确认 | 15% | ln(5日均量 / 20日均量) |
| 资金因子 | 15% | 北向资金 + 主力资金流 |
| 情绪因子 | 15% | 涨停占比 + 市场宽度 |
| 景气度 | 10% | 营收/利润增速初筛（一票否决） |
| 估值因子 | 5% | PE分位仓位调节 |

**风控规则：**
- ATR 波动率仓位管理：仓位 ∝ 1/ATR%
- MA60 仓位调节：低于均线 → 仓位减半
- 尖峰检测：TR > 2×ATR → 额外降仓
- 趋势-动量一致性过滤
- 景气度一票否决
- 估值仓位调节：PE分位 [0.5x, 1.5x]

## 架构

```
quant-aio/
├── config/
│   ├── settings.py              # 全局配置
│   └── strategy_params.yaml     # 策略参数（训练产出）
├── src/
│   ├── data/fetcher.py          # 数据获取层（腾讯+baostock+东方财富 多源 fallback）
│   ├── strategy/
│   │   ├── base.py              # 策略基类 BaseStrategy + SectorScore
│   │   ├── __init__.py          # 策略注册表 get_strategy()
│   │   ├── v1_simple_momentum.py  # V1 简单动量策略
│   │   ├── v2_three_factor.py   # V2 七因子策略
│   │   ├── trend_strength.py    # 趋势强度因子
│   │   ├── short_momentum.py    # 短期动量因子
│   │   ├── volume_confirm.py    # 量能确认因子
│   │   ├── multi_window_momentum.py  # 多窗口动量因子
│   │   ├── capital_flow.py      # 资金因子（北向+主力）
│   │   ├── sentiment.py         # 情绪因子（涨停+市场宽度）
│   │   ├── valuation.py         # 估值因子（PE分位）
│   │   ├── fundamental.py       # 景气度因子（营收+利润）
│   │   └── fusion.py            # 兼容层（旧接口重定向到V2）
│   ├── engine/
│   │   └── daily.py             # 每日分析引擎（策略无关）
│   └── backtest/
│       ├── engine.py            # 回测引擎（策略无关）
│       └── optimizer.py         # 优化器工具函数
├── scripts/
│   └── run_backtest.py          # 回测执行脚本（AI 调用）
├── data/                        # 缓存和结果（gitignore）
└── main.py                      # CLI 入口
```

## 快速开始

```bash
# 安装依赖
pip install -e ".[dev]"

# ── 列出可用策略 ──
python main.py strategies

# ── 每日分析（默认V2策略）──
python main.py daily

# ── 用V1策略分析 ──
python main.py daily --strategy v1

# ── 单次回测 ──
python main.py backtest --strategy v2 --start 20250101 --end 20250301

# ── V1策略回测 ──
python main.py backtest --strategy v1 --start 20250101 --end 20250301

# ── 查看训练计划 ──
python main.py train
```

## 数据源

多数据源自动 fallback，无需手动切换：

| 数据 | 主源 | 备源1 | 备源2 |
|------|------|-------|-------|
| 指数K线 | 腾讯(AKShare) | baostock | 东方财富 |
| 板块列表 | 东方财富 | baostock行业聚合 | 硬编码fallback |
| 板块K线 | 东方财富 | baostock个股聚合 | - |
| 北向资金 | AKShare(东方财富) | - | - |
| 主力资金 | AKShare(板块资金流) | - | - |
| 板块估值 | AKShare(板块PE) | - | - |
| 涨停统计 | AKShare(涨停池) | - | - |

## AI Agent 驱动的自循环训练

```bash
# 第 1 步：查看训练窗口
python main.py train --strategy v2

# 第 2 步：对每个窗口，执行回测
python scripts/run_backtest.py --strategy v2 --start 20250101 --end 20250301

# 第 3 步：AI 读取结果
cat data/backtest_results/runner_latest.json

# 第 4 步：AI 分析结果，调整 config/strategy_params.yaml

# 第 5 步：重复 2-4 直到达标
```

## 可调参数 (strategy_params.yaml)

| 参数 | 说明 | V1默认 | V2默认 |
|------|------|--------|--------|
| buy_threshold | 综合分>此值买入 | 0.10 | 0.10 |
| sell_threshold | 综合分<此值卖出 | -0.10 | -0.10 |
| stop_loss | 止损线 | -5% | -5% |
| take_profit | 止盈线 | 15% | 15% |
| position_per_sector | 单板块仓位 | 20% | 20% |
| top_n | 截面排名选前N板块 | 5 | 5 |
| rebalance_freq | 调仓频率 | monthly | monthly |
