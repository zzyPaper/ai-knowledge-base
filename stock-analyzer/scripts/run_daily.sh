#!/bin/bash
# 每日14:45 自动跑专业行业轮动策略
# Crontab: 45 14 * * 1-5 /Users/zhenzhiyuan/AI知识库/stock-analyzer/scripts/run_daily.sh

set -e
cd /Users/zhenzhiyuan/AI知识库/stock-analyzer

LOG=/tmp/stock_daily_v2.log
echo "===== $(date '+%Y-%m-%d %H:%M:%S') =====" >> "$LOG"

/usr/bin/python3 scripts/run_daily_v2.py --brief >> "$LOG" 2>&1
echo "--- done ---" >> "$LOG"

# Also save full report with date
DATE=$(date '+%Y-%m-%d')
echo "Full report: data/results/${DATE}_v2_report.md" >> "$LOG"
