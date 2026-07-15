@echo off
chcp 65001 >nul
cd /d e:\quant

if not exist e:\quant\logs mkdir e:\quant\logs

set YEAR=%date:~0,4%
set MONTH=%date:~5,2%
set DAY=%date:~8,2%
set TODAY=%YEAR%%MONTH%%DAY%
set LOGFILE=e:\quant\logs\daily_%TODAY%.txt

echo ======================================== > "%LOGFILE%"
echo  板块ETF轮动策略 - 每日分析 >> "%LOGFILE%"
echo  运行时间: %date% %time% >> "%LOGFILE%"
echo ======================================== >> "%LOGFILE%"
echo. >> "%LOGFILE%"

python -X utf8 e:\quant\daily_run.py >> "%LOGFILE%" 2>&1

echo. >> "%LOGFILE%"
echo ======================================== >> "%LOGFILE%"
echo  分析完成，请在15:00前执行交易 >> "%LOGFILE%"
echo  收盘后运行: python daily_run.py --confirm >> "%LOGFILE%"
echo ======================================== >> "%LOGFILE%"

start "" notepad "%LOGFILE%"
