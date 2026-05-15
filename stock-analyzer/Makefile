.PHONY: install verify backtest daily clean train

install:
	python3 -m pip install pandas akshare pyyaml tabulate

verify:
	python3 scripts/verify_all.py

train:
	python3 scripts/run_backtest.py --use-synthetic

backtest:
	python3 scripts/run_backtest.py $(ARGS)

daily:
	python3 scripts/run_analysis.py $(ARGS)

clean:
	rm -rf data/cache/*.parquet
	rm -rf data/results/*.md
	rm -rf data/backtest_results/*
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
