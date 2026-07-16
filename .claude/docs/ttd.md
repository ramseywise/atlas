Test
- make test-learner — bandit posterior updates, arm selection, save/load, reward function
-  make test-crypto — indicators, graders, agent smoke (single/multi-symbol, bandit policy)
- make test — full suite passes (181 tests)
- make lint — clean
- make crypto --symbol BTC/USDT — live run, actually hits Binance via CCXT, fetches real OHLCV, runs predictions end-to-end against Binance (requires network)
- make api then hit POST /crypto/predict — confirm the API endpoint works with real data
- make crypto-monitor — scores past predictions against fresh actuals
- make compare-learner — bandit vs rule-based head-to-head, runs walk-forward CV comparing bandit vs rule-based (CPU-intensive, ~2-5 min)
