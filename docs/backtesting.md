# Backtesting

Run the historical backtest with the same signal engine used live:

```powershell
python backtest/engine.py --config config.yaml --symbol AVAX/USDC:USDC --timeframe 5m --limit 1000
```

Optional exports:

```powershell
python backtest/engine.py --config config.yaml --symbol AVAX/USDC:USDC --timeframe 5m --limit 1000 --trades-out backtest_trades.csv --equity-out backtest_equity.csv
```

What it does:

- fetches OHLCV for the base timeframe and supporting MTFs
- rebuilds indicators candle by candle
- calls `generate_quant_signal()` on each closed candle
- simulates fills with spread and slippage
- tracks trade PnL, win rate, drawdown, and exit reasons

Notes:

- The backtester is conservative on fills.
- If Binance public endpoints are unavailable, historical fetch will fail.
- The module is intended for validation and tuning, not guaranteed live profit.
