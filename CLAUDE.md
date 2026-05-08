# CLAUDE.md

@.clinerules/caveman.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> ⚠️ See **AGENTS.md** for canonical architecture map, edit recipes, and TypedDict references. This file is supplementary.

## Quick Start

**Run the bot:**
```bash
python main.py
```

**Run backtest:**
```bash
python backtest/engine.py --config config.yaml --limit 1000
```

**Run paper (demo) mode:**
```bash
EXECUTION_MODE=paper python main.py
```

**Expected environment:** `.env` file with:
```
BINANCE_API_KEY=...
BINANCE_SECRET=...
```

## Architecture Overview

### Bot Flow (core/bot_loop.py)

The bot is a **hybrid system** that runs in three modes:
1. **live** — real Binance Futures trading with capital at risk
2. **paper** — simulated trading using `execution/paper.py` (same logic, no real positions)
3. **backtest** — historical candle-by-candle replay via `backtest/engine.py`

**Execution sequence (run_hybrid_bot)**:
1. Load `config.yaml` and environment variables (API keys)
2. Initialize exchange connection (ccxt Binance Futures USDM)
3. Create signal engine (`indicators/signal.py`) with multi-timeframe data
4. Create executor (`execution/futures.py` for live, `execution/paper.py` for paper)
5. Main loop: fetch OHLCV → generate signal → execute entry/exit → update position → repeat
6. On close: export trade log, print stats

### Signal Generation (indicators/signal.py)

**Composite signal** combines 10+ institutional indicators into a single score (-1.0 to +1.0):
- **Mean Reversion**: Z-Score + RSI (detects oversold/overbought)
- **VWAP**: Institutional anchor confirmation
- **MACD**: 3 patterns (flip, shrinking tower, zero bounce)
- **EMA Gradient**: Trend filter (9/21/50/200 stack)
- **Bollinger Bands**: Exhaustion at extremes
- **ADX, SAR, RSI Divergence, OBV, Volume Delta, KDJ**: Secondary confirmations
- **Support/Resistance Walls (SMC)**: Structural levels from `indicators/smc.py`

**Multi-timeframe stack** (MTF): Requires minimum agreement across 5m/15m/1h/4h to bias direction. Higher timeframes (1h/4h) act as swing bias, lower TF (5m/15m) for precise entry.

**Entry gate**: Signal score must exceed thresholds (typically ±0.05), with confidence multipliers based on ADX, Bollinger squeeze state, and session time.

### Execution (execution/)

**Base class** (`execution/base.py`): Abstract interface for all modes.

**Futures** (`execution/futures.py` — LIVE/PAPER Binance Futures USDM):
- Dynamic leverage based on signal confidence (1x–3x)
- TP/SL placement via exchange reduce-only orders (hard backstops)
- Native trailing stops with tight callback (0.25%)
- Breakeven trigger (0.08% profit → move SL to entry)
- Scalp runner: partial exit at 50% position, trail remainder aggressively
- Same-side reentry cooldown (120s) — prevent whipsaws after profitable exits
- All trades logged to `trade_log_futures.csv`

**Paper** (`execution/paper.py`): Simulates the same logic using historical prices and order fills.

**Spot** (`execution/spot.py`): Grid-based accumulation on support; co-exits with futures (not primary).

### Backtesting (backtest/engine.py)

Replays historical candles with **live signal logic** (not curve-fitted). Run:
```bash
python backtest/engine.py --config config.yaml --limit 1000 --trades-out results.csv
```

Key: Backtest uses exact same signal/execution code as live, so results are predictive (within spread/slippage assumptions).

### Configuration (config.yaml)

**Top-level sections:**
- `symbol`, `timeframe`, `macro_timeframe` — trading pair & chart timeframes
- `execution` — leverage, TP%, SL%, breakeven trigger, trailing settings, TTL (max hold)
- `leverage` — confidence-based dynamic leverage (maps signal strength to 1-3x)
- `mtf` — multi-timeframe control (which TF's to monitor, min agreement threshold)
- `strategy` — signal parameters (min_conf, max_spread, RSI gates, volatility scaling, etc.)
- `risk` — daily loss cap, max open positions (locked to 1)
- `intervals` — recalc frequency (indicator refresh, regime refresh)
- `ai` — AI trade gate (Deepseek/GPT gating high-cost entries)
- `dashboard` — web UI on 0.0.0.0:8765 with live candles & trade overlay

**Current mode**: **Swing-Filtered Scalping**
- Primary TF: 15m (scalp entries on swing structure)
- TP: 0.75%, SL: 0.18% (tight, 4.2:1 R/R)
- Breakeven: 0.08% (protect capital immediately)
- Trailing giveback: 8% (squeeze profits)
- Max hold: 20min TTL (no bag holding)
- Leverage: 3x max (tight SL makes it safe)

## Key Modules

| Module | Purpose |
|--------|---------|
| `indicators/signal.py` | Composite signal engine: synthesizes all indicators into [-1, 1] score |
| `indicators/calc.py` | Base indicators (MACD, RSI, EMA, Bollinger, ATR, etc.) |
| `indicators/smc.py` | Smart Money Concepts: support/resistance walls, structure levels |
| `indicators/helpers.py` | Utility functions (Z-Score, volume pressure, etc.) |
| `indicators/mtf.py` | Multi-timeframe aggregation & agreement logic |
| `execution/futures.py` | Live Binance Futures logic (entries, exits, trailing, TP/SL placement) |
| `execution/paper.py` | Paper trading (same logic, no real API calls) |
| `backtest/engine.py` | Historical replay with trade simulation |
| `core/bot_loop.py` | Main orchestration: init → loop → shutdown |
| `core/startup.py` | Logging setup, symbol candidates, bootstrap data |
| `core/singles.py` | Instance locking (port-specific) to prevent duplicate bots |
| `market/data.py` | OHLCV fetching, data caching |
| `safety/gate.py` | Paper-mode gates (prevent overfitting) |
| `safety/sr_wall.py` | Support/resistance wall detection |
| `ai/orchestrator.py` | AI trade gate (Deepseek/OpenAI evaluation of trades) |
| `dashboard/server.py` | Web UI: real-time candles, position monitor, trade overlay |
| `ui/terminal.py` | Terminal UI (ANSI status display) |

## Development Workflows

### Testing Entry Logic
```bash
python tests/check_entry.py  # Verify current position size & PnL
python tests/test_live_keys.py  # Validate API connectivity
```

### Backtesting a Strategy Change
1. Edit `config.yaml` (e.g., change `tp_pct`, `sl_pct`, `timeframe`)
2. Run:
   ```bash
   python backtest/engine.py --config config.yaml --limit 1000 --trades-out bt_results.csv
   ```
3. Check equity curve, win rate, Sharpe, etc. in output

### Paper-Trading Before Live
1. Set `execution.mode: "paper"` in config.yaml, OR run with `EXECUTION_MODE=paper python main.py`
2. Bot trades with simulated capital (`paper_starting_balance_usdt: 50` by default)
3. Trades still logged to `trade_log_futures.csv` — compare paper vs live behavior

### Switching Modes
- **Live**: `execution.mode: "live"` + valid API keys in `.env` → real capital at risk
- **Paper**: `execution.mode: "paper"` → simulated, no live API calls
- **Backtest**: Run `backtest/engine.py` directly (ignores `execution.mode`)

## Signal Debugging

**Inspect signal composition:**
- Signal code: `indicators/signal.py` lines ~200–400 (component scoring)
- MTF logic: `indicators/mtf.py` (agreement threshold for bias)
- Indicator feeds: `indicators/calc.py` (all base calculations)

**Example**: If entry doesn't trigger:
1. Check signal score (printed in logs): `Final signal: 0.042 (BUY)` vs threshold (usually ±0.05)
2. Check ADX filter: if ADX < 15, signals are weakened
3. Check MTF agreement: if 1h & 4h don't agree, may veto entry even on 15m
4. Check guards: spread too wide, RSI at extremes, session blackout, etc.

## Exchange & Account

**Exchange**: Binance Futures (USDM) via ccxt
- **Leverage**: Set in config, executed via `exchange.fapiPrivatePostLeverage()`
- **Order types**: Limit orders (preferred) with 15s TTL; market fallback if timeout
- **Trailing stops**: Native `TRAILING_STOP_MARKET` orders on exchange
- **Position limit**: 1 open position max (enforced in code)
- **Spot co-exit**: Optional grid on spot; futures position controls entry/exit logic

**Risk limits**:
- Daily loss cap (default 10% of initial capital)
- Max open positions: 1
- ATR-based volatility scaling (reduces leverage in high-vol)

## Common Edits

### Change TP/SL
Edit `config.yaml`:
```yaml
strategy:
  tp_pct: 0.0075        # 0.75% target
  sl_pct: 0.0018        # 0.18% max loss
```
Or in `execution/futures.py` defaults (lines 69–70, 146–147).

### Change timeframe
Edit `config.yaml`:
```yaml
timeframe: "15m"        # Primary execution
mtf.timeframes: ["5m", "15m", "1h", "4h"]  # Context stack
```

### Adjust leverage
Edit `config.yaml`:
```yaml
leverage:
  max_leverage: 3
  confidence_levels:
    0.50: 2.0  # 50% confidence → 2x
    0.65: 3.0  # 65% confidence → 3x
```

### Modify trailing stop
Edit `config.yaml`:
```yaml
execution:
  trailing_tp_giveback_pct: 0.08      # 8% giveback (lock in 92% of peak)
  trailing_callback_pct: 0.25         # 0.25% callback on native trailing
```

## AI Integration

**Trade gate** (`ai/orchestrator.py`): Before entering, optionally send signal context to Deepseek or GPT-4o to veto/approve trades. Reduces false signals but adds latency. Enable in config:
```yaml
ai:
  enabled: true
  trade_gate_enabled: true
  model: "deepseek-chat"
```

**AI overlay**: Concurrent analysis displayed on dashboard (not gating, informational).

**Cost**: ~$0.002 per gated trade (Deepseek) or ~$0.01 (GPT-4o). Only gate high-uncertainty trades to control cost.

## Logging & Monitoring

**Log file**: `bot.log` (rotates at 5 MB, keeps 3 backups)
- Levels: DEBUG (signal detail), INFO (trades, position updates), WARNING (gates, rejections), ERROR (crashes)

**Trade log**: `trade_log_futures.csv`
- Columns: entry_price, exit_price, pnl, fees, timestamp, signal_score, confidence, etc.

**Dashboard**: Open browser to `http://localhost:8765` (if enabled in config)
- Live 15m candle with indicators
- Position entry/exit price overlay
- Recent trades table
- Equity curve (paper mode) or account equity (live)

**Terminal UI**: Real-time ANSI status (signal, last trade, drawdown, etc.) — disabled if dashboard is on.

## Troubleshooting

| Issue | Check |
|-------|-------|
| "Another instance already running" | Process still alive on port 45678 (live) or 45679 (paper). Kill it or change `BOT_INSTANCE_PORT`. |
| No API connectivity | Validate `BINANCE_API_KEY` / `BINANCE_SECRET` in `.env`. Test with `tests/test_live_keys.py`. |
| Signal score always 0 | Check if indicators are calculating; verify `max_age_seconds` in MTF (data may be stale). |
| Limit orders never fill | Reduce `pending_entry_ttl_seconds`, or enable `market_fallback_on_timeout: true`. |
| Too many false entries | Raise `min_conf` in strategy, or increase ADX threshold, or tighten MTF agreement. |
| Backtest results don't match live | Backtest uses synthetic/historical fills; account for slippage & spread assumptions. |

## Important Files to Avoid Breaking

- `execution/futures.py`: Position lifecycle, TP/SL placement, trailing logic — changes here affect live capital
- `indicators/signal.py`: Signal composition — small changes can flip win rates
- `core/bot_loop.py`: Main loop — ensures clean shutdown, logging, error recovery
- `backtest/engine.py`: Must mirror live execution exactly (test any core logic changes here first)
