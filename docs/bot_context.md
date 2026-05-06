# Bot Context

Crypto trading bot. `main.py` is the runtime loop: market data → indicators → AI overlay → order execution. Supports futures, spot, paper.

## Runtime Flow
1. Load config, init market data + indicators + AI orchestrator
2. `generate_quant_signal()` produces actionable signal (indicators.py)
3. `HybridAIOrchestrator` overlays AI veto/bias — rules-first, AI as tiebreaker only
4. `execution_factory.create_executor()` places/manages orders
5. `DashboardRuntime` mirrors state to web UI

## Key Files
| File | Role |
|------|------|
| `main.py` | Entry point, runtime loop, dashboard wiring |
| `indicators.py` | Signal engine: 13 indicators, SMC, MTF |
| `execution.py` | Binance futures/spot/paper: orders, SL/TP, trailing |
| `execution_factory.py` | Chooses executor from config |
| `agents.py` | Multi-provider AI: regime, overlay, post-trade |
| `market_data.py` | Binance OHLCV, order book, funding |
| `ml_optimizer.py` | Logistic regression weight learning from trades |
| `dashboard_server.py` | HTTP dashboard + REST API |
| `news_data.py` | CryptoPanic news |
| `performance_gate.py` | Paper performance gate before live |
| `ai_context.py` | Loads task-specific AI prompts from `ai_context/` |
| `backtest.py` | Historical backtesting |

## Signal Contract
`generate_quant_signal()` returns: `action`, `score`, `confidence`, `reason`, `tp`, `sl`, `entry`, `structure_support`, `structure_resistance`, `hold_until_ts`, `market_bias`

`main.py` augments with: loss-tilt pauses, AI overlay restrictions, structure TP targeting.

## Execution Modes
- `BinanceFuturesExecution` — live futures
- `BinanceSpotExecution` — live/demo spot
- `PaperFuturesExecution` — simulation with bootstrapped candles

## Running
```
.\run_live.ps1   → config.yaml, dashboard :8765, bot port :45678
.\run_demo.ps1   → config.paper.test.yaml, dashboard :8766, bot port :45679
python main.py   → defaults to config.yaml
```
Live and demo must run separately — different ports, configs, trade logs. Use `tmux` for server runs.

## Config (`config.yaml`)
Key groups: `symbol`, `timeframe`, `macro_timeframe`, `data.market`, `execution`, `strategy`, `leverage`, `mtf`, `ai`, `auto_learning`, `dashboard`, `risk`, `spot`

Variant: `config.paper.test.yaml` (paper mode). `config.live.yaml` is deprecated.

## Artifacts
`trade_log_futures.csv`, `trade_log_demo_futures.csv`, `learning_state.json`, `weights.json`, `ui_state.json`, `bot.log`

## Dashboard API (from dashboard_server.py)
`GET /api/state`, `/api/config`, `/api/schema` | `POST /api/settings`, `/api/toggle-pause`, `/api/command`, `/api/emergency-exit`

## Operational Notes
- Single-instance enforcement via port lock
- Exchange SL/TP/trailing orders are position-scoped; stale orders cancelled on flat detection (pending maker entries preserved until TTL)
- Futures trailing is profit-gated: hard stop = emergency loss protection; trailing activates after `profit_trailing_activation_pct`
- `TRAIL_TP` tracks peak profit, closes on `trailing_tp_giveback_pct` giveback; `SCALP_EXIT` only tightens protection
- AI prompts split by task (regime, signal review, execution safety, post-trade). Keep them short.
- Paper gate warns on live failure but doesn't block execution
