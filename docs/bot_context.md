# Bot Context

Last reviewed: 2026-05-05

This repository is a crypto trading bot centered on `main.py`. It fetches market data, builds indicators, generates a trade signal, routes that signal through optional AI overlays, and forwards orders to either futures, spot, or paper execution depending on config.

The bot now also uses a lightweight ICM-style context layout under `ai_context/` so AI prompts can be routed by task instead of being assembled from one large shared prompt.

## Runtime Overview

The current live flow is:

1. `main.py` loads config and environment variables.
2. `MarketData` fetches OHLCV and builds the initial candle cache.
3. `calculate_base_indicators()` computes the indicator set used by the strategy and dashboard.
4. `build_mtf_timeframe_context()` prepares higher-timeframe confirmation data.
5. `generate_quant_signal()` returns the actionable signal.
6. Optional AI layers adjust or veto entries through `HybridAIOrchestrator`.
7. The selected executor places or manages orders.
8. `DashboardRuntime` mirrors state to the local web dashboard.

## Key Entry Points

- [`main.py`](../main.py): startup, runtime loop, dashboard wiring, AI overlay, signal-to-order handoff.
- [`execution_factory.py`](../execution_factory.py): chooses spot, futures, or paper execution and applies common settings.
- [`execution.py`](../execution.py): order management, trade logging, stops, take profit, trailing logic, paper mode.
- [`indicators.py`](../indicators.py): indicator calculation, multi-timeframe context, and signal generation.
- [`market_data.py`](../market_data.py): Binance market-data fetcher and OHLCV resampling.
- [`agents.py`](../agents.py): AI orchestration for macro regime, trade evaluation, and overlay bias.
- [`news_data.py`](../news_data.py): crypto news fetcher used for macro regime reads.
- [`dashboard_server.py`](../dashboard_server.py): local HTTP dashboard and settings API.
- [`performance_gate.py`](../performance_gate.py): paper-performance gate used before live runtime continues.
- [`ml_optimizer.py`](../ml_optimizer.py): weight learning from completed trades.
- [`ai_context.py`](../ai_context.py): loads compact workspace prompts for the AI layer.
- [`ai_context/`](../ai_context/): compact prompt workspaces and shared AI conventions.

## Strategy And Signal Contract

`generate_quant_signal()` is the core strategy function. The bot expects the signal object to keep these fields stable:

- `action`
- `score`
- `confidence`
- `reason`
- `tp`
- `sl`
- `entry`
- `structure_support`
- `structure_resistance`
- `hold_until_ts`
- `market_bias`

`main.py` also augments the signal with runtime controls such as loss-tilt pauses, AI overlay restrictions, and structural take-profit targeting.

## Execution Modes

The bot supports three execution paths:

- Live futures via `BinanceFuturesExecution`
- Live or demo spot via `BinanceSpotExecution`
- Paper simulation via `PaperFuturesExecution`

`execution_factory.create_executor()` resolves the correct executor from `config.yaml` and fills in shared risk and order-management settings.

## Configuration

The active default config is [`config.yaml`](../config.yaml). Related variants in the repo include:

- [`config.live.yaml`](../config.live.yaml): live-oriented config
- [`config.paper.test.yaml`](../config.paper.test.yaml): paper-testing config
- [`dashboard_overrides.yaml`](../dashboard_overrides.yaml): persisted dashboard overrides

Important config groups:

- `symbol`, `timeframe`, `macro_timeframe`
- `data.market`
- `execution`
- `strategy`
- `leverage`
- `mtf`
- `ai`
- `auto_learning`
- `dashboard`
- `risk`
- `spot`

## State, Logs, And Artifacts

- `trade_log_futures.csv` and `trade_log.csv` store execution history.
- `learning_state.json` persists auto-learning state.
- `weights.json` stores learned indicator weights.
- `ui_state.json` stores dashboard UI overrides.
- `bot.log` is the main rotating application log.
- `dashboard_debug.log` captures dashboard-side diagnostics.

## Dashboard

The dashboard server runs from `dashboard_server.py` and is started by `main.py` when `dashboard.enabled` is true. It exposes:

- `GET /api/state`
- `GET /api/config`
- `GET /api/schema`
- `POST /api/settings`
- `POST /api/toggle-pause`
- `POST /api/command`
- `POST /api/emergency-exit`

Default config uses port `8765`.

## Operational Notes

- `main.py` enforces a single running instance.
- The bot can fall back between spot and futures market-data sources if one source fails.
- Paper mode can bootstrap synthetic candles if historical fetch fails at startup.
- Auto-learning can refresh weights from recent completed trades and optionally call an AI advisor.
- Live mode includes a paper-performance gate, but the current runtime only warns when the gate fails.
- AI prompts are intentionally split by task: regime, signal review, execution safety, and post-trade learning.
- Keep these prompts short. The savings come from narrower contexts and fewer repeated instructions, not from the folder structure alone.

## Latest Runtime Context

- Use [`run_live.ps1`](../run_live.ps1) for the live bot and [`run_demo.ps1`](../run_demo.ps1) for the demo/paper bot.
- Live and demo must run separately. Do not reuse the same process, port, or session for both.
- `run_live.ps1` explicitly loads `config.yaml`.
- `run_live.ps1` starts the live bot on dashboard port `8765` and isolates it with `BOT_INSTANCE_PORT=45678`.
- `run_demo.ps1` starts the demo bot on dashboard port `8766` and isolates it with `BOT_INSTANCE_PORT=45679`.
- `run_live.ps1` now launches `main.py` directly instead of accidentally opening the Python REPL.
- `run_demo.ps1` uses `config.paper.test.yaml` and keeps paper trade logs separate from live logs.
- `config.yaml` remains the fallback/default profile for direct `python main.py` runs.
- `config.live.yaml` is kept only as a deprecated reference copy and is not used by the live launcher.
- If the bot is being run on the server, prefer a dedicated `tmux` session for the live process so it can be reattached later.
- Futures execution now treats exchange-side SL/TP/trailing orders as position-scoped protection. When a position is closed by the bot, by reversal banking, or detected flat on the exchange, stale reduce-only conditional orders are cancelled before the bot opens the next fresh position.
- A pending maker entry is preserved while flat until its TTL expires; only stale protection orders are cleared during that state.
- Runner exits can now scale out part of a futures position before handing the remainder back to the trailing-stop stack.
- Futures trailing is profit-gated: the hard stop remains as emergency loss protection, and local/native trailing arms only after `profit_trailing_activation_pct`.
- `TRAIL_TP` is the primary profit exit when enabled: it tracks peak profit and closes on `trailing_tp_giveback_pct` giveback; `SCALP_EXIT` only tightens protection in this mode.

## Files Worth Reading First

1. [`main.py`](../main.py)
2. [`indicators.py`](../indicators.py)
3. [`execution_factory.py`](../execution_factory.py)
4. [`execution.py`](../execution.py)
5. [`dashboard_server.py`](../dashboard_server.py)
6. [`agents.py`](../agents.py)
7. [`market_data.py`](../market_data.py)
