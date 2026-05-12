# Agent Context — Crypto AI Bot

> This file is read first by AI agents. It maps the entire codebase so agents
> never need to grep for "where is X defined?" or "what keys does Y have?".

## Canonical Agent Map

This file is the canonical architecture and agent-routing map for this repo.

Before using Claude, Codex, or any high-cost model:
1. Read this file first.
2. Use CodeGraph to identify affected modules.
3. Use Serena to inspect exact symbols.
4. Use RTK or a cheap log agent for command/test output.
5. Send premium models only a short handoff, not full repo context.

## Architecture

```
main.py → core/bot_loop.py (runtime loop)
  ├─ core/signal_gates.py      → loss tilt, scalp hold, confidence floor, gate trace
  ├─ core/ai_gates.py          → AI overlay gate, AI trade gate, regime veto, entry dispatch
  ├─ core/singles.py           → single instance lock, consecutive losses
  ├─ core/snapshot.py          → dashboard snapshot builder
  ├─ core/startup.py           → config init, OHLCV fetch, runtime config
  ├─ core/synthetic_data.py    → offline demo OHLCV fallback
  ├─ market/data.py          → OHLCV, order book, funding rate
  ├─ indicators/calc.py      → base indicator calculation
  ├─ indicators/signals/     → signal generation pipeline
  │   ├─ engine.py           → orchestrator (17 phases)
  │   ├─ context.py           → Phase 1: core context building
  │   ├─ synthesis.py         → Phase 6-7: score synthesis + action
  │   ├─ trend.py             → Phase 8: PSAR/MACD/MTF trend
  │   ├─ builder.py           → Phase 9: signal dict construction
  │   ├─ stops.py             → Phase 10-11: setup overrides + SL/TP
  │   ├─ scores.py            → indicator scoring (SMC, MR, VWAP, BB)
  │   ├─ setups.py            → ORB, VWAP, wick sweep, mean reversion
  │   ├─ mtf_bias.py          → MTF fast-score / RSI bias
  │   ├─ divergence.py        → MACD + CVD divergence
  │   ├─ alpha.py             → alpha overlay, integrity, pivots
  │   ├─ utils.py             → volume delta, OB pressure, session
  │   ├─ ctx.py               → SignalContext TypedDict (75+ keys)
  │   └─ gates/
  │       ├─ guards.py        → spread, chase, ATR, ADX, low-vol, session
  │       ├─ walls.py         → SR wall, range, entry mode, midrange
  │       ├─ confirmation.py  → strike zone, OB, trend confirmation
  │       ├─ sniper.py        → range reversal, exhaustion, MTF veto
  │       └─ bias.py           → trend continuation bias, range zones
  ├─ ai/orchestrator.py      → AI regime, overlay, trade gate
  ├─ execution/              → order placement & management
  │   ├─ futures/            → Binance USDⓈ-M futures (mixin package)
  │   │   ├─ _core.py        → class definition + __init__ + calculate_dynamic_leverage
  │   │   ├─ _trade_log.py   → trade recording, session, observe, logging
  │   │   ├─ _balance.py     → USDT/BTC balance fetch
  │   │   ├─ _orders.py      → order cancel, fetch, wipe orphans, cleanup
  │   │   ├─ _protection.py  → trailing stop, exchange SL/TP, cleanup orders
  │   │   ├─ _exit.py        → reduce-only exit, finalize, diagnostics, emergency
  │   │   ├─ _entry.py       → place_limit_order
  │   │   ├─ _position.py    → process_orders_and_positions (main loop)
  │   │   └─ _portfolio.py   → open orders, portfolio value, risk, close_all
  │   ├─ paper.py             → paper trading simulation
  │   ├─ spot.py              → Binance spot
  │   ├─ base.py              → base executor class
  │   └─ factory.py           → executor creation from config
  ├─ dashboard/server.py      → HTTP dashboard + REST API
  ├─ ml/optimizer.py          → weight learning from trades
  ├─ safety/gate.py           → paper performance gate
  ├─ safety/sr_wall.py        → SR wall escape detection
  └─ ui/terminal.py           → ANSI terminal dashboard
```

## TypedDict References

| Dict | TypedDict | File |
|------|-----------|------|
| `ctx` (shared mutable state across all 17 signal phases) | `SignalContext` | `indicators/signals/ctx.py` |
| `state` (order book / ticker data from `fetch_order_book_and_ticks`) | `MarketState` | `market/state_types.py` |
| `signal` (return value of `generate_quant_signal()`) | `QuantSignal` | `indicators/signals/signal_types.py` |
| `cfg` (YAML config dict loaded by `load_config()`) | `BotConfig` | `config/schema.py` |
| `strategy_config` (runtime copy of `cfg["strategy"]`) | `StrategyConfig` | `config/schema.py` |

## Edit Recipes

| WANT TO CHANGE... | EDIT THIS FILE |
|-------------------|----------------|
| Signal flow / phase order | `indicators/signals/engine.py` |
| Spread, chase, ATR guards | `indicators/signals/gates/guards.py` |
| SR wall veto, range position | `indicators/signals/gates/walls.py` |
| Strike zone / OB confirmation | `indicators/signals/gates/confirmation.py` |
| Range reversal, exhaustion | `indicators/signals/gates/sniper.py` |
| Trend continuation bias | `indicators/signals/gates/bias.py` |
| Indicator scores (SMC, MR, VWAP) | `indicators/signals/scores.py` |
| ORB, wick sweep, mean reversion | `indicators/signals/setups.py` |
| PSAR/MACD trend, article setups | `indicators/signals/trend.py` |
| Signal dict keys/construction | `indicators/signals/builder.py` |
| Stop-loss / take-profit | `indicators/signals/stops.py` |
| Add a new ctx key | `indicators/signals/ctx.py` |
| MTF fast-score / RSI bias | `indicators/signals/mtf_bias.py` |
| MACD + CVD divergence | `indicators/signals/divergence.py` |
| Alpha overlay / signal integrity | `indicators/signals/alpha.py` |
| Volume delta / OB pressure | `indicators/signals/utils.py` |
| Core context (support, location) | `indicators/signals/context.py` |
| Base indicators (EMAs, RSI, etc.) | `indicators/calc.py` |
| MTF timeframe context | `indicators/mtf.py` |
| SMC & S/R detection | `indicators/smc.py` |
| Market location / VWAP / CVD | `indicators/location.py` |
| Order book data fetching | `market/data.py` |
| News data fetching | `market/news.py` |
| AI regime / overlay / trade gate | `ai/orchestrator.py` |
| AI prompt templates | `ai/context.py` |
| Order placement, SL/TP, trailing | `execution/futures/_entry.py` |
| Reduce-only exit, finalize | `execution/futures/_exit.py` |
| Order management (cancel, fetch, wipe) | `execution/futures/_orders.py` |
| Trailing stop, exchange SL/TP | `execution/futures/_protection.py` |
| Position loop (main tick) | `execution/futures/_position.py` |
| Trade recording, logging | `execution/futures/_trade_log.py` |
| Balance fetching | `execution/futures/_balance.py` |
| Portfolio, risk limits, close_all | `execution/futures/_portfolio.py` |
| Futures class __init__ | `execution/futures/_core.py` |
| Dynamic leverage | `execution/futures/_core.py` |
| Paper trading simulation | `execution/paper.py` |
| Spot trading | `execution/spot.py` |
| Executor base class | `execution/base.py` |
| Executor creation from config | `execution/factory.py` |
| Web dashboard + REST API | `dashboard/server.py` |
| Weight learning from trades | `ml/optimizer.py` |
| Paper performance gate | `safety/gate.py` |
| SR wall escape detection | `safety/sr_wall.py` |
| Terminal UI rendering | `ui/terminal.py` |
| Main runtime loop | `core/bot_loop.py` |
| Loss tilt computation, scalp hold guard | `core/signal_gates.py` |
| AI overlay gate, AI trade gate, regime veto | `core/ai_gates.py` |
| Config loading | `config/loader.py` |
| Config schema (all keys) | `config/schema.py` |

## Running

```
.\run_live.ps1   → config.yaml, dashboard :8080, bot port :45678
.\run_demo.ps1   → config.paper.test.yaml, dashboard :8766, bot port :45679
python main.py    → defaults to config.yaml
BOT_CONFIG=other.yaml python main.py  → custom config
```

## Key Patterns

- **`ctx` dict**: all 17 signal phases read/write a shared `SignalContext` dict. Defined in `ctx.py`.
- **`state` dict**: returned by `market.fetch_order_book_and_ticks()`. Defined in `market/state_types.py`.
- **`signal` dict**: returned by `generate_quant_signal()`. Defined in `indicators/signals/signal_types.py`.
- **Config**: loaded by `config/loader.py`, schema in `config/schema.py`. Runtime defaults applied in `core/bot_loop.py`.
- **Futures mixins**: `BinanceFuturesExecution` inherits from 8 mixin classes defined in `execution/futures/_*.py`. The main class definition is in `_core.py`. All public API names stay identical.
- **`from __future__ import annotations`**: required in all files importing TypedDicts from `ctx.py`, `state_types.py`, `signal_types.py`, or `schema.py` to avoid circular import issues.

→ [`docs/bot_context.md`](docs/bot_context.md) for API details and operational notes.
