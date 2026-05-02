# Spot/Futures Coupling Map

This document captures the coupling that was present before the split so refactors can stay behavior-safe.

## Runtime Coupling

- `main.py` directly instantiated futures executors and set many mutable attributes one-by-one.
- `main.py` also performed runtime mode switching by importing `execution` inline.
- `execution.market` existed in config but was not used to select spot vs futures execution.

## Strategy/Execution Coupling

- `indicators.generate_quant_signal()` returns fields consumed implicitly by execution:
  - `action`, `score`, `confidence`, `sl`, `tp`, `structure_support`, `structure_resistance`, `hold_until_ts`.
- Execution logic assumes these keys exist and silently degrades if they do not.

## Data/Strategy Coupling

- Decisions were made from the newest indicator row, which can be an open candle.
- Tick-level polling (`1s`) and 5m strategy timeframe caused multiple decisions per candle.

## Logging Coupling

- Spot and futures were sharing the same trade log path and schema writer path.
- Analytics tools consuming logs depend on the existing CSV columns.

## Refactor Safety Rules

- Keep the existing trade-log column schema unchanged.
- Keep signal field names unchanged.
- Move wiring first, then enforce stronger guards and closed-candle logic.
