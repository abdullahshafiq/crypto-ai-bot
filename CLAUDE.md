# Claude Routing Rules

Primary objective: minimize token usage.

## Read First

Read `AGENTS.md` first. It is the canonical architecture map.

## Tool Order

1. CodeGraph: affected modules/files only, max 5 files.
2. Serena: exact symbols/functions/references only.
3. RTK: compressed logs/tests/git output only.
4. Full file read: last resort.

## Hard Rules

- Do not explore the repo from scratch.
- Do not read full logs.
- Do not read full CSV/backtest dumps.
- Do not read generated/cache files.
- Do not inspect more than 3 files before producing a reason.
- Do not edit the same function more than 2 times without new evidence.
- Do not change live trading, execution, strategy, signal, safety, or risk logic without explicit user approval.

## Running

```bash
python main.py              # live (config.yaml)
python main.py paper         # paper mode
python backtest/engine.py --config config.yaml --limit 1000  # backtest
```

## Premium Model Use

Use Claude/Codex only for:
- trading/risk/strategy decisions
- live execution/order logic
- multi-file risky edits
- final patch review

Cheap agents should handle:
- file discovery
- log/test summaries
- docs cleanup
- simple handoff writing
