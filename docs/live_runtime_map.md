# Live Runtime Map

The current entrypoint is still `main.py`, but new edits should use these boundaries:

- `bot/execution/`: live exchange adapters.
- `bot/strategy/`: indicator and signal generation surface.
- `bot/integrations/`: market data, news, and AI providers.
- `execution/factory.py`: executor selection and shared live settings.
- `config.yaml`: active local live configuration.
- `config.live.yaml`: deployment-oriented live configuration.

Paper execution remains in `execution/futures/_entry.py` and `execution/paper.py`, but the active runtime now refuses non-live modes and no longer falls back to paper when keys/equity checks fail.
