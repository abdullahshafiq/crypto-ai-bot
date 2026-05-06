import logging
from logging.handlers import RotatingFileHandler
import pandas as pd
import time
import os

from .synthetic_data import _fallback_bootstrap_ohlcv


def setup_logging(logging_cfg: dict | None = None):
    # Only log to file so we don't mess up the clean terminal dashboard
    logging_cfg = logging_cfg or {}
    max_mb = float(logging_cfg.get("max_mb", 5))
    backups = int(logging_cfg.get("backups", 3))
    max_bytes = int(max_mb * 1024 * 1024)
    if max_bytes <= 0:
        max_bytes = 5 * 1024 * 1024
    if backups < 1:
        backups = 1

    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            RotatingFileHandler("bot.log", maxBytes=max_bytes, backupCount=backups, encoding="utf-8")
        ]
    )
_NETWORK_FAIL_UNTIL = 0.0
_NETWORK_COOLDOWN_SECONDS = 60.0

def _runtime_fetch_ohlcv(market, symbol: str, timeframe: str, limit: int, *, paper_mode: bool, logger=None) -> pd.DataFrame:
    """
    Fetch OHLCV for runtime use. In paper mode, fall back to a synthetic bootstrap
    when Binance market data is unreachable so the demo can still run locally.
    """
    global _NETWORK_FAIL_UNTIL
    now = time.time()
    if now < _NETWORK_FAIL_UNTIL:
        if paper_mode:
            return _fallback_bootstrap_ohlcv(symbol, timeframe, limit=limit)
        return pd.DataFrame()

    try:
        df = market.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if df is not None and not df.empty:
            _NETWORK_FAIL_UNTIL = 0.0
            return df
    except Exception as e:
        if logger:
            logger.debug(f"Runtime OHLCV fetch failed for {symbol} {timeframe}: {e}")

    _NETWORK_FAIL_UNTIL = now + _NETWORK_COOLDOWN_SECONDS
    if paper_mode:
        if logger:
            logger.warning(f"Paper runtime falling back to synthetic OHLCV for {symbol} {timeframe}.")
        return _fallback_bootstrap_ohlcv(symbol, timeframe, limit=limit)
    return pd.DataFrame()


def _startup_symbol_candidates(symbol: str) -> list[str]:
    symbol = str(symbol or "").strip()
    if not symbol:
        return []

    candidates = [symbol]
    if "/USDC:USDC" in symbol:
        candidates.append(symbol.replace("/USDC:USDC", "/USDT:USDT"))
    elif symbol.endswith("/USDC"):
        candidates.append(symbol[:-5] + "/USDT")

    seen = set()
    ordered = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            ordered.append(candidate)
            seen.add(candidate)
    return ordered


def _reapply_runtime_executor_config(executor, cfg):
    exec_cfg = cfg.get("execution", {}) or {}
    executor.max_open_positions = cfg["risk"].get("max_open_positions", 1)
    executor.daily_loss_cap_pct = cfg["risk"].get("daily_loss_cap")
    executor.min_balance_floor = float(cfg["risk"].get("min_balance_floor", 90.0))
    leverage_cfg = cfg.get("leverage", {}) or {}
    executor.dynamic_leverage_enabled = bool(leverage_cfg.get("enabled", False))
    executor.leverage_min = float(leverage_cfg.get("min_leverage", 1.0))
    executor.leverage_max = float(leverage_cfg.get("max_leverage", 4.0))
    executor.leverage_use_score = bool(leverage_cfg.get("use_score_multiplier", False))
    executor.leverage_score_weight = float(leverage_cfg.get("score_weight", 0.3))
    executor.atr_volatility_scaling = bool(leverage_cfg.get("atr_volatility_scaling", False))
    executor.atr_reference_pct = float(leverage_cfg.get("atr_reference_pct", 0.02))
    executor.atr_min_multiplier = float(leverage_cfg.get("atr_min_multiplier", 0.3))
    conf_levels = leverage_cfg.get("confidence_levels", {})
    executor.leverage_confidence_levels = {float(k): float(v) for k, v in conf_levels.items()}
    executor.dca_enabled = bool(exec_cfg.get("dca_enabled", False))
    executor.dca_max_steps = int(exec_cfg.get("dca_max_steps", 0))
    executor.dca_distance_pct = float(exec_cfg.get("dca_distance_pct", 0.01))
