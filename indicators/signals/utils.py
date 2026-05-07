"""Utility functions: volume delta, order book pressure, session blackout."""

import time
import pandas as pd


def _calculate_volume_delta(df: pd.DataFrame) -> float:
    """
    Estimates the 'Aggression' within a candle.
    If price rises on high volume, it implies aggressive buying.
    If price falls on high volume, it implies aggressive selling.
    """
    latest = df.iloc[-1]
    avg_vol = df['volume'].rolling(window=20).mean().iloc[-1]

    vol_rel = latest['volume'] / avg_vol
    price_change = (latest['close'] - latest['open']) / latest['open']

    delta = price_change * vol_rel * 10.0
    return max(min(delta, 1.0), -1.0)


def _map_order_book_pressure(state: dict) -> float:
    """
    Uses real-time Order Book data to find 'The Wall of Money'.
    Bids > Asks = Buying Pressure.
    Asks > Bids = Selling Pressure.
    """
    if not state or 'orderbook' not in state:
        return 0.0

    ob = state['orderbook']
    bids = sum([v for k, v in ob.get('bids', [])[:10]])
    asks = sum([v for k, v in ob.get('asks', [])[:10]])

    if (bids + asks) == 0:
        return 0.0
    pressure = (bids - asks) / (bids + asks)
    return pressure


def _session_blackout_state(strategy_config: dict) -> tuple[bool, int, set[int]]:
    """Return whether the current UTC hour is inside a configured low-quality session."""
    cfg = strategy_config or {}
    enabled = bool(cfg.get("session_filter_enabled", False))
    if not enabled:
        return False, int(time.gmtime().tm_hour), set()

    raw_hours = cfg.get("session_block_hours_utc", []) or []
    blocked_hours: set[int] = set()
    if isinstance(raw_hours, str):
        raw_hours = [part.strip() for part in raw_hours.split(",") if part.strip()]
    for item in raw_hours:
        try:
            blocked_hours.add(int(item) % 24)
        except (TypeError, ValueError):
            continue

    current_hour = int(time.gmtime().tm_hour)
    return current_hour in blocked_hours, current_hour, blocked_hours