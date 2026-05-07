"""
Market state type definitions — TypedDicts for order book / ticker data.

Quick file finder:
  WANT TO KNOW...                   → READ THIS FILE
  ──────────────────────────────────────────────────────
  What keys does state dict have?    → MarketState (below)
  What keys does backtest state add? → BacktestMarketState (below)
  Order book structure for OB pressure → OrderBookData (below)
"""
from __future__ import annotations

from typing import Literal, NotRequired, TypedDict


class OrderBookLevel(TypedDict, total=False):
    """Single [price, volume] level."""
    price: float
    volume: float


class OrderBookData(TypedDict, total=False):
    """Subset of ccxt order book structure used by _map_order_book_pressure."""
    bids: list[list[float]]
    asks: list[list[float]]


class MarketState(TypedDict, total=False):
    """
    Dict returned by MarketData.fetch_order_book_and_ticks().

    Required keys are always present in live mode.
    Optional keys may be None during warmup or absent entirely.
    Backtest-only keys are never present in live mode.

    Live-only keys:  price, bid, ask, bid_vol, ask_vol, spread_pct,
                     ret_5s, ret_30s, vol_60s, volume_state, warmup
    Backtest extras: orderbook, session_open, previous_close,
                     previous_high, previous_low, control_zone, average_zone
    """

    # ── Core price / volume ───────────────────────────────────────
    price: float                    # Last ticker price
    bid: float                      # Best bid price
    ask: float                      # Best ask price
    bid_vol: float                  # Aggregated bid volume across top N levels
    ask_vol: float                  # Aggregated ask volume across top N levels
    spread_pct: float              # (ask - bid) / bid; 0.0 if bid == 0

    # ── Derived short-term metrics ─────────────────────────────────
    ret_5s: float                   # 5-second price return; None during warmup
    ret_30s: float                  # 30-second price return; None during warmup
    vol_60s: float                  # Sum of quoteVolume over last ~60 s
    volume_state: Literal["spike", "normal"]
    warmup: bool                    # True if tick history < 5 s / 30 s deep

    # ── Order book (backtest-only) ────────────────────────────────
    orderbook: OrderBookData        # Used by _map_order_book_pressure;
                                    #  guarded: returns 0.0 if absent

    # ── Session anchors (backtest-only; live falls back to indicators)
    session_open: float             # Day open price
    previous_close: float           # Previous bar close
    previous_high: float            # Previous bar high
    previous_low: float             # Previous bar low
    control_zone: float             # VWAP / control price level
    average_zone: float             # Mean price anchor (VWAP / BB mid)