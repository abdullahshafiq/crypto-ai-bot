"""
Signal output type definitions — TypedDicts for generate_quant_signal return value.

Quick file finder:
  WANT TO KNOW...                       → READ THIS FILE
  ────────────────────────────────────────────────────────────
  What keys does the signal dict have?    → QuantSignal (below)
  Rejection confirmation sub-structure?   → RejectionConfirmation (below)
  Market location sub-structure?          → MarketLocationInfo (below)
  Setup info sub-structure?              → SetupInfo (below)
"""
from __future__ import annotations

from typing import NotRequired, TypedDict


class RejectionConfirmation(TypedDict, total=False):
    """Rejection confirmation gate result — set by gates/confirmation.py."""
    confirmed: bool               # True = confirmation passed
    mode: str                     # "REVERSAL_SHORT" | "REVERSAL_LONG" | etc.
    action: str                   # "BUY" | "SELL"
    reason: str                   # Human-readable explanation
    bars_away: int                 # Bars since rejection
    pullback_pct: float           # Pullback from rejection level


class MarketLocationInfo(TypedDict, total=False):
    """Market location score breakdown — set by context.py."""
    score: float                  # Overall location score
    notes: str                    # Human-readable notes
    levels: dict                  # Support/resistance levels dict


class SetupInfo(TypedDict, total=False):
    """Pattern setup details — set by setups.py."""
    direction: str                # "LONG" | "SHORT"
    reason: str                   # Setup-specific reason
    entry_offset_pct: float       # Offset from current price
    sl_source: str                # "wick_sweep" | "mean_reversion" | etc.


class QuantSignal(TypedDict):
    """
    Complete signal dict returned by generate_quant_signal().

    Constructed in builder._build_signal_dict(), then mutated
    by stops, gates, and bot_loop post-processing.
    """

    # ── Action / score (always present) ──────────────────────────
    action: str                    # "BUY" | "SELL" | "HOLD"
    score: float                   # Composite score (neg=short, pos=long)
    confidence: float              # min(abs(score), 1.0)
    reason: str                    # Human-readable reason string
    hold_reason: str               # Empty string when action != HOLD

    # ── Stops (set for BUY/SELL by _compute_sl_tp) ───────────────
    sl: NotRequired[float]         # Stop-loss price
    sl_source: NotRequired[str]    # "wick_sweep" | "pivot_r1_guard" | etc.
    tp: NotRequired[float]        # Take-profit price
    structural_sl_pct: NotRequired[float]  # SL distance as % of current price
    reward_risk: NotRequired[float]         # reward:risk ratio
    tp_target: NotRequired[float]          # Structural TP target (set by bot_loop)

    # ── Gates / locks ────────────────────────────────────────────
    sr_wall_locked: NotRequired[bool]       # True when SR wall veto is active
    rejection_confirmation: NotRequired[RejectionConfirmation]
    gate_trace: NotRequired[str]            # Gate notes (set by bot_loop on HOLD)
    hold_until_ts: NotRequired[float]       # UNIX timestamp (set by AI overlay)

    # ── PSAR state ───────────────────────────────────────────────
    psar_streak: NotRequired[int]           # Consecutive SAR bars
    psar_exit: NotRequired[bool]           # True when psar_streak != 0
    psar_closed_bull: NotRequired[bool]    # Previous candle PSAR bullish
    psar_live_bull: NotRequired[bool]      # Current PSAR bullish
    psar_state_note: NotRequired[str]      # Descriptive note

    # ── MTF / market bias ────────────────────────────────────────
    market_bias: NotRequired[str]           # "LONG_ONLY" | "SHORT_ONLY" | "NEUTRAL"
    mtf_fast_bias: NotRequired[str]        # Fast TF bias
    mtf_rsi_bias: NotRequired[str]         # RSI-based bias
    mtf_rsi_score: NotRequired[float]      # RSI contribution score

    # ── Momentum / exhaustion ─────────────────────────────────────
    momentum_exhaustion: NotRequired[str]   # "BULL_EXHAUST" | "BEAR_EXHAUST" | ""
    cvd_state: NotRequired[str]             # CVD label
    body_ratio_score: NotRequired[float]   # Body-to-range ratio score

    # ── Order flow / volume ──────────────────────────────────────
    vpoc: NotRequired[float]               # Volume point of control price
    anchored_vwap: NotRequired[float]      # Anchored VWAP price
    order_block: NotRequired[dict]         # OB structure dict
    atr: NotRequired[float]                # ATR value
    atr_pct: NotRequired[float]            # ATR as % of price

    # ── Market location ───────────────────────────────────────────
    market_location: NotRequired[MarketLocationInfo]

    # ── SR structure ─────────────────────────────────────────────
    structure_support: NotRequired[float]   # Nearest support level
    structure_resistance: NotRequired[float]  # Nearest resistance level
    wall_state: NotRequired[dict]          # Wall state dict from context.py

    # ── Pivots ───────────────────────────────────────────────────
    pivot_classic: NotRequired[dict]        # Classic pivot levels dict

    # ── Setup overrides (set by stops.py) ─────────────────────────
    mean_reversion: NotRequired[SetupInfo]  # MR setup details
    wick_sweep: NotRequired[SetupInfo]     # Wick sweep setup details
    entry: NotRequired[float]              # Entry price override

    # ── Weights (only in "Warming Up" early return) ───────────────
    weights: NotRequired[dict]             # Indicator weights dict

    # ── Midrange policy (set by gates/walls.py) ───────────────────
    action_support: NotRequired[float]      # Support level for midrange
    action_resistance: NotRequired[float]   # Resistance level for midrange
    top_action_zone: NotRequired[float]     # Top of action zone
    bottom_action_zone: NotRequired[float]  # Bottom of action zone