"""
Signal generation package — produces the final BUY/SELL/HOLD signal.

Quick file finder — edit the file next to the thing you want to change:

  WANT TO CHANGE...                          → EDIT THIS FILE
  ─────────────────────────────────────────────────────────────────────
  Overall signal flow / phase order           → engine.py
  Spread, chase, ATR, ADX, session guards    → gates/guards.py
  SR wall veto, range position, entry mode    → gates/walls.py
  Midrange policy                            → gates/walls.py
  Wall rejection rescue                       → gates/walls.py
  Strike zone / OB / trend confirmation gate  → gates/confirmation.py
  Rejection confirmation gate                 → gates/confirmation.py
  Range reversal sniper                       → gates/sniper.py
  Exhaustion / divergence gate                → gates/sniper.py
  MTF trend veto                              → gates/sniper.py
  Trend continuation bias, range zones        → gates/bias.py
  Indicator scores (SMC, MR, VWAP, BB...)    → scores.py
  MTF fast-score / RSI bias                   → mtf_bias.py
  MACD / CVD divergence                       → divergence.py
  Mean reversion setup                        → setups.py
  Wick sweep setup                            → setups.py
  VWAP bounce setup                           → setups.py
  ORB breakout setup                          → setups.py
  Article setup overrides (ORB+VWAP in flow)  → trend.py
  PSAR/MACD/MTF trend confirmation logic     → trend.py
  Signal dict construction                    → builder.py
  Stop-loss / take-profit / reward-risk       → stops.py
  Score synthesis, action determination       → synthesis.py
  Core context building (support, location)   → context.py
  Alpha overlay / signal integrity / pivots   → alpha.py
  Volume delta / OB pressure / session        → utils.py
"""

from .engine import generate_quant_signal
from .alpha import generate_alpha_overlay, validate_signal_integrity, compute_advanced_pivots
from .gates import _apply_rejection_confirmation_gate
from .signal_types import QuantSignal, RejectionConfirmation, MarketLocationInfo, SetupInfo

__all__ = [
    "generate_quant_signal",
    "validate_signal_integrity",
    "generate_alpha_overlay",
    "_apply_rejection_confirmation_gate",
    "compute_advanced_pivots",
    "QuantSignal",
    "RejectionConfirmation",
    "MarketLocationInfo",
    "SetupInfo",
]