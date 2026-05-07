"""
Gate functions — each one reads/writes the shared `ctx` dict and returns either
an early-exit signal dict (for guard functions) or None (for mutate-in-place gates).

  WANT TO CHANGE...                          → EDIT THIS FILE
  ───────────────────────────────────────────────────────────
  Spread threshold                          → guards.py  apply_spread_guard
  Chase detection                           → guards.py  apply_chasing_guard
  ATR minimum filter                        → guards.py  apply_atr_guard
  ADX range filter                          → guards.py  apply_adx_range_filter
  Low-vol minimum score                     → guards.py  apply_low_vol_min_score
  Session blackout hours                    → guards.py  apply_session_blackout
  SR wall veto                              → walls.py   apply_sr_wall_veto
  Range position veto                       → walls.py   apply_range_position_veto
  Entry mode classification                 → walls.py   classify_entry_mode_and_walls
  Wall rejection rescue                     → walls.py   apply_wall_rejection_rescue
  Midrange policy                           → walls.py   apply_midrange_policy
  Strike zone check                         → confirmation.py  apply_strike_zone_check
  Order block gate                          → confirmation.py  apply_ob_gate
  Trend confirmation (2-of-3)               → confirmation.py  apply_trend_confirmation_gate
  Rejection confirmation gate               → confirmation.py  _apply_rejection_confirmation_gate
  Range reversal sniper (floor/ceiling)     → sniper.py  apply_range_reversal_sniper
  Exhaustion/divergence hard gate           → sniper.py  apply_exhaustion_divergence_gate
  MTF trend veto                           → sniper.py  apply_mtf_trend_veto
  Trend continuation bias nudge             → bias.py    apply_trend_continuation_bias
  Range zone computation                    → bias.py    compute_range_zones
"""

from .guards import (
    apply_spread_guard, apply_chasing_guard, apply_atr_guard,
    apply_adx_range_filter, apply_low_vol_min_score, apply_session_blackout,
)
from .walls import (
    apply_sr_wall_veto, apply_range_position_veto,
    classify_entry_mode_and_walls, apply_wall_rejection_rescue,
    apply_midrange_policy,
)
from .confirmation import (
    apply_strike_zone_check, apply_ob_gate,
    apply_trend_confirmation_gate, _apply_rejection_confirmation_gate,
)
from .sniper import (
    apply_range_reversal_sniper, apply_exhaustion_divergence_gate,
    apply_mtf_trend_veto,
)
from .bias import apply_trend_continuation_bias, compute_range_zones

__all__ = [
    "apply_spread_guard", "apply_chasing_guard", "apply_atr_guard",
    "apply_adx_range_filter", "apply_low_vol_min_score", "apply_session_blackout",
    "apply_sr_wall_veto", "apply_range_position_veto",
    "classify_entry_mode_and_walls", "apply_wall_rejection_rescue",
    "apply_midrange_policy",
    "apply_strike_zone_check", "apply_ob_gate",
    "apply_trend_confirmation_gate", "_apply_rejection_confirmation_gate",
    "apply_range_reversal_sniper", "apply_exhaustion_divergence_gate",
    "apply_mtf_trend_veto",
    "apply_trend_continuation_bias", "compute_range_zones",
]