"""
Configuration type definitions — TypedDicts mirrors of config.yaml structure.

Quick file finder:
  WANT TO KNOW...                        → READ THIS SECTION
  ───────────────────────────────────────────────────────────
  What keys does the config dict have?    → BotConfig (top-level)
  Execution/trading settings?              → ExecutionConfig
  Strategy/gate parameters?               → StrategyConfig
  Dynamic leverage?                        → LeverageConfig
  Risk limits?                             → RiskConfig
  MTF settings?                            → MTFConfig
  AI overlay settings?                     → AIConfig / AITradeConfig
  Intervals/delays?                        → IntervalsConfig
  Dashboard settings?                      → DashboardConfig
  Auto-learning?                           → AutoLearningConfig
  Rejection confirmation?                  → RejectionConfirmationConfig
  Spot/grid?                               → SpotConfig

Keys marked YAML-only exist in config.yaml but are not read at runtime.
Keys marked factory have defaults in execution/ rather than bot_loop defaulting.
"""
from __future__ import annotations

from typing import Literal, NotRequired, TypedDict


class DataConfig(TypedDict, total=False):
    market: str  # "auto" | "usdm" | "spot"


class ExecutionConfig(TypedDict, total=False):
    mode: str                                    # "live" | "paper"
    market: str                                  # "usdm" | "spot"
    leverage: int                                # Default 5
    paused: bool                                 # Default False
    close_positions_on_pause: bool               # Default False
    panic_exit: bool                             # Default False

    # Paper-specific
    paper_starting_balance_usdt: float           # Default 1000
    paper_futures_mode: str                      # YAML-only

    # Order management
    use_limit_orders: bool                       # Default True (factory)
    use_native_trailing_stop: bool               # Default False (factory)
    use_exchange_stop_loss: bool                 # Default True (factory)
    use_exchange_take_profit: bool               # Default True (factory)
    market_fallback_on_timeout: bool             # Default False (factory)

    # Prices & fees
    fee_rate: float                              # Default 0.0006
    fee_slippage_buffer_pct: float               # Default 0.0002 (factory)
    fee_edge_multiplier: float                   # Default 1.2 (factory)

    # Timing
    min_seconds_between_trades: int              # Default 60
    min_seconds_before_reversal: int              # Default 60
    ttl_exit_seconds: int                        # Default 0

    # Reversal
    reversal_min_confidence: float                # Default 0.45
    reversal_min_score: float                     # Default 0.25
    reversal_min_net_edge_pct: float             # Default 0.0015

    # Break-even & trailing
    break_even_trigger_pct: float                 # Default 0.0030
    break_even_buffer_pct: float                  # Default 0.0004
    profit_trailing_enabled: bool                 # Default True
    profit_trailing_activation_pct: float        # Default = break_even_trigger_pct
    trailing_tp_enabled: bool                     # Default True
    trailing_tp_giveback_pct: float               # Default 0.12
    trailing_tp_min_peak_pct: float               # Default = profit_trailing_activation_pct
    trail_tighten_1_pct: float                    # Default 0.0050
    trail_tighten_2_pct: float                    # Default 0.0100
    trail_t1_gap_pct: float                       # Default 0.0025
    trail_t2_gap_pct: float                       # Default 0.0020
    trailing_callback_pct: float                  # Default 0.6

    # Profit/exit gates
    min_profit_after_fees: float                  # Default 0.0012
    exit_on_reversal_only_in_profit: bool          # Default True

    # Pending / resting orders
    pending_entry_ttl_seconds: int                # Default 20
    use_resting_support_orders: bool              # YAML-only
    resting_entry_ttl_seconds: int                # Default 120
    resting_entry_min_conf: float                  # YAML-only
    resting_entry_offset_pct: float                # YAML-only
    resting_entry_max_distance_pct: float          # YAML-only

    # Same-side re-entry
    same_side_reentry_cooldown_seconds: int        # Default 180
    same_side_reentry_strong_confidence: float     # Default 0.85

    # Scalping
    scalp_min_hold_seconds: int                   # Default 30
    scalp_runner_enabled: bool                    # Default True
    scalp_runner_pullback_pct: float              # Default 0.0012
    scalp_runner_min_lock_pct: float              # Default 0.0018
    scalp_runner_exchange_tp_multiplier: float   # Default 3.0
    scalp_runner_partial_exit_pct: float          # Default 0.45

    # Spot balance overrides
    spot_balance_pct: float                       # Default 0.20
    spot_reserve_pct: float                       # Falls back to spot.reserve_quote_pct
    spot_max_layers: int                          # Falls back to spot.max_layers

    # DCA
    dca_enabled: bool                             # Default False
    dca_max_steps: int                            # Default 0
    dca_distance_pct: float                       # Default 0.01

    # Paper-gate (live mode safety check)
    paper_gate_min_trades: int                    # Default 100
    paper_gate_min_profit_factor: float           # Default 1.2
    paper_gate_max_drawdown: float                # Default 0.20

    # Paper-mode confidence override
    paper_min_conf: float                         # Default = strategy.min_conf

    # Trade log override
    trade_log_file: str                           # Default depends on market

    # Misc (YAML-only or factory-defaulted)
    dynamic_leverage: bool                        # YAML-only; use leverage.enabled instead
    ttl_exit_only_if_unprofitable: bool           # Default True (factory)
    ttl_exit_profit_cap_pct: float               # Default 0.0 (factory)


class LeverageConfig(TypedDict, total=False):
    enabled: bool                                 # Default False
    min_leverage: float                           # Default 1.0
    max_leverage: float                           # Default 4.0
    confidence_levels: dict[float, float]         # {0.30: 1.0, 0.65: 3.0}
    use_score_multiplier: bool                    # Default False
    score_weight: float                           # Default 0.3
    atr_volatility_scaling: bool                   # Default False
    atr_reference_pct: float                      # Default 0.02
    atr_min_multiplier: float                    # Default 0.3


class MTFConfig(TypedDict, total=False):
    enabled: bool                                 # Default False
    timeframes: list[str]                         # Default ["15m", "3h", "4h"]
    min_agree: int                                 # YAML-only
    require_full_confirmation: bool                 # YAML-only
    strict_veto_on_missing: bool                  # YAML-only
    veto_on_missing: bool                          # YAML-only
    max_age_seconds: int                           # YAML-only
    sr_buffer_pct: float                           # YAML-only
    mode: str                                      # YAML-only


class StrategyConfig(TypedDict, total=False):
    timeframe: str                                # Default = top-level timeframe
    candles_per_day: int                          # Default 96
    max_spread: float                             # Default 0.0005
    min_conf: float                               # Default 0.15
    signal_smoothing_ticks: int                    # YAML-only
    max_ret_30s: float                            # Default 0.005
    max_ret_5s: float                             # Default 0.0025
    block_on_volume_spike: bool                    # Default False
    fixed_trade_usdt: float                       # Default 0.0 (setdefault → 100)
    tp_pct: float                                 # Default 0.0035 (setdefault)
    sl_pct: float                                 # Default 0.0025 (setdefault)
    max_structural_sl_pct: float                  # Default 0.0120 (factory)
    min_reward_risk: float                         # Default 0.90
    range_action_zone_pct: float                  # Default 0.20
    wall_veto_zone_pct: float                     # YAML-only
    support_veto_pct: float                       # Default 0.0015
    resistance_veto_pct: float                    # Default 0.0015
    support_break_pct: float                      # Default 0.0010
    resistance_break_pct: float                   # Default 0.0010
    max_chase_pct: float                          # Default 0.0
    mtf_neutral_allow_score: float                # YAML-only
    entry_min_confidence_hard: float              # Default 0.20
    midrange_min_score: float                     # Default 0.28
    mtf_rsi_bull_level: float                    # Default 55
    mtf_rsi_bear_level: float                    # Default 45
    mtf_rsi_min_agree: int                        # Default 2
    vpoc_near_pct: float                          # Default 0.0010
    vpoc_break_pct: float                         # Default 0.0015
    session_filter_enabled: bool                   # YAML-only
    session_block_hours_utc: list[int]             # Default []
    session_block_min_score: float                  # Default 0.35
    entry_near_resistance_block_pct: float          # YAML-only
    entry_near_support_block_pct: float             # YAML-only
    macd_noise_threshold: float                    # Default 0.0001
    vol_filter_atr_pct: float                      # Default 0.0005
    vol_filter_atr_max_pct: float                   # Default 0.08
    low_vol_min_score: float                        # Default 0.45
    wick_sweep_enabled: bool                        # Default True
    wick_sweep_buffer_pct: float                    # Default 0.0008
    wick_sweep_reclaim_pct: float                   # Default 0.0002
    wick_sweep_wick_ratio: float                    # Default 1.6
    wall_reversal_assist: bool                      # Default True
    wall_reversal_score_gate: float                  # Default 0.12
    wall_breakout_score_gate: float                  # Default 0.15
    range_position_veto_enabled: bool                # Default True
    range_veto_top_pct: float                        # Default 0.75
    range_veto_bottom_pct: float                      # Default 0.25
    range_veto_escape_score: float                    # Default 0.80
    range_reversal_min_depth: float                    # Default 0.20
    range_reversal_max_boost: float                    # Default 0.45
    rsi_ob_entry_gate: int                             # Default 72
    rsi_os_entry_gate: int                              # Default 28
    orb_volume_mult: float                              # Default 1.25
    sr_wall_veto_enabled: bool                          # Default True
    spread_atr_ratio_max: float                         # Default 0.15
    ob_midpoint_tolerance_pct: float                    # Default 0.0015
    anchored_vwap_near_pct: float                        # Default 0.0010
    loss_tilt_min_losses: int                             # Default 3
    loss_tilt_pause_losses: int                            # Default 5
    loss_tilt_pause_minutes: int                           # Default 15

    # Location-only keys (accessed in indicators/location.py)
    location_support_near_pct: float                      # Default 0.0075
    location_support_far_pct: float                         # Default 0.0200
    location_resistance_near_pct: float                      # Default 0.0025
    location_resistance_far_pct: float                        # Default 0.0200


class RiskConfig(TypedDict, total=False):
    daily_loss_cap: float                          # No default; None means disabled
    disable_loss_cap: bool                         # YAML-only
    max_open_positions: int                        # Default 1
    min_balance_floor: float                        # Default 0.0 (bot_loop says 90.0)


class AIConfig(TypedDict, total=False):
    enabled: bool                                  # Required
    model: str                                     # Required
    overlay_enabled: bool                           # Default False
    overlay_refresh_seconds: int                    # Default 1800
    trade_gate_enabled: bool                        # Default False
    trade_gate_min_interval: int                    # Default 300
    trade_gate_on_error: str                        # Default "allow"
    trade_gate_max_hold_minutes: int                # Default 60 (paper-only)


class AITradeConfig(TypedDict, total=False):
    enabled: bool
    max_hold_minutes: int                           # Default 60
    on_error: str                                   # Default "allow"
    min_interval_seconds: int                       # Default 30
    model: str                                      # Default = ai.model


class UIConfig(TypedDict, total=False):
    mode: str                                       # Default "auto"
    compact: bool                                   # YAML-only
    status_lines: int                               # Default 3
    alt_screen: bool                                 # YAML-only
    chart_tf: str                                   # Default = top-level timeframe


class IntervalsConfig(TypedDict, total=False):
    indicator_refresh: int                          # Required
    regime_refresh: int                              # Required
    tick_delay_seconds: float                       # Required


class DashboardConfig(TypedDict, total=False):
    enabled: bool                                   # Default True
    host: str                                       # Default "127.0.0.1"
    port: int                                       # Default 8080
    candle_limit: int                               # Default 240


class AIAdvisorConfig(TypedDict, total=False):
    enabled: bool                                   # Default False
    model: str                                      # Default = ai.model
    max_weight_shift: float                          # Default 0.12


class AutoLearningConfig(TypedDict, total=False):
    enabled: bool                                   # Default False
    min_completed_trades: int                       # Default 5
    refresh_closed_trades: int                      # Default 3
    max_recent_trades: int                          # YAML-only
    shrinkage: float                                # Default 0.45
    ai_advisor: AIAdvisorConfig


class RejectionConfirmationConfig(TypedDict, total=False):
    enabled: bool                                   # YAML-only
    min_bars_away: int                               # YAML-only
    macd_threshold: float                            # YAML-only
    pullback_pct: float                               # YAML-only
    require_psar_flip: bool                           # YAML-only


class SpotConfig(TypedDict, total=False):
    mode: str                                        # Default "grid"
    max_layers: int                                  # Default 3
    layer_quote_pct: float                            # Default 0.20
    reserve_quote_pct: float                           # Default 0.30
    buy_near_support_pct: float                         # Default 0.0020
    sell_near_resistance_pct: float                      # Default 0.0020
    layer_spacing_pct: float                              # Default 0.0030
    emergency_break_pct: float                             # Default 0.0040
    min_take_profit_pct: float                              # Default 0.0035
    min_trail_profit_quote: float                           # YAML-only


class MemoryConfig(TypedDict, total=False):
    max_closed_trades: int                            # Default 5000


class LoggingConfig(TypedDict, total=False):
    max_mb: float                                    # Default 5
    backups: int                                     # Default 3


class BotConfig(TypedDict, total=False):
    """
    Complete bot configuration — mirrors config.yaml structure.

    Top-level required keys: symbol, timeframe
    All sub-sections are optional (have sensible defaults).
    """
    symbol: str                                     # Required
    timeframe: str                                  # Required
    macro_timeframe: str                            # Default "1h"

    data: DataConfig
    execution: ExecutionConfig
    leverage: LeverageConfig
    mtf: MTFConfig
    strategy: StrategyConfig
    risk: RiskConfig
    ai: AIConfig
    ai_trade: AITradeConfig
    ai_overlay: dict                                # Alternate overlay config
    ui: UIConfig
    intervals: IntervalsConfig
    dashboard: DashboardConfig
    auto_learning: AutoLearningConfig
    rejection_confirmation: RejectionConfirmationConfig
    spot: SpotConfig
    memory: MemoryConfig
    logging: LoggingConfig