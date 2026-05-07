"""
TypedDict for the shared `ctx` (SignalContext) dict passed through all signal phases.

Keys are grouped by the phase that first writes them. Since ctx is built up
progressively, all keys are NotRequired — but each downstream reader can assume
the keys from preceding phases exist.

If you add a new key to ctx, add it here with the correct type and phase comment.
"""

from __future__ import annotations

from typing import NotRequired, TypedDict

import pandas as pd


class SignalContext(TypedDict, total=False):
    # ── Phase 0: Initial seed (from engine.py) ──────────────────────────
    state: dict
    latest_indicators: dict
    strategy_config: dict
    df_indicators: pd.DataFrame
    latest_macro: dict | str
    mtf_context: dict
    mtf_config: dict
    pivot_data: dict

    # ── Phase 1: Core context (context.py) ────────────────────────────────
    current_price: float
    macro_bias: str
    range_action_zone_pct: float
    weights: dict
    support: float
    resistance: float
    wall_state: dict
    location_score: float
    location_notes: str
    location_levels: dict
    vpoc: float
    anchored_vwap: float
    vol_context: dict
    liquidity: dict
    funding_impact: float

    # ── Phase 2: MTF bias (engine.py → compute_mtf_bias) ────────────────
    mtf_fast_score: float
    mtf_fast_bias: str
    mtf_rsi_score: float
    mtf_rsi_bias: str

    # ── Phase 3: Guards (engine.py → chasing guard) ─────────────────────
    ema_21: float
    ema_9: float
    atr_pct_now: float
    trend_continuation: bool
    ema_dist_pct: float
    max_chase_pct: float

    # ── Phase 4: Indicator scores (scores.py → compute_indicator_scores) ─
    smc_score: float
    sr_score: float
    smc_label: str
    ob_context: dict
    z_score: float
    rsi_14: float
    mr_score: float
    vwap_score: float
    bb_score: float
    macd_diff: float
    prev_macd_diff: float
    prev2_macd_diff: float
    macd_val: float
    macd_score: float
    divergence_state: str
    bear_div: bool
    bull_div: bool
    hidden_bull_div: bool
    hidden_bear_div: bool
    cvd_state: str
    cvd_bonus: float
    momentum_exhaustion: str
    body_ratio_score: float
    power_bonus: float
    vol_spike: bool
    div_bonus: float
    htf_score: float
    htf_1h: dict
    htf_4h: dict
    pa_score: float
    psar_val: float
    adx_value: float
    adx_score: float
    volume_delta: float
    obv_score: float
    ob_score: float
    kdj_score: float
    st_score: float
    alpha: float
    score_without_sr: float
    sr_directional: float
    total_score_raw: float
    total_score: float

    # ── Phase 5: Guards (ATR / ADX) ────────────────────────────────────
    # (no new keys — guards return early-exit dicts)

    # ── Phase 6-7: Score synthesis + action (synthesis.py) ──────────────
    psar_bull: bool
    macd_diff_val: float
    action: str
    hold_reason: str
    sr_wall_locked: bool
    ema_9_val: float
    ema_21_val: float
    price_above_ema9: bool
    price_below_ema9: bool

    # ── Phase 8: Trend confirmation (trend.py) ──────────────────────────
    psar_closed_bull: bool
    psar_live_bull: bool
    psar_state_note: str
    mtf_macd_bull: bool
    mtf_macd_bear: bool
    mtf_structure_bull: bool
    mtf_structure_bear: bool
    macd_final_bull: bool
    macd_final_bear: bool
    psar_streak: int
    signal_reason_suffix: str
    article_sl_override: float

    # ── Phase 9: Signal dict (builder.py) ────────────────────────────────
    signal: dict

    # ── Phase 10: Setup overrides (stops.py) ────────────────────────────
    wick_setup: dict

    # ── Phase 7+: Gate-specific keys ─────────────────────────────────────
    entry_mode: str
    in_action_zone: bool