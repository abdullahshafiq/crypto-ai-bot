"""
Signal engine orchestrator — generates the final BUY/SELL/HOLD signal.

The `ctx` dict is the shared context passed through all phases. Keys populated:

  Phase 1 (core context): state, strategy_config, df_indicators, latest_indicators,
    current_price, macro_bias, range_action_zone_pct, weights, support, resistance,
    vpoc, anchored_vwap, wall_state, location_score, location_notes, location_levels,
    vol_context, liquidity, funding_impact, ema_21, ema_9, atr_pct_now

  Phase 2 (MTF bias): mtf_fast_score, mtf_fast_bias, mtf_rsi_score, mtf_rsi_bias

  Phase 3 (guards): trend_continuation, ema_dist_pct, max_chase_pct

  Phase 4 (scores): smc_score, sr_score, smc_label, ob_context, mr_score, vwap_score,
    bb_score, macd_score, macd_diff, prev_macd_diff, prev2_macd_diff, macd_val,
    divergence_state, bear_div, bull_div, hidden_bull_div, hidden_bear_div,
    cvd_state, cvd_bonus, momentum_exhaustion, body_ratio_score, power_bonus,
    vol_spike, div_bonus, htf_score, htf_1h, htf_4h, pa_score, psar_val,
    adx_value, adx_score, volume_delta, obv_score, ob_score, kdj_score, st_score,
    alpha, score_without_sr, sr_directional, total_score_raw, total_score, z_score, rsi_14

  Phase 7+ (gates/signal): action, hold_reason, sr_wall_locked, entry_mode,
    signal_reason_suffix, in_action_zone, psar_bull, psar_closed_bull, psar_live_bull,
    psar_state_note, mtf_macd_bull, mtf_macd_bear, mtf_structure_bull, mtf_structure_bear,
    macd_final_bull, macd_final_bear, psar_streak, ema_9_val, ema_21_val,
    price_above_ema9, price_below_ema9, signal
"""

from __future__ import annotations

from .ctx import SignalContext
from .context import _build_core_context
from .synthesis import _apply_score_synthesis, _determine_action
from .mtf_bias import compute_mtf_bias
from .scores import compute_indicator_scores
from .trend import _compute_trend_confirmation
from .builder import _build_signal_dict
from .stops import _apply_setup_overrides, _compute_sl_tp
from .gates import (
    apply_chasing_guard, apply_atr_guard, apply_adx_range_filter,
    apply_mtf_trend_veto, apply_range_reversal_sniper,
    apply_exhaustion_divergence_gate, apply_wall_rejection_rescue,
    apply_midrange_policy, _apply_rejection_confirmation_gate,
)
from .alpha import generate_alpha_overlay


def generate_quant_signal(state, latest_indicators, strategy_config, df_indicators, latest_macro, mtf_context=None, mtf_config=None, pivot_data=None) -> dict:
    """
    The Institutional Sniper Signal Engine.
    Orchestrates indicator scoring, gate checks, entry classification, and signal generation.
    """
    if df_indicators is None or len(df_indicators) < 50:
        return {"action": "HOLD", "score": 0, "confidence": 0, "reason": "Warming Up", "weights": {}}

    ctx = {
        'state': state,
        'latest_indicators': latest_indicators,
        'strategy_config': strategy_config,
        'df_indicators': df_indicators,
        'latest_macro': latest_macro,
        'mtf_context': mtf_context,
        'mtf_config': mtf_config,
        'pivot_data': pivot_data,
    }

    # ── Phase 1: Core context ──────────────────────────────────────────
    early_exit = _build_core_context(ctx)
    if early_exit:
        return early_exit

    # ── Phase 2: MTF bias ──────────────────────────────────────────────
    mtf_result = compute_mtf_bias(mtf_config, mtf_context, strategy_config)
    ctx['mtf_fast_score'] = mtf_result['mtf_fast_score']
    ctx['mtf_fast_bias'] = mtf_result['mtf_fast_bias']
    ctx['mtf_rsi_score'] = mtf_result['mtf_rsi_score']
    ctx['mtf_rsi_bias'] = mtf_result['mtf_rsi_bias']

    # ── Phase 3: Chasing guard ──────────────────────────────────────────
    current_price = ctx['current_price']
    ctx['ema_21'] = float(latest_indicators.get('ema_21', current_price) or current_price)
    ctx['ema_9'] = float(latest_indicators.get('ema_9', current_price) or current_price)
    ctx['atr_pct_now'] = float(latest_indicators.get("atr_pct", 0.5) or 0.5) / 100.0
    early_exit = apply_chasing_guard(ctx)
    if early_exit:
        return early_exit

    # ── Phase 4: Indicator scores ──────────────────────────────────────
    compute_indicator_scores(ctx)

    # ── Phase 5: ATR / ADX guards ───────────────────────────────────────
    early_exit = apply_atr_guard(ctx)
    if early_exit:
        return early_exit
    early_exit = apply_adx_range_filter(ctx)
    if early_exit:
        return early_exit

    # ── Phases 6-7: Score synthesis + action determination ──────────────
    early_exit = _apply_score_synthesis(ctx)
    if early_exit:
        return early_exit
    _determine_action(ctx)

    # ── Phase 8: Trend confirmation + article setups ────────────────────
    _compute_trend_confirmation(ctx)

    # ── Phase 9: Build signal dict ──────────────────────────────────────
    signal = _build_signal_dict(ctx)

    # ── Phase 10: Mean reversion + wick sweep setups ───────────────────
    _apply_setup_overrides(signal, ctx)

    # ── Phase 11: Stop loss / take profit ───────────────────────────────
    signal = _compute_sl_tp(signal, ctx)
    ctx['signal'] = signal

    # ── Phase 12: MTF trend veto ────────────────────────────────────────
    apply_mtf_trend_veto(ctx)
    signal = ctx['signal']

    # ── Phase 13: Range reversal sniper ─────────────────────────────────
    apply_range_reversal_sniper(ctx)
    signal = ctx['signal']

    # ── Phase 14: Midrange policy ───────────────────────────────────────
    apply_midrange_policy(ctx)
    signal = ctx['signal']

    # ── Phase 15: Rejection confirmation gate ───────────────────────────
    signal, _rejection_applied = _apply_rejection_confirmation_gate(
        signal, df_indicators, strategy_config, support=ctx['support'], resistance=ctx['resistance'],
    )

    # ── Phase 16: Exhaustion & divergence gate ──────────────────────────
    ctx['signal'] = signal
    apply_exhaustion_divergence_gate(ctx)
    signal = ctx['signal']

    # ── Phase 17: Wall rejection rescue ─────────────────────────────────
    ctx['signal'] = signal
    apply_wall_rejection_rescue(ctx)
    return ctx['signal']