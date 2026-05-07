"""Phases 6-7: Score synthesis (funding, ADX, PSAR/MACD bias, clipping) and action determination (BUY/SELL/HOLD, SR veto, range veto)."""

from __future__ import annotations

from .ctx import SignalContext
import numpy as np

from .gates import (
    apply_trend_continuation_bias, apply_low_vol_min_score,
    apply_session_blackout, apply_sr_wall_veto, apply_range_position_veto,
)


def _apply_score_synthesis(ctx: SignalContext) -> dict | None:
    """Synthesize raw score into total_score, apply trend bias and guards. Returns early-exit dict or None."""
    total_score = ctx['total_score_raw']
    total_score *= 1.0
    total_score += (ctx['funding_impact'] * 0.05)
    total_score += (ctx['mtf_fast_score'] * 0.20)

    adx_value = ctx['adx_value']
    if adx_value >= 25:
        total_score *= 1.10
    elif adx_value < 18:
        total_score *= 0.90

    current_price = ctx['current_price']
    ema_21 = ctx['ema_21']

    psar_closed_val = ctx['latest_indicators'].get('psar')
    psar_live_val = float(ctx['df_indicators']['psar'].iloc[-1]) if "psar" in ctx['df_indicators'].columns and len(ctx['df_indicators']) else psar_closed_val
    macd_diff_val = float(ctx['latest_indicators'].get('macd_diff', 0) or 0)

    psar_bull = True
    if psar_closed_val is not None:
        try:
            psar_bull = float(psar_closed_val) < current_price
        except (TypeError, ValueError):
            psar_bull = current_price >= ema_21
    elif psar_live_val is not None:
        try:
            psar_bull = float(psar_live_val) < current_price
        except (TypeError, ValueError):
            psar_bull = current_price >= ema_21

    ctx['psar_bull'] = psar_bull
    ctx['macd_diff_val'] = macd_diff_val
    ctx['total_score'] = total_score

    apply_trend_continuation_bias(ctx)
    ctx['total_score'] = float(np.clip(ctx['total_score'], -1.0, 1.0))

    early_exit = apply_low_vol_min_score(ctx)
    if early_exit:
        return early_exit

    early_exit = apply_session_blackout(ctx)
    if early_exit:
        return early_exit

    return None


def _determine_action(ctx: SignalContext) -> None:
    """Determine initial BUY/SELL/HOLD action, then apply SR wall and range position vetoes."""
    total_score = ctx['total_score']
    action = "HOLD"
    if total_score > 0.05:
        action = "BUY"
    elif total_score < -0.05:
        action = "SELL"
    ctx['action'] = action
    ctx['hold_reason'] = ""
    ctx['sr_wall_locked'] = False

    current_price = ctx['current_price']
    latest_indicators = ctx['latest_indicators']
    ema_9_val = float(latest_indicators.get('ema_9', current_price) or current_price)
    ema_21_val = float(latest_indicators.get('ema_21', current_price) or current_price)
    ctx['ema_9_val'] = ema_9_val
    ctx['ema_21_val'] = ema_21_val
    ctx['price_above_ema9'] = current_price > ema_9_val
    ctx['price_below_ema9'] = current_price < ema_9_val

    apply_sr_wall_veto(ctx)
    apply_range_position_veto(ctx)