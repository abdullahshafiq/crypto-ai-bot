"""Bias adjustments: MTF trend-continuation bias nudge and range zone computation."""

from __future__ import annotations

from ..ctx import SignalContext


def apply_trend_continuation_bias(ctx: SignalContext) -> None:
    """Apply MTF trend-continuation bias — nudges flat scores in the trend direction."""
    mtf_fast_bias = ctx.get('mtf_fast_bias', 'NEUTRAL')
    mtf_fast_score = ctx.get('mtf_fast_score', 0.0)
    total_score = ctx.get('total_score', 0.0)
    current_price = ctx['current_price']
    ema_21 = ctx.get('ema_21', ctx['latest_indicators'].get('ema_21', current_price))
    pa_score = ctx.get('pa_score', 0.0)
    macd_diff_val = ctx.get('macd_diff', 0.0)
    psar_bull = ctx.get('psar_bull', True)

    if mtf_fast_bias == "LONG_ONLY" and total_score < 0.10:
        if current_price >= ema_21 and pa_score >= 0:
            ctx['total_score'] = max(total_score, 0.12)
    elif mtf_fast_bias == "SHORT_ONLY" and total_score > -0.10:
        if current_price <= ema_21 and pa_score <= 0:
            ctx['total_score'] = min(total_score, -0.12)
    elif mtf_fast_bias == "NEUTRAL":
        bearish_context_votes = int(current_price <= ema_21) + int(not psar_bull) + int(macd_diff_val < 0)
        bullish_context_votes = int(current_price >= ema_21) + int(psar_bull) + int(macd_diff_val > 0)
        if bearish_context_votes >= 2 and current_price < ema_21 and pa_score <= 0 and total_score > -0.08:
            ctx['total_score'] = min(total_score, -0.08)
        elif bullish_context_votes >= 2 and current_price > ema_21 and pa_score >= 0 and total_score < 0.08:
            ctx['total_score'] = max(total_score, 0.08)


def compute_range_zones(ctx: SignalContext) -> dict | None:
    """Compute action zone boundaries. Returns dict with zone info or None."""
    m5_s = ctx.get('signal', {}).get("structure_support")
    m5_r = ctx.get('signal', {}).get("structure_resistance")
    if not (m5_s and m5_r):
        return None

    current_price = ctx['current_price']
    range_action_zone_pct = ctx.get('range_action_zone_pct', 0.20)
    range_action_zone_pct = max(0.05, min(0.45, range_action_zone_pct))

    range_width = m5_r - m5_s
    action_zone_size = range_width * range_action_zone_pct
    top_action_zone = m5_r - action_zone_size
    bottom_action_zone = m5_s + action_zone_size

    return {
        'range_width': range_width,
        'action_zone_size': action_zone_size,
        'top_action_zone': top_action_zone,
        'bottom_action_zone': bottom_action_zone,
        'at_top': current_price >= top_action_zone,
        'at_bottom': current_price <= bottom_action_zone,
        'in_middle': not (current_price >= top_action_zone) and not (current_price <= bottom_action_zone),
    }