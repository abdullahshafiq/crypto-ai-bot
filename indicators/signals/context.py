"""Phase 1: Build core market context — spread guard, macro bias, support/resistance, location, volatility, funding."""

from __future__ import annotations

from .ctx import SignalContext
from ..calc import get_signal_weights
from ..mtf import _pick_structural_levels
from ..location import (
    compute_volatility_context, identify_liquidity_pools, calculate_funding_impact,
    _compute_vpoc, _compute_anchored_vwap,
    _compute_market_location_score, _compute_wall_state,
)
from .gates import apply_spread_guard


def _build_core_context(ctx: SignalContext) -> dict | None:
    """Populate ctx with market context. Returns early-exit dict if spread guard fires, else None."""
    current_price = ctx['state']['price']
    ctx['current_price'] = current_price

    early_exit = apply_spread_guard(ctx)
    if early_exit:
        return early_exit

    latest_macro = ctx['latest_macro']
    if isinstance(latest_macro, dict):
        macro_bias = str(latest_macro.get("regime") or latest_macro.get("bias") or "NEUTRAL").upper()
    else:
        macro_bias = str(latest_macro or "NEUTRAL").upper()
    ctx['macro_bias'] = macro_bias

    strategy_config = ctx['strategy_config']
    ctx['range_action_zone_pct'] = float(strategy_config.get("range_action_zone_pct", 0.20) or 0.20)
    ctx['weights'] = get_signal_weights()

    support, resistance = _pick_structural_levels(
        current_price,
        mtf_context=ctx.get('mtf_context'),
        pivot_data=ctx.get('pivot_data'),
        max_sl_pct=float(strategy_config.get('max_structural_sl_pct', 0.012)),
    )
    ctx['support'] = support
    ctx['resistance'] = resistance

    df_indicators = ctx['df_indicators']
    latest_indicators = ctx['latest_indicators']
    state = ctx['state']

    vpoc = _compute_vpoc(df_indicators)
    anchored_vwap = _compute_anchored_vwap(df_indicators)
    last_close = float(df_indicators['close'].iloc[-1]) if len(df_indicators) else float(current_price)
    wall_state = _compute_wall_state(
        current_price=current_price,
        last_close=last_close,
        support=support,
        resistance=resistance,
        strategy_config=strategy_config,
    )
    ctx['wall_state'] = wall_state

    location_score, location_notes, location_levels = _compute_market_location_score(
        current_price,
        support=support,
        resistance=resistance,
        vpoc=vpoc,
        anchored_vwap=anchored_vwap,
        state=state,
        latest_indicators=latest_indicators,
        strategy_config=strategy_config,
    )
    ctx['location_score'] = location_score
    ctx['location_notes'] = location_notes
    ctx['location_levels'] = location_levels
    ctx['vpoc'] = vpoc
    ctx['anchored_vwap'] = anchored_vwap

    vol_context = compute_volatility_context(df_indicators)
    liquidity = identify_liquidity_pools(df_indicators)
    funding_impact = calculate_funding_impact(latest_macro)
    ctx['vol_context'] = vol_context
    ctx['liquidity'] = liquidity
    ctx['funding_impact'] = funding_impact

    return None