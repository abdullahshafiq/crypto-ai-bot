"""Pre-trade guards: spread, chasing, ATR, ADX range, low-vol min score, session blackout."""

from __future__ import annotations

from ..ctx import SignalContext
from ..utils import _session_blackout_state


def apply_spread_guard(ctx: SignalContext) -> dict | None:
    """Return a HOLD signal dict if spread/ATR guard triggers, else None."""
    state = ctx['state']
    strategy_config = ctx['strategy_config']
    latest_indicators = ctx['latest_indicators']

    current_price = state['price']
    spread_pct = float(state.get("spread_pct", 0.0) or 0.0)
    max_spread = float(strategy_config.get("max_spread", 0.0007) or 0.0007)
    atr_pct_now = float(latest_indicators.get("atr_pct", 0.5) or 0.5) / 100.0
    spread_atr_ratio_max = float(strategy_config.get("spread_atr_ratio_max", 0.15) or 0.15)
    dynamic_max_spread = max(max_spread, atr_pct_now * spread_atr_ratio_max)
    if spread_pct > dynamic_max_spread:
        return {"action": "HOLD", "score": 0.0, "confidence": 0.0, "reason": f"Spread/ATR guard ({spread_pct:.4%}>{dynamic_max_spread:.4%})", "weights": {}}

    ret_30s = state.get("ret_30s")
    max_ret_30s = float(strategy_config.get("max_ret_30s", 0.0050) or 0.0050)
    if isinstance(ret_30s, (int, float)) and abs(float(ret_30s)) > max_ret_30s:
        return {"action": "HOLD", "score": 0.0, "confidence": 0.0, "reason": f"Return guard 30s ({float(ret_30s):+.3%})", "weights": {}}

    ret_5s = state.get("ret_5s")
    max_ret_5s = float(strategy_config.get("max_ret_5s", 0.0025) or 0.0025)
    if isinstance(ret_5s, (int, float)) and abs(float(ret_5s)) > max_ret_5s:
        return {"action": "HOLD", "score": 0.0, "confidence": 0.0, "reason": f"Return guard 5s ({float(ret_5s):+.3%})", "weights": {}}

    if bool(strategy_config.get("block_on_volume_spike", False)) and str(state.get("volume_state", "normal")) == "spike":
        return {"action": "HOLD", "score": 0.0, "confidence": 0.0, "reason": "Volume spike guard", "weights": {}}

    return None


def apply_chasing_guard(ctx: SignalContext) -> dict | None:
    """Return a HOLD signal dict if anti-chasing guard triggers, else None."""
    current_price = ctx['current_price']
    latest_indicators = ctx['latest_indicators']
    strategy_config = ctx['strategy_config']
    mtf_fast_bias = ctx['mtf_fast_bias']
    mtf_fast_score = ctx['mtf_fast_score']
    atr_pct_now = ctx['atr_pct_now']
    ema_21 = ctx['ema_21']
    ema_9 = ctx.get('ema_9', latest_indicators.get('ema_9', current_price))
    df_indicators = ctx['df_indicators']
    action = ctx.get('action', 'HOLD')

    ema_dist_pct = abs(ema_9 - ema_21) / ema_21
    configured_max_chase_pct = float(strategy_config.get("max_chase_pct", 0.0) or 0.0)
    base_max_chase_pct = configured_max_chase_pct if configured_max_chase_pct > 0 else max(0.0015, min(0.004, atr_pct_now * 0.5))

    trend_continuation = (
        (current_price < ema_21 and (mtf_fast_bias == "SHORT_ONLY" or mtf_fast_score <= -0.15))
        or (current_price > ema_21 and (mtf_fast_bias == "LONG_ONLY" or mtf_fast_score >= 0.15))
    )
    ctx['trend_continuation'] = trend_continuation
    if trend_continuation:
        max_chase_pct = max(base_max_chase_pct, min(0.0065, max(0.0045, atr_pct_now * 0.75)))
    else:
        max_chase_pct = base_max_chase_pct
    ctx['ema_dist_pct'] = ema_dist_pct
    ctx['max_chase_pct'] = max_chase_pct

    if ema_dist_pct > max_chase_pct:
        return {"action": "HOLD", "score": 0.0, "confidence": 0.0, "reason": f"Chasing Guard ({ema_dist_pct:.3%}>{max_chase_pct:.3%})", "weights": {}}

    # Near-extreme guard: block BUY near 20c high, SELL near 20c low
    extreme_lookback = strategy_config.get("chase_recent_extreme_lookback")
    if extreme_lookback is None:
        extreme_lookback = 20
    else:
        extreme_lookback = int(extreme_lookback)

    near_extreme_pct = strategy_config.get("chase_near_extreme_pct")
    if near_extreme_pct is None:
        near_extreme_pct = 0.0025
    else:
        near_extreme_pct = float(near_extreme_pct)

    if near_extreme_pct > 0 and action in {"BUY", "SELL"} and len(df_indicators) >= extreme_lookback:
        recent_ext = df_indicators.tail(extreme_lookback)
        if action == "BUY":
            recent_high = float(recent_ext["high"].max())
            if current_price >= recent_high * (1 - near_extreme_pct):
                return {"action": "HOLD", "score": 0.0, "confidence": 0.0, "reason": f"Near-Extreme Guard: BUY too close to {extreme_lookback}c high (${recent_high:.3f})", "weights": {}}
        elif action == "SELL":
            recent_low = float(recent_ext["low"].min())
            if current_price <= recent_low * (1 + near_extreme_pct):
                return {"action": "HOLD", "score": 0.0, "confidence": 0.0, "reason": f"Near-Extreme Guard: SELL too close to {extreme_lookback}c low (${recent_low:.3f})", "weights": {}}

    # Block entries after extended uninterrupted run (price already exhausted)
    max_consec = int(strategy_config.get("max_consecutive_candles_chase", 4) or 4)
    if action in {"BUY", "SELL"} and len(df_indicators) >= max_consec:
        recent = df_indicators.tail(max_consec)
        if action == "BUY":
            all_green = all(recent['close'].values[i] >= recent['open'].values[i] for i in range(max_consec))
            if all_green:
                return {"action": "HOLD", "score": 0.0, "confidence": 0.0, "reason": f"Chase Guard: {max_consec} consecutive green candles — wait for pullback", "weights": {}}
        elif action == "SELL":
            all_red = all(recent['close'].values[i] <= recent['open'].values[i] for i in range(max_consec))
            if all_red:
                return {"action": "HOLD", "score": 0.0, "confidence": 0.0, "reason": f"Chase Guard: {max_consec} consecutive red candles — wait for bounce", "weights": {}}

    return None


def apply_atr_guard(ctx: SignalContext) -> dict | None:
    """Return a HOLD signal dict if ATR guard triggers, else None."""
    strategy_config = ctx['strategy_config']
    atr_pct_now = ctx['atr_pct_now']
    atr_min = float(strategy_config.get("vol_filter_atr_pct", 0.0005) or 0.0005)
    atr_max = float(strategy_config.get("vol_filter_atr_max_pct", 0.08) or 0.08)
    if atr_pct_now < atr_min or atr_pct_now > atr_max:
        return {"action": "HOLD", "score": 0.0, "confidence": 0.0, "reason": f"ATR guard ({atr_pct_now:.3%})", "weights": {}}
    return None


def apply_adx_range_filter(ctx: SignalContext) -> dict | None:
    """Return a HOLD signal dict if ADX ranging filter triggers, else None."""
    adx_value = ctx.get('adx_value', 0.0)
    total_score = ctx.get('total_score', 0.0)
    if adx_value < 15:
        # S/R edge bounce entries are valid in ranging markets — skip block when at the walls
        support = ctx.get('support')
        resistance = ctx.get('resistance')
        current_price = ctx['current_price']
        range_action_zone_pct = float(ctx.get('range_action_zone_pct', 0.20) or 0.20)
        if support and resistance:
            zone = (float(resistance) - float(support)) * range_action_zone_pct
            if current_price <= float(support) + zone or current_price >= float(resistance) - zone:
                return None
        return {"action": "HOLD", "score": total_score, "confidence": min(abs(total_score), 1.0), "reason": f"Ranging market (ADX:{adx_value:.0f}<15)", "weights": {}}
    return None


def apply_low_vol_min_score(ctx: SignalContext) -> dict | None:
    """Return a HOLD signal dict if low-vol score threshold not met, else None."""
    strategy_config = ctx['strategy_config']
    atr_pct_now = ctx['atr_pct_now']
    atr_min = float(strategy_config.get("vol_filter_atr_pct", 0.0005) or 0.0005)
    total_score = ctx.get('total_score', 0.0)
    if atr_pct_now < atr_min * 2.0:
        low_vol_min_score = float(strategy_config.get("low_vol_min_score", 0.45) or 0.45)
        if abs(total_score) < low_vol_min_score:
            return {
                "action": "HOLD",
                "score": total_score,
                "confidence": min(abs(total_score), 1.0),
                "reason": f"Low vol: need score >= {low_vol_min_score:.2f}",
                "weights": {},
            }
    return None


def apply_session_blackout(ctx: SignalContext) -> dict | None:
    """Return a HOLD signal dict if session blackout triggers, else None."""
    strategy_config = ctx['strategy_config']
    total_score = ctx.get('total_score', 0.0)
    session_blocked, utc_hour, blocked_hours = _session_blackout_state(strategy_config)
    if session_blocked:
        session_min_score = float(strategy_config.get("session_block_min_score", 0.35) or 0.35)
        if abs(total_score) < session_min_score:
            return {
                "action": "HOLD",
                "score": total_score,
                "confidence": min(abs(total_score), 1.0),
                "reason": f"Session filter UTC{utc_hour:02d}: need score >= {session_min_score:.2f}",
                "weights": {},
            }
    return None