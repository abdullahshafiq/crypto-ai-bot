"""Pre-trade guards: spread, chasing, ATR, ADX range, low-vol min score, session blackout."""

from __future__ import annotations

from ..ctx import SignalContext
from ..utils import _session_blackout_state


def _is_structural_range_edge(ctx: SignalContext) -> bool:
    support = ctx.get('support')
    resistance = ctx.get('resistance')
    if not support or not resistance:
        return False

    try:
        support_f = float(support)
        resistance_f = float(resistance)
        current_price = float(ctx['current_price'])
    except (KeyError, TypeError, ValueError):
        return False

    if resistance_f <= support_f:
        return False

    zone_pct = float(ctx.get('range_action_zone_pct', 0.20) or 0.20)
    zone = (resistance_f - support_f) * zone_pct
    return current_price <= support_f + zone or current_price >= resistance_f - zone


def _is_local_range_edge(ctx: SignalContext) -> bool:
    df_indicators = ctx.get('df_indicators')
    if df_indicators is None or len(df_indicators) < 2:
        return False

    try:
        if "high" not in df_indicators.columns or "low" not in df_indicators.columns:
            return False
        current_price = float(ctx['current_price'])
        local_lookback = max(20, min(40, len(df_indicators)))
        local_recent = df_indicators.iloc[-(local_lookback + 1):-1]
        if len(local_recent) == 0:
            local_recent = df_indicators.iloc[-local_lookback:]
        local_hi = float(local_recent["high"].max())
        local_lo = float(local_recent["low"].min())
        local_width = local_hi - local_lo
    except (KeyError, TypeError, ValueError, AttributeError):
        return False

    if local_width <= 0:
        return False

    strategy_config = ctx.get('strategy_config', {}) or {}
    bottom = float(strategy_config.get("range_veto_bottom_pct", 0.25) or 0.25)
    top = float(strategy_config.get("range_veto_top_pct", 0.75) or 0.75)
    local_pos = (current_price - local_lo) / local_width
    if local_pos < -0.10 or local_pos > 1.10:
        return False
    return local_pos <= bottom or local_pos >= top


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
        max_chase_pct = max(base_max_chase_pct, min(0.0070, max(0.0045, atr_pct_now * 0.75)))
    else:
        max_chase_pct = base_max_chase_pct
    ctx['ema_dist_pct'] = ema_dist_pct
    ctx['max_chase_pct'] = max_chase_pct

    # Skip chasing guard at floor/ceiling — sniper handles range reversals
    range_edge = _is_structural_range_edge(ctx) or _is_local_range_edge(ctx)

    if ema_dist_pct > max_chase_pct and not range_edge:
        return {"action": "HOLD", "score": 0.0, "confidence": 0.0, "reason": f"Chasing Guard ({ema_dist_pct:.3%}>{max_chase_pct:.3%})", "weights": {}}

    # --- Institutional Confluence of Danger Guard ---
    # Block only when multiple danger signs agree to prevent over-filtering.
    danger_signals = []

    # 1. Price distance from EMA21 (Overextension)
    price_dist_ema21 = (current_price - ema_21) / ema_21
    max_price_dist = max(0.006, atr_pct_now * 1.8)
    overextended = (action == "BUY" and price_dist_ema21 > max_price_dist) or \
                   (action == "SELL" and price_dist_ema21 < -max_price_dist)
    if overextended:
        danger_signals.append(f"Overextended({price_dist_ema21:.2%})")

    # 2. RSI Extreme (Momentum Exhaustion)
    rsi_val = ctx.get('rsi_14', 50.0)
    rsi_extreme = (action == "BUY" and rsi_val > 72) or \
                  (action == "SELL" and rsi_val < 28)
    if rsi_extreme:
        danger_signals.append(f"RSI-Extreme({rsi_val:.1f})")

    # 3. MACD Momentum Weakening
    macd_diff = ctx.get('macd_diff', 0.0)
    prev_macd_diff = ctx.get('prev_macd_diff', 0.0)
    macd_weakening = (action == "BUY" and macd_diff > 0 and macd_diff < prev_macd_diff) or \
                     (action == "SELL" and macd_diff < 0 and macd_diff > prev_macd_diff)
    if macd_weakening:
        danger_signals.append("MACD-Weakening")

    # 4. Near SR Wall
    sr_score = ctx.get('sr_score', 0.0)
    wall_state = ctx.get('wall_state', {}) or {}
    support_broken = bool(wall_state.get("support_broken"))
    resistance_broken = bool(wall_state.get("resistance_broken"))
    near_wall = (
        (action == "BUY" and sr_score <= -1.0 and not resistance_broken)
        or
        (action == "SELL" and sr_score >= 1.0 and not support_broken)
    )
    if near_wall:
        danger_signals.append(f"Near-Wall(sr={sr_score:.1f})")

    if len(danger_signals) >= 2 and not range_edge:
        reasons = " + ".join(danger_signals)
        return {"action": "HOLD", "score": 0.0, "confidence": 0.0, "reason": f"Confluence of Danger: {reasons}", "weights": {}}

    # Near-extreme guard: block BUY near 30c high, SELL near 30c low
    extreme_lookback = int(strategy_config.get("chase_recent_extreme_lookback") or 30)
    near_extreme_pct = float(strategy_config.get("chase_near_extreme_pct") or 0.0035)

    if near_extreme_pct > 0 and action in {"BUY", "SELL"} and len(df_indicators) >= extreme_lookback:
        recent_ext = df_indicators.tail(extreme_lookback)
        recent_high = float(recent_ext["high"].max())
        recent_low = float(recent_ext["low"].min())

        if recent_high <= 0 or recent_low <= 0:
            pass  # skip near-extreme on bad data
        elif recent_high < current_price / 5 and action == "BUY":
            pass  # stale high from prior symbol — skip
        elif recent_low > current_price * 5 and action == "SELL":
            pass  # stale low from prior symbol — skip
        elif action == "BUY":
            if current_price >= recent_high * (1 - near_extreme_pct):
                ctx['action'] = "HOLD"
                ctx['hold_reason'] = f"Near-Extreme: BUY too close to {extreme_lookback}c high (${recent_high:.3f})"
                return None
        elif action == "SELL":
            if current_price <= recent_low * (1 + near_extreme_pct):
                ctx['action'] = "HOLD"
                ctx['hold_reason'] = f"Near-Extreme: SELL too close to {extreme_lookback}c low (${recent_low:.3f})"
                return None

    # Block entries after extended uninterrupted run (price already exhausted)
    max_consec = int(strategy_config.get("max_consecutive_candles_chase", 4) or 4)
    if action in {"BUY", "SELL"} and len(df_indicators) >= max_consec:
        recent = df_indicators.tail(max_consec)
        if action == "BUY":
            all_green = all(recent['close'].values[i] >= recent['open'].values[i] for i in range(max_consec))
            if all_green and not range_edge:
                return {"action": "HOLD", "score": 0.0, "confidence": 0.0, "reason": f"Chase Guard: {max_consec} consecutive green candles — wait for pullback", "weights": {}}
        elif action == "SELL":
            all_red = all(recent['close'].values[i] <= recent['open'].values[i] for i in range(max_consec))
            if all_red and not range_edge:
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
        if _is_structural_range_edge(ctx) or _is_local_range_edge(ctx):
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
