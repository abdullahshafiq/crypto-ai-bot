"""Entry confirmation gates: strike zone check, order block gate, trend confirmation (2-of-3), rejection confirmation gate."""

from __future__ import annotations

from ..ctx import SignalContext

def apply_strike_zone_check(ctx: SignalContext) -> None:
    """Apply strike zone override check for action zone entries."""
    strategy_config = ctx['strategy_config']
    action = ctx.get('action', 'HOLD')
    current_price = ctx['current_price']
    support = ctx.get('support')
    resistance = ctx.get('resistance')
    wall_state = ctx.get('wall_state', {})
    psar_bull = ctx.get('psar_bull', True)
    mtf_macd_bear = ctx.get('mtf_macd_bear', False)
    mtf_structure_bear = ctx.get('mtf_structure_bear', False)
    mtf_macd_bull = ctx.get('mtf_macd_bull', False)
    mtf_structure_bull = ctx.get('mtf_structure_bull', False)
    macd_diff_val = ctx.get('macd_diff', 0.0)
    bull_div = ctx.get('bull_div', False)
    bear_div = ctx.get('bear_div', False)
    df_indicators = ctx['df_indicators']
    range_action_zone_pct = ctx.get('range_action_zone_pct', 0.20)

    range_action_zone_pct = max(0.05, min(0.45, range_action_zone_pct))

    in_action_zone = False
    if support and resistance:
        range_w = float(resistance) - float(support)
        zone_size = range_w * range_action_zone_pct
        support_edge = current_price <= float(support) + zone_size
        resistance_edge = current_price >= float(resistance) - zone_size
        support_zone_top = float(wall_state.get("support_zone_top") or (float(support) * 1.0015))
        resistance_zone_bottom = float(wall_state.get("resistance_zone_bottom") or (float(resistance) * 0.9985))
        support_reclaim = bool(wall_state.get("support_touching")) or current_price <= support_zone_top
        resistance_reclaim = bool(wall_state.get("resistance_touching")) or current_price >= resistance_zone_bottom
        in_action_zone = (action == "BUY" and support_edge and support_reclaim) or \
                         (action == "SELL" and resistance_edge and resistance_reclaim)

    ctx['in_action_zone'] = in_action_zone

    if not in_action_zone or action not in {"BUY", "SELL"}:
        return

    macd_fast_bull = macd_diff_val > 0
    macd_fast_bear = macd_diff_val < 0
    current_vol = float(df_indicators['volume'].iloc[-1])
    avg_vol_fast = df_indicators['volume'].rolling(window=10).mean().iloc[-1]
    volume_surge = current_vol > avg_vol_fast * 1.1
    ema_9_val = ctx.get('ema_9_val', ctx['latest_indicators'].get('ema_9', current_price))
    price_above_ema9 = current_price > ema_9_val
    price_below_ema9 = current_price < ema_9_val

    rsi_val = float(ctx['latest_indicators'].get('rsi', 50) or 50)
    rsi_ob = rsi_val > 65
    rsi_os = rsi_val < 35

    if action == "BUY":
        bull_momentum = psar_bull or macd_fast_bull or (rsi_os and volume_surge)
        if mtf_macd_bear and mtf_structure_bear and not bull_div:
            ctx['action'] = "HOLD"
            ctx['hold_reason'] = "MTF Gate: 15m/10m MACD bearish with lower-high structure"
        if not (price_above_ema9 and bull_momentum):
            ctx['action'] = "HOLD"
            ctx['hold_reason'] = "Strike Zone: Waiting for Price>EMA9 + Bull Momentum (SAR/Hist/RSI)"
    elif action == "SELL":
        bear_momentum = (not psar_bull) or macd_fast_bear or (rsi_ob and volume_surge)
        if mtf_macd_bull and mtf_structure_bull and not bear_div:
            ctx['action'] = "HOLD"
            ctx['hold_reason'] = "MTF Gate: 15m/10m MACD bullish with higher-low structure"
        if not (price_below_ema9 and bear_momentum):
            ctx['action'] = "HOLD"
            ctx['hold_reason'] = "Strike Zone: Waiting for Price<EMA9 + Bear Momentum (SAR/Hist/RSI)"


def apply_ob_gate(ctx: SignalContext) -> None:
    """Apply order block midpoint gate."""
    action = ctx.get('action', 'HOLD')
    if action not in {"BUY", "SELL"}:
        return
    ob_context = ctx.get('ob_context', {})
    ob_active = bool(ob_context.get("active"))
    ob_mid = ob_context.get("mid")
    ob_dir = str(ob_context.get("direction", "NEUTRAL") or "NEUTRAL").upper()
    if not (ob_active and ob_mid):
        return

    current_price = ctx['current_price']
    strategy_config = ctx['strategy_config']
    ob_mid = float(ob_mid)
    ob_mid_tolerance = float(strategy_config.get("ob_midpoint_tolerance_pct", 0.0015) or 0.0015)
    near_ob_mid = abs(current_price - ob_mid) / ob_mid <= ob_mid_tolerance

    if action == "BUY" and ob_dir == "BULLISH":
        if not near_ob_mid:
            ctx['action'] = "HOLD"
            ctx['hold_reason'] = f"OB Gate: wait for 50% bullish OB retest near {ob_mid:.3f}"
        else:
            suffix = ctx.get('signal_reason_suffix', '')
            ctx['signal_reason_suffix'] = suffix + " [OB Mid Retest]"
    elif action == "SELL" and ob_dir == "BEARISH":
        if not near_ob_mid:
            ctx['action'] = "HOLD"
            ctx['hold_reason'] = f"OB Gate: wait for 50% bearish OB retest near {ob_mid:.3f}"
        else:
            suffix = ctx.get('signal_reason_suffix', '')
            ctx['signal_reason_suffix'] = suffix + " [OB Mid Retest]"


def apply_trend_confirmation_gate(ctx: SignalContext) -> None:
    """2-of-3 EMA/PSAR/MACD confirmation gate for non-strike-zone entries."""
    action = ctx.get('action', 'HOLD')
    in_action_zone = ctx.get('in_action_zone', False)
    if action not in {"BUY", "SELL"} or in_action_zone:
        return

    ema_9_val = ctx.get('ema_9_val', ctx['current_price'])
    ema_21_val = ctx.get('ema_21', ctx['latest_indicators'].get('ema_21', ctx['current_price']))
    psar_bull = ctx.get('psar_bull', True)
    macd_final_bull = ctx.get('macd_final_bull', True)
    macd_final_bear = ctx.get('macd_final_bear', False)
    mtf_macd_bull = ctx.get('mtf_macd_bull', False)
    mtf_macd_bear = ctx.get('mtf_macd_bear', False)
    mtf_structure_bull = ctx.get('mtf_structure_bull', False)
    mtf_structure_bear = ctx.get('mtf_structure_bear', False)
    macd_diff_val = ctx.get('macd_diff', 0.0)
    current_price = ctx['current_price']
    pa_score = ctx.get('pa_score', 0.0)

    ema_cross_bull = ema_9_val > ema_21_val
    bull_votes = int(ema_cross_bull) + int(psar_bull) + int(macd_final_bull)
    bear_votes = int(not ema_cross_bull) + int(not psar_bull) + int(macd_final_bear)

    _not_at_range_bottom = True
    _sup = ctx.get('support')
    _res = ctx.get('resistance')
    if _sup and _res and float(_res) > float(_sup):
        _pos = (current_price - float(_sup)) / (float(_res) - float(_sup))
        _not_at_range_bottom = _pos > 0.25

    sell_momentum_escape = (
        current_price < ema_21_val
        and pa_score <= 0
        and (not psar_bull or macd_diff_val <= 0)
        and _not_at_range_bottom
    )
    buy_momentum_escape = (
        current_price > ema_21_val
        and (psar_bull or macd_diff_val > 0)
        and pa_score >= 0
    )

    if action == "BUY" and bull_votes < 2:
        if buy_momentum_escape:
            pass
        else:
            ctx['action'] = "HOLD"
            ctx['hold_reason'] = "Trend Gate: Waiting for 2-of-3 EMA/PSAR/MACD alignment"
    elif action == "BUY" and mtf_macd_bear and mtf_structure_bear:
        ctx['action'] = "HOLD"
        ctx['hold_reason'] = "MTF Gate: 15m/10m MACD bearish with lower-high structure"
    elif action == "SELL" and bear_votes < 2:
        if sell_momentum_escape:
            pass
        else:
            ctx['action'] = "HOLD"
            ctx['hold_reason'] = "Trend Gate: Waiting for 2-of-3 EMA/PSAR/MACD alignment"
    elif action == "SELL" and mtf_macd_bull and mtf_structure_bull:
        ctx['action'] = "HOLD"
        ctx['hold_reason'] = "MTF Gate: 15m/10m MACD bullish with higher-low structure"


def _apply_rejection_confirmation_gate(
    signal: dict,
    df_indicators,
    strategy_config: dict,
    support: float = None,
    resistance: float = None,
) -> tuple[dict, bool]:
    """
    Final entry filter for reversal-style entries.
    Requires a confirmation candle before reversal entries are allowed to execute.
    """
    cfg = dict((strategy_config or {}).get("rejection_confirmation", {}) or {})
    if not bool(cfg.get("enabled", False)):
        return signal, False

    action = str(signal.get("action", "HOLD") or "HOLD").upper()
    if action not in {"BUY", "SELL"}:
        return signal, False

    reason = str(signal.get("reason", "") or "")
    entry_mode = "TREND"
    if "[Mode:" in reason:
        try:
            entry_mode = reason.split("[Mode:")[-1].split("]")[0].strip().upper() or "TREND"
        except Exception:
            entry_mode = "TREND"

    if "REVERSAL" not in entry_mode:
        return signal, False

    if df_indicators is None or len(df_indicators) < 3:
        signal["action"] = "HOLD"
        signal["hold_reason"] = "RejectionGate: insufficient candles for confirmation"
        signal["reason"] = f"{reason} [RejectionGate: insufficient candles]"
        return signal, True

    min_bars_away = max(1, int(cfg.get("min_bars_away", 2) or 2))
    macd_threshold = float(cfg.get("macd_threshold", 0.0) or 0.0)
    pullback_pct = float(cfg.get("pullback_pct", 0.002) or 0.002)
    require_psar_flip = bool(cfg.get("require_psar_flip", True))

    last = df_indicators.iloc[-1]
    prev = df_indicators.iloc[-2]
    current_price = float(last["close"])
    current_open = float(last["open"])
    current_macd = float(last["macd"]) if "macd" in df_indicators.columns else 0.0
    current_macd_diff = float(last["macd_diff"]) if "macd_diff" in df_indicators.columns else 0.0
    prev_macd_diff = float(prev["macd_diff"]) if "macd_diff" in df_indicators.columns else 0.0
    current_psar = float(last["psar"]) if "psar" in df_indicators.columns else current_price
    recent = df_indicators.tail(min_bars_away)

    if action == "BUY":
        support_level = float(support) if support is not None else None
        macd_confirmed = current_macd > macd_threshold and current_macd_diff > 0 and prev_macd_diff <= 0
        psar_confirmed = (current_price > current_psar) if require_psar_flip else True
        candle_confirmed = current_price >= current_open
        pullback_confirmed = True
        if support_level and support_level > 0:
            pullback_confirmed = bool((recent["low"] > support_level * (1 + pullback_pct)).all())
        confirmed = macd_confirmed and psar_confirmed and candle_confirmed and pullback_confirmed
        if not confirmed:
            reasons = []
            if not macd_confirmed: reasons.append("MACD not yet bullish")
            if not psar_confirmed: reasons.append("PSAR has not flipped bullish")
            if not candle_confirmed: reasons.append("Candle not bullish")
            if not pullback_confirmed: reasons.append("Price still too close to support")
            signal["action"] = "HOLD"
            signal["hold_reason"] = f"RejectionGate: {' | '.join(reasons)}"
            signal["reason"] = f"{reason} [RejectionGate: {' | '.join(reasons)}]"
            signal["rejection_confirmation"] = {"confirmed": False, "mode": entry_mode, "action": action, "reason": " | ".join(reasons)}
            return signal, True
    else:
        resistance_level = float(resistance) if resistance is not None else None
        macd_confirmed = current_macd < macd_threshold and current_macd_diff < 0 and prev_macd_diff >= 0
        psar_confirmed = (current_price < current_psar) if require_psar_flip else True
        candle_confirmed = current_price <= current_open
        pullback_confirmed = True
        if resistance_level and resistance_level > 0:
            pullback_confirmed = bool((recent["high"] < resistance_level * (1 - pullback_pct)).all())
        confirmed = macd_confirmed and psar_confirmed and candle_confirmed and pullback_confirmed
        if not confirmed:
            reasons = []
            if not macd_confirmed: reasons.append("MACD not yet bearish")
            if not psar_confirmed: reasons.append("PSAR has not flipped bearish")
            if not candle_confirmed: reasons.append("Candle not bearish")
            if not pullback_confirmed: reasons.append("Price still too close to resistance")
            signal["action"] = "HOLD"
            signal["hold_reason"] = f"RejectionGate: {' | '.join(reasons)}"
            signal["reason"] = f"{reason} [RejectionGate: {' | '.join(reasons)}]"
            signal["rejection_confirmation"] = {"confirmed": False, "mode": entry_mode, "action": action, "reason": " | ".join(reasons)}
            return signal, True

    signal["reason"] = f"{reason} [RejectionGate: Confirmed]"
    signal["rejection_confirmation"] = {"confirmed": True, "mode": entry_mode, "action": action, "reason": "Confirmed"}
    return signal, True