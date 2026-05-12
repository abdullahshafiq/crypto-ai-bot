"""Phase 9: Build the final signal dict from all computed context."""

from __future__ import annotations

from .ctx import SignalContext




def _build_trade_intent(action: str, entry_mode: str, hold_reason: str) -> str:
    """Return a short human-readable intent for the current setup."""
    mode = str(entry_mode or "TREND").upper()
    action = str(action or "HOLD").upper()
    hold_reason_l = str(hold_reason or "").lower()
    wants_retest = "retest" in hold_reason_l or "test" in hold_reason_l

    if "BREAKDOWN_SHORT" in mode:
        return "Wait for retest, then short" if action == "HOLD" and wants_retest else "Breakdown short"
    if "BREAKOUT_SHORT" in mode:
        return "Wait for retest, then short" if action == "HOLD" and wants_retest else "Breakout short"
    if "BREAKOUT_LONG" in mode:
        return "Wait for retest, then long" if action == "HOLD" and wants_retest else "Breakout long"
    if "REVERSAL_SHORT" in mode:
        return "Wait for rejection, then short" if action == "HOLD" else "Reversal short"
    if "REVERSAL_LONG" in mode:
        return "Wait for reclaim, then long" if action == "HOLD" else "Reversal long"
    if action == "BUY":
        return "Trend long"
    if action == "SELL":
        return "Trend short"
    if "SHORT" in mode:
        return "Waiting for short setup"
    if "LONG" in mode:
        return "Waiting for long setup"
    return "Waiting for alignment"

def _build_signal_dict(ctx: SignalContext) -> dict:
    """Construct the signal dict with all metadata. Reads from ctx, returns the signal dict."""
    total_score = ctx['total_score']
    action = ctx['action']
    hold_reason = ctx.get('hold_reason', '')
    entry_mode = ctx.get('entry_mode', 'TREND')
    sr_wall_locked = ctx.get('sr_wall_locked', False)
    signal_reason_suffix = ctx.get('signal_reason_suffix', '')
    support = ctx['support']
    resistance = ctx['resistance']
    current_price = ctx['current_price']
    df_indicators = ctx['df_indicators']
    latest_indicators = ctx['latest_indicators']
    strategy_config = ctx['strategy_config']
    pivot_data = ctx.get('pivot_data')
    wall_state = ctx['wall_state']
    location_score = ctx['location_score']
    location_notes = ctx['location_notes']
    location_levels = ctx['location_levels']
    vpoc = ctx['vpoc']
    anchored_vwap = ctx['anchored_vwap']
    article_sl_override = ctx.get('article_sl_override')

    reason = (
        f"{signal_reason_suffix} | {ctx.get('psar_state_note', '')} | Score:{total_score:.3f} "
        f"SMC:{ctx.get('smc_label', '')} Pivot:Gated:IndicatorsOnly MTF:{ctx.get('mtf_fast_bias', 'NEUTRAL')} "
        f"(MR:{ctx.get('mr_score', 0):.1f} OB:{ctx.get('smc_score', 0):.1f} "
        f"SR:{ctx.get('sr_score', 0):.1f} VWAP:{ctx.get('vwap_score', 0):.1f} "
        f"ADX:{ctx.get('adx_score', 0):.1f} LOC:{ctx.get('location_score', 0):.1f} "
        f"VOL:{ctx.get('volume_delta', 0):.1f} OBV:{ctx.get('obv_score', 0):.1f} "
        f"BB:{ctx.get('bb_score', 0):.1f} MACD:{ctx.get('macd_score', 0):.1f} "
        f"PA:{ctx.get('pa_score', 0):.1f} KDJ:{ctx.get('kdj_score', 0):.1f} "
        f"ST:{ctx.get('st_score', 0):.1f} DIV:{ctx.get('divergence_state', 'NONE')} "
        f"CVD:{ctx.get('cvd_state', '')} EXH:{ctx.get('momentum_exhaustion', '')} "
        f"RSI:{ctx.get('mtf_rsi_bias', 'NEUTRAL')} BR:{ctx.get('body_ratio_score', 0):.2f})"
    )

    psar_val_raw = latest_indicators.get('psar_streak', 0)
    psar_streak = int(psar_val_raw) if psar_val_raw is not None and str(psar_val_raw) != 'nan' else 0

    signal = {
        "action": action,
        "score": total_score,
        "confidence": min(abs(total_score), 1.0),
        "reason": reason,
        "intent": _build_trade_intent(action, entry_mode, hold_reason),
        "psar_streak": psar_streak,
        "psar_exit": True if psar_streak != 0 else False,
        "psar_closed_bull": bool(ctx.get('psar_closed_bull', True)),
        "psar_live_bull": bool(ctx.get('psar_live_bull', True)),
        "psar_state_note": ctx.get('psar_state_note', ''),
        "article_sl_override": float(article_sl_override) if article_sl_override is not None else None,
        "market_bias": ctx.get('mtf_fast_bias', 'NEUTRAL') if ctx.get('mtf_fast_bias') != "NEUTRAL" else "NEUTRAL",
        "mtf_fast_bias": ctx.get('mtf_fast_bias', 'NEUTRAL'),
        "mtf_rsi_bias": ctx.get('mtf_rsi_bias', 'NEUTRAL'),
        "mtf_rsi_score": float(ctx.get('mtf_rsi_score', 0.0)),
        "momentum_exhaustion": ctx.get('momentum_exhaustion', ''),
        "cvd_state": ctx.get('cvd_state', ''),
        "body_ratio_score": float(ctx.get('body_ratio_score', 0.0)),
        "vpoc": float(vpoc) if vpoc else 0.0,
        "anchored_vwap": float(anchored_vwap) if anchored_vwap else 0.0,
        "order_block": ctx.get('ob_context', {}),
        "hold_reason": hold_reason,
    }

    ctx['signal'] = signal

    gate_locked = bool(hold_reason) or sr_wall_locked
    if support is not None:
        signal["structure_support"] = float(support)
    if resistance is not None:
        signal["structure_resistance"] = float(resistance)

    atr_pct = float(latest_indicators.get("atr_pct", 0.0) or 0.0) / 100.0
    latest_price = float(df_indicators["close"].iloc[-1]) if len(df_indicators) > 0 else 0.0
    if atr_pct > 0 and latest_price > 0:
        signal["atr"] = atr_pct * latest_price
    if isinstance(pivot_data, dict):
        signal["pivot_classic"] = dict(pivot_data.get("classic", {}) or {})
    signal["wall_state"] = wall_state
    signal["market_location"] = {
        "score": float(location_score),
        "notes": location_notes,
        "levels": location_levels,
    }
    signal["sr_wall_locked"] = bool(sr_wall_locked)

    # Apply EMA 200 confidence adjustment for trend alignment
    ema_200_bull = ctx.get('ema_200_bull', False)
    if signal.get("action") in {"BUY", "SELL"} and ema_200_bull is not None:
        base_conf = float(signal.get("confidence", 0.0))
        if ema_200_bull and signal["action"] == "BUY":
            # Aligned with bullish trend: boost confidence
            signal["confidence"] = min(base_conf * 1.10, 1.0)
            signal["reason"] = f"{signal['reason']} [TrendAlign:BULL↑]"
        elif ema_200_bull and signal["action"] == "SELL":
            # Against bullish trend: slightly reduce confidence
            signal["confidence"] = max(base_conf * 0.95, base_conf - 0.05)
            signal["reason"] = f"{signal['reason']} [TrendContra:BULL]"
        elif not ema_200_bull and signal["action"] == "SELL":
            # Aligned with bearish trend: boost confidence
            signal["confidence"] = min(base_conf * 1.10, 1.0)
            signal["reason"] = f"{signal['reason']} [TrendAlign:BEAR↓]"
        elif not ema_200_bull and signal["action"] == "BUY":
            # Against bearish trend: slightly reduce confidence
            signal["confidence"] = max(base_conf * 0.95, base_conf - 0.05)
            signal["reason"] = f"{signal['reason']} [TrendContra:BEAR]"

    return signal