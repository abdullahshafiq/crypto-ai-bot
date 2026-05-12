"""Phases 10-11: Mean reversion / wick sweep setup overrides and stop-loss / take-profit computation."""

from __future__ import annotations

from .ctx import SignalContext
from .setups import _detect_mean_reversion_setup, _detect_wick_sweep_setup
from .scalper import detect_best_scalper_signal
from ..mtf import _pick_structural_levels


def _apply_setup_overrides(signal: dict, ctx: SignalContext) -> None:
    """Apply mean reversion, wick sweep, and ultimate scalper setups to the signal dict. Mutates signal in-place."""
    df_indicators = ctx['df_indicators']
    current_price = ctx['current_price']
    strategy_config = ctx['strategy_config']
    support = ctx['support']
    resistance = ctx['resistance']
    wall_state = ctx['wall_state']
    mtf_fast_bias = ctx.get('mtf_fast_bias', 'NEUTRAL')
    hold_reason = ctx.get('hold_reason', '')
    sr_wall_locked = ctx.get('sr_wall_locked', False)
    gate_locked = bool(hold_reason) or sr_wall_locked

    # --- Setup 1: Best Scalper (Volman/Brooks/Cameron Hybrid) ---
    scalp_setup = detect_best_scalper_signal(df_indicators, strategy_config, support=support, resistance=resistance)
    if scalp_setup.get("triggered"):
        signal["scalper_setup"] = scalp_setup
        signal["reason"] = f"{signal['reason']} Scalp:{scalp_setup['reason']}"
        signal["score"] = float(signal["score"]) + float(scalp_setup.get("score", 0.0))
        signal["confidence"] = min(abs(float(signal["score"])), 1.0)
        
        # Override action — S/R scalps are contra-MTF by design (shorting at resistance when MTF is still bullish)
        if not gate_locked:
            if scalp_setup["direction"] == "LONG" and mtf_fast_bias != "SHORT_ONLY":
                signal["action"] = "BUY"
                signal["entry_mode"] = "SR_SCALP"
            elif scalp_setup["direction"] == "SHORT" and mtf_fast_bias != "LONG_ONLY":
                signal["action"] = "SELL"
                signal["entry_mode"] = "SR_SCALP"
            elif scalp_setup["direction"] == "SHORT" and mtf_fast_bias == "LONG_ONLY":
                # Resistance rejection overrides bullish MTF bias — price AT resistance is the signal
                signal["action"] = "SELL"
                signal["entry_mode"] = "SR_SCALP_CONTRA"

    # --- Setup 2: Mean Reversion ---
    mr_setup = _detect_mean_reversion_setup(df_indicators, strategy_config)
    if mr_setup.get("triggered"):
        signal["mean_reversion"] = {
            "direction": mr_setup.get("direction"),
            "reason": mr_setup.get("reason"),
        }
        signal["reason"] = f"{signal['reason']} MR:{mr_setup.get('reason', '')}"
        signal["score"] = float(signal["score"]) + float(mr_setup.get("score", 0.0) or 0.0)
        signal["confidence"] = min(abs(float(signal["score"])), 1.0)
        mr_long_allowed = (ctx['sr_score'] > -1.0) and not bool(wall_state.get("resistance_touching"))
        mr_short_allowed = (ctx['sr_score'] < 1.0) and not bool(wall_state.get("support_touching"))
        if not gate_locked and mr_setup.get("direction") == "LONG" and mtf_fast_bias != "SHORT_ONLY" and mr_long_allowed:
            signal["action"] = "BUY" if signal["action"] == "HOLD" or float(signal["score"]) > 0 else signal["action"]
        elif not gate_locked and mr_setup.get("direction") == "SHORT" and mtf_fast_bias != "LONG_ONLY" and mr_short_allowed:
            signal["action"] = "SELL" if signal["action"] == "HOLD" or float(signal["score"]) < 0 else signal["action"]

    wick_setup = _detect_wick_sweep_setup(
        df_indicators, current_price,
        support=support, resistance=resistance, config=strategy_config,
    )
    ctx['wick_setup'] = wick_setup
    if wick_setup.get("triggered"):
        signal["wick_sweep"] = {
            "direction": wick_setup.get("direction"),
            "reason": wick_setup.get("reason"),
        }
        signal["reason"] = f"{signal['reason']} Wick:{wick_setup.get('reason', '')}"
        signal["score"] = float(signal["score"]) + float(wick_setup.get("score", 0.0) or 0.0)
        signal["confidence"] = min(abs(float(signal["score"])), 1.0)
        if wick_setup.get("sl") is not None:
            signal["sl"] = float(wick_setup.get("sl"))
            signal["sl_source"] = "wick_sweep"
        wick_long_allowed = (ctx['sr_score'] > -1.0) and not bool(wall_state.get("resistance_touching"))
        wick_short_allowed = (ctx['sr_score'] < 1.0) and not bool(wall_state.get("support_touching"))
        if not gate_locked and wick_setup.get("direction") == "LONG" and mtf_fast_bias != "SHORT_ONLY" and wick_long_allowed:
            signal["action"] = "BUY" if signal["action"] == "HOLD" or float(signal["score"]) > 0 else signal["action"]
        elif not gate_locked and wick_setup.get("direction") == "SHORT" and mtf_fast_bias != "LONG_ONLY" and wick_short_allowed:
            signal["action"] = "SELL" if signal["action"] == "HOLD" or float(signal["score"]) < 0 else signal["action"]


def _compute_sl_tp(signal: dict, ctx: SignalContext) -> dict:
    """Compute stop-loss and take-profit for the signal. Returns modified signal dict."""
    current_price = ctx['current_price']
    support = ctx['support']
    resistance = ctx['resistance']
    strategy_config = ctx['strategy_config']
    pivot_data = ctx.get('pivot_data')
    article_sl_override = ctx.get('article_sl_override')
    wick_setup = ctx.get('wick_setup', {})
    max_sl_pct = float(strategy_config.get("max_structural_sl_pct", 0.0040) or 0.0040)
    atr_pct_now = float(ctx.get("atr_pct_now", 0.0) or 0.0)
    atr_buffer = current_price * atr_pct_now * 1.5 if current_price > 0 and atr_pct_now > 0 else 0.0
    strong_support, strong_resistance = _pick_structural_levels(
        current_price,
        ctx.get("mtf_context"),
        pivot_data,
        max_sl_pct=max_sl_pct,
    )

    stop_buffer = 0.0005
    final_action = signal["action"]
    if article_sl_override and final_action in {"BUY", "SELL"}:
        signal["sl"] = float(article_sl_override)
    elif final_action == "BUY":
        anchor_support = strong_support if strong_support is not None else support
        if anchor_support is not None:
            signal["sl"] = max(0.0, float(anchor_support) - atr_buffer)
            signal["sl_source"] = "strong_support_atr" if strong_support is not None else "support_atr"
    elif final_action == "SELL":
        anchor_resistance = strong_resistance if strong_resistance is not None else resistance
        if anchor_resistance is not None:
            signal["sl"] = float(anchor_resistance) + atr_buffer
            signal["sl_source"] = "strong_resistance_atr" if strong_resistance is not None else "resistance_atr"
        classic_pivots = pivot_data.get("classic", {}) if isinstance(pivot_data, dict) else {}
        try:
            r1_level = float(classic_pivots.get("r1", 0.0) or 0.0)
        except (TypeError, ValueError):
            r1_level = 0.0
        if r1_level > current_price and float(signal["sl"]) < r1_level * 1.001:
            signal["sl"] = r1_level * 1.002
            signal["sl_source"] = "pivot_r1_guard"

    final_action = signal["action"]
    if final_action in {"BUY", "SELL"} and signal.get("sl") and current_price > 0:
        max_sl_pct = float(strategy_config.get("max_structural_sl_pct", 0.0040) or 0.0040)
        fallback_sl_pct = float(strategy_config.get("sl_pct", 0.0015) or 0.0015)
        sl_dist_pct = abs(float(signal["sl"]) - float(current_price)) / float(current_price)

        min_sl_pct = float(strategy_config.get("sl_pct", 0.0025) or 0.0025)
        if sl_dist_pct < min_sl_pct:
            if final_action == "BUY":
                signal["sl"] = current_price * (1 - min_sl_pct)
            else:
                signal["sl"] = current_price * (1 + min_sl_pct)
            sl_dist_pct = min_sl_pct

        tp_pct_cfg = float(strategy_config.get("tp_pct", 0.0025) or 0.0025)
        if final_action == "BUY" and resistance is not None:
            structural_tp = float(resistance) * (1 - stop_buffer)
            structural_tp_pct = (structural_tp - current_price) / current_price
            if structural_tp_pct > 0.001:
                signal["tp"] = structural_tp
                tp_pct_used = structural_tp_pct
            else:
                signal["tp"] = current_price * (1 + tp_pct_cfg)
                tp_pct_used = tp_pct_cfg
        elif final_action == "SELL" and support is not None:
            structural_tp = float(support) * (1 + stop_buffer)
            structural_tp_pct = (current_price - structural_tp) / current_price
            if structural_tp_pct > 0.001:
                signal["tp"] = structural_tp
                tp_pct_used = structural_tp_pct
            else:
                signal["tp"] = current_price * (1 - tp_pct_cfg)
                tp_pct_used = tp_pct_cfg
        else:
            signal["tp"] = float(current_price) * (1 + tp_pct_cfg) if final_action == "BUY" else float(current_price) * (1 - tp_pct_cfg)
            tp_pct_used = tp_pct_cfg

        reward_risk = tp_pct_used / sl_dist_pct if sl_dist_pct > 0 else 0.0
        signal["structural_sl_pct"] = sl_dist_pct
        signal["reward_risk"] = reward_risk
        signal["reason"] = f"{signal['reason']} RR:{reward_risk:.2f} SLd:{sl_dist_pct:.2%}"
        wick_mode = bool(wick_setup.get("triggered"))
        if sl_dist_pct > max_sl_pct and not wick_mode:
            signal["action"] = "HOLD"
            signal["hold_reason"] = f"Structural SL too far ({sl_dist_pct:.2%} > {max_sl_pct:.2%})"
            signal["reason"] = f"{signal['reason']} SLTooFar:{sl_dist_pct:.2%}"

    return signal
