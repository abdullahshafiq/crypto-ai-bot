from __future__ import annotations

import time
import logging

logger = logging.getLogger(__name__)


def apply_ai_overlay(signal: dict, ai_overlay_state: dict, executor, symbol: str, status_fn) -> dict:
    overlay_bias = str(ai_overlay_state.get("bias", "NEUTRAL") or "NEUTRAL").upper()
    overlay_risk = str(ai_overlay_state.get("risk_mode", "NORMAL") or "NORMAL").upper()
    overlay_avoid = bool(ai_overlay_state.get("avoid_new_entries", False))
    overlay_hold_minutes = int(ai_overlay_state.get("max_hold_minutes", 0) or 0)
    overlay_note = str(ai_overlay_state.get("rationale", "") or "")[:120]

    active_positions = getattr(executor, 'active_positions', [])
    if active_positions:
        current_pos = active_positions[0]
        if overlay_bias == "SHORT_ONLY" and current_pos['side'] == "LONG":
            signal["action"] = "HOLD"
            status_fn("AI EMERGENCY: Liquidating LONG (Bias: SHORT_ONLY)")
            executor.close_all_positions(symbol)
        elif overlay_bias == "LONG_ONLY" and current_pos['side'] == "SHORT":
            signal["action"] = "HOLD"
            status_fn("AI EMERGENCY: Liquidating SHORT (Bias: LONG_ONLY)")
            executor.close_all_positions(symbol)

    if overlay_avoid and signal.get("action") in {"BUY", "SELL"}:
        signal["action"] = "HOLD"
        signal["hold_reason"] = "AI overlay: no new entries"
        signal["reason"] = f"{signal.get('reason', '')} [AI Overlay] {overlay_note}"
    elif overlay_bias == "LONG_ONLY" and signal.get("action") == "SELL":
        signal["reason"] = f"{signal.get('reason', '')} [AI Overlay Soft Bias] {overlay_note}"
    elif overlay_bias == "SHORT_ONLY" and signal.get("action") == "BUY":
        signal["reason"] = f"{signal.get('reason', '')} [AI Overlay Soft Bias] {overlay_note}"

    if signal.get("action") in {"BUY", "SELL"} and overlay_hold_minutes > 0:
        signal["hold_until_ts"] = time.time() + (overlay_hold_minutes * 60)

    return signal


def apply_ai_trade_gate(
    signal: dict,
    ai_orch,
    ai_trade_cfg: dict,
    ai_model: str,
    executor,
    state: dict,
    mtf_context: dict,
    runtime_strategy_config: dict,
    ai_overlay_state: dict,
    last_ai_trade_ts: float,
    last_ai_trade_key: str | None,
    last_ai_trade_resp: dict | None,
    symbol: str,
    status_fn,
) -> tuple:
    if not ai_trade_cfg.get("enabled", False) or signal.get("action") not in {"BUY", "SELL"}:
        return signal, last_ai_trade_ts, last_ai_trade_key, last_ai_trade_resp

    now = time.time()
    ai_resp = None
    use_cached = False
    max_hold_minutes = int(ai_trade_cfg.get("max_hold_minutes", 60))
    on_error = str(ai_trade_cfg.get("on_error", "allow")).strip().lower()
    current_pos = executor.active_positions[0] if getattr(executor, "active_positions", []) else None
    is_reversal = False
    if isinstance(current_pos, dict):
        if (signal["action"] == "BUY" and current_pos.get("side") == "SHORT") or (signal["action"] == "SELL" and current_pos.get("side") == "LONG"):
            is_reversal = True

    should_skip_ai = False
    if current_pos and not is_reversal:
        should_skip_ai = True

    if not should_skip_ai:
        min_ivl = int(ai_trade_cfg.get("min_interval_seconds", 30))
        key = str(signal.get("action"))
        use_cached = (last_ai_trade_key == key) and (last_ai_trade_resp is not None) and ((now - last_ai_trade_ts) < float(min_ivl))

        if use_cached:
            ai_resp = last_ai_trade_resp
        else:
            status_fn("Asking AI to evaluate trade...")
            ai_model_trade = str(ai_trade_cfg.get("model", ai_model))

            ctx = {
                "symbol": symbol,
                "mode": getattr(executor, "label", ""),
                "proposed_action": signal.get("action"),
                "is_reversal": is_reversal,
                "price": state.get("price"),
                "spread_pct": state.get("spread_pct"),
                "ret_30s": state.get("ret_30s"),
                "signal": {
                    "score": signal.get("score"),
                    "confidence": signal.get("confidence"),
                    "tp": signal.get("tp"),
                    "sl": signal.get("sl"),
                    "reason": str(signal.get("reason", ""))[:200],
                },
                "fees": {
                    "fee_rate_per_side": getattr(executor, "fee_rate", None),
                    "fee_slippage_buffer_pct": getattr(executor, "fee_slippage_buffer_pct", None),
                    "fee_edge_multiplier": getattr(executor, "fee_edge_multiplier", None),
                },
                "mtf": mtf_context,
                "position": current_pos or None,
            }
            ai_resp = ai_orch.evaluate_trade(ctx, model=ai_model_trade)
            last_ai_trade_ts = now
            last_ai_trade_key = key
            last_ai_trade_resp = ai_resp

    decision = str((ai_resp or {}).get("decision", "ALLOW")).upper()
    hold_minutes = int((ai_resp or {}).get("hold_minutes", 0) or 0)
    if hold_minutes < 0:
        hold_minutes = 0
    if hold_minutes > max_hold_minutes:
        hold_minutes = max_hold_minutes
    scalp_friendly = (
        signal.get("action") in {"BUY", "SELL"}
        and float(signal.get("confidence", 0.0) or 0.0) >= float(runtime_strategy_config.get("min_conf", 0.05))
        and not bool(ai_overlay_state.get("avoid_new_entries", False))
        and str(ai_overlay_state.get("risk_mode", "NORMAL") or "NORMAL").upper() not in {"HIGH", "EXTREME"}
    )

    if decision == "VETO":
        veto_note = str((ai_resp or {}).get("rationale", "") or "")[:120]
        if scalp_friendly:
            signal["reason"] = f"{signal.get('reason', '')} [AI Soft Veto Ignored] {veto_note}"
            if not use_cached and not should_skip_ai:
                status_fn("AI: SOFT ALLOW")
        else:
            signal["action"] = "HOLD"
            signal["reason"] = f"{signal.get('reason', '')} [AI Veto] {veto_note}"
            if not use_cached and not should_skip_ai:
                status_fn("AI: VETO")
    else:
        if not use_cached and not should_skip_ai:
            status_fn("AI: ALLOW")

    if decision not in {"ALLOW", "VETO"} and on_error == "veto":
        signal["action"] = "HOLD"
        signal["reason"] = f"{signal.get('reason', '')} [AI Error Veto]"

    return signal, last_ai_trade_ts, last_ai_trade_key, last_ai_trade_resp


def apply_regime_veto(signal: dict, is_ai_enabled: bool, regime: str) -> dict:
    if not is_ai_enabled or signal.get("action") == "HOLD":
        return signal
    if regime == "BEARISH" and signal["action"] == "BUY":
        signal["reason"] += " [AI Regime Soft Bias: Bearish]"
    elif regime == "BULLISH" and signal["action"] == "SELL":
        signal["reason"] += " [AI Regime Soft Bias: Bullish]"
    elif regime == "VOLATILE":
        signal["reason"] += " [AI Regime Caution: Volatile]"
    return signal


def dispatch_entry(signal: dict, ai_overlay_state: dict, state: dict, executor, symbol: str) -> None:
    entry_style = str(ai_overlay_state.get('entry_style', 'MIXED')).upper()
    target_price = float(signal.get('entry', state['price']) or state['price'])

    if entry_style == "BUY_PULLBACKS" and signal['action'] == "BUY":
        target_price = min(target_price, state['price'] * 0.999)
    elif entry_style == "SELL_RALLIES" and signal['action'] == "SELL":
        target_price = max(target_price, state['price'] * 1.001)

    if signal['action'] == "BUY" and signal.get("structure_resistance"):
        signal["tp_target"] = float(signal["structure_resistance"]) * 0.999
    elif signal['action'] == "SELL" and signal.get("structure_support"):
        signal["tp_target"] = float(signal["structure_support"]) * 1.001

    executor.place_limit_order(signal, symbol, target_price)