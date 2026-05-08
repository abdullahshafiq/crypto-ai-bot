from __future__ import annotations

import time


def compute_loss_tilt_override(
    consecutive_losses: int,
    base_min_conf: float,
    strategy_config: dict,
) -> dict:
    tilt_min_losses = max(1, int(strategy_config.get("loss_tilt_min_losses", 3) or 3))
    tilt_pause_losses = max(tilt_min_losses + 1, int(strategy_config.get("loss_tilt_pause_losses", 5) or 5))
    tilt_pause_minutes = max(1, int(strategy_config.get("loss_tilt_pause_minutes", 15) or 15))

    overrides = {}
    pauses = {}

    if consecutive_losses >= tilt_min_losses:
        loss_tilt_depth = max(0, consecutive_losses - tilt_min_losses)
        overrides["min_conf"] = min(
            0.90,
            base_min_conf + 0.05 + (0.03 * loss_tilt_depth),
        )
        overrides["entry_min_confidence_hard"] = min(
            0.90,
            float(strategy_config.get("entry_min_confidence_hard", 0.20) or 0.20) + 0.05 + (0.03 * loss_tilt_depth),
        )
        overrides["midrange_min_score"] = max(
            float(strategy_config.get("midrange_min_score", 0.28) or 0.28),
            0.32,
        )
        overrides["session_block_min_score"] = max(
            float(strategy_config.get("session_block_min_score", 0.35) or 0.35),
            0.35,
        )

    if consecutive_losses >= tilt_pause_losses:
        pauses["tilt_pause_until"] = time.time() + float(tilt_pause_minutes * 60)
        pauses["tilt_pause_minutes"] = tilt_pause_minutes
        pauses["should_pause"] = True
    else:
        pauses["should_pause"] = False

    return {"overrides": overrides, "pauses": pauses}


def apply_loss_tilt_pause(signal: dict, loss_tilt_pause_until: float) -> dict:
    if time.time() < loss_tilt_pause_until and signal.get("action") in {"BUY", "SELL"}:
        signal["action"] = "HOLD"
        signal["hold_reason"] = f"Consecutive loss tilt: entry pause"
        signal["reason"] = f"{signal.get('reason', '')} LOSS_TILT_PAUSE"
    return signal


def apply_scalp_hold_guard(signal: dict, executor, exec_cfg: dict) -> dict:
    _scalp_min_hold = float(exec_cfg.get("scalp_min_hold_seconds", 30) or 30)
    _active_pos = getattr(executor, "active_positions", [])
    if _active_pos and signal.get("action") in {"BUY", "SELL"}:
        _pos_entry_ts = float(_active_pos[0].get("entry_ts", 0.0) or 0.0)
        _pos_age = time.time() - _pos_entry_ts
        _pos_side = str(_active_pos[0].get("side", "")).upper()
        _is_reversal_signal = (
            (_pos_side == "LONG" and signal["action"] == "SELL")
            or (_pos_side == "SHORT" and signal["action"] == "BUY")
        )
        if _is_reversal_signal and _pos_age < _scalp_min_hold:
            signal["action"] = "HOLD"
            signal["hold_reason"] = f"Scalp hold guard: {int(_scalp_min_hold - _pos_age)}s remaining"
            signal["reason"] = f"{signal.get('reason', '')} SCALP_HOLD_GUARD"
    return signal


def apply_confidence_floor(signal: dict, min_conf_floor: float) -> dict:
    if signal.get("action") in {"BUY", "SELL"} and float(signal.get("confidence", 0.0) or 0.0) < min_conf_floor:
        signal["action"] = "HOLD"
        signal["hold_reason"] = f"Weak confidence ({float(signal.get('confidence', 0.0) or 0.0):.1%} < {min_conf_floor:.0%})"
    return signal


def apply_reward_risk_gate(signal: dict, min_rr: float) -> dict:
    if signal.get("action") not in {"BUY", "SELL"}:
        return signal
    rr = float(signal.get("reward_risk", 0.0) or 0.0)
    if rr <= 0.0 or rr < min_rr:
        signal["action"] = "HOLD"
        signal["hold_reason"] = f"R/R too low ({rr:.2f} < {min_rr:.2f})"
        signal["reason"] = f"{signal.get('reason', '')} RR_GATE"
    return signal


def format_gate_trace(signal: dict, ai_overlay_state: dict, is_paused: bool, runtime_strategy_config: dict, loss_tilt_pause_until: float) -> str:
    gate_notes = []
    hold_reason = str(signal.get("hold_reason", "") or "")
    if hold_reason:
        gate_notes.append(hold_reason)
    if bool(signal.get("sr_wall_locked", False)):
        gate_notes.append("SR_WALL_LOCK")
    rejection = signal.get("rejection_confirmation")
    if isinstance(rejection, dict):
        if not bool(rejection.get("confirmed", True)):
            rejection_reason = str(rejection.get("reason", "") or "")
            if len(rejection_reason) > 48:
                rejection_reason = rejection_reason[:45] + "..."
            gate_notes.append(f"REJECTION:{rejection_reason}")
        elif rejection.get("mode"):
            gate_notes.append(f"REJECTION_OK:{str(rejection.get('mode', ''))[:16]}")
    if time.time() < loss_tilt_pause_until:
        gate_notes.append("LOSS_TILT_PAUSE")
    if bool(ai_overlay_state.get("avoid_new_entries", False)):
        gate_notes.append("AI_NO_NEW_ENTRIES")
    min_conf_floor = float(runtime_strategy_config.get("min_conf", 0.05))
    if float(signal.get("confidence", 0.0) or 0.0) < min_conf_floor:
        gate_notes.append(f"CONF<{min_conf_floor:.2f}")
    min_rr = float(runtime_strategy_config.get("min_reward_risk", 0.0) or 0.0)
    rr = float(signal.get("reward_risk", 0.0) or 0.0)
    if min_rr > 0.0 and 0.0 < rr < min_rr:
        gate_notes.append(f"RR<{min_rr:.2f}")
    if bool(is_paused):
        gate_notes.append("PAUSED")

    if gate_notes:
        gate_text = " | ".join(gate_notes)
        if len(gate_text) > 160:
            gate_text = gate_text[:157] + "..."
        return gate_text
    return ""