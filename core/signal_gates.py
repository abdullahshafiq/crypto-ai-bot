from __future__ import annotations

import time

_AGGRESSIVE_SCALP_ALLOW_HOLD_REASON = (
    "conf<",
    "hard score gate",
    "waiting for alignment",
    "ranging market",
    "trend gate",
)

_AGGRESSIVE_SCALP_BLOCK_HOLD_REASON = (
    "loss tilt",
    "scalp hold",
    "cooldown",
    "near-extreme",
    "wall veto",
    "range veto",
    "structural sl too far",
    "r/r too low",
    "spread",
    "confluence of danger",
)


def _is_aggressive_scalp_hold_reason_allowed(hold_reason: str) -> bool:
    reason = str(hold_reason or "").strip().lower()
    if not reason:
        return False
    if any(token in reason for token in _AGGRESSIVE_SCALP_BLOCK_HOLD_REASON):
        return False
    return any(token in reason for token in _AGGRESSIVE_SCALP_ALLOW_HOLD_REASON)


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


def apply_confidence_floor(signal: dict, min_conf_floor: float, strategy_config: dict = None) -> dict:
    _action = signal.get("action")
    if _action not in {"BUY", "SELL"}:
        return signal
    _conf = float(signal.get("confidence", 0.0) or 0.0)
    if _conf >= min_conf_floor:
        return signal
    _reason = str(signal.get("reason", "") or "")
    _is_sniper = any(tag in _reason for tag in ("RangeFloor", "EarlyFloor", "RangeCeil", "EarlyCeil"))
    if _is_sniper:
        _sniper_min = 0.08
        if isinstance(strategy_config, dict):
            _sniper_min = float(strategy_config.get("range_sniper_min_conf", 0.08) or 0.08)
        if _conf >= _sniper_min:
            return signal
    signal["action"] = "HOLD"
    signal["hold_reason"] = f"Weak confidence ({_conf:.1%} < {min_conf_floor:.0%})"
    return signal


def apply_loss_tilt_hard_gate(signal: dict, hard_min: float, raw_score: float = None, strategy_config: dict = None) -> dict:
    _action = signal.get("action")
    if _action not in {"BUY", "SELL"}:
        return signal
    _hard_min = float(hard_min or 0.0)
    if _hard_min <= 0.0:
        return signal
    _score = float(raw_score if raw_score is not None else (signal.get("score", signal.get("confidence", 0.0)) or 0.0))
    _reason = str(signal.get("reason", "") or "")
    _is_sniper = any(tag in _reason for tag in ("RangeFloor", "EarlyFloor", "RangeCeil", "EarlyCeil"))
    if _is_sniper:
        _sniper_min = 0.08
        if isinstance(strategy_config, dict):
            _sniper_min = float(strategy_config.get("range_sniper_min_conf", 0.08) or 0.08)
        _hard_min = min(_hard_min, _sniper_min, 0.30)
    if abs(_score) < _hard_min:
        signal["action"] = "HOLD"
        signal["hold_reason"] = f"Hard score gate ({abs(_score):.2f} < {_hard_min:.2f})"
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


def apply_aggressive_scalp_gate(
    signal: dict,
    state: dict,
    exec_cfg: dict,
    strategy_config: dict,
    executor,
) -> dict:
    """Paper-only gate: flip HOLD -> BUY/SELL when a tight scalp opportunity
    exists that clears fees + buffer, has tight structural SL, and is not
    buying into resistance / selling into support.
    """
    if signal.get("action") != "HOLD":
        return signal
    if bool(signal.get("sr_wall_locked", False)):
        return signal
    if not bool(strategy_config.get("aggressive_scalp_enabled", False)):
        return signal
    if not _is_aggressive_scalp_hold_reason_allowed(signal.get("hold_reason", "")):
        return signal

    score = float(signal.get("score", 0.0) or 0.0)
    min_conf = float(strategy_config.get("aggressive_scalp_min_conf", 0.10) or 0.10)
    if score > min_conf:
        direction = "BUY"
    elif score < -min_conf:
        direction = "SELL"
    else:
        return signal

    price = float(state.get("price", 0.0) or 0.0)
    if price <= 0.0:
        return signal

    tp = float(signal.get("tp", 0.0) or 0.0)
    sl = float(signal.get("sl", 0.0) or 0.0)
    if tp <= 0.0 or sl <= 0.0:
        return signal

    # Expected TP move must exceed round-trip fees + buffer
    fee_rate = float(exec_cfg.get("fee_rate", 0.0006) or 0.0006)
    roundtrip_fee_pct = 2.0 * fee_rate
    buffer_pct = float(strategy_config.get("aggressive_scalp_fee_buffer_pct", 0.0003) or 0.0003)
    min_move_pct = roundtrip_fee_pct + buffer_pct

    if direction == "BUY":
        tp_move_pct = (tp - price) / price
    else:
        tp_move_pct = (price - tp) / price
    if tp_move_pct <= min_move_pct:
        return signal

    # SL distance must be within max_structural_sl_pct * 1.2
    max_structural_sl_pct = float(strategy_config.get("max_structural_sl_pct", 0.0030) or 0.0030)
    if direction == "BUY":
        sl_dist_pct = (price - sl) / price
    else:
        sl_dist_pct = (sl - price) / price
    if sl_dist_pct <= 0.0 or sl_dist_pct > max_structural_sl_pct * 1.2:
        return signal

    # Do not BUY into resistance / SELL into support
    resistance = float(
        signal.get("structure_resistance", 0.0) or signal.get("resistance", 0.0) or 0.0
    )
    support = float(
        signal.get("structure_support", 0.0) or signal.get("support", 0.0) or 0.0
    )
    if direction == "BUY" and resistance > 0.0 and price >= resistance:
        return signal
    if direction == "SELL" and support > 0.0 and price <= support:
        return signal

    # Cooldown respected
    min_seconds_between_trades = float(exec_cfg.get("min_seconds_between_trades", 60) or 60)
    last_trade_ts = float(getattr(executor, "_last_trade_ts", 0.0) or 0.0)
    if last_trade_ts > 0.0 and (time.time() - last_trade_ts) < min_seconds_between_trades:
        return signal

    # All checks passed — flip to aggressive scalp
    signal["action"] = direction
    signal["hold_reason"] = ""
    signal["aggressive_scalp"] = True
    signal["reason"] = f"{signal.get('reason', '')} [AGGRESSIVE_SCALP]"
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
