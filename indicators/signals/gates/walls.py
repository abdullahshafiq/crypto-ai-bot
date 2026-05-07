"""Wall/SR gates: SR wall veto, range position veto, entry mode classification, wall rejection rescue, midrange policy."""

from __future__ import annotations

from ..ctx import SignalContext
import numpy as np
import logging

from safety import sr_wall_escape_ready as _sr_wall_escape_ready

logger = logging.getLogger(__name__)


def apply_sr_wall_veto(ctx: SignalContext) -> None:
    """Apply SR wall veto — mutates ctx['action'] and ctx['hold_reason'] in place."""
    strategy_config = ctx['strategy_config']
    action = ctx.get('action', 'HOLD')
    if not bool(strategy_config.get("sr_wall_veto_enabled", True)):
        return
    if action not in {"BUY", "SELL"}:
        return

    sr_score = ctx.get('sr_score', 0.0)
    total_score = ctx.get('total_score', 0.0)
    mtf_fast_bias = ctx.get('mtf_fast_bias', 'NEUTRAL')
    macd_diff_val = ctx.get('macd_diff', 0.0)
    psar_bull = ctx.get('psar_bull', True)
    current_price = ctx['current_price']
    ema_9_val = ctx.get('ema_9_val', ctx['latest_indicators'].get('ema_9', current_price))
    wall_escape_gate = float(strategy_config.get("wall_breakout_score_gate", 0.15) or 0.15)

    if sr_score >= 1.0 and action == "SELL" and not _sr_wall_escape_ready(
        action, sr_score, total_score, mtf_fast_bias, macd_diff_val, psar_bull, current_price, ema_9_val, wall_escape_gate,
    ):
        ctx['action'] = "HOLD"
        ctx['hold_reason'] = f"SR Wall Veto: price near support (sr={sr_score:.1f}) — no SHORT"
        ctx['sr_wall_locked'] = True
    elif sr_score <= -1.0 and action == "BUY" and not _sr_wall_escape_ready(
        action, sr_score, total_score, mtf_fast_bias, macd_diff_val, psar_bull, current_price, ema_9_val, wall_escape_gate,
    ):
        ctx['action'] = "HOLD"
        ctx['hold_reason'] = f"SR Wall Veto: price near resistance (sr={sr_score:.1f}) — no LONG"
        ctx['sr_wall_locked'] = True


def apply_range_position_veto(ctx: SignalContext) -> None:
    """Apply range position veto — mutates ctx['action'] and ctx['hold_reason'] in place."""
    strategy_config = ctx['strategy_config']
    action = ctx.get('action', 'HOLD')
    if action not in {"BUY", "SELL"}:
        return
    if not bool(strategy_config.get("range_position_veto_enabled", True)):
        return

    df_indicators = ctx['df_indicators']
    current_price = ctx['current_price']
    sr_score = ctx.get('sr_score', 0.0)
    mtf_fast_bias = ctx.get('mtf_fast_bias', 'NEUTRAL')
    total_score = ctx.get('total_score', 0.0)

    _tf_str = str(strategy_config.get("timeframe", "15m") or "15m").strip().lower()
    _tf_map = {"1m": 1440, "3m": 480, "5m": 288, "10m": 144, "15m": 96, "1h": 24, "4h": 6}
    _cpd = int(strategy_config.get("candles_per_day", _tf_map.get(_tf_str, 96)))
    _lookback = max(50, min(_cpd, len(df_indicators)))

    _recent = df_indicators.iloc[-_lookback:]
    _range_high = float(_recent["high"].max())
    _range_low = float(_recent["low"].min())
    _range_width = _range_high - _range_low

    if _range_width <= 0:
        return

    _pos = (current_price - _range_low) / _range_width
    _veto_top = float(strategy_config.get("range_veto_top_pct", 0.75) or 0.75)
    _veto_bottom = float(strategy_config.get("range_veto_bottom_pct", 0.25) or 0.25)
    _veto_escape_score = max(0.80, float(strategy_config.get("range_veto_escape_score", 0.80) or 0.80))

    _mtf_unanimous = mtf_fast_bias in {"LONG_ONLY", "SHORT_ONLY"}
    _is_strong = (
        abs(total_score) >= _veto_escape_score
        and _mtf_unanimous
        and abs(sr_score) < 1.0
    )

    if action == "BUY" and _pos >= _veto_top:
        if _is_strong and mtf_fast_bias == "LONG_ONLY" and sr_score > -0.5:
            pass
        else:
            ctx['action'] = "HOLD"
            ctx['hold_reason'] = (
                f"Range Veto: BUY in top {int(_veto_top*100)}% of {_lookback}c range "
                f"(pos={_pos:.0%} score={total_score:.2f} sr={sr_score:.1f}). "
                f"Need score>={_veto_escape_score:.2f} + unanimous MTF + sr>-0.5 for breakout."
            )

    elif action == "SELL" and _pos <= _veto_bottom:
        if _is_strong and mtf_fast_bias == "SHORT_ONLY" and sr_score < 0.5:
            pass
        else:
            ctx['action'] = "HOLD"
            ctx['hold_reason'] = (
                f"Range Veto: SELL in bottom {int((1-_veto_bottom)*100)}% of {_lookback}c range "
                f"(pos={_pos:.0%} score={total_score:.2f} sr={sr_score:.1f}). "
                f"Need score>={_veto_escape_score:.2f} + unanimous MTF + sr<0.5 for breakdown."
            )


def classify_entry_mode_and_walls(ctx: SignalContext) -> None:
    """
    Classify entry mode (TREND/REVERSAL/BREAKOUT/BREAKDOWN) and apply wall logic.
    Mutates ctx['action'], ctx['entry_mode'], ctx['hold_reason'], ctx['sr_wall_locked'], ctx['signal_reason_suffix'].
    """
    strategy_config = ctx['strategy_config']
    action = ctx.get('action', 'HOLD')
    total_score = ctx.get('total_score', 0.0)
    current_price = ctx['current_price']
    df_indicators = ctx['df_indicators']
    support = ctx.get('support')
    resistance = ctx.get('resistance')
    wall_state = ctx.get('wall_state', {})
    sr_score = ctx.get('sr_score', 0.0)
    mtf_fast_bias = ctx.get('mtf_fast_bias', 'NEUTRAL')
    macd_diff_val = ctx.get('macd_diff', 0.0)
    psar_bull = ctx.get('psar_bull', True)
    macd_final_bull = ctx.get('macd_final_bull', True)
    macd_final_bear = ctx.get('macd_final_bear', False)
    ema_9_val = ctx.get('ema_9_val', ctx['latest_indicators'].get('ema_9', current_price))

    sup_touch = bool(wall_state.get("support_touching"))
    res_touch = bool(wall_state.get("resistance_touching"))
    sup_broken = bool(wall_state.get("support_broken"))
    res_broken = bool(wall_state.get("resistance_broken"))

    entry_mode = "TREND"
    signal_reason_suffix = ""
    hold_reason = ctx.get('hold_reason', '')
    sr_wall_locked = ctx.get('sr_wall_locked', False)

    ctx['signal_reason_suffix'] = signal_reason_suffix

    if support and resistance and action in {"BUY", "SELL"}:
        if res_broken and action == "BUY":
            entry_mode = "BREAKOUT_LONG"
        elif sup_broken and action == "SELL":
            entry_mode = "BREAKOUT_SHORT"
        elif res_touch and not res_broken and action == "SELL":
            entry_mode = "REVERSAL_SHORT"
        elif sup_touch and not sup_broken and action == "BUY":
            entry_mode = "REVERSAL_LONG"
        elif res_touch and not res_broken and action == "BUY":
            wall_reversal_gate = float(strategy_config.get("wall_reversal_score_gate", 0.12) or 0.12)
            resistance_rejection_ready = (
                bool(strategy_config.get("wall_reversal_assist", True))
                and total_score <= -wall_reversal_gate
                and (
                    (macd_final_bear and not psar_bull)
                    or (
                        current_price < float(df_indicators["open"].iloc[-1])
                        and (macd_diff_val <= 0 or not psar_bull or current_price < ema_9_val)
                    )
                )
            )
            if resistance_rejection_ready:
                action = "SELL"
                entry_mode = "REVERSAL_SHORT"
                signal_reason_suffix += " [Wall Rejection Short]"
            else:
                action = "HOLD"
                hold_reason = (
                    f"Wall Veto: BUY blocked — price at resistance "
                    f"(${current_price:.3f} / wall ${float(resistance):.3f}). "
                    f"Wait for break>${float(wall_state.get('resistance_break_level', resistance)):.3f}."
                )
                sr_wall_locked = True
        elif sup_touch and not sup_broken and action == "SELL":
            wall_reversal_gate = float(strategy_config.get("wall_reversal_score_gate", 0.12) or 0.12)
            support_rejection_ready = (
                bool(strategy_config.get("wall_reversal_assist", True))
                and total_score >= wall_reversal_gate
                and (
                    (macd_final_bull and psar_bull)
                    or (
                        current_price > float(df_indicators["open"].iloc[-1])
                        and (macd_diff_val >= 0 or psar_bull or current_price > ema_9_val)
                    )
                )
            )
            if support_rejection_ready:
                action = "BUY"
                entry_mode = "REVERSAL_LONG"
                signal_reason_suffix += " [Wall Reclaim Long]"
            else:
                breakdown_likely = (
                    (macd_diff_val < 0)
                    and (mtf_fast_bias == "SHORT_ONLY")
                    and (total_score < -0.25)
                )
                if not breakdown_likely:
                    action = "HOLD"
                    hold_reason = (
                        f"Wall Veto: SELL near intact support (${current_price:.3f} / floor ${float(support):.3f}). "
                        f"Need SHORT_ONLY MTF + score<-0.25 for breakdown entry."
                    )
                    sr_wall_locked = True
                else:
                    entry_mode = "BREAKDOWN_SHORT"
                    signal_reason_suffix += " [Breakdown Short]"

    ctx['action'] = action
    ctx['entry_mode'] = entry_mode
    ctx['hold_reason'] = hold_reason
    ctx['sr_wall_locked'] = sr_wall_locked
    ctx['signal_reason_suffix'] = signal_reason_suffix


def apply_wall_rejection_rescue(ctx: SignalContext) -> None:
    """Promote HOLD into short/long when wall rejection is confirmed."""
    signal = ctx.get('signal', {})
    strategy_config = ctx['strategy_config']
    total_score = ctx.get('total_score', 0.0)
    support = ctx.get('support')
    resistance = ctx.get('resistance')
    wall_state = ctx.get('wall_state', {})
    df_indicators = ctx['df_indicators']
    current_price = ctx['current_price']

    if signal.get("action") != "HOLD":
        return
    if not bool(strategy_config.get("wall_reversal_assist", True)):
        return
    if support is None or resistance is None:
        return

    _reversal_assist_score_ok = total_score < -0.10
    if not _reversal_assist_score_ok:
        return

    res_touch = bool(wall_state.get("resistance_touching"))
    res_broken = bool(wall_state.get("resistance_broken"))
    if not (res_touch and not res_broken):
        return

    reject_cfg = dict((strategy_config or {}).get("rejection_confirmation", {}) or {})
    min_bars_away = max(1, int(reject_cfg.get("min_bars_away", 2) or 2))
    pullback_pct = float(reject_cfg.get("pullback_pct", 0.002) or 0.002)
    require_psar_flip = bool(reject_cfg.get("require_psar_flip", True))
    macd_threshold = float(reject_cfg.get("macd_threshold", 0.0) or 0.0)
    last = df_indicators.iloc[-1]
    prev = df_indicators.iloc[-2]
    current_macd = float(last["macd"]) if "macd" in df_indicators.columns else 0.0
    current_macd_diff = float(last["macd_diff"]) if "macd_diff" in df_indicators.columns else 0.0
    prev_macd_diff_f = float(prev["macd_diff"]) if "macd_diff" in df_indicators.columns else 0.0
    current_psar = float(last["psar"]) if "psar" in df_indicators.columns else current_price
    recent = df_indicators.tail(min_bars_away)

    macd_confirmed = current_macd < macd_threshold and current_macd_diff < 0 and prev_macd_diff_f >= 0
    psar_confirmed = (current_price < current_psar) if require_psar_flip else True
    candle_confirmed = current_price <= float(last["open"])
    pullback_confirmed = bool((recent["high"] < resistance * (1 - pullback_pct)).all())
    rejection_confirmed = macd_confirmed and psar_confirmed and candle_confirmed and pullback_confirmed

    if rejection_confirmed:
        signal["action"] = "SELL"
        signal["hold_reason"] = ""
        signal["reason"] = f"{signal.get('reason','')} [Wall Rejection Short]"
        signal["score"] = float(total_score)
        signal["confidence"] = min(abs(float(total_score)), 1.0)
        signal["sr_wall_locked"] = False
        signal["rejection_confirmation"] = {
            "confirmed": True,
            "mode": "REVERSAL_SHORT",
            "action": "SELL",
            "reason": "Wall rejection short confirmed",
        }
    else:
        signal["rejection_confirmation"] = {
            "confirmed": False,
            "mode": "REVERSAL_SHORT",
            "action": "SELL",
            "reason": "Wall rejection not yet confirmed",
        }


def apply_midrange_policy(ctx: SignalContext) -> None:
    """Apply midrange entry policy for signals in the middle of the range."""
    signal = ctx.get('signal', {})
    current_price = ctx['current_price']
    strategy_config = ctx['strategy_config']
    mtf_fast_bias = ctx.get('mtf_fast_bias', 'NEUTRAL')
    sr_score = ctx.get('sr_score', 0.0)
    total_score = ctx.get('total_score', 0.0)
    macd_diff_val = ctx.get('macd_diff', 0.0)
    psar_bull = ctx.get('psar_bull', True)
    ema_9_val = ctx.get('ema_9_val', ctx['latest_indicators'].get('ema_9', current_price))

    m5_s = signal.get("structure_support")
    m5_r = signal.get("structure_resistance")

    if m5_s and m5_r:
        range_action_zone_pct = ctx.get('range_action_zone_pct', 0.20)
        range_action_zone_pct = max(0.05, min(0.45, range_action_zone_pct))
        range_w = m5_r - m5_s
        zone_size = range_w * range_action_zone_pct
        top_action_zone = m5_r - zone_size
        bottom_action_zone = m5_s + zone_size
        in_middle = not (current_price >= top_action_zone) and not (current_price <= bottom_action_zone)

        signal["action_support"] = float(bottom_action_zone)
        signal["action_resistance"] = float(top_action_zone)
        signal["top_action_zone"] = float(top_action_zone)
        signal["bottom_action_zone"] = float(bottom_action_zone)

        final_action = signal.get("action", "HOLD")
        if final_action not in {"BUY", "SELL"} or not in_middle:
            return

        wall_escape_gate = float(strategy_config.get("wall_breakout_score_gate", 0.15) or 0.15)
        _reason_str = signal.get("reason", "") or ""
        entry_mode = _reason_str.split("[Mode:")[-1].split("]")[0] if "[Mode:" in _reason_str else "TREND"

        mid_min_score = float(strategy_config.get("midrange_min_score", 0.28) or 0.28)
        mtf_aligned = mtf_fast_bias == ("LONG_ONLY" if final_action == "BUY" else "SHORT_ONLY")
        score_ok = abs(float(signal.get("score", 0.0) or 0.0)) >= mid_min_score

        if "REVERSAL" in entry_mode or "BREAKOUT" in entry_mode:
            signal["action"] = "HOLD"
            signal["hold_reason"] = (
                f"Gate: {entry_mode} needs edge "
                f"(price ${current_price:.2f} between zones ${bottom_action_zone:.2f}–${top_action_zone:.2f})."
            )
        elif not mtf_aligned:
            signal["action"] = "HOLD"
            signal["hold_reason"] = (
                f"Midrange Gate: MTF ({mtf_fast_bias}) ≠ {final_action}. "
                f"Wait for edge or unanimous MTF bias."
            )
        elif not score_ok:
            signal["action"] = "HOLD"
            signal["hold_reason"] = (
                f"Midrange Gate: score {float(signal.get('score', 0)):.3f} < {mid_min_score:.2f}. "
                f"Need stronger signal for midrange trend entry."
            )
        elif final_action == "BUY" and sr_score <= -1.0 and not _sr_wall_escape_ready(
            final_action, sr_score, float(signal.get("score", 0.0) or 0.0), mtf_fast_bias,
            float(macd_diff_val or 0.0), bool(psar_bull), current_price, ema_9_val, wall_escape_gate,
        ):
            signal["action"] = "HOLD"
            signal["hold_reason"] = (
                f"Midrange SR Block: BUY near resistance wall (sr={sr_score:.1f}) "
                f"even with TREND mode — no long into ceiling."
            )
            signal["sr_wall_locked"] = True
        elif final_action == "SELL" and sr_score >= 1.0 and not _sr_wall_escape_ready(
            final_action, sr_score, float(signal.get("score", 0.0) or 0.0), mtf_fast_bias,
            float(macd_diff_val or 0.0), bool(psar_bull), current_price, ema_9_val, wall_escape_gate,
        ):
            signal["action"] = "HOLD"
            signal["hold_reason"] = (
                f"Midrange SR Block: SELL near support wall (sr={sr_score:.1f}) "
                f"even with TREND mode — no short into floor."
            )
            signal["sr_wall_locked"] = True
    else:
        _reason_str = signal.get("reason", "") or ""
        entry_mode = _reason_str.split("[Mode:")[-1].split("]")[0] if "[Mode:" in _reason_str else "TREND"
        wall_escape_gate = float(strategy_config.get("wall_breakout_score_gate", 0.15) or 0.15)

        if "REVERSAL" in entry_mode:
            signal["action"] = "HOLD"
            signal["hold_reason"] = "Range Gate: No S/R boundaries — reversal entry skipped."
        elif signal.get("action") == "BUY" and sr_score <= -1.0 and not _sr_wall_escape_ready(
            signal.get("action"), sr_score, float(signal.get("score", 0.0) or 0.0), mtf_fast_bias,
            float(macd_diff_val or 0.0), bool(psar_bull), current_price, ema_9_val, wall_escape_gate,
        ):
            signal["action"] = "HOLD"
            signal["hold_reason"] = f"No-SR fallback SR block: BUY near resistance (sr={sr_score:.1f})"
            signal["sr_wall_locked"] = True
        elif signal.get("action") == "SELL" and sr_score >= 1.0 and not _sr_wall_escape_ready(
            signal.get("action"), sr_score, float(signal.get("score", 0.0) or 0.0), mtf_fast_bias,
            float(macd_diff_val or 0.0), bool(psar_bull), current_price, ema_9_val, wall_escape_gate,
        ):
            signal["action"] = "HOLD"
            signal["hold_reason"] = f"No-SR fallback SR block: SELL near support (sr={sr_score:.1f})"
            signal["sr_wall_locked"] = True