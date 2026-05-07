"""Phase 8: Trend confirmation — PSAR/MACD/MTF hierarchy, entry mode classification, strike zone, OB gate, article setups (ORB + VWAP), sniper checklist."""

from __future__ import annotations

from .ctx import SignalContext
import numpy as np

from .gates import (
    classify_entry_mode_and_walls, apply_strike_zone_check, apply_ob_gate,
    apply_trend_confirmation_gate,
)
from .setups import _detect_orb_breakout_setup, _detect_vwap_bounce_setup


def _compute_trend_confirmation(ctx: SignalContext) -> None:
    """Compute PSAR/MACD/MTF trend signals, run entry gates, apply article setups. Mutates ctx in-place."""
    current_price = ctx['current_price']
    df_indicators = ctx['df_indicators']
    latest_indicators = ctx['latest_indicators']
    strategy_config = ctx['strategy_config']
    mtf_context = ctx.get('mtf_context') or {}
    total_score = ctx['total_score']
    action = ctx['action']
    hold_reason = ctx.get('hold_reason', '')
    ema_9_val = ctx['ema_9_val']
    ema_21_val = ctx['ema_21_val']

    psar_closed_val2 = latest_indicators.get('psar')
    psar_live_val2 = (
        float(df_indicators['psar'].iloc[-1])
        if "psar" in df_indicators.columns and len(df_indicators)
        else psar_closed_val2
    )
    macd_val = float(latest_indicators.get('macd', 0) or 0)
    macd_diff_val2 = float(latest_indicators.get('macd_diff', 0) or 0)

    psar_closed_bull = True
    psar_live_bull = True
    if psar_closed_val2 is not None:
        try:
            psar_closed_bull = float(psar_closed_val2) < current_price
        except (TypeError, ValueError):
            pass
    if psar_live_val2 is not None:
        try:
            psar_live_bull = float(psar_live_val2) < current_price
        except (TypeError, ValueError):
            pass
    ctx['psar_bull'] = psar_closed_bull
    ctx['psar_closed_bull'] = psar_closed_bull
    ctx['psar_live_bull'] = psar_live_bull
    ctx['psar_state_note'] = f"PSAR(C:{'BULL' if psar_closed_bull else 'BEAR'} L:{'BULL' if psar_live_bull else 'BEAR'})"

    article_sl_override = None

    if action in {"BUY", "SELL"}:
        ema_bull = ema_9_val > ema_21_val
        macd_noise_threshold = float(strategy_config.get('macd_noise_threshold', 0.0001) or 0.0001)

        mtf_15m = mtf_context.get('15m', {})
        mtf_10m = mtf_context.get('10m', {})
        macd_15m = float(mtf_15m.get('macd', 0) or 0)
        macd_10m = float(mtf_10m.get('macd', 0) or 0)

        macd_pos_bull = macd_val > 0
        macd_cross_bull = macd_diff_val2 > 0
        mtf_macd_bull = (macd_15m > 0) and (macd_10m > 0) and abs(macd_10m) > macd_noise_threshold
        mtf_macd_bear = (macd_15m < 0) and (macd_10m < 0) and abs(macd_10m) > macd_noise_threshold
        mtf_structure_bull = str(mtf_15m.get("structure", "")).upper() in {"HH_HL", "HIGHER_LOW"}
        mtf_structure_bear = str(mtf_15m.get("structure", "")).upper() in {"LH_LL", "LOWER_HIGH"}
        ctx['mtf_macd_bull'] = mtf_macd_bull
        ctx['mtf_macd_bear'] = mtf_macd_bear
        ctx['mtf_structure_bull'] = mtf_structure_bull
        ctx['mtf_structure_bear'] = mtf_structure_bear

        lookback_macds = df_indicators['macd'].tail(3).values
        macd_zero_cross_bull = any(
            (lookback_macds[i] > 0 and lookback_macds[i - 1] <= 0)
            for i in range(1, len(lookback_macds))
        )
        macd_zero_cross_bear = any(
            (lookback_macds[i] < 0 and lookback_macds[i - 1] >= 0)
            for i in range(1, len(lookback_macds))
        )
        macd_outside_noise = abs(macd_val) > macd_noise_threshold
        macd_inst_bull = (macd_pos_bull or macd_zero_cross_bull) and macd_outside_noise
        macd_inst_bear = ((not macd_pos_bull) or macd_zero_cross_bear) and macd_outside_noise
        ctx['macd_final_bull'] = macd_inst_bull
        ctx['macd_final_bear'] = macd_inst_bear

        psar_val_raw = latest_indicators.get('psar_streak', 0)
        psar_streak = int(psar_val_raw) if psar_val_raw is not None and str(psar_val_raw) != 'nan' else 0
        ctx['psar_streak'] = psar_streak

        classify_entry_mode_and_walls(ctx)
        action = ctx['action']
        hold_reason = ctx.get('hold_reason', hold_reason)

        apply_strike_zone_check(ctx)
        action = ctx['action']
        hold_reason = ctx.get('hold_reason', hold_reason)

        apply_ob_gate(ctx)
        action = ctx['action']
        hold_reason = ctx.get('hold_reason', hold_reason)

        if not ctx.get('in_action_zone', False):
            apply_trend_confirmation_gate(ctx)
            action = ctx['action']
            hold_reason = ctx.get('hold_reason', hold_reason)

        orb_breakout_setup = _detect_orb_breakout_setup(
            df_indicators, current_price,
            latest_indicators=latest_indicators,
            strategy_config=strategy_config,
            mtf_fast_bias=ctx['mtf_fast_bias'],
        )
        vwap_bounce_setup = _detect_vwap_bounce_setup(
            df_indicators, current_price,
            latest_indicators=latest_indicators,
            strategy_config=strategy_config,
            anchored_vwap=ctx['anchored_vwap'],
            mtf_fast_bias=ctx['mtf_fast_bias'],
        )
        article_notes = []
        article_override = None

        if orb_breakout_setup.get("triggered"):
            orb_dir = str(orb_breakout_setup.get("direction", "NEUTRAL") or "NEUTRAL").upper()
            if action == "HOLD" or (action == "BUY" and orb_dir == "LONG") or (action == "SELL" and orb_dir == "SHORT"):
                action = "BUY" if orb_dir == "LONG" else "SELL"
                article_override = "ORB_BREAKOUT"
                article_notes.append(f"ORB:{orb_breakout_setup.get('reason','')}")
                if orb_breakout_setup.get("score") is not None:
                    total_score = float(np.clip(float(total_score) + float(orb_breakout_setup.get("score", 0.0) or 0.0), -1.0, 1.0))
                if orb_breakout_setup.get("sl") is not None:
                    article_sl_override = float(orb_breakout_setup.get("sl"))

        if vwap_bounce_setup.get("triggered"):
            vwap_dir = str(vwap_bounce_setup.get("direction", "NEUTRAL") or "NEUTRAL").upper()
            if action == "HOLD" or (action == "BUY" and vwap_dir == "LONG") or (action == "SELL" and vwap_dir == "SHORT"):
                action = "BUY" if vwap_dir == "LONG" else "SELL"
                article_override = article_override or "VWAP_BOUNCE"
                article_notes.append(f"VWAP:{vwap_bounce_setup.get('reason','')}")
                if vwap_bounce_setup.get("score") is not None:
                    total_score = float(np.clip(float(total_score) + float(vwap_bounce_setup.get("score", 0.0) or 0.0), -1.0, 1.0))
                if vwap_bounce_setup.get("sl") is not None and article_sl_override is None:
                    article_sl_override = float(vwap_bounce_setup.get("sl"))

        signal_reason_suffix = " [Strike Override]" if ctx.get('in_action_zone') and action != "HOLD" else (" [Indicators OK]" if action != "HOLD" else " [Waiting for alignment]")
        if article_override:
            signal_reason_suffix += f" [{article_override}]"
        if article_notes:
            article_note_text = " | ".join(article_notes)
            if len(article_note_text) > 120:
                article_note_text = article_note_text[:117] + "..."
            signal_reason_suffix += f" [{article_note_text}]"

        is_bearish_attempt = total_score < 0
        checklist = []
        macd_inst_bull = ctx.get('macd_final_bull', False)
        macd_inst_bear = ctx.get('macd_final_bear', False)
        if is_bearish_attempt:
            checklist.append(f"EMA:{'OK' if not ema_bull else 'Wait'}")
            checklist.append(f"MACD:{'OK' if macd_inst_bear else 'Wait'}")
            checklist.append(f"SAR:C={'OK' if not psar_closed_bull else 'Wait'}")
            checklist.append(f"L={'OK' if not psar_live_bull else 'Wait'}")
        else:
            checklist.append(f"EMA:{'OK' if ema_bull else 'Wait'}")
            checklist.append(f"MACD:{'OK' if macd_inst_bull else 'Wait'}")
            checklist.append(f"SAR:C={'OK' if psar_closed_bull else 'Wait'}")
            checklist.append(f"L={'OK' if psar_live_bull else 'Wait'}")
        checklist.append(f"Vol:{'OK' if ctx.get('vol_spike') else 'Low'}")
        checklist_str = " | ".join(checklist)
        status_msg = f"[{checklist_str}]"
        if action != "HOLD":
            status_msg = f"READY: {checklist_str}"
            if abs(ctx.get('div_bonus', 0)) > 0:
                status_msg += " +DIV"
        signal_reason_suffix = f" {status_msg}"
    else:
        psar_closed_bull = ctx.get('psar_closed_bull', True)
        psar_live_bull = ctx.get('psar_live_bull', True)
        psar_state_note = ctx.get('psar_state_note', f"PSAR(C:{'BULL' if psar_closed_bull else 'BEAR'} L:{'BULL' if psar_live_bull else 'BEAR'})")
        signal_reason_suffix = f" [Waiting for alignment | {psar_state_note}]"

    ctx['action'] = action
    ctx['total_score'] = total_score
    ctx['hold_reason'] = hold_reason
    ctx['signal_reason_suffix'] = signal_reason_suffix
    ctx['article_sl_override'] = article_sl_override