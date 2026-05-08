"""Sniper logic: range reversal (floor bounce / ceiling rejection), exhaustion/divergence hard gate, MTF trend veto."""

from __future__ import annotations

from ..ctx import SignalContext
import numpy as np
import logging

logger = logging.getLogger(__name__)


def apply_range_reversal_sniper(ctx: SignalContext) -> None:
    """Apply range reversal sniper for range-floor/ceiling bounce signals."""
    signal = ctx.get('signal', {})
    strategy_config = ctx['strategy_config']
    df_indicators = ctx['df_indicators']
    current_price = ctx['current_price']
    mtf_fast_bias = ctx.get('mtf_fast_bias', 'NEUTRAL')
    latest_indicators = ctx['latest_indicators']
    macd_diff_val = ctx.get('macd_diff', 0.0)

    if signal.get("action") != "HOLD":
        return
    if not bool(strategy_config.get("range_position_veto_enabled", True)):
        return

    _tf_str_rv = str(strategy_config.get("timeframe", "15m") or "15m").strip().lower()
    _tf_map_rv = {"1m": 1440, "3m": 480, "5m": 288, "10m": 144, "15m": 96, "1h": 24, "4h": 6}
    _default_cpd = _tf_map_rv.get(_tf_str_rv, 96)
    _cpd_rv = int(strategy_config.get("sniper_lookback") or strategy_config.get("candles_per_day", _default_cpd))
    _lookback_rv = max(20, min(_cpd_rv, len(df_indicators)))
    _recent_rv = df_indicators.iloc[-_lookback_rv:]
    _rng_hi_rv = float(_recent_rv["high"].max())
    _rng_lo_rv = float(_recent_rv["low"].min())
    _rng_w_rv = _rng_hi_rv - _rng_lo_rv

    if _rng_w_rv <= 0:
        return

    _pos_rv = (current_price - _rng_lo_rv) / _rng_w_rv
    _veto_bot_rv = float(strategy_config.get("range_veto_bottom_pct", 0.25) or 0.25)
    _veto_top_rv = float(strategy_config.get("range_veto_top_pct", 0.75) or 0.75)

    _psar_raw_rv = df_indicators["psar"].iloc[-1] if "psar" in df_indicators.columns and len(df_indicators) else None
    psar_bull_rv = ctx.get('psar_bull', True)
    try:
        _psar_bull_rv = float(_psar_raw_rv) < current_price if _psar_raw_rv is not None else psar_bull_rv
    except (TypeError, ValueError):
        _psar_bull_rv = True

    _rsi_rv = float(latest_indicators.get("rsi", 50) or 50)
    _rsi_os_rv = float(strategy_config.get("rsi_os_entry_gate", 28) or 28)
    _rsi_ob_rv = float(strategy_config.get("rsi_ob_entry_gate", 72) or 72)
    _prev_md_rv = float(macd_diff_val)

    # Score formula combines stab (evidence count) and depth (how deep into veto zone):
    #   sc = clip(0.20 + 0.12 * (stab - 1) + depth * 0.20, 0.20, 0.85)
    # Hits 0.55 (min_conf) at stab=4 even with shallow depth at the veto edge.

    if _pos_rv <= _veto_bot_rv:
        _depth_rv = max(0.0, 1.0 - (_pos_rv / max(_veto_bot_rv, 1e-9)))
        _s1 = current_price >= float(df_indicators["open"].iloc[-1])
        _s2 = _rsi_rv <= (_rsi_os_rv + 10)
        _s3 = _psar_bull_rv
        _s4 = macd_diff_val > _prev_md_rv
        _s5 = len(df_indicators) > 2 and current_price >= float(df_indicators["low"].iloc[-2])
        _stab_rv = int(_s1) + int(_s2) + int(_s3) + int(_s4) + int(_s5)

        _mtf_against = (mtf_fast_bias == "SHORT_ONLY")
        _min_stab = 4 if _mtf_against else 2

        if _stab_rv >= _min_stab:
            _rev_sc = float(np.clip(0.20 + 0.12 * (_stab_rv - 1) + _depth_rv * 0.20, 0.20, 0.85))
            signal["action"] = "BUY"
            signal["score"] = _rev_sc
            signal["confidence"] = min(_rev_sc, 1.0)
            signal["hold_reason"] = ""
            _mtf_tag = " contraMTF" if _mtf_against else ""
            signal["reason"] = (
                f"{signal.get('reason', '')} "
                f"[RangeFloor{_mtf_tag} pos={_pos_rv:.0%} depth={_depth_rv:.2f} stab={_stab_rv}/5 sc={_rev_sc:.2f}]"
            ).strip()
            logger.debug(f"[RANGE_REV] Floor bounce: pos={_pos_rv:.1%} depth={_depth_rv:.2f} stab={_stab_rv}/5 mtf_against={_mtf_against} → BUY score={_rev_sc:.3f}")
        elif _stab_rv >= 2:
            signal["reason"] = (
                f"{signal.get('reason', '')} "
                f"[Watching FloorBounce pos={_pos_rv:.0%} stab={_stab_rv}/5 need>={_min_stab}{' contraMTF' if _mtf_against else ''}]"
            ).strip()

    elif _pos_rv >= _veto_top_rv:
        _depth_rv = max(0.0, (_pos_rv - _veto_top_rv) / max(1.0 - _veto_top_rv, 1e-9))
        _s1 = current_price <= float(df_indicators["open"].iloc[-1])
        _s2 = _rsi_rv >= (_rsi_ob_rv - 10)
        _s3 = not _psar_bull_rv
        _s4 = macd_diff_val < _prev_md_rv
        _s5 = len(df_indicators) > 2 and current_price <= float(df_indicators["high"].iloc[-2])
        _stab_rv = int(_s1) + int(_s2) + int(_s3) + int(_s4) + int(_s5)

        _mtf_against = (mtf_fast_bias == "LONG_ONLY")
        _min_stab = 4 if _mtf_against else 2

        if _stab_rv >= _min_stab:
            _rev_sc = float(np.clip(0.20 + 0.12 * (_stab_rv - 1) + _depth_rv * 0.20, 0.20, 0.85))
            signal["action"] = "SELL"
            signal["score"] = -_rev_sc
            signal["confidence"] = min(_rev_sc, 1.0)
            signal["hold_reason"] = ""
            _mtf_tag = " contraMTF" if _mtf_against else ""
            signal["reason"] = (
                f"{signal.get('reason', '')} "
                f"[RangeCeil{_mtf_tag} pos={_pos_rv:.0%} depth={_depth_rv:.2f} stab={_stab_rv}/5 sc={_rev_sc:.2f}]"
            ).strip()
            logger.debug(f"[RANGE_REV] Ceiling rejection: pos={_pos_rv:.1%} depth={_depth_rv:.2f} stab={_stab_rv}/5 mtf_against={_mtf_against} → SELL score={-_rev_sc:.3f}")
        elif _stab_rv >= 2:
            signal["reason"] = (
                f"{signal.get('reason', '')} "
                f"[Watching CeilRejection pos={_pos_rv:.0%} stab={_stab_rv}/5 need>={_min_stab}{' contraMTF' if _mtf_against else ''}]"
            ).strip()


def apply_exhaustion_divergence_gate(ctx: SignalContext) -> None:
    """Apply the final exhaustion & divergence hard gate."""
    signal = ctx.get('signal', {})
    strategy_config = ctx['strategy_config']
    total_score = ctx.get('total_score', 0.0)
    momentum_exhaustion = ctx.get('momentum_exhaustion')
    bear_div = ctx.get('bear_div', False)
    bull_div = ctx.get('bull_div', False)
    rsi_14 = ctx.get('rsi_14', 50.0)
    macd_diff = ctx.get('macd_diff', 0)
    prev_macd_diff = ctx.get('prev_macd_diff', 0)
    prev2_macd_diff = ctx.get('prev2_macd_diff', 0)

    if signal.get("action") not in {"BUY", "SELL"}:
        return

    _final_action = signal["action"]
    _exhaust_block = False
    _div_block = False
    _rsi_block = False

    if _final_action == "BUY" and momentum_exhaustion == "BULL_EXHAUST":
        _exhaust_block = True
    elif _final_action == "SELL" and momentum_exhaustion == "BEAR_EXHAUST":
        _exhaust_block = True

    try:
        _h0 = float(macd_diff)
        _h1 = float(prev_macd_diff)
        _h2 = float(prev2_macd_diff)
        if _final_action == "BUY" and _h0 > 0 and _h1 > 0 and _h0 < _h1 < _h2:
            _exhaust_block = True
        elif _final_action == "SELL" and _h0 < 0 and _h1 < 0 and _h0 > _h1 > _h2:
            _exhaust_block = True
    except (TypeError, ValueError):
        pass

    if _final_action == "BUY" and bear_div:
        _div_block = True
    elif _final_action == "SELL" and bull_div:
        _div_block = True

    _rsi_ob_gate = float(strategy_config.get("rsi_ob_entry_gate", 72) or 72)
    _rsi_os_gate = float(strategy_config.get("rsi_os_entry_gate", 28) or 28)
    if _final_action == "BUY" and rsi_14 > _rsi_ob_gate:
        _rsi_block = True
    elif _final_action == "SELL" and rsi_14 < _rsi_os_gate:
        _rsi_block = True

    if _exhaust_block and _div_block:
        signal["action"] = "HOLD"
        signal["hold_reason"] = f"ExhaustDiv Gate: {momentum_exhaustion} + divergence — top/bottom entry blocked"
    elif _exhaust_block and _rsi_block:
        signal["action"] = "HOLD"
        signal["hold_reason"] = f"ExhaustRSI Gate: {momentum_exhaustion} + RSI {rsi_14:.0f} — top/bottom entry blocked"
    elif _div_block and _rsi_block:
        signal["action"] = "HOLD"
        signal["hold_reason"] = f"DivRSI Gate: divergence + RSI {rsi_14:.0f} — top/bottom entry blocked"
    elif _rsi_block and abs(total_score) < 0.70:
        signal["action"] = "HOLD"
        signal["hold_reason"] = f"RSI Extreme Gate: RSI {rsi_14:.0f} — no entry without strong breakout score (>0.70)"


def apply_mtf_trend_veto(ctx: SignalContext) -> None:
    """Apply MTF fast-bias trend veto."""
    mtf_fast_bias = ctx.get('mtf_fast_bias', 'NEUTRAL')
    smc_score = ctx.get('smc_score', 0.0)
    total_score = ctx.get('total_score', 0.0)
    signal = ctx.get('signal', {})
    action = signal.get('action', 'HOLD')

    if mtf_fast_bias == "LONG_ONLY" and action == "SELL":
        signal['action'] = "HOLD"
        signal['hold_reason'] = "MTF trend veto: fast timeframes bullish"
    elif mtf_fast_bias == "SHORT_ONLY" and action == "BUY":
        signal['action'] = "HOLD"
        signal['hold_reason'] = "MTF trend veto: fast timeframes bearish"
    elif mtf_fast_bias == "NEUTRAL" and action in {"BUY", "SELL"}:
        if action == "SELL" and smc_score > 0:
            signal['action'] = "HOLD"
            signal['hold_reason'] = "NEUTRAL MTF + Bullish structure = no short"
        elif action == "BUY" and smc_score < 0:
            signal['action'] = "HOLD"
            signal['hold_reason'] = "NEUTRAL MTF + Bearish structure = no long"
        elif abs(total_score) < 0.05:
            signal['action'] = "HOLD"
            signal['hold_reason'] = "MTF trend veto: fast timeframes not aligned"
        else:
            signal['reason'] = f"{signal.get('reason', '')} MTFPartial"
    elif mtf_fast_bias != "NEUTRAL":
        signal['reason'] = f"{signal.get('reason', '')} MTFConfirm:{mtf_fast_bias}"