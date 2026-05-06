import pandas as pd
import numpy as np
import time
import math
import logging

from .calc import get_signal_weights, TUNING
from .smc import detect_smc_and_sr
from .mtf import _pick_structural_levels
from .location import (
    compute_volatility_context, identify_liquidity_pools, calculate_funding_impact,
    _compute_vpoc, _compute_anchored_vwap, _detect_cvd_divergence,
    _compute_market_location_score, _compute_wall_state,
    _detect_momentum_exhaustion, _body_range_ratio_score,
)
from safety import sr_wall_escape_ready as _sr_wall_escape_ready

logger = logging.getLogger(__name__)

def generate_alpha_overlay(df: pd.DataFrame, smc_score: float, macro_bias: str) -> float:
    """
    MASTER RECONSTRUCTION: Alpha Overlay Engine.
    Synthesizes structural bias with momentum to find 'The Edge'.
    Uses EMA 50/200 golden/death cross for swing-speed trend confirmation.
    """
    if len(df) < 200: return 0.0

    # 1. Momentum Cross Check (EMA 50/200) — swing-speed trend filter
    ema_50 = df['ema_50'].iloc[-1]
    ema_200 = df['ema_200'].iloc[-1]
    mom_bias = 1.0 if ema_50 > ema_200 else -1.0

    # 2. Relative Volume Spike
    # Confirms if the move is backed by real money.
    avg_vol = df['volume'].rolling(window=20).mean().iloc[-1]
    current_vol = df['volume'].iloc[-1]
    vol_spike_mult = float(TUNING.get("vol_spike_mult", 1.5) or 1.5)
    vol_spike = current_vol > (avg_vol * vol_spike_mult)

    # 3. Triple Alignment Strategy
    # When SMC, Momentum, and Macro Bias all point in the same direction.
    alpha = 0.0
    if mom_bias > 0 and smc_score > 0 and macro_bias == "BULLISH":
        alpha = 0.5 if vol_spike else 0.25
    elif mom_bias < 0 and smc_score < 0 and macro_bias == "BEARISH":
        alpha = -0.5 if vol_spike else -0.25

    return alpha

def validate_signal_integrity(signal: dict, vol_context: dict) -> dict:
    """
    MASTER RECONSTRUCTION: Signal Integrity Validation.
    Final filter to ensure we aren't trading in 'Dangerous' conditions.
    """
    # For swing trading, a volatility squeeze is a potential breakout setup — not a veto.
    if vol_context.get('squeeze'):
        signal['squeeze_warning'] = "Volatility Squeeze: Potential Breakout Setup"

    # Block signals if ATR is too low (not enough movement to cover fees)
    if vol_context.get('atr_rank', 1.0) < 0.05:
        signal['action'] = "HOLD"
        signal['hold_reason'] = "Low Volatility: Insufficient Profit Potential"

    return signal

def compute_advanced_pivots(df: pd.DataFrame) -> dict:
    """
    MASTER RECONSTRUCTION: Calculates institutional pivot points.
    Includes Classic, Woodie, and Camarilla levels for the 725-line version.
    """
    if df is None or len(df) < 2: return {}

    # Use previous day/period for levels
    prev = df.iloc[-2]
    high, low, close = prev['high'], prev['low'], prev['close']

    pp = (high + low + close) / 3
    r1 = (2 * pp) - low
    s1 = (2 * pp) - high
    r2 = pp + (high - low)
    s2 = pp - (high - low)

    return {
        'pp': pp, 'r1': r1, 's1': s1, 'r2': r2, 's2': s2,
        'classic': {'pp': pp, 'r1': r1, 's1': s1, 'r2': r2, 's2': s2}
    }


def _apply_rejection_confirmation_gate(
    signal: dict,
    df_indicators: pd.DataFrame,
    strategy_config: dict,
    support: float = None,
    resistance: float = None,
) -> tuple[dict, bool]:
    """
    Final entry filter for reversal-style entries.

    This gate keeps the existing structure engine intact, but requires a
    confirmation candle before reversal entries are allowed to execute.
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
            if not macd_confirmed:
                reasons.append("MACD not yet bullish")
            if not psar_confirmed:
                reasons.append("PSAR has not flipped bullish")
            if not candle_confirmed:
                reasons.append("Candle not bullish")
            if not pullback_confirmed:
                reasons.append("Price still too close to support")
            signal["action"] = "HOLD"
            signal["hold_reason"] = f"RejectionGate: {' | '.join(reasons)}"
            signal["reason"] = f"{reason} [RejectionGate: {' | '.join(reasons)}]"
            signal["rejection_confirmation"] = {
                "confirmed": False,
                "mode": entry_mode,
                "action": action,
                "reason": " | ".join(reasons),
            }
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
            if not macd_confirmed:
                reasons.append("MACD not yet bearish")
            if not psar_confirmed:
                reasons.append("PSAR has not flipped bearish")
            if not candle_confirmed:
                reasons.append("Candle not bearish")
            if not pullback_confirmed:
                reasons.append("Price still too close to resistance")
            signal["action"] = "HOLD"
            signal["hold_reason"] = f"RejectionGate: {' | '.join(reasons)}"
            signal["reason"] = f"{reason} [RejectionGate: {' | '.join(reasons)}]"
            signal["rejection_confirmation"] = {
                "confirmed": False,
                "mode": entry_mode,
                "action": action,
                "reason": " | ".join(reasons),
            }
            return signal, True

    signal["reason"] = f"{reason} [RejectionGate: Confirmed]"
    signal["rejection_confirmation"] = {
        "confirmed": True,
        "mode": entry_mode,
        "action": action,
        "reason": "Confirmed",
    }
    return signal, True

def generate_quant_signal(state, latest_indicators, strategy_config, df_indicators, latest_macro, mtf_context=None, mtf_config=None, pivot_data=None) -> dict:
    """
    MASTER RECONSTRUCTION: The 725-Line Institutional Sniper Signal Engine.
    Exhaustive synthesis of 10 primary indicators with 4 advanced alpha overlays.

    This engine is designed for high-frequency scalping where every basis point
    of confidence matters. It factors in structural breaks, volatility squeezes,
    institutional pivot penalties, and liquidity pool proximity.
    """
    if df_indicators is None or len(df_indicators) < 50:
        return {"action": "HOLD", "score": 0, "confidence": 0, "reason": "Warming Up", "weights": {}}

    # --- 1. CORE PARAMETERS & CONTEXT ---
    current_price = state['price']
    range_action_zone_pct = float(strategy_config.get("range_action_zone_pct", 0.20) or 0.20)
    range_action_zone_pct = max(0.05, min(0.45, range_action_zone_pct))
    wall_veto_zone_pct = float(strategy_config.get("wall_veto_zone_pct", 0.20) or 0.20)
    wall_veto_zone_pct = max(0.05, min(0.35, wall_veto_zone_pct))
    support_veto_pct = float(strategy_config.get("support_veto_pct", 0.0015) or 0.0015)
    resistance_veto_pct = float(strategy_config.get("resistance_veto_pct", 0.0015) or 0.0015)
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
    weights = get_signal_weights()
    if isinstance(latest_macro, dict):
        macro_bias = str(latest_macro.get("regime") or latest_macro.get("bias") or "NEUTRAL").upper()
    else:
        macro_bias = str(latest_macro or "NEUTRAL").upper()
    support, resistance = _pick_structural_levels(current_price, mtf_context=mtf_context, pivot_data=pivot_data)
    vpoc = _compute_vpoc(df_indicators)
    anchored_vwap = _compute_anchored_vwap(df_indicators)
    last_close = float(df_indicators['close'].iloc[-1]) if len(df_indicators) else float(current_price)
    wall_state = _compute_wall_state(
        current_price=current_price,
        last_close=last_close,
        support=support,
        resistance=resistance,
        strategy_config=strategy_config,
    )
    location_score, location_notes, location_levels = _compute_market_location_score(
        current_price,
        support=support,
        resistance=resistance,
        vpoc=vpoc,
        anchored_vwap=anchored_vwap,
        state=state,
        latest_indicators=latest_indicators,
        strategy_config=strategy_config,
    )

    # --- 2. ADVANCED CONTEXTUAL MODULES ---
    # We call our institutional modules to understand the 'Regime'
    vol_context = compute_volatility_context(df_indicators)
    liquidity = identify_liquidity_pools(df_indicators)
    funding_impact = calculate_funding_impact(latest_macro)

    # --- 2b. FAST MTF BIAS ---
    # For scalping, fast timeframes lead; higher timeframes remain background context only.
    mtf_fast_score = 0.0
    mtf_fast_bias = "NEUTRAL"
    if mtf_config and mtf_config.get("enabled", False) and isinstance(mtf_context, dict):
        min_agree = int(mtf_config.get("min_agree", 2) or 2)
        min_agree = max(2, min(4, min_agree))
        tf_weights = [
            ("3m", 0.35),
            ("5m", 0.30),
            ("10m", 0.20),
            ("15m", 0.15),
        ]
        bull_votes = 0
        bear_votes = 0
        for tf, weight in tf_weights:
            ctx = mtf_context.get(tf)
            if not isinstance(ctx, dict):
                continue
            trend = str(ctx.get("trend", "NEUT") or "NEUT").upper()
            if trend == "BULL":
                bull_votes += 1
                mtf_fast_score += weight
            elif trend == "BEAR":
                bear_votes += 1
                mtf_fast_score -= weight

        if bull_votes >= min_agree:
            mtf_fast_bias = "LONG_ONLY"
        elif bear_votes >= min_agree:
            mtf_fast_bias = "SHORT_ONLY"

    mtf_rsi_score = 0.0
    mtf_rsi_bias = "NEUTRAL"
    if mtf_config and mtf_config.get("enabled", False) and isinstance(mtf_context, dict):
        rsi_bull_level = float(strategy_config.get("mtf_rsi_bull_level", 55) or 55)
        rsi_bear_level = float(strategy_config.get("mtf_rsi_bear_level", 45) or 45)
        rsi_min_agree = int(strategy_config.get("mtf_rsi_min_agree", 2) or 2)
        rsi_min_agree = max(1, min(4, rsi_min_agree))
        rsi_checks = [
            ("3m", 0.30),
            ("5m", 0.25),
            ("15m", 0.25),
            ("1h", 0.20),
        ]
        bull_votes = 0
        bear_votes = 0
        for tf, weight in rsi_checks:
            ctx = mtf_context.get(tf)
            if not isinstance(ctx, dict):
                continue
            try:
                rsi_val = float(ctx.get("rsi_14", 50.0) or 50.0)
            except (TypeError, ValueError):
                continue
            if rsi_val >= rsi_bull_level:
                mtf_rsi_score += weight
                bull_votes += 1
            elif rsi_val <= rsi_bear_level:
                mtf_rsi_score -= weight
                bear_votes += 1
        if bull_votes >= rsi_min_agree:
            mtf_rsi_bias = "BULLISH"
        elif bear_votes >= rsi_min_agree:
            mtf_rsi_bias = "BEARISH"

    # --- 3. PRIMARY SIGNAL CALCULATION ---
    # Structural & SMC Bias
    smc_score, sr_score, smc_label, ob_context = detect_smc_and_sr(df_indicators, current_price)

    # Mean Reversion (MR) - Enhanced with Z-Score & RSI
    # Identifies statistically overextended price action likely to snap back.
    ema_9 = latest_indicators.get('ema_9', current_price)
    ema_21 = latest_indicators.get('ema_21', current_price)

    # ANTI-CHASING GUARD: Prevent entering when EMAs already diverged (move already started)
    ema_dist_pct = abs(ema_9 - ema_21) / ema_21
    configured_max_chase_pct = float(strategy_config.get("max_chase_pct", 0.0) or 0.0)
    base_max_chase_pct = configured_max_chase_pct if configured_max_chase_pct > 0 else max(0.0015, min(0.004, atr_pct_now * 0.5))
    # Allow a slightly wider chase window when the fast MTF stack is already leaning
    # in the same direction as price. This keeps the guard from blocking obvious
    # trend continuation entries on strong breakdowns/breakouts.
    trend_continuation = (
        (current_price < ema_21 and (mtf_fast_bias == "SHORT_ONLY" or mtf_fast_score <= -0.15))
        or (current_price > ema_21 and (mtf_fast_bias == "LONG_ONLY" or mtf_fast_score >= 0.15))
    )
    if trend_continuation:
        max_chase_pct = max(base_max_chase_pct, min(0.0065, max(0.0045, atr_pct_now * 0.75)))
    else:
        max_chase_pct = base_max_chase_pct
    if ema_dist_pct > max_chase_pct:
        # If EMAs are already wide apart, the move has already started. Don't chase.
        return {"action": "HOLD", "score": 0.0, "confidence": 0.0, "reason": f"Chasing Guard ({ema_dist_pct:.3%}>{max_chase_pct:.3%})", "weights": {}}
    z_score = float(latest_indicators.get('z_score', 0.0) or 0.0)
    rsi_14 = float(latest_indicators.get('rsi_14', 50.0) or 50.0)

    mr_score = 0.0
    if current_price < ema_21 * 0.998:
        # FIX 8: Require BOTH z_score AND rsi for strong reversion — rsi alone fires
        # during mid-downtrend momentum candles and generates premature longs.
        if z_score <= -2.0 and rsi_14 < 35:
            mr_score = 1.0   # Strong bullish reversion (both conditions)
        elif z_score <= -1.5 or (z_score <= -1.0 and rsi_14 < 38):
            mr_score = 0.5   # Moderate stretch
    elif current_price > ema_21 * 1.002:
        if z_score >= 2.0 and rsi_14 > 65:
            mr_score = -1.0  # Strong bearish reversion (both conditions)
        elif z_score >= 1.5 or (z_score >= 1.0 and rsi_14 > 62):
            mr_score = -0.5  # Moderate stretch

    # Volume Weighted Average Price (VWAP)
    # The 'Institutional Value' anchor.
    vwap = latest_indicators.get('vwap', current_price)
    if mtf_fast_bias == "LONG_ONLY":
        vwap_score = 1.0 if current_price >= vwap else -1.0
    elif mtf_fast_bias == "SHORT_ONLY":
        vwap_score = -1.0 if current_price >= vwap else 1.0
    else:
        # FIX 6: In neutral MTF, use trend-following VWAP (above = bullish, below = bearish).
        # The old mean-reversion logic (below vwap = bullish) was causing longs near tops
        # when price was pulling back to VWAP from above during a downtrend.
        vwap_score = 1.0 if current_price >= vwap else -1.0

    # Bollinger Bands (BB)
    # Detecting exhaustion at the 2-sigma deviations.
    bb_low = latest_indicators.get('bb_low', current_price)
    bb_high = latest_indicators.get('bb_high', current_price)
    bb_score = 1.0 if current_price < bb_low * 1.001 else (-1.0 if current_price > bb_high * 0.999 else 0.0)

    # --- MACD ADVANCED PATTERNS ---
    # We detect Pattern 1 (Flip), Pattern 2 (Shrinking Tower), and Pattern 3 (Zero Bounce)
    macd_diff = latest_indicators.get('macd_diff', 0)
    prev_macd_diff = df_indicators['macd_diff'].iloc[-2] if len(df_indicators) > 2 else 0
    prev2_macd_diff = df_indicators['macd_diff'].iloc[-3] if len(df_indicators) > 3 else 0
    macd_val = latest_indicators.get('macd', 0)

    macd_score = 0.0
    # Pattern 1: The Flip (Standard Cross)
    if macd_diff > 0 and prev_macd_diff <= 0:
        macd_score += 0.8  # Strong flip
    elif macd_diff < 0 and prev_macd_diff >= 0:
        macd_score -= 0.8

    # Pattern 2: Shrinking Tower (Selling is Dying)
    # If histogram is negative but getting closer to zero
    if macd_diff < 0 and macd_diff > prev_macd_diff:
        macd_score += 0.3  # Bullish bonus (selling dying)
    elif macd_diff > 0 and macd_diff < prev_macd_diff:
        macd_score -= 0.3  # Bearish bonus (buying dying)

    # Pattern 3: Zero Bounce (Trend Has Power)
    # Price pulls back, histogram drops toward 0, then bounces back UP while still above 0
    if macd_diff > 0 and prev_macd_diff > 0 and macd_diff > prev_macd_diff and prev_macd_diff < prev2_macd_diff:
        macd_score += 1.0  # Very strong continuation
    elif macd_diff < 0 and prev_macd_diff < 0 and macd_diff < prev_macd_diff and prev_macd_diff > prev2_macd_diff:
        macd_score -= 1.0  # Very strong bearish continuation

    # Fallback/Base score
    if macd_score == 0:
        macd_score = np.sign(macd_diff) if abs(macd_diff) > 0.0001 else 0.0

    # --- EARLY DIVERGENCE / EXHAUSTION DETECTION ---
    divergence_state = _detect_macd_divergence(df_indicators)
    bear_div = divergence_state == "BEARISH"
    bull_div = divergence_state == "BULLISH"
    hidden_bull_div = divergence_state == "HIDDEN_BULLISH"
    hidden_bear_div = divergence_state == "HIDDEN_BEARISH"
    cvd_state, cvd_bonus = _detect_cvd_momentum_divergence(df_indicators)
    momentum_exhaustion = _detect_momentum_exhaustion(df_indicators)
    body_ratio_score = _body_range_ratio_score(df_indicators, lookback=3)

    # --- INSTITUTIONAL POWER SUITE (ALPHA) ---
    power_bonus = 0.0

    # 1. Volume Footprint (Whale Participation)
    avg_vol = df_indicators['volume'].rolling(window=20).mean().iloc[-1]
    curr_vol = df_indicators['volume'].iloc[-1]
    vol_spike = curr_vol > (avg_vol * 1.5)
    if vol_spike: power_bonus += 0.15

    # 2. EMA Gradient/Slope (Trading with the 15m Tide)
    htf_15m = mtf_context.get('15m', {}) if mtf_context else {}
    prev_ema_15m = float(htf_15m.get('ema_21_prev', 0) or 0)
    curr_ema_15m = float(htf_15m.get('ema_21', 0) or 0)
    slope_15m_up = curr_ema_15m > prev_ema_15m if prev_ema_15m > 0 else False
    if slope_15m_up: power_bonus += 0.10

    # 3. Bollinger Squeeze — reduce confidence, don't boost chop
    is_squeezed = vol_context.get("squeeze", False)
    if is_squeezed:
        power_bonus -= 0.15  # squeeze = no direction yet, wait for breakout
    else:
        # Check if squeeze just broke (width expanding)
        bb_width_now = df_indicators['bb_width'].iloc[-1]
        bb_width_prev = df_indicators['bb_width'].iloc[-2]
        if bb_width_now > bb_width_prev * 1.05:
            power_bonus += 0.20  # breakout confirmed

    # 4. Candle conviction and MTF RSI confluence
    power_bonus += body_ratio_score * 0.18
    power_bonus += mtf_rsi_score * 0.12

    # --- DIVERGENCE BONUS (ALPHA) ---
    div_bonus = 0.0
    if bull_div: div_bonus += 0.25
    if bear_div: div_bonus -= 0.25
    if hidden_bull_div: div_bonus += 0.18
    if hidden_bear_div: div_bonus -= 0.18
    if cvd_bonus != 0.0:
        div_bonus += cvd_bonus
    if momentum_exhaustion == "BULL_EXHAUST":
        div_bonus -= 0.12
    elif momentum_exhaustion == "BEAR_EXHAUST":
        div_bonus += 0.12

    # --- HIGH TIMEFRAME STRICT BIAS (1H / 4H) ---
    # Keep this as low-weight background context.
    # Use the latest closed candle from the trimmed HTF frame.
    htf_1h = mtf_context.get('1h', {}) if mtf_context else {}
    htf_4h = mtf_context.get('4h', {}) if mtf_context else {}

    macd_1h = float(htf_1h.get('macd', 0) or 0)
    macd_4h = float(htf_4h.get('macd', 0) or 0)

    htf_score = 0.0
    htf_score += 0.10 if macd_1h > 0 else (-0.10 if macd_1h < 0 else 0.0)
    htf_score += 0.10 if macd_4h > 0 else (-0.10 if macd_4h < 0 else 0.0)

    # Price Action (PA) + SAR Alignment
    psar_val = latest_indicators.get('psar', current_price)
    pa_bullish = df_indicators['close'].iloc[-1] > df_indicators['open'].iloc[-1] and current_price > psar_val
    pa_bearish = df_indicators['close'].iloc[-1] < df_indicators['open'].iloc[-1] and current_price < psar_val
    pa_score = 1.0 if pa_bullish else (-1.0 if pa_bearish else 0.0)

    # ADX / Volume Flow
    adx_value = float(latest_indicators.get('adx', 0.0) or 0.0)
    adx_pos = float(latest_indicators.get('adx_pos', 0.0) or 0.0)
    adx_neg = float(latest_indicators.get('adx_neg', 0.0) or 0.0)
    trend_dir = 1.0 if current_price >= ema_21 else -1.0
    if adx_value >= 25:
        adx_score = trend_dir
    elif adx_value >= 18:
        adx_score = trend_dir * 0.5
    else:
        adx_score = trend_dir * 0.15

    volume_delta = float(np.clip(_calculate_volume_delta(df_indicators), -1.0, 1.0))
    obv = float(latest_indicators.get('obv', 0.0) or 0.0)
    obv_ema = float(latest_indicators.get('obv_ema', obv) or obv)
    obv_score = 1.0 if obv > obv_ema else -1.0
    ob_score = _map_order_book_pressure(state)

    # KDJ & SuperTrend
    kdj_j = latest_indicators.get('j', 50)
    kdj_score = 1.0 if kdj_j < 20 else (-1.0 if kdj_j > 80 else 0.0)
    st_score = latest_indicators.get('trend_bias', 0)
    atr_min = float(strategy_config.get("vol_filter_atr_pct", 0.0005) or 0.0005)
    atr_max = float(strategy_config.get("vol_filter_atr_max_pct", 0.08) or 0.08)
    if atr_pct_now < atr_min or atr_pct_now > atr_max:
        return {"action": "HOLD", "score": 0.0, "confidence": 0.0, "reason": f"ATR guard ({atr_pct_now:.3%})", "weights": {}}

    # ADX Ranging Market Filter: below 15 = choppy range, indicators are noise
    if adx_value < 15:
        return {"action": "HOLD", "score": 0.0, "confidence": 0.0, "reason": f"Ranging market (ADX:{adx_value:.0f}<15)", "weights": {}}

    # --- 4. ALPHA OVERLAY & REFINEMENT ---
    # The 'Alpha Overlay' adds extra weight when multiple trends align.
    alpha = generate_alpha_overlay(df_indicators, smc_score, macro_bias)

    # --- 5. INSTITUTIONAL PENALTIES ---
    # DISABLED for testing: No-Man's Land Guard
    penalty = 1.0
    pivot_msg = "Gated:IndicatorsOnly"

    # --- 6. FINAL WEIGHTED SYNTHESIS ---
    # Support walls should help longs and hurt shorts; resistance walls should do the opposite.
    # Use the non-wall score to infer the current direction, then sign the wall contribution accordingly.
    score_without_sr = (
        (mr_score * weights['mr']) +
        (vwap_score * weights['vwap']) +
        (adx_score * weights['adx']) +
        (volume_delta * weights['vol']) +
        (obv_score * weights['obv']) +
        (ob_score * weights['ob']) +
        (bb_score * weights['bb']) +
        (macd_score * weights['macd']) +
        (pa_score * weights['pa']) +
        (smc_score * weights['smc']) +
        (location_score * weights['loc']) +
        (kdj_score * weights['kdj']) +
        (st_score * weights['st']) +
        alpha + # Add the Alpha Overlay contribution
        htf_score + # Add the Low-Weight 1H/4H Macro Bias
        div_bonus + # Add the Divergence Bonus Priority
        power_bonus # Add the Institutional Power Suite
    )
    # FIX BUG2: sr_directional sign must reflect the trade direction being evaluated.
    # score_without_sr already contains smc_score which is biased near walls,
    # so using its sign here can amplify the wrong side.
    # Correct logic: support wall (sr_score > 0) helps BUYs and hurts SELLs.
    # We still don't know `action` yet (set at line ~1487), so we use score_without_sr
    # sign as the best proxy — but we also add a hard wall veto on the action below (line ~1510).
    sr_directional = sr_score if score_without_sr >= 0 else -sr_score
    total_score = score_without_sr + (sr_directional * weights['sr'])

    # Apply context-based multipliers
    total_score *= penalty
    total_score += (funding_impact * 0.05) # Subtle adjustment for funding
    total_score += (mtf_fast_score * 0.20)
    if adx_value >= 25:
        total_score *= 1.10
    elif adx_value < 18:
        total_score *= 0.90

    # Precompute psar_bull and macd_diff_val for trend-continuation bias check
    psar_closed_val = latest_indicators.get('psar')
    psar_live_val = float(df_indicators['psar'].iloc[-1]) if "psar" in df_indicators.columns and len(df_indicators) else psar_closed_val
    macd_diff_val = float(latest_indicators.get('macd_diff', 0) or 0)
    psar_bull = True
    if psar_closed_val is not None:
        try:
            psar_bull = float(psar_closed_val) < current_price
        except (TypeError, ValueError):
            psar_bull = current_price >= ema_21
    elif psar_live_val is not None:
        try:
            psar_bull = float(psar_live_val) < current_price
        except (TypeError, ValueError):
            psar_bull = current_price >= ema_21

    # Trend-continuation bias:
    # If the fast scalp cluster is unanimous but the composite score is too flat,
    # give it a small push instead of freezing on the fence.
    if mtf_fast_bias == "LONG_ONLY" and total_score < 0.10:
        if current_price >= ema_21 and pa_score >= 0:
            total_score = max(total_score, 0.12)
    elif mtf_fast_bias == "SHORT_ONLY" and total_score > -0.10:
        if current_price <= ema_21 and pa_score <= 0:
            total_score = min(total_score, -0.12)
    elif mtf_fast_bias == "NEUTRAL":
        bearish_context_votes = int(current_price <= ema_21) + int(not psar_bull) + int(macd_diff_val < 0)
        bullish_context_votes = int(current_price >= ema_21) + int(psar_bull) + int(macd_diff_val > 0)
        # If the fast MTF stack is neutral but the local tape is already leaning hard
        # one way, bias the score a little so we do not sit flat through a trend leg.
        if bearish_context_votes >= 2 and current_price < ema_21 and pa_score <= 0 and total_score > -0.08:
            total_score = min(total_score, -0.08)
        elif bullish_context_votes >= 2 and current_price > ema_21 and pa_score >= 0 and total_score < 0.08:
            total_score = max(total_score, 0.08)

    # Cap score to [-1, 1] — prevents unbounded additive bonuses inflating confidence
    total_score = float(np.clip(total_score, -1.0, 1.0))

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

    signal_reason_suffix = ""
    hold_reason = ""
    sr_wall_locked = False

    # --- 7. SIGNAL INTEGRITY (Indicators Only) ---
    action = "HOLD"
    if total_score > 0.05: action = "BUY"
    elif total_score < -0.05: action = "SELL"

    # Precompute the fast indicator aliases before any wall-veto branch can use them.
    ema_9_val = float(latest_indicators.get('ema_9', current_price) or current_price)
    ema_21_val = float(latest_indicators.get('ema_21', current_price) or current_price)
    macd_val = float(latest_indicators.get('macd', 0) or 0)
    price_above_ema9 = current_price > ema_9_val
    price_below_ema9 = current_price < ema_9_val

    # FIX BUG2b: Hard wall veto now that we know action direction.
    # SR walls normally block counter-direction entries, but confirmed breakdowns/breakouts
    # are allowed to pass so the bot does not sit on obvious moves.
    if bool(strategy_config.get("sr_wall_veto_enabled", True)):
        wall_escape_gate = float(strategy_config.get("wall_breakout_score_gate", 0.15) or 0.15)
        if sr_score >= 1.0 and action == "SELL" and not _sr_wall_escape_ready(
            action,
            sr_score,
            total_score,
            mtf_fast_bias,
            macd_diff_val,
            psar_bull,
            current_price,
            ema_9_val,
            wall_escape_gate,
        ):
            action = "HOLD"
            hold_reason = f"SR Wall Veto: price near support (sr={sr_score:.1f}) — no SHORT"
            sr_wall_locked = True
        elif sr_score <= -1.0 and action == "BUY" and not _sr_wall_escape_ready(
            action,
            sr_score,
            total_score,
            mtf_fast_bias,
            macd_diff_val,
            psar_bull,
            current_price,
            ema_9_val,
            wall_escape_gate,
        ):
            action = "HOLD"
            hold_reason = f"SR Wall Veto: price near resistance (sr={sr_score:.1f}) — no LONG"
            sr_wall_locked = True

    # --- RANGE POSITION VETO ---
    # Blocks weak BUY entries near the top of the recent range and weak SELL entries
    # near the bottom. Directly targets the "long at top / short at bottom" failure pattern.
    #
    # Escape conditions (entry still allowed):
    #   1. Score is strong AND MTF is unanimously aligned (real breakout/breakdown)
    #   2. Veto is disabled via config (range_position_veto_enabled: false)
    #
    # Timeframe-aware: uses candles_per_day from config so the lookback is correct
    # on 1m (1440), 3m (480), 5m (288), etc. Falls back to 480 (3m equivalent).
    if action in {"BUY", "SELL"} and bool(strategy_config.get("range_position_veto_enabled", True)):
        _tf_str = str(strategy_config.get("timeframe", "3m") or "3m").strip().lower()
        _tf_map = {"1m": 1440, "3m": 480, "5m": 288, "10m": 144, "15m": 96, "1h": 24, "4h": 6}
        _cpd = int(strategy_config.get("candles_per_day", _tf_map.get(_tf_str, 480)))
        _lookback = max(50, min(_cpd, len(df_indicators)))

        _recent = df_indicators.iloc[-_lookback:]
        _range_high = float(_recent["high"].max())
        _range_low = float(_recent["low"].min())
        _range_width = _range_high - _range_low

        if _range_width > 0:
            _pos = (current_price - _range_low) / _range_width   # 0.0 = bottom, 1.0 = top
            _veto_top = float(strategy_config.get("range_veto_top_pct", 0.75) or 0.75)
            _veto_bottom = float(strategy_config.get("range_veto_bottom_pct", 0.25) or 0.25)
            _veto_min_score = float(strategy_config.get("range_veto_min_score", 0.55) or 0.55)
            _veto_escape_score = max(0.80, float(strategy_config.get("range_veto_escape_score", 0.80) or 0.80))

            _mtf_unanimous = mtf_fast_bias in {"LONG_ONLY", "SHORT_ONLY"}
            _is_strong = (
                abs(total_score) >= _veto_escape_score
                and _mtf_unanimous
                and abs(sr_score) < 1.0
            )

            if action == "BUY" and _pos >= _veto_top:
                if _is_strong and mtf_fast_bias == "LONG_ONLY" and sr_score > -0.5:
                    # Unanimous bull MTF + very strong score + clear resistance = real breakout.
                    pass
                else:
                    action = "HOLD"
                    hold_reason = (
                        f"Range Veto: BUY in top {int(_veto_top*100)}% of {_lookback}c range "
                        f"(pos={_pos:.0%} score={total_score:.2f} sr={sr_score:.1f}). "
                        f"Need score>={_veto_escape_score:.2f} + unanimous MTF + sr>-0.5 for breakout."
                    )

            elif action == "SELL" and _pos <= _veto_bottom:
                if _is_strong and mtf_fast_bias == "SHORT_ONLY" and sr_score < 0.5:
                    # Unanimous bear MTF + very strong score + clear support = real breakdown.
                    pass
                else:
                    action = "HOLD"
                    hold_reason = (
                        f"Range Veto: SELL in bottom {int((1-_veto_bottom)*100)}% of {_lookback}c range "
                        f"(pos={_pos:.0%} score={total_score:.2f} sr={sr_score:.1f}). "
                        f"Need score>={_veto_escape_score:.2f} + unanimous MTF + sr<0.5 for breakdown."
                    )

    # --- TREND CONFIRMATION: EMA + PSAR + MACD must agree ---
    # We follow the "MACD Zero Line Rule" from your screenshots:
    # 1. MACD > 0 => LONGS ONLY
    # 2. MACD < 0 => SHORTS ONLY
    # 3. Ignore crossovers inside the "Noise Channel"
    ema_9_val = float(latest_indicators.get('ema_9', current_price) or current_price)
    ema_21_val = float(latest_indicators.get('ema_21', current_price) or current_price)
    psar_closed_val = latest_indicators.get('psar')
    psar_live_val = float(df_indicators['psar'].iloc[-1]) if "psar" in df_indicators.columns and len(df_indicators) else psar_closed_val
    macd_val = float(latest_indicators.get('macd', 0) or 0)
    macd_diff_val = float(latest_indicators.get('macd_diff', 0) or 0)
    price_above_ema9 = current_price > ema_9_val
    price_below_ema9 = current_price < ema_9_val

    # Noise Channel Threshold: Ignore signals too close to the zero line.
    # For DOGE at $0.11, 0.0001 is a meaningful filter (similar to 0.5 on Forex)
    macd_noise_threshold = float(strategy_config.get('macd_noise_threshold', 0.0001) or 0.0001)

    psar_state_note = ""
    article_sl_override = None

    if action in {"BUY", "SELL"}:
        ema_bull = ema_9_val > ema_21_val
        psar_closed_bull = True  # default if PSAR unavailable
        psar_live_bull = True
        if psar_closed_val is not None:
            try:
                psar_closed_bull = float(psar_closed_val) < current_price
            except (TypeError, ValueError):
                pass
        if psar_live_val is not None:
            try:
                psar_live_bull = float(psar_live_val) < current_price
            except (TypeError, ValueError):
                pass
        # Backward-compatible alias used by the rest of the signal engine.
        psar_bull = psar_closed_bull
        psar_state_note = f"PSAR(C:{'BULL' if psar_closed_bull else 'BEAR'} L:{'BULL' if psar_live_bull else 'BEAR'})"

        # MACD Strategy from Screenshots:
        # Rule 1: Zero Line Bias (macd > 0 for Long, macd < 0 for Short)
        # Rule 2: Crossover (macd_diff > 0 for Long, macd_diff < 0 for Short)
        # Rule 3: Channel Filter (abs(macd) > noise threshold)
        # Rule 4: Divergence (Price HH + MACD LH = Bearish Divergence)
        macd_pos_bull = macd_val > 0
        macd_cross_bull = macd_diff_val > 0

        # --- MTF MACD & PRICE ACTION HIERARCHY ---
        # 1. 15m Bias: Above/Below Zero
        # 2. 10m Confirmation: Same direction + Outside Noise Channel
        # 3. 3m Trigger: Histogram Flip + Price Action (Candle Color)
        mtf_15m = mtf_context.get('15m', {})
        mtf_10m = mtf_context.get('10m', {})

        macd_15m = float(mtf_15m.get('macd', 0) or 0)
        macd_10m = float(mtf_10m.get('macd', 0) or 0)

        # Trend Bias & Confirmation Layers
        bias_bull = macd_15m > 0
        conf_bull = macd_10m > 0 and abs(macd_10m) > macd_noise_threshold
        mtf_macd_bull = bias_bull and conf_bull
        mtf_macd_bear = (macd_15m < 0) and (macd_10m < 0) and abs(macd_10m) > macd_noise_threshold
        mtf_structure_bull = str(mtf_15m.get("structure", "")).upper() in {"HH_HL", "HIGHER_LOW"}
        mtf_structure_bear = str(mtf_15m.get("structure", "")).upper() in {"LH_LL", "LOWER_HIGH"}

        # Price Action (PA) Rule: Candle must match direction
        pa_bull = current_price > float(df_indicators['open'].iloc[-1])

        # Structural Break Rule: Price must be pushing the range
        # (Using 10m high/low as the breakout level)
        ten_min_high = float(mtf_10m.get('high', 0) or 0)
        ten_min_low = float(mtf_10m.get('low', 0) or 0)
        breakout_bull = current_price >= ten_min_high * 0.9998 if ten_min_high > 0 else True
        breakout_bear = current_price <= ten_min_low * 1.0002 if ten_min_low > 0 else True

        # --- CORE 3 STRATEGY (EMA + MACD + SAR) ---
        # 1. EMA (9/21 Agreement)
        # 2. MACD (Crossover + Bias)
        # 3. PSAR (Direction + 3-Dot Streak)

        # --- INSTITUTIONAL MACD GATE ---
        # Use the configured threshold directly.
        macd_scalp_threshold = macd_noise_threshold
        macd_outside_noise = abs(macd_val) > macd_scalp_threshold

        # *** Zero Line Cross Detection (3-Candle Window) ***
        # We look back 3 candles; if any of them crossed zero, we are in the "Impact Zone"
        lookback_macds = df_indicators['macd'].tail(3).values
        macd_zero_cross_bull = any((lookback_macds[i] > 0 and lookback_macds[i-1] <= 0) for i in range(1, len(lookback_macds)))
        macd_zero_cross_bear = any((lookback_macds[i] < 0 and lookback_macds[i-1] >= 0) for i in range(1, len(lookback_macds)))

        # Combined Institutional MACD Signal (Pure Side-of-Zero logic)
        macd_inst_bull = (macd_pos_bull or macd_zero_cross_bull) and macd_outside_noise
        macd_inst_bear = ((not macd_pos_bull) or macd_zero_cross_bear) and macd_outside_noise

        # Divergence already computed above — skip duplicate re-detection
        # Momentum Only Gate (No Divergence Veto)
        macd_final_bull = macd_inst_bull
        macd_final_bear = macd_inst_bear

        # PSAR Streak (Instant Flip Rule)
        psar_val_raw = latest_indicators.get('psar_streak', 0)
        psar_streak = int(psar_val_raw) if pd.notnull(psar_val_raw) else 0
        psar_1_dot_bull = abs(psar_streak) >= 1

        # --- ENTRY MODE CLASSIFIER ---
        # Three distinct entry modes, each with its own gate logic:
        #
        #  REVERSAL_LONG   — price at support (intact), score bullish → BUY
        #  REVERSAL_SHORT  — price at resistance (intact), score bearish → SELL
        #  BREAKOUT_LONG   — resistance just broken with close above → BUY momentum
        #  BREAKOUT_SHORT  — support just broken with close below → SELL momentum
        #  BREAKDOWN_SHORT — support being tested with strong bearish alignment
        #  TREND           — MTF unanimous, enter anywhere with EMA9+MACD confirm
        #
        # The only hard directional veto: BUY into intact resistance.
        # All other combos are evaluated by the midrange gate below.

        sup_touch  = bool(wall_state.get("support_touching"))
        res_touch  = bool(wall_state.get("resistance_touching"))
        sup_broken = bool(wall_state.get("support_broken"))
        res_broken = bool(wall_state.get("resistance_broken"))

        entry_mode = "TREND"
        if support and resistance:
            if res_broken and action == "BUY":
                entry_mode = "BREAKOUT_LONG"
            elif sup_broken and action == "SELL":
                entry_mode = "BREAKOUT_SHORT"
            elif res_touch and not res_broken and action == "SELL":
                entry_mode = "REVERSAL_SHORT"
            elif sup_touch and not sup_broken and action == "BUY":
                entry_mode = "REVERSAL_LONG"
            elif res_touch and not res_broken and action == "BUY":
                # If the long is being rejected at resistance, optionally flip into
                # a short when the rejection is confirmed by momentum/structure.
                wall_reversal_gate = float(strategy_config.get("wall_reversal_score_gate", 0.12) or 0.12)
                resistance_rejection_ready = (
                    bool(strategy_config.get("wall_reversal_assist", True))
                    and total_score <= -wall_reversal_gate
                    and (
                        (macd_final_bear and not psar_bull)
                        or (
                            current_price < float(df_indicators["open"].iloc[-1])
                            and (
                                macd_diff_val <= 0
                                or not psar_bull
                                or current_price < ema_9_val
                            )
                        )
                    )
                )
                if resistance_rejection_ready:
                    action = "SELL"
                    entry_mode = "REVERSAL_SHORT"
                    signal_reason_suffix += " [Wall Rejection Short]"
                else:
                    # Only hard veto: buying INTO an intact ceiling
                    action = "HOLD"
                    hold_reason = (
                        f"Wall Veto: BUY blocked — price at resistance "
                        f"(${current_price:.3f} / wall ${float(resistance):.3f}). "
                        f"Wait for break>${float(wall_state.get('resistance_break_level', resistance)):.3f}."
                    )
                    sr_wall_locked = True
            elif sup_touch and not sup_broken and action == "SELL":
                # If a short is being rejected at support, optionally flip into a
                # long when the reclaim is confirmed by momentum/structure.
                wall_reversal_gate = float(strategy_config.get("wall_reversal_score_gate", 0.12) or 0.12)
                support_rejection_ready = (
                    bool(strategy_config.get("wall_reversal_assist", True))
                    and total_score >= wall_reversal_gate
                    and (
                        (macd_final_bull and psar_bull)
                        or (
                            current_price > float(df_indicators["open"].iloc[-1])
                            and (
                                macd_diff_val >= 0
                                or psar_bull
                                or current_price > ema_9_val
                            )
                        )
                    )
                )
                if support_rejection_ready:
                    action = "BUY"
                    entry_mode = "REVERSAL_LONG"
                    signal_reason_suffix += " [Wall Reclaim Long]"
                else:
                    # Shorting into an intact floor: only allow with strong breakdown evidence
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

        signal_reason_suffix += f" [Mode:{entry_mode}]"

        # --- STRIKE ZONE OVERRIDE ---
        # Only use the fast strike gate at the actual edge of the range. Buying the
        # middle of the lower half caused late longs before a real support reclaim.
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

        # Final indicators-only gate (Relaxed if in Action Zone)
        if in_action_zone:
            # AGGRESSIVE SCALP MODE: In the Strike Zone, we use the EMA 9 line as a dynamic barrier.
            # We also check for Volume Surge and RSI overextension for high-conviction reversals.
            macd_fast_bull = macd_diff_val > 0
            macd_fast_bear = macd_diff_val < 0

            # Fast Volume Confirmation
            current_vol = float(df_indicators['volume'].iloc[-1])
            avg_vol_fast = df_indicators['volume'].rolling(window=10).mean().iloc[-1]
            volume_surge = current_vol > avg_vol_fast * 1.1  # 10% volume surge

            # RSI Overextension (Oversold for Long, Overbought for Short)
            rsi_val = float(latest_indicators.get('rsi', 50) or 50)
            rsi_ob = rsi_val > 65
            rsi_os = rsi_val < 35

            if action == "BUY":
                # Primary: Price > EMA9
                # Secondary (Need 1/3): PSAR Bull, Hist Bull, or RSI OS + Vol Surge
                bull_momentum = psar_bull or macd_fast_bull or (rsi_os and volume_surge)
                if mtf_macd_bear and mtf_structure_bear and not bull_div:
                    action = "HOLD"
                    hold_reason = "MTF Gate: 15m/10m MACD bearish with lower-high structure"
                if not (price_above_ema9 and bull_momentum):
                    action = "HOLD"
                    hold_reason = "Strike Zone: Waiting for Price>EMA9 + Bull Momentum (SAR/Hist/RSI)"
            elif action == "SELL":
                # Primary: Price < EMA9
                # Secondary (Need 1/3): PSAR Bear, Hist Bear, or RSI OB + Vol Surge
                bear_momentum = (not psar_bull) or macd_fast_bear or (rsi_ob and volume_surge)
                if mtf_macd_bull and mtf_structure_bull and not bear_div:
                    action = "HOLD"
                    hold_reason = "MTF Gate: 15m/10m MACD bullish with higher-low structure"
                if not (price_below_ema9 and bear_momentum):
                    action = "HOLD"
                    hold_reason = "Strike Zone: Waiting for Price<EMA9 + Bear Momentum (SAR/Hist/RSI)"

        ob_active = bool(ob_context.get("active"))
        ob_mid = ob_context.get("mid")
        ob_dir = str(ob_context.get("direction", "NEUTRAL") or "NEUTRAL").upper()
        if action in {"BUY", "SELL"} and ob_active and ob_mid:
            ob_mid = float(ob_mid)
            ob_mid_tolerance = float(strategy_config.get("ob_midpoint_tolerance_pct", 0.0015) or 0.0015)
            near_ob_mid = abs(current_price - ob_mid) / ob_mid <= ob_mid_tolerance
            if action == "BUY" and ob_dir == "BULLISH":
                if not near_ob_mid:
                    action = "HOLD"
                    hold_reason = f"OB Gate: wait for 50% bullish OB retest near {ob_mid:.3f}"
                else:
                    signal_reason_suffix += " [OB Mid Retest]"
            elif action == "SELL" and ob_dir == "BEARISH":
                if not near_ob_mid:
                    action = "HOLD"
                    hold_reason = f"OB Gate: wait for 50% bearish OB retest near {ob_mid:.3f}"
                else:
                    signal_reason_suffix += " [OB Mid Retest]"
        else:
            # CONSERVATIVE TREND MODE: Outside Strike Zone, require 2-of-3 confirmation
            # instead of waiting for all three lagging trend filters to agree.
            ema_cross_bull = ema_9_val > ema_21_val
            bull_votes = int(ema_cross_bull) + int(psar_bull) + int(macd_final_bull)
            bear_votes = int(not ema_cross_bull) + int(not psar_bull) + int(macd_final_bear)
            sell_momentum_escape = (
                current_price < ema_21_val
                and (not psar_bull or macd_diff_val < 0)
                and pa_score <= 0
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
                    action = "HOLD"
                    hold_reason = "Trend Gate: Waiting for 2-of-3 EMA/PSAR/MACD alignment"
            elif action == "BUY" and mtf_macd_bear and mtf_structure_bear:
                action = "HOLD"
                hold_reason = "MTF Gate: 15m/10m MACD bearish with lower-high structure"
            elif action == "SELL" and bear_votes < 2:
                if sell_momentum_escape:
                    pass
                else:
                    action = "HOLD"
                    hold_reason = "Trend Gate: Waiting for 2-of-3 EMA/PSAR/MACD alignment"
            elif action == "SELL" and mtf_macd_bull and mtf_structure_bull:
                action = "HOLD"
                hold_reason = "MTF Gate: 15m/10m MACD bullish with higher-low structure"

        orb_breakout_setup = _detect_orb_breakout_setup(
            df_indicators,
            current_price,
            latest_indicators=latest_indicators,
            strategy_config=strategy_config,
            mtf_fast_bias=mtf_fast_bias,
        )
        vwap_bounce_setup = _detect_vwap_bounce_setup(
            df_indicators,
            current_price,
            latest_indicators=latest_indicators,
            strategy_config=strategy_config,
            anchored_vwap=anchored_vwap,
            mtf_fast_bias=mtf_fast_bias,
        )
        article_notes = []
        article_override = None
        article_sl_override = None

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

        signal_reason_suffix = " [Strike Override]" if in_action_zone and action != "HOLD" else (" [Indicators OK]" if action != "HOLD" else " [Waiting for alignment]")
        if article_override:
            signal_reason_suffix += f" [{article_override}]"
        if article_notes:
            article_note_text = " | ".join(article_notes)
            if len(article_note_text) > 120:
                article_note_text = article_note_text[:117] + "..."
            signal_reason_suffix += f" [{article_note_text}]"

        # --- SNIPER CHECKLIST (Direction-Aware) ---
        is_bearish_attempt = total_score < 0
        checklist = []

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

        checklist.append(f"Vol:{'OK' if vol_spike else 'Low'}")

        checklist_str = " | ".join(checklist)

        status_msg = f"[{checklist_str}]"
        if action != "HOLD":
            status_msg = f"READY: {checklist_str}"
            if abs(div_bonus) > 0: status_msg += " +DIV"

        signal_reason_suffix = f" {status_msg}"
    else:
        psar_closed_bull = True
        psar_live_bull = True
        if psar_closed_val is not None:
            try:
                psar_closed_bull = float(psar_closed_val) < current_price
            except (TypeError, ValueError):
                pass
        if psar_live_val is not None:
            try:
                psar_live_bull = float(psar_live_val) < current_price
            except (TypeError, ValueError):
                pass
        psar_state_note = f"PSAR(C:{'BULL' if psar_closed_bull else 'BEAR'} L:{'BULL' if psar_live_bull else 'BEAR'})"
        signal_reason_suffix = f" [Waiting for alignment | {psar_state_note}]"

    # The reason string used for the professional dashboard.
    reason = (
        f"{signal_reason_suffix} | {psar_state_note} | Score:{total_score:.3f} SMC:{smc_label} Pivot:{pivot_msg} MTF:{mtf_fast_bias} "
        f"(MR:{mr_score:.1f} OB:{smc_score:.1f} SR:{sr_score:.1f} VWAP:{vwap_score:.1f} ADX:{adx_score:.1f} "
        f"LOC:{location_score:.1f} VOL:{volume_delta:.1f} OBV:{obv_score:.1f} BB:{bb_score:.1f} MACD:{macd_score:.1f} PA:{pa_score:.1f} "
        f"KDJ:{kdj_score:.1f} ST:{st_score:.1f} DIV:{divergence_state} CVD:{cvd_state} EXH:{momentum_exhaustion} "
        f"RSI:{mtf_rsi_bias} BR:{body_ratio_score:.2f})"
    )

    # Exit Rule: Trigger closure if SAR flips
    psar_val_raw = latest_indicators.get('psar_streak', 0)
    psar_streak = int(psar_val_raw) if pd.notnull(psar_val_raw) else 0
    signal = {
        "action": action,
        "score": total_score,
        "confidence": min(abs(total_score), 1.0),
        "reason": reason,
        "psar_streak": psar_streak,
        "psar_exit": True if psar_streak != 0 else False, # will be used by main loop for exit
        "psar_closed_bull": bool(psar_closed_bull) if 'psar_closed_bull' in locals() else None,
        "psar_live_bull": bool(psar_live_bull) if 'psar_live_bull' in locals() else None,
        "psar_state_note": psar_state_note,
        "article_sl_override": float(article_sl_override) if article_sl_override is not None else None,
        "market_bias": mtf_fast_bias if mtf_fast_bias != "NEUTRAL" else "NEUTRAL",
        "mtf_fast_bias": mtf_fast_bias,
        "mtf_rsi_bias": mtf_rsi_bias,
        "mtf_rsi_score": float(mtf_rsi_score),
        "momentum_exhaustion": momentum_exhaustion,
        "cvd_state": cvd_state,
        "body_ratio_score": float(body_ratio_score),
        "vpoc": float(vpoc) if vpoc else 0.0,
        "anchored_vwap": float(anchored_vwap) if anchored_vwap else 0.0,
        "order_block": ob_context,
        "hold_reason": hold_reason
    }
    gate_locked = bool(hold_reason) or sr_wall_locked

    if support is not None:
        signal["structure_support"] = float(support)
    if resistance is not None:
        signal["structure_resistance"] = float(resistance)
    if isinstance(pivot_data, dict):
        signal["pivot_classic"] = dict(pivot_data.get("classic", {}) or {})
    signal["wall_state"] = wall_state
    signal["market_location"] = {
        "score": float(location_score),
        "notes": location_notes,
        "levels": location_levels,
    }
    signal["sr_wall_locked"] = bool(sr_wall_locked)

    mr_setup = _detect_mean_reversion_setup(df_indicators, strategy_config)
    if mr_setup.get("triggered"):
        signal["mean_reversion"] = {
            "direction": mr_setup.get("direction"),
            "reason": mr_setup.get("reason"),
        }
        signal["reason"] = f"{signal['reason']} MR:{mr_setup.get('reason', '')}"
        signal["score"] = float(signal["score"]) + float(mr_setup.get("score", 0.0) or 0.0)
        signal["confidence"] = min(abs(float(signal["score"])), 1.0)
        mr_long_allowed = (sr_score > -1.0) and not bool(wall_state.get("resistance_touching"))
        mr_short_allowed = (sr_score < 1.0) and not bool(wall_state.get("support_touching"))
        if not gate_locked and mr_setup.get("direction") == "LONG" and mtf_fast_bias != "SHORT_ONLY" and mr_long_allowed:
            signal["action"] = "BUY" if signal["action"] == "HOLD" or float(signal["score"]) > 0 else signal["action"]
        elif not gate_locked and mr_setup.get("direction") == "SHORT" and mtf_fast_bias != "LONG_ONLY" and mr_short_allowed:
            signal["action"] = "SELL" if signal["action"] == "HOLD" or float(signal["score"]) < 0 else signal["action"]

    wick_setup = _detect_wick_sweep_setup(
        df_indicators,
        current_price,
        support=support,
        resistance=resistance,
        config=strategy_config,
    )
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
        wick_long_allowed = (sr_score > -1.0) and not bool(wall_state.get("resistance_touching"))
        wick_short_allowed = (sr_score < 1.0) and not bool(wall_state.get("support_touching"))
        if not gate_locked and wick_setup.get("direction") == "LONG" and mtf_fast_bias != "SHORT_ONLY" and wick_long_allowed:
            signal["action"] = "BUY" if signal["action"] == "HOLD" or float(signal["score"]) > 0 else signal["action"]
        elif not gate_locked and wick_setup.get("direction") == "SHORT" and mtf_fast_bias != "LONG_ONLY" and wick_short_allowed:
            signal["action"] = "SELL" if signal["action"] == "HOLD" or float(signal["score"]) < 0 else signal["action"]

    # --- STRUCTURAL STOP LOSS ---
    # Place SL at structural level, but cap at max_structural_sl_pct
    stop_buffer = 0.0005
    final_action = signal["action"]
    article_sl_override = signal.get("article_sl_override")
    if article_sl_override and final_action in {"BUY", "SELL"}:
        signal["sl"] = float(article_sl_override)
    elif final_action == "BUY" and support is not None:
        signal["sl"] = float(support) * (1 - stop_buffer)  # just below support
    elif final_action == "SELL" and resistance is not None:
        signal["sl"] = float(resistance) * (1 + stop_buffer)  # just above resistance
        classic_pivots = pivot_data.get("classic", {}) if isinstance(pivot_data, dict) else {}
        try:
            r1_level = float(classic_pivots.get("r1", 0.0) or 0.0)
        except (TypeError, ValueError):
            r1_level = 0.0
        if r1_level > current_price and float(signal["sl"]) < r1_level * 1.001:
            signal["sl"] = r1_level * 1.002
            signal["sl_source"] = "pivot_r1_guard"

    # Recalculate final_action
    final_action = signal["action"]

    if final_action in {"BUY", "SELL"} and signal.get("sl") and current_price > 0:
        max_sl_pct = float(strategy_config.get("max_structural_sl_pct", 0.0040) or 0.0040)
        min_reward_risk = float(strategy_config.get("min_reward_risk", 0.75) or 0.75)
        fallback_sl_pct = float(strategy_config.get("sl_pct", 0.0015) or 0.0015)
        sl_dist_pct = abs(float(signal["sl"]) - float(current_price)) / float(current_price)

        # TP target = structural resistance for LONG, support for SHORT
        # This anchors TP to real chart levels, not arbitrary percentages
        tp_pct_cfg = float(strategy_config.get("tp_pct", 0.0025) or 0.0025)
        if final_action == "BUY" and resistance is not None:
            structural_tp = float(resistance) * (1 - stop_buffer)  # slightly inside resistance
            structural_tp_pct = (structural_tp - current_price) / current_price
            # Use structural TP if it's meaningful, otherwise fall back to config pct
            if structural_tp_pct > 0.001:  # at least 0.1% room
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
        # If structural SL is too far away, VETO the entry rather than using a fake SL
        if sl_dist_pct > max_sl_pct and not wick_mode:
            signal["action"] = "HOLD"
            signal["hold_reason"] = f"Structural SL too far ({sl_dist_pct:.2%} > {max_sl_pct:.2%})"
            signal["reason"] = f"{signal['reason']} SLTooFar:{sl_dist_pct:.2%}"

    if mtf_fast_bias == "LONG_ONLY" and signal["action"] == "SELL":
        signal["action"] = "HOLD"
        signal["hold_reason"] = "MTF trend veto: fast timeframes bullish"
    elif mtf_fast_bias == "SHORT_ONLY" and signal["action"] == "BUY":
        signal["action"] = "HOLD"
        signal["hold_reason"] = "MTF trend veto: fast timeframes bearish"
    elif mtf_fast_bias == "NEUTRAL" and signal["action"] in {"BUY", "SELL"}:
        if signal["action"] == "SELL" and smc_score > 0:
            signal["action"] = "HOLD"
            signal["hold_reason"] = "NEUTRAL MTF + Bullish structure = no short"
        elif signal["action"] == "BUY" and smc_score < 0:
            signal["action"] = "HOLD"
            signal["hold_reason"] = "NEUTRAL MTF + Bearish structure = no long"
        elif abs(total_score) < 0.05:
            signal["action"] = "HOLD"
            signal["hold_reason"] = "MTF trend veto: fast timeframes not aligned"
        else:
            signal["reason"] = f"{signal['reason']} MTFPartial"
    elif mtf_fast_bias != "NEUTRAL":
        signal["reason"] = f"{signal['reason']} MTFConfirm:{mtf_fast_bias}"

    # --- RANGE REVERSAL SNIPER MODE (DYNAMIC OUTER EDGE) ---
    # We measure the total range width and only allow entries in the configured outer zone.
    m5_s = signal.get("structure_support")
    m5_r = signal.get("structure_resistance")

    if m5_s and m5_r:
        range_width        = m5_r - m5_s
        action_zone_size   = range_width * range_action_zone_pct
        top_action_zone    = m5_r - action_zone_size
        bottom_action_zone = m5_s + action_zone_size
        signal["action_support"]    = float(bottom_action_zone)
        signal["action_resistance"] = float(top_action_zone)

        at_top    = current_price >= top_action_zone
        at_bottom = current_price <= bottom_action_zone
        in_middle = not at_top and not at_bottom

        final_action = signal["action"]
        _reason_str  = signal.get("reason", "") or ""
        entry_mode   = _reason_str.split("[Mode:")[-1].split("]")[0] if "[Mode:" in _reason_str else "TREND"

        if final_action in {"BUY", "SELL"} and in_middle:
            # ── MIDRANGE POLICY ──────────────────────────────────────────────────
            # REVERSAL / BREAKOUT modes must be at the edge of the range.
            # TREND mode is allowed in midrange, but requires:
            #   1. MTF fast bias unanimously agrees with direction
            #   2. Score meets a higher midrange threshold
            # This unlocks trend-following scalps while keeping bad midrange
            # reversals filtered out.
            mid_min_score = float(strategy_config.get("midrange_min_score", 0.28) or 0.28)
            mtf_aligned   = mtf_fast_bias == ("LONG_ONLY" if final_action == "BUY" else "SHORT_ONLY")
            score_ok      = abs(float(signal.get("score", 0.0) or 0.0)) >= mid_min_score

            if "REVERSAL" in entry_mode or "BREAKOUT" in entry_mode:
                signal["action"]      = "HOLD"
                signal["hold_reason"] = (
                    f"Gate: {entry_mode} needs edge "
                    f"(price ${current_price:.2f} between zones ${bottom_action_zone:.2f}–${top_action_zone:.2f})."
                )
            elif not mtf_aligned:
                signal["action"]      = "HOLD"
                signal["hold_reason"] = (
                    f"Midrange Gate: MTF ({mtf_fast_bias}) ≠ {final_action}. "
                    f"Wait for edge or unanimous MTF bias."
                )
            elif not score_ok:
                signal["action"]      = "HOLD"
                signal["hold_reason"] = (
                    f"Midrange Gate: score {float(signal.get('score', 0)):.3f} < {mid_min_score:.2f}. "
                    f"Need stronger signal for midrange trend entry."
                )
            # else: TREND entry in midrange approved — but only if SR wall allows it.
            # Confirmed breakouts/breakdowns may override the wall if the trend is already aligned.
            elif final_action == "BUY" and sr_score <= -1.0 and not _sr_wall_escape_ready(
                final_action,
                sr_score,
                float(signal.get("score", 0.0) or 0.0),
                mtf_fast_bias,
                float(macd_diff_val or 0.0),
                bool(psar_bull),
                current_price,
                ema_9_val,
                float(strategy_config.get("wall_breakout_score_gate", 0.15) or 0.15),
            ):
                signal["action"]      = "HOLD"
                signal["hold_reason"] = (
                    f"Midrange SR Block: BUY near resistance wall (sr={sr_score:.1f}) "
                    f"even with TREND mode — no long into ceiling."
                )
                signal["sr_wall_locked"] = True
            elif final_action == "SELL" and sr_score >= 1.0 and not _sr_wall_escape_ready(
                final_action,
                sr_score,
                float(signal.get("score", 0.0) or 0.0),
                mtf_fast_bias,
                float(macd_diff_val or 0.0),
                bool(psar_bull),
                current_price,
                ema_9_val,
                float(strategy_config.get("wall_breakout_score_gate", 0.15) or 0.15),
            ):
                signal["action"]      = "HOLD"
                signal["hold_reason"] = (
                    f"Midrange SR Block: SELL near support wall (sr={sr_score:.1f}) "
                    f"even with TREND mode — no short into floor."
                )
                signal["sr_wall_locked"] = True

        signal["top_action_zone"]    = float(top_action_zone)
        signal["bottom_action_zone"] = float(bottom_action_zone)

    else:
        # No S/R found — allow TREND entries, gate reversals
        _reason_str = signal.get("reason", "") or ""
        entry_mode  = _reason_str.split("[Mode:")[-1].split("]")[0] if "[Mode:" in _reason_str else "TREND"
        if "REVERSAL" in entry_mode:
            signal["action"]      = "HOLD"
            signal["hold_reason"] = "Range Gate: No S/R boundaries — reversal entry skipped."
        # FIX 4: Even with no structural S/R, respect the SMC wall score.
        # sr_score is set from the swing-based wall detection in detect_smc_and_sr,
        # so it can fire even when no clean S/R zones were found.
        elif signal.get("action") == "BUY" and sr_score <= -1.0 and not _sr_wall_escape_ready(
            signal.get("action"),
            sr_score,
            float(signal.get("score", 0.0) or 0.0),
            mtf_fast_bias,
            float(macd_diff_val or 0.0),
            bool(psar_bull),
            current_price,
            ema_9_val,
            float(strategy_config.get("wall_breakout_score_gate", 0.15) or 0.15),
        ):
            signal["action"]      = "HOLD"
            signal["hold_reason"] = f"No-SR fallback SR block: BUY near resistance (sr={sr_score:.1f})"
            signal["sr_wall_locked"] = True
        elif signal.get("action") == "SELL" and sr_score >= 1.0 and not _sr_wall_escape_ready(
            signal.get("action"),
            sr_score,
            float(signal.get("score", 0.0) or 0.0),
            mtf_fast_bias,
            float(macd_diff_val or 0.0),
            bool(psar_bull),
            current_price,
            ema_9_val,
            float(strategy_config.get("wall_breakout_score_gate", 0.15) or 0.15),
        ):
            signal["action"]      = "HOLD"
            signal["hold_reason"] = f"No-SR fallback SR block: SELL near support (sr={sr_score:.1f})"
            signal["sr_wall_locked"] = True

    signal, _rejection_applied = _apply_rejection_confirmation_gate(
        signal,
        df_indicators,
        strategy_config,
        support=support,
        resistance=resistance,
    )

    # --- EXHAUSTION & DIVERGENCE HARD GATE ---
    # This is the final guard against entering at the very top (BUY) or very bottom (SELL).
    # All lagging indicators (EMA cross, PSAR flip, MACD cross) turn bullish/bearish exactly
    # AT the peak/trough because that's when the move completes. These checks detect that the
    # move has already peaked BEFORE committing to entry.
    if signal.get("action") in {"BUY", "SELL"}:
        _final_action = signal["action"]
        _exhaust_block = False
        _div_block = False
        _rsi_block = False

        # 1. Momentum exhaustion: ROC decelerating for 3 bars = move running out of steam
        if _final_action == "BUY" and momentum_exhaustion == "BULL_EXHAUST":
            _exhaust_block = True
        elif _final_action == "SELL" and momentum_exhaustion == "BEAR_EXHAUST":
            _exhaust_block = True

        # 1b. MACD histogram dying: histogram in the right direction but declining 2 bars straight
        # (buying positive-but-shrinking histogram = buying into an already fading move)
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

        # 2. Classic divergence: price making new high/low but MACD is not = top/bottom confirmation
        if _final_action == "BUY" and bear_div:
            _div_block = True
        elif _final_action == "SELL" and bull_div:
            _div_block = True

        # 3. RSI extremes at entry: overbought longs and oversold shorts are chasing the move
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

    # Final wall-rejection rescue:
    # If the system ended up HOLD because the long was rejected at resistance,
    # promote it into a short when the rejection is already confirmed by candle
    # structure and momentum. This keeps the bot from sitting idle at tops.
    # Wall reversal assist: only fire if the underlying score is actually bearish.
    # A wall veto on a bullish signal should NOT flip to SELL unless we have genuine
    # bearish evidence (negative total_score) plus the candle/MACD confirmation below.
    _reversal_assist_score_ok = total_score < -0.10  # score must have a bearish lean
    if (
        signal.get("action") == "HOLD"
        and bool(strategy_config.get("wall_reversal_assist", True))
        and support is not None
        and resistance is not None
        and _reversal_assist_score_ok
    ):
        res_touch = bool(wall_state.get("resistance_touching"))
        res_broken = bool(wall_state.get("resistance_broken"))
        if res_touch and not res_broken:
            reject_cfg = dict((strategy_config or {}).get("rejection_confirmation", {}) or {})
            min_bars_away = max(1, int(reject_cfg.get("min_bars_away", 2) or 2))
            pullback_pct = float(reject_cfg.get("pullback_pct", 0.002) or 0.002)
            require_psar_flip = bool(reject_cfg.get("require_psar_flip", True))
            macd_threshold = float(reject_cfg.get("macd_threshold", 0.0) or 0.0)
            last = df_indicators.iloc[-1]
            prev = df_indicators.iloc[-2]
            current_macd = float(last["macd"]) if "macd" in df_indicators.columns else 0.0
            current_macd_diff = float(last["macd_diff"]) if "macd_diff" in df_indicators.columns else 0.0
            prev_macd_diff = float(prev["macd_diff"]) if "macd_diff" in df_indicators.columns else 0.0
            current_psar = float(last["psar"]) if "psar" in df_indicators.columns else current_price
            recent = df_indicators.tail(min_bars_away)
            macd_confirmed = current_macd < macd_threshold and current_macd_diff < 0 and prev_macd_diff >= 0
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

    return signal
def _detect_macd_divergence(df: pd.DataFrame) -> str:
    """
    MASTER RECONSTRUCTION: MACD Divergence Engine.
    Detects when Price and Momentum are moving in opposite directions.
    As per screenshots:
    - Bearish Divergence: Higher Highs in Price + Lower Highs in MACD
    - Bullish Divergence: Lower Lows in Price + Higher Lows in MACD
    """
    if len(df) < 50:
        return "NONE"

    # Extract last 40 candles for swing detection
    data = df.iloc[-40:].copy()

    # 1. Detect peaks (Highs) for Bearish Divergence
    # We look for two distinct local highs
    highs = []
    macd_peaks = []

    for i in range(2, len(data) - 2):
        if data['high'].iloc[i] > data['high'].iloc[i-1] and data['high'].iloc[i] > data['high'].iloc[i+1]:
            highs.append((i, data['high'].iloc[i]))
            macd_peaks.append(data['macd'].iloc[i])

    if len(highs) >= 2:
        # Check the last two peaks
        p1_idx, p1_price = highs[-2]
        p2_idx, p2_price = highs[-1]
        m1 = macd_peaks[-2]
        m2 = macd_peaks[-1]
        # Bearish Divergence: Price HH + MACD LH
        if p2_price > p1_price and m2 < m1 and (p2_idx - p1_idx) > 3:
            return "BEARISH"
        # Hidden Bearish: Price LH + MACD HH (trend continuation short)
        if p2_price < p1_price and m2 > m1 and (p2_idx - p1_idx) > 3:
            return "HIDDEN_BEARISH"

    # 2. Detect troughs (Lows) for Bullish Divergence
    lows = []
    macd_troughs = []

    for i in range(2, len(data) - 2):
        if data['low'].iloc[i] < data['low'].iloc[i-1] and data['low'].iloc[i] < data['low'].iloc[i+1]:
            lows.append((i, data['low'].iloc[i]))
            macd_troughs.append(data['macd'].iloc[i])

    if len(lows) >= 2:
        p1_idx, p1_price = lows[-2]
        p2_idx, p2_price = lows[-1]
        m1 = macd_troughs[-2]
        m2 = macd_troughs[-1]
        # Bullish Divergence: Price LL + MACD HL
        if p2_price < p1_price and m2 > m1 and (p2_idx - p1_idx) > 3:
            return "BULLISH"
        # Hidden Bullish: Price HL + MACD LL (trend continuation long)
        if p2_price > p1_price and m2 < m1 and (p2_idx - p1_idx) > 3:
            return "HIDDEN_BULLISH"

    return "NONE"

def _detect_cvd_momentum_divergence(df: pd.DataFrame) -> tuple[str, float]:
    """Detect simple price/CVD divergence using OHLCV approximation."""
    return _detect_cvd_divergence(df)


def _session_blackout_state(strategy_config: dict) -> tuple[bool, int, set[int]]:
    """Return whether the current UTC hour is inside a configured low-quality session."""
    cfg = strategy_config or {}
    enabled = bool(cfg.get("session_filter_enabled", False))
    if not enabled:
        return False, int(time.gmtime().tm_hour), set()

    raw_hours = cfg.get("session_block_hours_utc", []) or []
    blocked_hours: set[int] = set()
    if isinstance(raw_hours, str):
        raw_hours = [part.strip() for part in raw_hours.split(",") if part.strip()]
    for item in raw_hours:
        try:
            blocked_hours.add(int(item) % 24)
        except (TypeError, ValueError):
            continue

    current_hour = int(time.gmtime().tm_hour)
    return current_hour in blocked_hours, current_hour, blocked_hours
def _detect_mean_reversion_setup(df: pd.DataFrame, config: dict) -> dict:
    """
    Detects extreme statistical deviations for an explicit Mean Reversion setup.
    Triggers when Z-score and RSI indicate deep oversold/overbought conditions.
    """
    if df is None or len(df) < 20:
        return {"triggered": False}

    z_score = df['z_score'].iloc[-1]
    rsi = df['rsi_14'].iloc[-1]

    mr_trigger = False
    direction = None
    reason = ""
    score_boost = 0.0

    if z_score < -2.5 and rsi < 25:
        mr_trigger = True
        direction = "LONG"
        reason = f"DeepOversold(Z:{z_score:.1f},RSI:{rsi:.0f})"
        score_boost = 0.5
    elif z_score > 2.5 and rsi > 75:
        mr_trigger = True
        direction = "SHORT"
        reason = f"DeepOverbought(Z:{z_score:.1f},RSI:{rsi:.0f})"
        score_boost = -0.5

    if mr_trigger:
        return {
            "triggered": True,
            "direction": direction,
            "reason": reason,
            "score": score_boost
        }

    return {"triggered": False}


def _detect_wick_sweep_setup(
    df: pd.DataFrame,
    current_price: float,
    support: float = None,
    resistance: float = None,
    config: dict = None,
) -> dict:
    """
    Detect a fast wick sweep + reclaim setup for scalping.
    Bullish: sweep below support and close back above it.
    Bearish: sweep above resistance and close back below it.
    """
    if df is None or len(df) < 3:
        return {"direction": "NEUTRAL", "score": 0.0, "reason": "", "sl": None, "triggered": False}

    cfg = config or {}
    enabled = bool(cfg.get("wick_sweep_enabled", True))
    if not enabled:
        return {"direction": "NEUTRAL", "score": 0.0, "reason": "", "sl": None, "triggered": False}

    last = df.iloc[-1]
    prev = df.iloc[-2]
    body = abs(float(last["close"]) - float(last["open"]))
    body = max(body, max(float(last["close"]) * 0.00005, 1e-9))
    upper_wick = float(last["high"]) - max(float(last["open"]), float(last["close"]))
    lower_wick = min(float(last["open"]), float(last["close"])) - float(last["low"])
    wick_ratio = float(cfg.get("wick_sweep_wick_ratio", 1.6) or 1.6)
    sweep_buffer = float(cfg.get("wick_sweep_buffer_pct", 0.0008) or 0.0008)
    reclaim_buffer = float(cfg.get("wick_sweep_reclaim_pct", 0.0002) or 0.0002)
    min_body_dir = float(cfg.get("wick_sweep_body_dir_pct", 0.0) or 0.0)

    bullish_sweep = False
    bearish_sweep = False
    bullish_reason = ""
    bearish_reason = ""

    if support is not None and float(support) > 0:
        support = float(support)
        swept_below = float(last["low"]) < support * (1 - sweep_buffer)
        reclaimed = float(last["close"]) > support * (1 + reclaim_buffer)
        bullish_candle = float(last["close"]) >= float(last["open"])
        prev_bearish_or_flat = float(prev["close"]) <= float(prev["open"])
        strong_lower_wick = lower_wick >= body * wick_ratio
        if swept_below and reclaimed and bullish_candle and strong_lower_wick and prev_bearish_or_flat:
            bullish_sweep = True
            bullish_reason = "Wick sweep below support and reclaim"

    if resistance is not None and float(resistance) > 0:
        resistance = float(resistance)
        swept_above = float(last["high"]) > resistance * (1 + sweep_buffer)
        reclaimed = float(last["close"]) < resistance * (1 - reclaim_buffer)
        bearish_candle = float(last["close"]) <= float(last["open"])
        prev_bullish_or_flat = float(prev["close"]) >= float(prev["open"])
        strong_upper_wick = upper_wick >= body * wick_ratio
        if swept_above and reclaimed and bearish_candle and strong_upper_wick and prev_bullish_or_flat:
            bearish_sweep = True
            bearish_reason = "Wick sweep above resistance and reclaim"

    if bullish_sweep and not bearish_sweep:
        stop = float(last["low"]) * (1 - max(0.0005, sweep_buffer))
        return {
            "direction": "LONG",
            "score": 0.35,
            "reason": bullish_reason,
            "sl": stop,
            "triggered": True,
        }

    if bearish_sweep and not bullish_sweep:
        stop = float(last["high"]) * (1 + max(0.0005, sweep_buffer))
        return {
            "direction": "SHORT",
            "score": -0.35,
            "reason": bearish_reason,
            "sl": stop,
            "triggered": True,
        }

    return {"direction": "NEUTRAL", "score": 0.0, "reason": "", "sl": None, "triggered": False}

def _detect_vwap_bounce_setup(
    df: pd.DataFrame,
    current_price: float,
    latest_indicators: dict = None,
    strategy_config: dict = None,
    anchored_vwap: float = None,
    mtf_fast_bias: str = "NEUTRAL",
) -> dict:
    """
    Detect a VWAP bounce/reclaim setup aligned with the article:
    - Trend context
    - Pullback to VWAP
    - Rejection candle
    - Volume confirmation
    """
    if df is None or len(df) < 5:
        return {"direction": "NEUTRAL", "score": 0.0, "reason": "", "sl": None, "triggered": False, "mode": "VWAP_BOUNCE"}

    cfg = strategy_config or {}
    if not bool(cfg.get("vwap_bounce_enabled", True)):
        return {"direction": "NEUTRAL", "score": 0.0, "reason": "", "sl": None, "triggered": False, "mode": "VWAP_BOUNCE"}

    last = df.iloc[-1]
    prev = df.iloc[-2]
    latest_indicators = latest_indicators or {}

    vwap_ref = float(anchored_vwap or 0.0)
    if vwap_ref <= 0:
        try:
            vwap_ref = float(last["vwap"]) if "vwap" in df.columns else 0.0
        except (TypeError, ValueError):
            vwap_ref = 0.0
    if vwap_ref <= 0 or current_price <= 0:
        return {"direction": "NEUTRAL", "score": 0.0, "reason": "", "sl": None, "triggered": False, "mode": "VWAP_BOUNCE"}

    near_pct = float(cfg.get("vwap_bounce_near_pct", 0.0015) or 0.0015)
    reclaim_pct = float(cfg.get("vwap_bounce_reclaim_pct", 0.0002) or 0.0002)
    vol_mult = float(cfg.get("vwap_bounce_vol_mult", 1.15) or 1.15)
    slope_ok = True
    try:
        if "ema_9" in df.columns and "ema_21" in df.columns:
            slope_ok = bool(float(last["ema_9"]) > float(last["ema_21"]))
    except Exception:
        slope_ok = True

    current_vol = float(last["volume"]) if "volume" in df.columns else 0.0
    avg_vol = float(df["volume"].rolling(window=min(20, len(df))).mean().iloc[-1]) if "volume" in df.columns else 0.0
    volume_ok = avg_vol > 0 and current_vol >= avg_vol * vol_mult
    candle_bull = float(last["close"]) >= float(last["open"])
    candle_bear = float(last["close"]) <= float(last["open"])
    price_vs_vwap = (current_price - vwap_ref) / vwap_ref
    recent = df.tail(3)

    long_touch = bool((recent["low"] <= vwap_ref * (1 + near_pct)).any()) if "low" in recent.columns else False
    short_touch = bool((recent["high"] >= vwap_ref * (1 - near_pct)).any()) if "high" in recent.columns else False

    long_confirmed = (
        price_vs_vwap >= -near_pct
        and long_touch
        and current_price > vwap_ref * (1 + reclaim_pct)
        and candle_bull
        and volume_ok
        and slope_ok
        and mtf_fast_bias != "SHORT_ONLY"
    )
    short_confirmed = (
        price_vs_vwap <= near_pct
        and short_touch
        and current_price < vwap_ref * (1 - reclaim_pct)
        and candle_bear
        and volume_ok
        and (not slope_ok or mtf_fast_bias != "LONG_ONLY")
    )

    if long_confirmed and not short_confirmed:
        sl = float(vwap_ref) * (1 - max(near_pct, 0.001))
        return {
            "triggered": True,
            "mode": "VWAP_BOUNCE",
            "direction": "LONG",
            "score": 0.30,
            "reason": f"VWAP bounce above {vwap_ref:.5f}",
            "sl": sl,
        }

    if short_confirmed and not long_confirmed:
        sl = float(vwap_ref) * (1 + max(near_pct, 0.001))
        return {
            "triggered": True,
            "mode": "VWAP_BOUNCE",
            "direction": "SHORT",
            "score": -0.30,
            "reason": f"VWAP rejection below {vwap_ref:.5f}",
            "sl": sl,
        }

    return {"direction": "NEUTRAL", "score": 0.0, "reason": "", "sl": None, "triggered": False, "mode": "VWAP_BOUNCE"}

def _detect_orb_breakout_setup(
    df: pd.DataFrame,
    current_price: float,
    latest_indicators: dict = None,
    strategy_config: dict = None,
    mtf_fast_bias: str = "NEUTRAL",
) -> dict:
    """
    Detect a rolling opening-range breakout setup.
    Uses the first N minutes of the current UTC day as the opening range.
    """
    if df is None or len(df) < 10:
        return {"direction": "NEUTRAL", "score": 0.0, "reason": "", "sl": None, "triggered": False, "mode": "ORB_BREAKOUT"}

    cfg = strategy_config or {}
    if not bool(cfg.get("orb_enabled", True)):
        return {"direction": "NEUTRAL", "score": 0.0, "reason": "", "sl": None, "triggered": False, "mode": "ORB_BREAKOUT"}

    if "timestamp" not in df.columns:
        return {"direction": "NEUTRAL", "score": 0.0, "reason": "", "sl": None, "triggered": False, "mode": "ORB_BREAKOUT"}

    latest_indicators = latest_indicators or {}
    last = df.iloc[-1]
    try:
        last_ts = pd.to_datetime(last["timestamp"], utc=True, errors="coerce")
    except Exception:
        last_ts = pd.NaT
    if pd.isna(last_ts):
        return {"direction": "NEUTRAL", "score": 0.0, "reason": "", "sl": None, "triggered": False, "mode": "ORB_BREAKOUT"}

    orb_minutes = int(cfg.get("orb_minutes", 15) or 15)
    orb_minutes = max(5, min(60, orb_minutes))
    breakout_buffer_pct = float(cfg.get("orb_breakout_buffer_pct", 0.0005) or 0.0005)
    volume_mult = float(cfg.get("orb_volume_mult", 1.25) or 1.25)

    day_start = last_ts.floor("D")
    day_mask = pd.to_datetime(df["timestamp"], utc=True, errors="coerce") >= day_start
    day_df = df.loc[day_mask].copy()
    if len(day_df) < 3:
        return {"direction": "NEUTRAL", "score": 0.0, "reason": "", "sl": None, "triggered": False, "mode": "ORB_BREAKOUT"}

    orb_end = day_start + pd.Timedelta(minutes=orb_minutes)
    orb_df = day_df[pd.to_datetime(day_df["timestamp"], utc=True, errors="coerce") < orb_end]
    if len(orb_df) < 2:
        return {"direction": "NEUTRAL", "score": 0.0, "reason": "", "sl": None, "triggered": False, "mode": "ORB_BREAKOUT"}

    orb_high = float(orb_df["high"].max())
    orb_low = float(orb_df["low"].min())
    if orb_high <= 0 or orb_low <= 0 or current_price <= 0:
        return {"direction": "NEUTRAL", "score": 0.0, "reason": "", "sl": None, "triggered": False, "mode": "ORB_BREAKOUT"}

    current_vol = float(last["volume"]) if "volume" in df.columns else 0.0
    avg_vol = float(day_df["volume"].rolling(window=min(20, len(day_df))).mean().iloc[-1]) if "volume" in df.columns else 0.0
    volume_ok = avg_vol > 0 and current_vol >= avg_vol * volume_mult
    candle_bull = float(last["close"]) >= float(last["open"])
    candle_bear = float(last["close"]) <= float(last["open"])
    macd_up = float(latest_indicators.get("macd_diff", 0.0) or 0.0) > 0
    macd_down = float(latest_indicators.get("macd_diff", 0.0) or 0.0) < 0

    breakout_long = (
        current_price >= orb_high * (1 + breakout_buffer_pct)
        and volume_ok
        and candle_bull
        and (mtf_fast_bias != "SHORT_ONLY")
        and (macd_up or float(last.get("ema_9", current_price)) > float(last.get("ema_21", current_price)))
    )
    breakout_short = (
        current_price <= orb_low * (1 - breakout_buffer_pct)
        and volume_ok
        and candle_bear
        and (mtf_fast_bias != "LONG_ONLY")
        and (macd_down or float(last.get("ema_9", current_price)) < float(last.get("ema_21", current_price)))
    )

    if breakout_long and not breakout_short:
        sl = orb_high * (1 - max(breakout_buffer_pct, 0.001))
        return {
            "triggered": True,
            "mode": "ORB_BREAKOUT",
            "direction": "LONG",
            "score": 0.40,
            "reason": f"ORB breakout above {orb_high:.5f}",
            "sl": sl,
        }

    if breakout_short and not breakout_long:
        sl = orb_low * (1 + max(breakout_buffer_pct, 0.001))
        return {
            "triggered": True,
            "mode": "ORB_BREAKOUT",
            "direction": "SHORT",
            "score": -0.40,
            "reason": f"ORB breakdown below {orb_low:.5f}",
            "sl": sl,
        }

    return {"direction": "NEUTRAL", "score": 0.0, "reason": "", "sl": None, "triggered": False, "mode": "ORB_BREAKOUT"}
def _calculate_volume_delta(df: pd.DataFrame) -> float:
    """
    Estimates the 'Aggression' within a candle.
    If price rises on high volume, it implies aggressive buying.
    If price falls on high volume, it implies aggressive selling.
    """
    latest = df.iloc[-1]
    avg_vol = df['volume'].rolling(window=20).mean().iloc[-1]

    vol_rel = latest['volume'] / avg_vol
    price_change = (latest['close'] - latest['open']) / latest['open']

    # Delta score
    delta = price_change * vol_rel * 10.0
    return max(min(delta, 1.0), -1.0)

def _map_order_book_pressure(state: dict) -> float:
    """
    Uses real-time Order Book data to find 'The Wall of Money'.
    Bids > Asks = Buying Pressure.
    Asks > Bids = Selling Pressure.
    """
    if not state or 'orderbook' not in state: return 0.0

    ob = state['orderbook']
    bids = sum([v for k, v in ob.get('bids', [])[:10]])
    asks = sum([v for k, v in ob.get('asks', [])[:10]])

    if (bids + asks) == 0: return 0.0
    pressure = (bids - asks) / (bids + asks)
    return pressure
