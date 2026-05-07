"""Indicator scoring: computes SMC, MR, VWAP, BB, MACD, PA, ADX, OBV, KDJ, SuperTrend scores and synthesizes the raw total_score."""

from __future__ import annotations

from .ctx import SignalContext
import numpy as np
import pandas as pd
import logging

from ..calc import get_signal_weights, TUNING
from ..smc import detect_smc_and_sr
from ..location import (
    compute_volatility_context, identify_liquidity_pools, calculate_funding_impact,
    _detect_momentum_exhaustion, _body_range_ratio_score,
)
from .alpha import generate_alpha_overlay
from .divergence import _detect_macd_divergence, _detect_cvd_momentum_divergence
from .utils import _calculate_volume_delta, _map_order_book_pressure

logger = logging.getLogger(__name__)


def compute_indicator_scores(ctx: SignalContext) -> None:
    """
    Compute all indicator scores and store them in ctx.

    Expects ctx to contain:
        df_indicators, latest_indicators, current_price, strategy_config,
        state, mtf_context, mtf_fast_bias, mtf_fast_score, mtf_rsi_score,
        atr_pct_now, ema_21, ema_9, macro_bias

    Adds to ctx:
        smc_score, sr_score, smc_label, ob_context,
        mr_score, vwap_score, bb_score, macd_score,
        macd_diff, prev_macd_diff, prev2_macd_diff, macd_val,
        divergence_state, bear_div, bull_div, hidden_bull_div, hidden_bear_div,
        cvd_state, cvd_bonus, momentum_exhaustion, body_ratio_score,
        power_bonus, div_bonus, htf_score, pa_score, psar_val,
        adx_value, adx_score, volume_delta, obv_score, ob_score,
        kdj_score, st_score, vol_spike, z_score, rsi_14, alpha,
        score_without_sr, sr_directional, total_score,
        ema_dist_pct, max_chase_pct, trend_continuation,
        htf_1h, htf_4h,
    """
    df_indicators = ctx['df_indicators']
    latest_indicators = ctx['latest_indicators']
    current_price = ctx['current_price']
    strategy_config = ctx['strategy_config']
    state = ctx['state']
    mtf_context = ctx.get('mtf_context') or {}
    mtf_fast_bias = ctx['mtf_fast_bias']
    mtf_rsi_score = ctx['mtf_rsi_score']
    atr_pct_now = ctx['atr_pct_now']
    macro_bias = ctx['macro_bias']
    weights = ctx['weights']
    vol_context = ctx['vol_context']

    last = df_indicators.iloc[-1] if len(df_indicators) > 0 else None

    # --- Structural & SMC Bias ---
    smc_score, sr_score, smc_label, ob_context = detect_smc_and_sr(df_indicators, current_price)
    ctx['smc_score'] = smc_score
    ctx['sr_score'] = sr_score
    ctx['smc_label'] = smc_label
    ctx['ob_context'] = ob_context

    # --- Mean Reversion (MR) ---
    ema_9 = ctx.get('ema_9', latest_indicators.get('ema_9', current_price))
    ema_21 = ctx.get('ema_21', latest_indicators.get('ema_21', current_price))
    z_score = float(latest_indicators.get('z_score', 0.0) or 0.0)
    rsi_14 = float(latest_indicators.get('rsi_14', 50.0) or 50.0)
    ctx['z_score'] = z_score
    ctx['rsi_14'] = rsi_14

    mr_score = 0.0
    if current_price < ema_21 * 0.998:
        if z_score <= -2.0 and rsi_14 < 35:
            mr_score = 1.0
        elif z_score <= -1.5 or (z_score <= -1.0 and rsi_14 < 38):
            mr_score = 0.5
    elif current_price > ema_21 * 1.002:
        if z_score >= 2.0 and rsi_14 > 65:
            mr_score = -1.0
        elif z_score >= 1.5 or (z_score >= 1.0 and rsi_14 > 62):
            mr_score = -0.5
    ctx['mr_score'] = mr_score

    # --- VWAP ---
    vwap = latest_indicators.get('vwap', current_price)
    if mtf_fast_bias == "LONG_ONLY":
        vwap_score = 1.0 if current_price >= vwap else -1.0
    elif mtf_fast_bias == "SHORT_ONLY":
        vwap_score = -1.0 if current_price >= vwap else 1.0
    else:
        vwap_score = 1.0 if current_price >= vwap else -1.0
    ctx['vwap_score'] = vwap_score

    # --- Bollinger Bands (BB) ---
    bb_low = latest_indicators.get('bb_low', current_price)
    bb_high = latest_indicators.get('bb_high', current_price)
    bb_score = 1.0 if current_price < bb_low * 1.001 else (-1.0 if current_price > bb_high * 0.999 else 0.0)
    ctx['bb_score'] = bb_score

    # --- MACD Advanced Patterns ---
    macd_diff = latest_indicators.get('macd_diff', 0)
    prev_macd_diff = df_indicators['macd_diff'].iloc[-2] if len(df_indicators) > 2 else 0
    prev2_macd_diff = df_indicators['macd_diff'].iloc[-3] if len(df_indicators) > 3 else 0
    macd_val = latest_indicators.get('macd', 0)
    ctx['macd_diff'] = macd_diff
    ctx['prev_macd_diff'] = prev_macd_diff
    ctx['prev2_macd_diff'] = prev2_macd_diff
    ctx['macd_val'] = macd_val

    macd_score = 0.0
    if macd_diff > 0 and prev_macd_diff <= 0:
        macd_score += 0.8
    elif macd_diff < 0 and prev_macd_diff >= 0:
        macd_score -= 0.8

    if macd_diff < 0 and macd_diff > prev_macd_diff:
        macd_score += 0.3
    elif macd_diff > 0 and macd_diff < prev_macd_diff:
        macd_score -= 0.3

    if macd_diff > 0 and prev_macd_diff > 0 and macd_diff > prev_macd_diff and prev_macd_diff < prev2_macd_diff:
        macd_score += 1.0
    elif macd_diff < 0 and prev_macd_diff < 0 and macd_diff < prev_macd_diff and prev_macd_diff > prev2_macd_diff:
        macd_score -= 1.0

    if macd_score == 0:
        macd_score = np.sign(macd_diff) if abs(macd_diff) > 0.0001 else 0.0
    ctx['macd_score'] = macd_score

    # --- Divergence / Exhaustion Detection ---
    divergence_state = _detect_macd_divergence(df_indicators)
    ctx['divergence_state'] = divergence_state
    ctx['bear_div'] = divergence_state == "BEARISH"
    ctx['bull_div'] = divergence_state == "BULLISH"
    ctx['hidden_bull_div'] = divergence_state == "HIDDEN_BULLISH"
    ctx['hidden_bear_div'] = divergence_state == "HIDDEN_BEARISH"

    cvd_state, cvd_bonus = _detect_cvd_momentum_divergence(df_indicators)
    momentum_exhaustion = _detect_momentum_exhaustion(df_indicators)
    body_ratio_score = _body_range_ratio_score(df_indicators, lookback=3)
    ctx['cvd_state'] = cvd_state
    ctx['cvd_bonus'] = cvd_bonus
    ctx['momentum_exhaustion'] = momentum_exhaustion
    ctx['body_ratio_score'] = body_ratio_score

    # --- Institutional Power Suite ---
    power_bonus = 0.0
    avg_vol = df_indicators['volume'].rolling(window=20).mean().iloc[-1]
    curr_vol = df_indicators['volume'].iloc[-1]
    vol_spike = curr_vol > (avg_vol * 1.5)
    if vol_spike:
        power_bonus += 0.15

    htf_15m = mtf_context.get('15m', {}) if mtf_context else {}
    prev_ema_15m = float(htf_15m.get('ema_21_prev', 0) or 0)
    curr_ema_15m = float(htf_15m.get('ema_21', 0) or 0)
    slope_15m_up = curr_ema_15m > prev_ema_15m if prev_ema_15m > 0 else False
    if slope_15m_up:
        power_bonus += 0.10

    is_squeezed = vol_context.get("squeeze", False)
    if is_squeezed:
        power_bonus -= 0.15
    else:
        bb_width_now = df_indicators['bb_width'].iloc[-1]
        bb_width_prev = df_indicators['bb_width'].iloc[-2]
        if bb_width_now > bb_width_prev * 1.05:
            power_bonus += 0.20

    power_bonus += body_ratio_score * 0.18
    power_bonus += mtf_rsi_score * 0.12
    ctx['power_bonus'] = power_bonus
    ctx['vol_spike'] = vol_spike

    # --- Divergence Bonus ---
    div_bonus = 0.0
    if ctx['bull_div']:
        div_bonus += 0.25
    if ctx['bear_div']:
        div_bonus -= 0.25
    if ctx['hidden_bull_div']:
        div_bonus += 0.18
    if ctx['hidden_bear_div']:
        div_bonus -= 0.18
    if cvd_bonus != 0.0:
        div_bonus += cvd_bonus
    if momentum_exhaustion == "BULL_EXHAUST":
        div_bonus -= 0.12
    elif momentum_exhaustion == "BEAR_EXHAUST":
        div_bonus += 0.12
    ctx['div_bonus'] = div_bonus

    # --- High Timeframe Strict Bias ---
    htf_1h = mtf_context.get('1h', {}) if mtf_context else {}
    htf_4h = mtf_context.get('4h', {}) if mtf_context else {}

    macd_1h = float(htf_1h.get('macd', 0) or 0)
    macd_4h = float(htf_4h.get('macd', 0) or 0)
    ctx['htf_1h'] = htf_1h
    ctx['htf_4h'] = htf_4h

    htf_score = 0.0
    htf_score += 0.10 if macd_1h > 0 else (-0.10 if macd_1h < 0 else 0.0)
    htf_score += 0.10 if macd_4h > 0 else (-0.10 if macd_4h < 0 else 0.0)
    ctx['htf_score'] = htf_score

    # --- Price Action (PA) + SAR Alignment ---
    psar_val = latest_indicators.get('psar', current_price)
    ctx['psar_val'] = psar_val
    pa_bullish = df_indicators['close'].iloc[-1] > df_indicators['open'].iloc[-1] and current_price > psar_val
    pa_bearish = df_indicators['close'].iloc[-1] < df_indicators['open'].iloc[-1] and current_price < psar_val
    pa_score = 1.0 if pa_bullish else (-1.0 if pa_bearish else 0.0)
    ctx['pa_score'] = pa_score

    # --- ADX / Volume Flow ---
    adx_value = float(latest_indicators.get('adx', 0.0) or 0.0)
    trend_dir = 1.0 if current_price >= ema_21 else -1.0
    if adx_value >= 25:
        adx_score = trend_dir
    elif adx_value >= 18:
        adx_score = trend_dir * 0.5
    else:
        adx_score = trend_dir * 0.15
    ctx['adx_value'] = adx_value
    ctx['adx_score'] = adx_score

    volume_delta = float(np.clip(_calculate_volume_delta(df_indicators), -1.0, 1.0))
    obv = float(latest_indicators.get('obv', 0.0) or 0.0)
    obv_ema = float(latest_indicators.get('obv_ema', obv) or obv)
    obv_score = 1.0 if obv > obv_ema else -1.0
    ob_score = _map_order_book_pressure(state)
    ctx['volume_delta'] = volume_delta
    ctx['obv_score'] = obv_score
    ctx['ob_score'] = ob_score

    # --- KDJ & SuperTrend ---
    kdj_j = latest_indicators.get('j', 50)
    kdj_score = 1.0 if kdj_j < 20 else (-1.0 if kdj_j > 80 else 0.0)
    st_score = latest_indicators.get('trend_bias', 0)
    ctx['kdj_score'] = kdj_score
    ctx['st_score'] = st_score

    # --- Alpha Overlay & Score Synthesis ---
    alpha = generate_alpha_overlay(df_indicators, smc_score, macro_bias)
    ctx['alpha'] = alpha

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
        (0 * weights['loc']) +  # location_score applied separately
        (kdj_score * weights['kdj']) +
        (st_score * weights['st']) +
        alpha +
        htf_score +
        div_bonus +
        power_bonus
    )

    # Add location_score separately (it's computed in core_context phase)
    location_score = ctx.get('location_score', 0.0)
    score_without_sr += location_score * weights['loc']

    sr_directional = sr_score if score_without_sr >= 0 else -sr_score
    total_score = score_without_sr + (sr_directional * weights['sr'])
    ctx['score_without_sr'] = score_without_sr
    ctx['sr_directional'] = sr_directional
    ctx['total_score_raw'] = total_score