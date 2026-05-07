"""Alpha overlay engine, signal integrity validation, and institutional pivot points."""

import pandas as pd
from ..calc import TUNING


def generate_alpha_overlay(df: pd.DataFrame, smc_score: float, macro_bias: str) -> float:
    """
    Alpha Overlay Engine.
    Synthesizes structural bias with momentum to find 'The Edge'.
    Uses EMA 50/200 golden/death cross for swing-speed trend confirmation.
    """
    if len(df) < 200:
        return 0.0

    ema_50 = df['ema_50'].iloc[-1]
    ema_200 = df['ema_200'].iloc[-1]
    mom_bias = 1.0 if ema_50 > ema_200 else -1.0

    avg_vol = df['volume'].rolling(window=20).mean().iloc[-1]
    current_vol = df['volume'].iloc[-1]
    vol_spike_mult = float(TUNING.get("vol_spike_mult", 1.5) or 1.5)
    vol_spike = current_vol > (avg_vol * vol_spike_mult)

    alpha = 0.0
    if mom_bias > 0 and smc_score > 0 and macro_bias == "BULLISH":
        alpha = 0.5 if vol_spike else 0.25
    elif mom_bias < 0 and smc_score < 0 and macro_bias == "BEARISH":
        alpha = -0.5 if vol_spike else -0.25

    return alpha


def validate_signal_integrity(signal: dict, vol_context: dict) -> dict:
    """
    Signal Integrity Validation.
    Final filter to ensure we aren't trading in 'Dangerous' conditions.
    """
    if vol_context.get('squeeze'):
        signal['squeeze_warning'] = "Volatility Squeeze: Potential Breakout Setup"

    if vol_context.get('atr_rank', 1.0) < 0.05:
        signal['action'] = "HOLD"
        signal['hold_reason'] = "Low Volatility: Insufficient Profit Potential"

    return signal


def compute_advanced_pivots(df: pd.DataFrame) -> dict:
    """
    Calculates institutional pivot points.
    Includes Classic, Woodie, and Camarilla levels.
    """
    if df is None or len(df) < 2:
        return {}

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