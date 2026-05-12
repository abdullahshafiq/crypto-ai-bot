"""
SCALPER STRATEGY (V2.0)
S/R-aware bounce and rejection detector.
Fires at structural levels BEFORE trend re-establishes, not after.

Sources: Volman buildup (squeeze), Brooks H2/L2 (second leg), Cameron momentum.
"""

import pandas as pd
import numpy as np


def detect_best_scalper_signal(
    df: pd.DataFrame,
    strategy_config: dict = None,
    support: float = None,
    resistance: float = None,
) -> dict:
    """
    Detects high-probability scalp setups at structural S/R levels.

    LONG: price near support + consolidation squeeze + H2 (second attempt up) + RSI rising
    SHORT: price near resistance + consolidation squeeze + L2 (second attempt down) + RSI falling

    Does NOT require trend pre-confirmation — fires at the inflection point.
    """
    if df is None or len(df) < 30:
        return {"triggered": False, "direction": "NEUTRAL", "reason": "Warming up"}

    cfg = strategy_config or {}
    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]

    current_price = float(last['close'])
    avg_atr = float(df['atr'].tail(20).mean()) if 'atr' in df.columns else current_price * 0.003

    # --- 1. VOLMAN BUILDUP: tight consolidation near a level ---
    recent_range = float(df['high'].tail(5).max() - df['low'].tail(5).min())
    is_buildup = recent_range < avg_atr * 1.5

    # --- 2. S/R PROXIMITY: determine which zone price is in ---
    sr_zone_pct = float(cfg.get("scalper_sr_zone_pct", 0.0030) or 0.0030)
    near_support = False
    near_resistance = False

    if support and float(support) > 0:
        near_support = abs(current_price - float(support)) / float(support) <= sr_zone_pct

    if resistance and float(resistance) > 0:
        near_resistance = abs(current_price - float(resistance)) / float(resistance) <= sr_zone_pct

    # --- 3. H2/L2 DETECTION: second leg after pullback to level ---
    h2_triggered = False
    l2_triggered = False

    if near_support:
        # H2: price touched near support in last 5 bars, now breaking prior bar high
        touched_support = bool((df['low'].tail(5) <= (float(support) * 1.002)).any()) if support else False
        breaking_up = float(last['close']) > float(prev['high'])
        bullish_candle = float(last['close']) > float(last['open'])
        if touched_support and breaking_up and bullish_candle and is_buildup:
            h2_triggered = True

    if near_resistance:
        # L2: price touched near resistance in last 5 bars, now breaking prior bar low
        touched_resistance = bool((df['high'].tail(5) >= (float(resistance) * 0.998)).any()) if resistance else False
        breaking_down = float(last['close']) < float(prev['low'])
        bearish_candle = float(last['close']) < float(last['open'])
        if touched_resistance and breaking_down and bearish_candle and is_buildup:
            l2_triggered = True

    # --- 4. MOMENTUM: volume + RSI direction ---
    rsi = float(last['rsi_14']) if 'rsi_14' in df.columns else 50.0
    rsi_prev = float(prev['rsi_14']) if 'rsi_14' in df.columns else 50.0

    vol_spike = float(last['volume']) > float(df['volume'].tail(20).mean()) * 1.2
    rsi_rising = rsi > rsi_prev
    rsi_falling = rsi < rsi_prev
    rsi_not_overbought = rsi < 70
    rsi_not_oversold = rsi > 30

    # --- 5. MACD histogram direction (momentum confirmation) ---
    macd_hist = float(last['macd_diff']) if 'macd_diff' in df.columns else 0.0
    macd_hist_prev = float(prev['macd_diff']) if 'macd_diff' in df.columns else 0.0
    macd_turning_up = macd_hist > macd_hist_prev
    macd_turning_down = macd_hist < macd_hist_prev

    # --- SYNTHESIS ---
    triggered = False
    direction = "NEUTRAL"
    score = 0.0
    reason_parts = []

    if h2_triggered and rsi_rising and rsi_not_overbought:
        triggered = True
        direction = "LONG"
        score = 0.40
        reason_parts = ["H2@Support"]
        if is_buildup:
            score += 0.05
            reason_parts.append("Buildup")
        if vol_spike:
            score += 0.08
            reason_parts.append("VolSpike")
        if macd_turning_up:
            score += 0.05
            reason_parts.append("MACD↑")

    elif l2_triggered and rsi_falling and rsi_not_oversold:
        triggered = True
        direction = "SHORT"
        score = -0.40
        reason_parts = ["L2@Resistance"]
        if is_buildup:
            score -= 0.05
            reason_parts.append("Buildup")
        if vol_spike:
            score -= 0.08
            reason_parts.append("VolSpike")
        if macd_turning_down:
            score -= 0.05
            reason_parts.append("MACD↓")

    return {
        "triggered": triggered,
        "direction": direction,
        "score": score,
        "reason": "+".join(reason_parts) if reason_parts else "",
        "sl_offset_atr": 1.0,
        "tp_offset_atr": 1.5,
    }
