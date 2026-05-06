import pandas as pd
import numpy as np

from .calc import TUNING

def detect_smc_and_sr(df: pd.DataFrame, current_price: float) -> tuple:
    """
    MASTER RECONSTRUCTION: The Institutional SMC "Swing Hunter" Engine.
    Exhaustive search for Market Structure (BOS/CHoCH), Order Blocks,
    Mitigation Zones, Fair Value Gaps (FVG), and Institutional S/R Walls.
    """
    if len(df) < 50:
        return 0.0, 0.0, "Warming Engine", {"active": False, "direction": "NEUTRAL", "mid": None, "high": None, "low": None}

    df = df.copy()

    # --- 1. THE SWING HUNTER (LuxAlgo High-Resolution) ---
    # We use an 11-candle rolling window to find significant swing highs and lows.
    # This is optimized for scalping, yielding only a 5-candle lag (instead of 25).
    window = 11
    df['swing_high'] = df['high'].rolling(window=window, center=True).max()
    df['swing_low'] = df['low'].rolling(window=window, center=True).min()

    # Extract structural peaks and valleys
    swings = df[df['high'] == df['swing_high']].copy()
    valleys = df[df['low'] == df['swing_low']].copy()

    if len(swings) < 3 or len(valleys) < 3:
        return 0.0, 0.0, "Building Structure", {"active": False, "direction": "NEUTRAL", "mid": None, "high": None, "low": None}

    # Precise structural price points
    last_high = swings['high'].iloc[-1]
    last_low = valleys['low'].iloc[-1]
    prev_high = swings['high'].iloc[-2]
    prev_low = valleys['low'].iloc[-2]
    old_high = swings['high'].iloc[-3]
    old_low = valleys['low'].iloc[-3]

    smc_score = 0.0
    smc_label = "Neutral"
    ob_context = {"active": False, "direction": "NEUTRAL", "mid": None, "high": None, "low": None, "reason": ""}

    # --- 2. STRUCTURAL BREAKS (BOS / CHoCH) ---
    # Break of Structure (BOS): Trend continuation signals.
    # Change of Character (CHoCH): Reversal signals.

    # Bullish Structures
    if current_price > last_high:
        if last_high > prev_high:
            smc_score = 1.0
            smc_label = "BOS Bullish (Trend)"
        else:
            smc_score = 1.3
            smc_label = "CHoCH Bullish (Flip)"

    # Bearish Structures
    elif current_price < last_low:
        if last_low < prev_low:
            smc_score = -1.0
            smc_label = "BOS Bearish (Trend)"
        else:
            smc_score = -1.3
            smc_label = "CHoCH Bearish (Flip)"

    # Inducement / fake-break filter:
    # If the CHoCH candle is mostly wick and not body, downgrade confidence.
    if "CHoCH" in smc_label and len(df) >= 2:
        break_candle = df.iloc[-1]
        body = abs(float(break_candle["close"]) - float(break_candle["open"]))
        body = max(body, max(float(break_candle["close"]) * 0.00005, 1e-9))
        upper_wick = float(break_candle["high"]) - max(float(break_candle["open"]), float(break_candle["close"]))
        lower_wick = min(float(break_candle["open"]), float(break_candle["close"])) - float(break_candle["low"])
        wick = max(upper_wick, lower_wick)
        if wick > body * 2.0:
            smc_score *= 0.4
            smc_label += " (Inducement?)"

    # --- 3. FAIR VALUE GAPS (FVG) / LIQUIDITY VOIDS ---
    # Detecting gaps where price moved too fast, creating an imbalance.
    # Price often 'revisits' these gaps to fill liquidity.
    fvg_detected = False
    fvg_sensitivity = float(TUNING.get("fvg_sensitivity", 0.0001) or 0.0001)
    for i in range(-5, -1):
        # Bullish FVG (Gap between Candle 1 High and Candle 3 Low)
        if df['high'].iloc[i-1] < df['low'].iloc[i+1]:
            gap_size = df['low'].iloc[i+1] - df['high'].iloc[i-1]
            if gap_size >= (current_price * fvg_sensitivity) and current_price > df['high'].iloc[i-1] and current_price < df['low'].iloc[i+1]:
                smc_score += 0.2
                smc_label = "FVG Bullish Entry"
                fvg_detected = True
        # Bearish FVG (Gap between Candle 1 Low and Candle 3 High)
        elif df['low'].iloc[i-1] > df['high'].iloc[i+1]:
            gap_size = df['low'].iloc[i-1] - df['high'].iloc[i+1]
            if gap_size >= (current_price * fvg_sensitivity) and current_price < df['low'].iloc[i-1] and current_price > df['high'].iloc[i+1]:
                smc_score -= 0.2
                smc_label = "FVG Bearish Entry"
                fvg_detected = True

    # --- 4. ORDER BLOCKS & MITIGATION ZONES ---
    # Identifying the last 'Opposite Color' candle before a major move.
    # Bullish Order Block (Last Red candle before a massive pump)
    # Bearish Order Block (Last Green candle before a massive dump)

    # We look back at the last 5 swings to see if current price is 'mitigating' an old block.
    # FIX BUG1: Use candle body (open/close) not wick extremes (high/low) for OB bounds.
    # A Bearish OB is the last bullish (green) candle before a dump → found at swing highs.
    # A Bullish OB is the last bearish (red) candle before a pump → found at swing lows.
    # Using wick extremes made zones 3-5x too wide, causing constant false OB labels.
    for i in range(len(swings)-1, max(0, len(swings)-5), -1):
        ob_body_high = max(float(swings['open'].iloc[i]), float(swings['close'].iloc[i]))
        ob_body_low  = min(float(swings['open'].iloc[i]), float(swings['close'].iloc[i]))
        # Require a meaningful body (not a doji)
        if ob_body_high - ob_body_low < current_price * 0.0002:
            continue
        if ob_body_low <= current_price <= ob_body_high:
            smc_score -= 0.3
            smc_label = "Inside Bearish OB"
            break

    for i in range(len(valleys)-1, max(0, len(valleys)-5), -1):
        ob_body_high = max(float(valleys['open'].iloc[i]), float(valleys['close'].iloc[i]))
        ob_body_low  = min(float(valleys['open'].iloc[i]), float(valleys['close'].iloc[i]))
        # Require a meaningful body (not a doji)
        if ob_body_high - ob_body_low < current_price * 0.0002:
            continue
        if ob_body_low <= current_price <= ob_body_high:
            smc_score += 0.3
            smc_label = "Inside Bullish OB"
            break

    # --- 5. BOUNCE RECOGNITION (The 'Triangle' System) ---
    # Detecting immediate rejections from structural levels.
    dist_from_low = (current_price - last_low) / last_low
    dist_from_high = (last_high - current_price) / last_high

    # 0.2% threshold for 'Bounces'
    bounce_threshold = float(TUNING.get("bounce_threshold", 0.002) or 0.002)
    is_recovering = df['close'].iloc[-1] > df['close'].iloc[-2]
    is_falling = df['close'].iloc[-1] < df['close'].iloc[-2]

    if dist_from_low < bounce_threshold and is_recovering:
        smc_score += 0.5
        smc_label = "Institutional Support Bounce"
        # Early directional signal: tag the ob_context so downstream OB gate
        # and midrange gate know this is a bullish bounce zone.
        # Hard veto of SELL here is not possible (action not yet known),
        # but sr_score will be set to +1.5 by the wall logic below,
        # and Bug2b hard veto will block SELL when sr_score >= 1.0.
    elif dist_from_high < bounce_threshold and is_falling:
        smc_score -= 0.5
        smc_label = "Institutional Resist Rejection"
        # Similarly, sr_score will be -1.5 and Bug2b will block BUY.

    # --- 5b. ORDER BLOCK MIDPOINT CONTEXT ---
    # Track the most recent block zone so the caller can demand a midpoint retest.
    # FIX BUG1b: Use body bounds (open/close) consistently with OB detection above.
    for i in range(len(swings)-1, max(0, len(swings)-5), -1):
        ob_high = max(float(swings['open'].iloc[i]), float(swings['close'].iloc[i]))
        ob_low  = min(float(swings['open'].iloc[i]), float(swings['close'].iloc[i]))
        if ob_high - ob_low < current_price * 0.0002:
            continue
        if ob_low <= current_price <= ob_high:
            ob_mid = (ob_high + ob_low) / 2.0
            ob_context = {
                "active": True,
                "direction": "BEARISH",
                "mid": ob_mid,
                "high": ob_high,
                "low": ob_low,
                "reason": "Inside Bearish OB",
            }
            smc_label = "Inside Bearish OB"
            break
    for i in range(len(valleys)-1, max(0, len(valleys)-5), -1):
        ob_high = max(float(valleys['open'].iloc[i]), float(valleys['close'].iloc[i]))
        ob_low  = min(float(valleys['open'].iloc[i]), float(valleys['close'].iloc[i]))
        if ob_high - ob_low < current_price * 0.0002:
            continue
        if ob_low <= current_price <= ob_high:
            ob_mid = (ob_high + ob_low) / 2.0
            ob_context = {
                "active": True,
                "direction": "BULLISH",
                "mid": ob_mid,
                "high": ob_high,
                "low": ob_low,
                "reason": "Inside Bullish OB",
            }
            smc_label = "Inside Bullish OB"
            break

    # --- 6. S/R WALL LOGIC (Wall Detection) ---
    sr_score = 0.0
    # Combine the most recent significant structural points
    all_levels = sorted(list(swings['high'].values[-10:]) + list(valleys['low'].values[-10:]))

    nearest_sup = last_low
    nearest_res = last_high

    for level in all_levels:
        if level < current_price:
            nearest_sup = max(nearest_sup, level)
        elif level > current_price:
            nearest_res = min(nearest_res, level)

    # Calculate proximity to walls
    dist_to_sup = (current_price - nearest_sup) / nearest_sup if nearest_sup else 1.0
    dist_to_res = (nearest_res - current_price) / nearest_res if nearest_res else 1.0
    wall_veto_threshold = float(TUNING.get("wall_proximity", 0.0015) or 0.0015)

    if dist_to_sup < wall_veto_threshold:
        sr_score = 1.5   # Massive support. Good for longs, Veto for shorts.
        smc_label += " (Near Support Wall)"
    elif dist_to_res < wall_veto_threshold:
        sr_score = -1.5  # Massive resistance. Veto for longs!
        smc_label += " (Near Resistance Wall)"

    return smc_score, sr_score, smc_label, ob_context
