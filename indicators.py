# ==================================================================================================
# INSTITUTIONAL SNIPER BOT: MASTER ANALYTICAL ENGINE (V4.2.0)
# ==================================================================================================
# VERSION HISTORY:
# V1.0.0: Initial quantitative engine with basic RSI/EMA filters.
# V2.5.0: Integrated 50-candle Swing Hunter and BOS detection.
# V3.8.0: Added Institutional Pivot Discipline and MTF S/R mapping.
# V4.1.0: Implemented Fair Value Gap (FVG) and Liquidity Pool mapping.
# V4.2.0: Restored Alpha Overlay and Institutional Rejection Logic.
# ==================================================================================================
# INSTITUTIONAL THEORY:
# This engine operates on the principle of 'Institutional Order Flow'. 
# By identifying areas where large players accumulate or distribute liquidity 
# (Order Blocks and FVG), we can enter trades with high-probability 'Edge'. 
# Scalping requires both structural verification (SMC) and momentum confirmation (EMA).
# ==================================================================================================

import pandas as pd
import numpy as np
import ta
import math
import time
import logging
import json
import os

logger = logging.getLogger(__name__)

# Try to load ML weights if they exist
WEIGHTS_FILE = os.path.join(os.path.dirname(__file__), 'weights.json')

# --- INSTITUTIONAL TUNING PARAMETERS ---
# These parameters define the bot's 'Patience' and 'Aggression' levels.
# Tuning these values affects how the bot perceives 'Bounces' and 'Walls'.
TUNING = {
    "swing_window": 50,       # Window for LuxAlgo-style structural detection
    "bounce_threshold": 0.002, # 0.2% price movement for 'Bounce' recognition
    "wall_proximity": 0.001,  # 0.1% proximity for 'Wall' detection
    "fvg_sensitivity": 0.0001, # Minimum gap size for FVG identification
    "vol_spike_mult": 1.5,     # Volume multiplier for 'Alpha' confirmation
    "squeeze_buffer": 1.1      # Buffer for Bollinger Band Squeeze detection
}

# MASTER SIGNAL WEIGHTS (Institutional Tuning)
SIGNAL_WEIGHTS = {
    'mr': 0.10,    # Mean Reversion (EMA 21/50)
    'ob': 0.10,    # Order Block / SMC
    'vwap': 0.10,  # Volume Weighted Average Price
    'bb': 0.05,    # Bollinger Band Exhaustion
    'macd': 0.05,  # Momentum Diff
    'pa': 0.15,    # Price Action (Candle Structure)
    'smc': 0.15,   # Market Structure (BOS/CHoCH)
    'sr': 0.10,    # Support/Resistance Walls
    'kdj': 0.10,   # Stochastic Momentum
    'st': 0.10     # SuperTrend Alignment
}

_WEIGHTS_CACHE = None
_WEIGHTS_MTIME = None
_WEIGHTS_LAST_CHECK = 0.0


def get_signal_weights() -> dict:
    """
    Load ML-optimized weights if ml_optimizer has produced weights.json.
    The cache keeps the 1-second trading loop from hitting disk every tick.
    """
    global _WEIGHTS_CACHE, _WEIGHTS_MTIME, _WEIGHTS_LAST_CHECK

    now = time.time()
    if _WEIGHTS_CACHE is not None and (now - _WEIGHTS_LAST_CHECK) < 30:
        return dict(_WEIGHTS_CACHE)

    _WEIGHTS_LAST_CHECK = now
    weights = dict(SIGNAL_WEIGHTS)

    try:
        if os.path.exists(WEIGHTS_FILE):
            mtime = os.path.getmtime(WEIGHTS_FILE)
            if _WEIGHTS_CACHE is not None and _WEIGHTS_MTIME == mtime:
                return dict(_WEIGHTS_CACHE)

            with open(WEIGHTS_FILE, "r", encoding="utf-8") as f:
                learned = json.load(f)

            for key in ["mr", "vwap", "bb", "macd", "pa", "smc", "sr", "kdj", "st"]:
                if key in learned:
                    weights[key] = max(0.0, float(learned[key]))
            # Historical key kept for compatibility; the signal formula uses smc.
            weights["ob"] = 0.0
            _WEIGHTS_MTIME = mtime
    except Exception as e:
        logger.warning(f"Failed to load learned weights: {e}")

    _WEIGHTS_CACHE = dict(weights)
    return weights

def calculate_base_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    MASTER RECONSTRUCTION: The 725-Line Analytical Engine (Phase 1).
    Calculates the full institutional suite required by main.py and the dashboard.
    """
    if df is None or df.empty:
        return df
        
    df = df.copy()
    
    # 1. Moving Averages (The Trend Core)
    df['ema_9'] = ta.trend.ema_indicator(df['close'], window=9)
    df['ema_21'] = ta.trend.ema_indicator(df['close'], window=21)
    df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)
    df['ema_200'] = ta.trend.ema_indicator(df['close'], window=200)
    
    # 2. Bollinger Bands (Volatility & Exhaustion)
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_mid'] = bb.bollinger_mavg()
    df['bb_low'] = bb.bollinger_lband()
    df['bb_width'] = bb.bollinger_wband()
    
    # 3. RSI (The Overbought/Oversold Guard)
    df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)
    df['rsi'] = df['rsi_14'] # Legacy alias
    
    # 4. MACD (Momentum Convergence)
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    
    # 5. KDJ (Fast Stochastic Momentum)
    # KDJ is a standard institutional tool for scalpers
    low_min = df['low'].rolling(window=9).min()
    high_max = df['high'].rolling(window=9).max()
    rsv = (df['close'] - low_min) / (high_max - low_min) * 100
    df['k'] = rsv.ewm(com=2, adjust=False).mean()
    df['d'] = df['k'].ewm(com=2, adjust=False).mean()
    df['j'] = 3 * df['k'] - 2 * df['d']
    
    # 6. ADX (Trend Strength - Required by AI Context)
    adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
    df['adx'] = adx.adx()
    df['adx_pos'] = adx.adx_pos()
    df['adx_neg'] = adx.adx_neg()
    
    # 7. ATR (Volatility-Based Stops)
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    df['atr_pct'] = (df['atr'] / df['close']) * 100
    
    # 8. VWAP (Intraday Value Anchor)
    # Using cumulative typical price for accurate intraday tracking
    df['vwap'] = ta.volume.volume_weighted_average_price(
        df['high'], df['low'], df['close'], df['volume'], window=14
    )
    
    # 9. SuperTrend Proxy (Trend Bias)
    # Simplified SuperTrend for high-speed calculation
    df['st_upper'] = df['bb_mid'] + (df['atr'] * 3)
    df['st_lower'] = df['bb_mid'] - (df['atr'] * 3)
    df['trend_bias'] = np.where(df['close'] > df['ema_21'], 1, -1)

    return df

def compute_volatility_context(df: pd.DataFrame) -> dict:
    """
    MASTER RECONSTRUCTION: Deep Volatility Analysis.
    Detects 'Squeezes' and 'Expansion' phases to prevent trading in chop.
    By identifying when Bollinger Band width contracts below historical norms, 
    we can avoid the 'Chop Zones' that drain scalping balances.
    """
    if len(df) < 20: return {"squeeze": False, "volatility": "Normal", "atr_rank": 0.5}
    
    # 1. Bollinger Band Squeeze (Width < 20-period Low)
    # This identifies periods of extreme low volatility preceding a breakout.
    bb_width = df['bb_width']
    is_squeeze = bb_width.iloc[-1] < bb_width.rolling(window=20).min().iloc[-1] * 1.1
    
    # 2. ATR Ranking (Normalized 0.0 to 1.0)
    # We rank the current ATR against its 100-candle history to determine if 
    # volatility is rising or falling in a macro sense.
    atr = df['atr']
    atr_min = atr.rolling(window=100).min().iloc[-1]
    atr_max = atr.rolling(window=100).max().iloc[-1]
    atr_rank = (atr.iloc[-1] - atr_min) / (atr_max - atr_min + 0.0001)
    
    vol_status = "High" if atr_rank > 0.7 else ("Low" if atr_rank < 0.3 else "Normal")
    
    return {
        "squeeze": is_squeeze,
        "volatility": vol_status,
        "atr_rank": atr_rank,
        "raw_width": bb_width.iloc[-1]
    }

def identify_liquidity_pools(df: pd.DataFrame) -> list:
    """
    MASTER RECONSTRUCTION: Liquidity Pool Identification.
    Locates price zones where 'Stop Hunts' are likely to occur.
    These are areas of extreme structural significance where buy/sell stops cluster.
    """
    if len(df) < 100: return []
    
    # Liquidity often sits just above/below significant swing points.
    # Institutional traders hunt these levels to fill large orders.
    window = 50
    highs = df['high'].rolling(window=window).max()
    lows = df['low'].rolling(window=window).min()
    
    # Calculate Buy-Side and Sell-Side Liquidity Zones
    pools = [
        {"type": "Buy Side Liquidity", "level": highs.iloc[-1] * 1.0005, "strength": "Strong"},
        {"type": "Sell Side Liquidity", "level": lows.iloc[-1] * 0.9995, "strength": "Strong"}
    ]
    
    # Add secondary pools based on intermediate swings
    pools.append({"type": "Minor Buy Side", "level": df['high'].iloc[-20:].max(), "strength": "Weak"})
    pools.append({"type": "Minor Sell Side", "level": df['low'].iloc[-20:].min(), "strength": "Weak"})
    
    return pools

def calculate_funding_impact(latest_macro: dict) -> float:
    """
    MASTER RECONSTRUCTION: Funding Rate Impact.
    Adjusts signal confidence based on the cost of holding a position.
    In Futures trading, high funding rates can eat into scalping profits quickly.
    """
    if not latest_macro: return 0.0
    
    funding_rate = latest_macro.get('funding_rate', 0.0)
    
    # Threshold for intervention (0.01% per 8h is standard)
    # High positive funding penalizes long entries.
    # High negative funding penalizes short entries.
    impact = 0.0
    if funding_rate > 0.01:
        impact = -1.0 # Longs are expensive
    elif funding_rate < -0.01:
        impact = 1.0  # Shorts are expensive
        
    return impact

def generate_alpha_overlay(df: pd.DataFrame, smc_score: float, macro_bias: str) -> float:
    """
    MASTER RECONSTRUCTION: Alpha Overlay Engine.
    Synthesizes structural bias with momentum to find 'The Edge'.
    This is the 'Secret Sauce' that confirms structural breaks with 
    volume-supported momentum crosses.
    """
    if len(df) < 50: return 0.0
    
    # 1. Momentum Cross Check (EMA 9/21)
    # Rapid trend confirmation for scalpers.
    ema_9 = df['ema_9'].iloc[-1]
    ema_21 = df['ema_21'].iloc[-1]
    mom_bias = 1.0 if ema_9 > ema_21 else -1.0
    
    # 2. Relative Volume Spike
    # Confirms if the move is backed by real money.
    avg_vol = df['volume'].rolling(window=20).mean().iloc[-1]
    current_vol = df['volume'].iloc[-1]
    vol_spike = current_vol > (avg_vol * 1.5)
    
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
    # Block signals during extreme Bollinger Squeezes to prevent 'fakeouts'
    if vol_context.get('squeeze') and abs(signal.get('score', 0)) < 0.3:
        signal['action'] = "HOLD"
        signal['hold_reason'] = "Volatility Squeeze: High Fakeout Risk"
        
    # Block signals if ATR is too low (not enough movement to cover fees)
    if vol_context.get('atr_rank', 1.0) < 0.1:
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

def detect_smc_and_sr(df: pd.DataFrame, current_price: float) -> tuple:
    """
    MASTER RECONSTRUCTION: The Institutional SMC "Swing Hunter" Engine.
    Exhaustive search for Market Structure (BOS/CHoCH), Order Blocks,
    Mitigation Zones, Fair Value Gaps (FVG), and Institutional S/R Walls.
    """
    if len(df) < 50: 
        return 0.0, 0.0, "Warming Engine"

    df = df.copy()
    
    # --- 1. THE SWING HUNTER (LuxAlgo High-Resolution) ---
    # We use a 50-candle rolling window to find significant swing highs and lows.
    # This matches the 'Swing Hunter' logic used by professional trading desks.
    window = 50
    df['swing_high'] = df['high'].rolling(window=window, center=True).max()
    df['swing_low'] = df['low'].rolling(window=window, center=True).min()
    
    # Extract structural peaks and valleys
    swings = df[df['high'] == df['swing_high']].copy()
    valleys = df[df['low'] == df['swing_low']].copy()
    
    if len(swings) < 3 or len(valleys) < 3:
        return 0.0, 0.0, "Building Structure"

    # Precise structural price points
    last_high = swings['high'].iloc[-1]
    last_low = valleys['low'].iloc[-1]
    prev_high = swings['high'].iloc[-2]
    prev_low = valleys['low'].iloc[-2]
    old_high = swings['high'].iloc[-3]
    old_low = valleys['low'].iloc[-3]

    smc_score = 0.0
    smc_label = "Neutral"

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

    # --- 3. FAIR VALUE GAPS (FVG) / LIQUIDITY VOIDS ---
    # Detecting gaps where price moved too fast, creating an imbalance.
    # Price often 'revisits' these gaps to fill liquidity.
    fvg_detected = False
    for i in range(-5, -1):
        # Bullish FVG (Gap between Candle 1 High and Candle 3 Low)
        if df['high'].iloc[i-1] < df['low'].iloc[i+1]:
            gap_size = df['low'].iloc[i+1] - df['high'].iloc[i-1]
            if current_price > df['high'].iloc[i-1] and current_price < df['low'].iloc[i+1]:
                smc_score += 0.2
                smc_label = "FVG Bullish Entry"
                fvg_detected = True
        # Bearish FVG (Gap between Candle 1 Low and Candle 3 High)
        elif df['low'].iloc[i-1] > df['high'].iloc[i+1]:
            gap_size = df['low'].iloc[i-1] - df['high'].iloc[i+1]
            if current_price < df['low'].iloc[i-1] and current_price > df['high'].iloc[i+1]:
                smc_score -= 0.2
                smc_label = "FVG Bearish Entry"
                fvg_detected = True

    # --- 4. ORDER BLOCKS & MITIGATION ZONES ---
    # Identifying the last 'Opposite Color' candle before a major move.
    # Bullish Order Block (Last Red candle before a massive pump)
    # Bearish Order Block (Last Green candle before a massive dump)
    
    # We look back at the last 5 swings to see if current price is 'mitigating' an old block
    for i in range(len(swings)-1, max(0, len(swings)-5), -1):
        block_high = swings['high'].iloc[i]
        block_low = swings['low'].iloc[i]
        if current_price >= block_low and current_price <= block_high:
            smc_score -= 0.3
            smc_label = "Inside Bearish OB"
            break
            
    for i in range(len(valleys)-1, max(0, len(valleys)-5), -1):
        block_high = valleys['high'].iloc[i]
        block_low = valleys['low'].iloc[i]
        if current_price >= block_low and current_price <= block_high:
            smc_score += 0.3
            smc_label = "Inside Bullish OB"
            break

    # --- 5. BOUNCE RECOGNITION (The 'Triangle' System) ---
    # Detecting immediate rejections from structural levels.
    dist_from_low = (current_price - last_low) / last_low
    dist_from_high = (last_high - current_price) / last_high
    
    # 0.2% threshold for 'Bounces'
    bounce_threshold = 0.002 
    is_recovering = df['close'].iloc[-1] > df['close'].iloc[-2]
    is_falling = df['close'].iloc[-1] < df['close'].iloc[-2]
    
    if dist_from_low < bounce_threshold and is_recovering:
        smc_score += 0.5
        smc_label = "Institutional Support Bounce"
    elif dist_from_high < bounce_threshold and is_falling:
        smc_score -= 0.5
        smc_label = "Institutional Resist Rejection"

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
    dist_to_sup = (current_price - nearest_sup) / nearest_sup
    dist_to_res = (nearest_res - current_price) / nearest_res
    
    if dist_to_sup < 0.001:
        sr_score = 1.0 # Bouncing off a wall
    elif dist_to_res < 0.001:
        sr_score = -1.0 # Hitting a ceiling

    return smc_score, sr_score, smc_label

def build_mtf_timeframe_context(df: pd.DataFrame) -> dict:
    """
    MASTER RECONSTRUCTION: Analyzes a timeframe for the dashboard.
    Calculates percentage distance to nearest S/R levels.
    """
    if df is None or len(df) < 50:
        return {"trend": "NEUT", "support_levels": [], "resistance_levels": [], "s_dist": "-", "r_dist": "-"}
        
    ema_21 = ta.trend.ema_indicator(df['close'], window=21).iloc[-1]
    current_price = df['close'].iloc[-1]
    
    # Structural S/R
    window = 50
    highs = df['high'].rolling(window=window).max()
    lows = df['low'].rolling(window=window).min()
    
    res = highs.iloc[-1]
    sup = lows.iloc[-1]
    
    r_dist = (res - current_price) / current_price * 100
    s_dist = (sup - current_price) / current_price * 100
    
    trend = "BULL" if current_price > ema_21 else "BEAR"
    
    return {
        "trend": trend,
        "support_levels": [sup],
        "resistance_levels": [res],
        "s_dist": f"{s_dist:+.1f}%",
        "r_dist": f"{r_dist:+.1f}%"
    }

def generate_quant_signal(state: dict, latest_indicators: dict, strategy_config: dict, df_indicators: pd.DataFrame, latest_macro: dict = None, mtf_context: dict = None, mtf_config: dict = None, pivot_data: dict = None) -> dict:
    """
    MASTER RECONSTRUCTION: The 725-Line Institutional Sniper Signal Engine.
    Exhaustive synthesis of 10 primary indicators with 4 advanced alpha overlays.
    
    This engine is designed for high-frequency scalping where every basis point 
    of confidence matters. It factors in structural breaks, volatility squeezes, 
    institutional pivot penalties, and liquidity pool proximity.
    """
    signal = {"action": "HOLD", "score": 0, "confidence": 0, "reason": "Warming Up", "weights": {}}
    if df_indicators is None or len(df_indicators) < 50:
        return {"action": "HOLD", "score": 0, "confidence": 0, "reason": "Warming Up", "weights": {}}

    # --- 1. CORE PARAMETERS & CONTEXT ---
    current_price = state['price']
    weights = get_signal_weights()
    macro_bias = str(latest_macro or "NEUTRAL").upper()
    
    # --- 2. ADVANCED CONTEXTUAL MODULES ---
    # We call our institutional modules to understand the 'Regime'
    vol_context = compute_volatility_context(df_indicators)
    liquidity = identify_liquidity_pools(df_indicators)
    funding_impact = calculate_funding_impact(latest_macro)
    
    # --- 3. PRIMARY SIGNAL CALCULATION ---
    # Structural & SMC Bias
    smc_score, sr_score, smc_label = detect_smc_and_sr(df_indicators, current_price)
    
    # Mean Reversion (MR)
    # Scalpers buy the 'Stretch' away from the EMA 21.
    ema_21 = latest_indicators.get('ema_21', current_price)
    mr_score = 1.0 if current_price < ema_21 * 0.998 else (-1.0 if current_price > ema_21 * 1.002 else 0.0)
    
    # Volume Weighted Average Price (VWAP)
    # The 'Institutional Value' anchor.
    vwap = latest_indicators.get('vwap', current_price)
    vwap_score = 1.0 if current_price < vwap else -1.0
    
    # Bollinger Bands (BB)
    # Detecting exhaustion at the 2-sigma deviations.
    bb_low = latest_indicators.get('bb_low', current_price)
    bb_high = latest_indicators.get('bb_high', current_price)
    bb_score = 1.0 if current_price < bb_low * 1.001 else (-1.0 if current_price > bb_high * 0.999 else 0.0)
    
    # MACD & Price Action Momentum
    macd_diff = latest_indicators.get('macd_diff', 0)
    macd_score = np.sign(macd_diff) if abs(macd_diff) > 0.0001 else 0.0
    pa_score = 1.0 if df_indicators['close'].iloc[-1] > df_indicators['open'].iloc[-1] else -1.0
    
    # KDJ & SuperTrend
    kdj_j = latest_indicators.get('j', 50)
    kdj_score = 1.0 if kdj_j < 20 else (-1.0 if kdj_j > 80 else 0.0)
    st_score = latest_indicators.get('trend_bias', 0)
    
    # --- 4. ALPHA OVERLAY & REFINEMENT ---
    # The 'Alpha Overlay' adds extra weight when multiple trends align.
    alpha = generate_alpha_overlay(df_indicators, smc_score, macro_bias)
    
    # --- 5. INSTITUTIONAL PENALTIES ---
    # The 'No-Man's Land' Guard prevents trading in the center of the trading range.
    penalty = 1.0
    pivot_msg = "Near Level"
    hard_veto = False
    
    if pivot_data:
        # Combine all institutional levels (Daily + Intraday 4H)
        daily = pivot_data.get('daily', {})
        h4 = pivot_data.get('h4', {})
        
        levels = []
        # Add Daily levels (Heavy Armor)
        levels += [daily.get('pp'), daily.get('s1'), daily.get('r1'), daily.get('s2'), daily.get('r2')]
        # Add 4H levels (Scalper Shields)
        levels += [h4.get('pp'), h4.get('s1'), h4.get('r1')]
        
        levels = [l for l in levels if l is not None]
        
        # Calculate distance to the nearest level
        min_dist = min([abs(current_price - l) / l for l in levels]) if levels else 1.0
        
        # Proximity threshold
        threshold = float(strategy_config.get('pivot_proximity_threshold', 0.0020))
        
        if min_dist > threshold:
            penalty = 0.5 
            pivot_msg = "Between Levels [Caution: Mid-Range]"
            if strategy_config.get('pivot_discipline', True):
                hard_veto = True
        else:
            # We are near a level! Identify if it's Daily or 4H
            is_daily = any([abs(current_price - daily.get(k,0))/current_price < threshold for k in ['pp','s1','r1','s2','r2']])
            pivot_msg = "Near DAILY Level" if is_daily else "Near 4H Level"
    
            
    # --- 6. FINAL WEIGHTED SYNTHESIS ---
    total_score = (
        (mr_score * weights['mr']) +
        (vwap_score * weights['vwap']) +
        (bb_score * weights['bb']) +
        (macd_score * weights['macd']) +
        (pa_score * weights['pa']) +
        (smc_score * weights['smc']) +
        (sr_score * weights['sr']) +
        (kdj_score * weights['kdj']) +
        (st_score * weights['st']) +
        alpha # Add the Alpha Overlay contribution
    )
    
    # Apply context-based multipliers
    total_score *= penalty
    total_score += (funding_impact * 0.05) # Subtle adjustment for funding
    
    # --- 7. SIGNAL INTEGRITY & VALIDATION ---
    action = "HOLD"
    # --- 8. SMART VETO BYPASS (The Sniper Logic) ---
    # If we are at a strong Institutional Wall (Daily/4H) AND Book Pressure is high,
    # we bypass the MTF Trend Veto to front-run the reversal.
    
    is_at_wall = (pivot_msg != "Between Levels [Caution: Mid-Range]")
    book_pressure = map_order_book_pressure(state)
    
    # RUTHLESS BYPASS: If we are at a wall AND we have a strong structural signal (SMC/Score),
    # we enter IMMEDIATELY. We don't wait for the trend or even the order book.
    bypass_veto = False
    if is_at_wall:
        # If score is very strong (>0.3) at a wall, we trust the structure
        if abs(total_score) > 0.30: 
            bypass_veto = True
            pivot_msg += " [RUTHLESS SNIPE]"
        # Or if book pressure is starting to build
        elif (total_score > 0 and book_pressure > 0.05) or (total_score < 0 and book_pressure < -0.05):
            bypass_veto = True
            pivot_msg += " [BOOK BYPASS]"

    if hard_veto and not bypass_veto:
        action = "HOLD"
        signal['action'] = "HOLD"
        signal['hold_reason'] = "MTF Veto (No Structural Edge)"
    elif total_score > float(strategy_config.get('min_conf', 0.15)):
        action = "BUY"
    elif total_score < -float(strategy_config.get('min_conf', 0.15)):
        action = "SELL"
    
    

    # --- MTF ALIGNMENT VETO ---
    if mtf_context and mtf_config and mtf_config.get('enabled') and not bypass_veto:
        mode = mtf_config.get('mode', 'soft')
        min_agree = int(mtf_config.get('min_agree', 2))
        tfs = mtf_config.get('timeframes', [])
        
        bull_count = 0
        bear_count = 0
        for tf in tfs:
            ctx = mtf_context.get(tf)
            if isinstance(ctx, dict) and ctx.get('trend'):
                if ctx['trend'] == 'BULL': bull_count += 1
                elif ctx['trend'] == 'BEAR': bear_count += 1
        
        if mode == 'hard':
            if action == 'BUY' and bull_count < min_agree:
                action = "HOLD"
                pivot_msg += f" [MTF Veto: Bulls {bull_count}<{min_agree}]"
            elif action == 'SELL' and bear_count < min_agree:
                action = "HOLD"
                pivot_msg += f" [MTF Veto: Bears {bear_count}<{min_agree}]"
                
            # Sniper Guard: Specifically veto if the fastest timeframe (e.g. 3m) disagrees
            if tfs and action != "HOLD":
                fastest_tf = tfs[0]
                fast_ctx = mtf_context.get(fastest_tf)
                if isinstance(fast_ctx, dict) and fast_ctx.get('trend'):
                    if action == "BUY" and fast_ctx['trend'] == "BEAR":
                        action = "HOLD"
                        pivot_msg += f" [MTF Veto: {fastest_tf} BEAR]"
                    elif action == "SELL" and fast_ctx['trend'] == "BULL":
                        action = "HOLD"
                        pivot_msg += f" [MTF Veto: {fastest_tf} BULL]"

    # --- 8. ORDER BOOK PRESSURE ANALYSIS ---
    # Detecting 'The Wall of Money' to prevent trading into massive institutional walls.
    book_pressure = map_order_book_pressure(state)
    book_msg = "Balanced"
    
    if book_pressure > 0.3: book_msg = "Heavy Buy Pressure"
    elif book_pressure < -0.3: book_msg = "Heavy Sell Pressure"
    
    # Order Book Veto: Never buy into a sell wall, never sell into a buy wall.
    if action == "BUY" and book_pressure < -0.3:
        action = "HOLD"
        pivot_msg += f" [Book Veto: Sell Wall {book_pressure:.2f}]"
    elif action == "SELL" and book_pressure > 0.3:
        action = "HOLD"
        pivot_msg += f" [Book Veto: Buy Wall {book_pressure:.2f}]"

    # The reason string used for the professional dashboard.
    reason = (
        f"Score:{total_score:.3f} SMC:{smc_label} Pivot:{pivot_msg} Book:{book_msg} "
        f"(MR:{mr_score:.1f} OB:{smc_score:.1f} SR:{sr_score:.1f} VWAP:{vwap_score:.1f} BB:{bb_score:.1f} "
        f"MACD:{macd_score:.1f} PA:{pa_score:.1f} KDJ:{kdj_score:.1f} ST:{st_score:.1f})"
    )
    
    # Build the final signal object without overwriting the existing one
    signal.update({
        "action": action,
        "score": total_score,
        "confidence": min(abs(total_score), 1.0),
        "reason": reason,
        "market_bias": "LONG_ONLY" if total_score > 0 else "SHORT_ONLY"
    })
    
    # Only set the default hold reason if it wasn't already set by a veto
    if not signal.get('hold_reason'):
        signal['hold_reason'] = "Pivot Discipline: Mid-Range" if hard_veto else ""
    
    
    # Final Order Book refinement to Confidence
    if action != "HOLD":
        # If book pressure aligns with our direction, boost confidence
        if (action == "BUY" and book_pressure > 0.2) or (action == "SELL" and book_pressure < -0.2):
            signal['confidence'] = min(1.0, signal['confidence'] * 1.2)
    
    try:
        tp_pct = float(strategy_config.get("tp_pct", 0.0030))
        sl_pct = float(strategy_config.get("sl_pct", 0.0050))
    except (TypeError, ValueError, AttributeError):
        tp_pct = 0.0030
        sl_pct = 0.0050

    if action == "BUY":
        signal.update({
            "entry": current_price,
            "tp": current_price * (1 + tp_pct),
            "sl": current_price * (1 - sl_pct),
        })
    elif action == "SELL":
        signal.update({
            "entry": current_price,
            "tp": current_price * (1 - tp_pct),
            "sl": current_price * (1 + sl_pct),
        })
    
    # Final filter for dangerous volatility conditions
    signal = validate_signal_integrity(signal, vol_context)
    
    
    return signal

def get_trend_status(df: pd.DataFrame) -> str:
    """
    MASTER RECONSTRUCTION: Detailed Trend Analysis.
    Generates a linguistic summary of the current price action relative to EMAs.
    Used for the 'Thinking' section of the professional dashboard.
    """
    if len(df) < 50: return "Warming Up"
    
    current_price = df['close'].iloc[-1]
    ema_21 = df['ema_21'].iloc[-1]
    ema_200 = df['ema_200'].iloc[-1]
    
    if current_price > ema_21 > ema_200:
        return "Strong Bullish (Institutional Alignment)"
    elif current_price < ema_21 < ema_200:
        return "Strong Bearish (Institutional Alignment)"
    elif current_price > ema_21:
        return "Short-Term Bullish (Recovery Phase)"
    else:
        return "Short-Term Bearish (Distribution Phase)"

def get_volatility_status(vol_context: dict) -> str:
    """
    MASTER RECONSTRUCTION: Volatility Status Report.
    Translates mathematical ATR/BB metrics into trading regime labels.
    """
    if vol_context.get('squeeze'):
        return "Squeeze Phase (Extreme Contraction - Breakout Imminent)"
    
    vol = vol_context.get('volatility', 'Normal')
    if vol == "High":
        return "High Volatility (Expansion Phase - Wide Stops Required)"
    elif vol == "Low":
        return "Low Volatility (Chop Zone - Scalping Risk)"
    return "Normal Volatility (Steady Flow)"

def get_momentum_status(latest_indicators: dict) -> str:
    """
    MASTER RECONSTRUCTION: Momentum Relationship Report.
    Analyzes the interaction between RSI and MACD for exhaustion signals.
    """
    rsi = latest_indicators.get('rsi_14', 50)
    macd_diff = latest_indicators.get('macd_diff', 0)
    
    if rsi > 70 and macd_diff < 0:
        return "Bearish Divergence (Overbought Exhaustion)"
    elif rsi < 30 and macd_diff > 0:
        return "Bullish Divergence (Oversold Exhaustion)"
    elif macd_diff > 0:
        return "Bullish Momentum (Rising Flow)"
    return "Bearish Momentum (Falling Flow)"

def get_structural_status(smc_label: str) -> str:
    """
    MASTER RECONSTRUCTION: Structural Context Report.
    Explains the current SMC phase for the trading dashboard.
    """
    if "BOS" in smc_label:
        return "Trend Continuation (Structure Break Verified)"
    elif "CHoCH" in smc_label:
        return "Structural Flip (Trend Reversal Verified)"
    elif "Bounce" in smc_label:
        return "Level Rejection (Liquidity Reaction Verified)"
    return "Neutral Structure (Range Bound)"

# ==================================================================================================
# LEGACY COMPATIBILITY & ALIASES
# ==================================================================================================

def calculate_indicators(df):
    """Legacy alias for calculate_base_indicators."""
    return calculate_base_indicators(df)

def get_quant_signal(*args, **kwargs):
    """Legacy alias for generate_quant_signal."""
    return generate_quant_signal(*args, **kwargs)

# ==================================================================================================
# INTERNAL INSTITUTIONAL HELPERS (THE 'GRANULAR EDGE')
# ==================================================================================================

def _get_institutional_bias(df: pd.DataFrame) -> float:
    """
    MASTER RECONSTRUCTION: EMA Stack Analysis.
    Calculates the 'Institutional Stack' bias based on the hierarchical 
    alignment of the 9, 21, 50, and 200 EMA levels. 
    
    A perfectly stacked EMA sequence indicates 'High Probability' trend flow.
    Returns 1.0 if EMA 9 > 21 > 50 > 200 (Extreme Bullish),
    Returns -1.0 if EMA 9 < 21 < 50 < 200 (Extreme Bearish).
    """
    if len(df) < 200: return 0.0
    
    ema9 = df['ema_9'].iloc[-1]
    ema21 = df['ema_21'].iloc[-1]
    ema50 = df['ema_50'].iloc[-1]
    ema200 = df['ema_200'].iloc[-1]
    
    if ema9 > ema21 > ema50 > ema200: return 1.0
    if ema9 < ema21 < ema50 < ema200: return -1.0
    return 0.0

def _analyze_wick_rejection(df: pd.DataFrame) -> float:
    """
    Identifies 'Price Pushes' where long wicks indicate institutional absorption.
    A long bottom wick suggests 'Hidden Demand', a long top wick suggests 'Hidden Supply'.
    """
    latest = df.iloc[-1]
    body_size = abs(latest['close'] - latest['open'])
    upper_wick = latest['high'] - max(latest['open'], latest['close'])
    lower_wick = min(latest['open'], latest['close']) - latest['low']
    
    # Rejection Score (0.0 to 1.0)
    if lower_wick > (body_size * 2): return 0.5 # Bullish Absorption
    if upper_wick > (body_size * 2): return -0.5 # Bearish Absorption
    return 0.0

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

def map_order_book_pressure(state: dict) -> float:
    """
    MASTER RECONSTRUCTION: Order Book Pressure Mapping.
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
    

# ==================================================================================================
# RISK WARNING & LEGAL DISCLAIMER
# ==================================================================================================
# Trading cryptocurrencies and futures involves significant risk and can result in the loss 
# of your invested capital. This bot is provided as an analytical tool for professional 
# traders and does not guarantee profits. Past performance is not indicative of future 
# results. Ensure all API keys have appropriate permissions and use a dedicated sub-account 
# for automated trading. THE AUTHORS ARE NOT RESPONSIBLE FOR ANY TRADING LOSSES.
# ==================================================================================================
# END OF MASTER ANALYTICAL ENGINE
# ==================================================================================================
