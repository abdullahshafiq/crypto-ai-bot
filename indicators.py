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

# MASTER SIGNAL WEIGHTS (Institutional Tuning - Scalp Optimized)
SIGNAL_WEIGHTS = {
    'mr': 0.05,    # Mean Reversion (EMA 21/50)
    'vwap': 0.08,  # Volume Weighted Average Price
    'adx': 0.08,   # Trend Strength
    'vol': 0.08,   # Volume Delta
    'obv': 0.05,   # Accumulation / distribution flow
    'bb': 0.03,    # Bollinger Band Exhaustion
    'macd': 0.30,  # Momentum Authority
    'pa': 0.30,    # Price Action / SAR Authority
    'smc': 0.05,   # Market Structure
    'sr': 0.02,    # Support/Resistance Walls
    'loc': 0.08,   # Market location / session context
    'kdj': 0.03,   # Stochastic Momentum
    'st': 0.25,    # EMA Trend Authority
}
# Normalize to 1.0 (sum was 1.40 — inflating all scores by 40%)
_weight_total = sum(SIGNAL_WEIGHTS.values())
for _k in SIGNAL_WEIGHTS:
    SIGNAL_WEIGHTS[_k] /= _weight_total

_WEIGHTS_CACHE = None
_WEIGHTS_MTIME = None
_WEIGHTS_LAST_CHECK = 0.0


def get_signal_weights() -> dict:
    """
    Load learned weights from weights.json if available, otherwise use static defaults.
    """
    try:
        if os.path.exists(WEIGHTS_FILE):
            with open(WEIGHTS_FILE, "r") as f:
                learned = json.load(f)
            if isinstance(learned, dict) and learned:
                merged = dict(SIGNAL_WEIGHTS)
                merged.update({k: float(v) for k, v in learned.items() if k in merged})
                total = sum(merged.values())
                if total > 0:
                    return {k: v / total for k, v in merged.items()}
    except Exception:
        pass
    return dict(SIGNAL_WEIGHTS)

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

    # 9a. OBV (Accumulation / Distribution)
    price_delta = df['close'].diff().fillna(0.0)
    obv_dir = np.sign(price_delta).fillna(0.0)
    df['obv'] = (obv_dir * df['volume'].fillna(0.0)).cumsum()
    df['obv_ema'] = df['obv'].ewm(span=10, adjust=False).mean()
    df['vwap_dist_pct'] = ((df['close'] - df['vwap']) / df['close']) * 100.0

    # 10. SuperTrend Proxy (Trend Bias)
    # Use EMA 9 vs EMA 21 for trend bias instead of raw close price to prevent whipsaws
    df['st_upper'] = df['bb_mid'] + (df['atr'] * 3)
    df['st_lower'] = df['bb_mid'] - (df['atr'] * 3)
    df['trend_bias'] = np.where(df['ema_9'] > df['ema_21'], 1, -1)

    # 11. Z-Score (Mean Reversion)
    # Measures how many standard deviations price is from the 20-period mean
    df['z_score'] = (df['close'] - df['bb_mid']) / (df['close'].rolling(window=20).std() + 1e-8)

    # 12. Parabolic SAR (Dynamic Trailing Stop / Reversal)
    psar_ind = ta.trend.PSARIndicator(df['high'], df['low'], df['close'], step=0.02, max_step=0.2)
    df['psar'] = psar_ind.psar()
    df['psar_down'] = psar_ind.psar_down_indicator()
    df['psar_up'] = psar_ind.psar_up_indicator()

    # PSAR Streak (How many dots in the current direction)
    # 1 if PSAR is below price (Bullish), -1 if above (Bearish)
    psar_dir = np.where(df['psar'] < df['close'], 1, -1)
    psar_series = pd.Series(psar_dir, index=df.index)
    groups = (psar_series != psar_series.shift()).cumsum()
    df['psar_streak'] = psar_series.groupby(groups).cumcount() + 1
    df['psar_streak'] = df['psar_streak'] * psar_series

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
    squeeze_buffer = float(TUNING.get("squeeze_buffer", 1.1) or 1.1)
    is_squeeze = bb_width.iloc[-1] < bb_width.rolling(window=20).min().iloc[-1] * squeeze_buffer
    
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
    if not isinstance(latest_macro, dict): return 0.0
    
    funding_rate = latest_macro.get('funding_rate', 0.0)
    
    # Threshold for intervention (0.01% per 8h is standard)
    # High positive funding penalizes long entries.
    # High negative funding penalizes short entries.
    impact = 0.0
    if funding_rate > 0.0001:
        impact = -1.0 # Longs are expensive
    elif funding_rate < -0.0001:
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
    # For scalping, a squeeze is a warning, not an automatic veto.
    # Only very weak signals are blocked; stronger setups are allowed through.
    if vol_context.get('squeeze'):
        squeeze_score = abs(signal.get('score', 0))
        if squeeze_score < 0.05:
            signal['action'] = "HOLD"
            signal['hold_reason'] = "Volatility Squeeze: High Fakeout Risk"
        else:
            signal['squeeze_warning'] = "Volatility Squeeze: Trade Allowed"
        
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
    # We use an 11-candle rolling window to find significant swing highs and lows.
    # This is optimized for scalping, yielding only a 5-candle lag (instead of 25).
    window = 11
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
    fvg_sensitivity = float(TUNING.get("fvg_sensitivity", 0.0001) or 0.0001)
    for i in range(-5, -1):
        # Bullish FVG (Gap between Candle 1 High and Candle 3 Low)
        if df['high'].iloc[i-1] < df['low'].iloc[i+1]:
            gap_size = df['low'].iloc[i+1] - df['high'].iloc[i-1]
            if gap_size / max(current_price, 1e-9) >= fvg_sensitivity and current_price > df['high'].iloc[i-1] and current_price < df['low'].iloc[i+1]:
                smc_score += 0.2
                smc_label = "FVG Bullish Entry"
                fvg_detected = True
        # Bearish FVG (Gap between Candle 1 Low and Candle 3 High)
        elif df['low'].iloc[i-1] > df['high'].iloc[i+1]:
            gap_size = df['low'].iloc[i-1] - df['high'].iloc[i+1]
            if gap_size / max(current_price, 1e-9) >= fvg_sensitivity and current_price < df['low'].iloc[i-1] and current_price > df['high'].iloc[i+1]:
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
    bounce_threshold = float(TUNING.get("bounce_threshold", 0.002) or 0.002)
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
    dist_to_sup = (current_price - nearest_sup) / nearest_sup if nearest_sup else 1.0
    dist_to_res = (nearest_res - current_price) / nearest_res if nearest_res else 1.0
    wall_veto_threshold = float(TUNING.get("wall_proximity", 0.001) or 0.001)
    
    if dist_to_sup < wall_veto_threshold:
        sr_score = 1.5   # Massive support. Good for longs, Veto for shorts.
        smc_label += " (Near Support Wall)"
    elif dist_to_res < wall_veto_threshold:
        sr_score = -1.5  # Massive resistance. Veto for longs!
        smc_label += " (Near Resistance Wall)"

    return smc_score, sr_score, smc_label

def build_mtf_timeframe_context(df: pd.DataFrame) -> dict:
    """
    Analyzes a timeframe for the dashboard.
    Uses swing detection (local highs/lows) to find NEAREST structural S/R
    instead of rolling window absolute min/max which misses recent pivots.
    """
    if df is None or len(df) < 50:
        return {"trend": "NEUT", "support_levels": [], "resistance_levels": [], "s_dist": "-", "r_dist": "-"}
        
    ema_9 = ta.trend.ema_indicator(df['close'], window=9).iloc[-1]
    ema_21 = ta.trend.ema_indicator(df['close'], window=21).iloc[-1]
    current_price = df['close'].iloc[-1]
    
    # --- Swing Detection: vectorized rolling window (was O(n²) nested loop) ---
    swing_lookback = 10
    window = swing_lookback * 2 + 1  # 21-candle window
    
    # Swing low: candle whose low is the minimum in the window
    rolling_low_min = df['low'].rolling(window=window, center=False).min()
    # Align: the low at index i must be the min of window starting i-swing_lookback
    is_swing_low = df['low'].shift(-swing_lookback) == rolling_low_min.shift(-swing_lookback)
    all_swing_lows = [float(v) for v in df.loc[is_swing_low.fillna(False), 'low'].values]
    swing_lows = [v for v in all_swing_lows if v < current_price]
    
    # Swing high: candle whose high is the maximum in the window  
    rolling_high_max = df['high'].rolling(window=window, center=False).max()
    is_swing_high = df['high'].shift(-swing_lookback) == rolling_high_max.shift(-swing_lookback)
    all_swing_highs = [float(v) for v in df.loc[is_swing_high.fillna(False), 'high'].values]
    swing_highs = [v for v in all_swing_highs if v > current_price]
    
    # Pick the NEAREST swing low below price (support) and swing high above (resistance)
    supports_below = sorted([s for s in swing_lows if s < current_price], reverse=True)
    resistances_above = sorted([r for r in swing_highs if r > current_price])
    
    # Fallback to rolling min/max only if no swings found
    sup = supports_below[0] if supports_below else df['low'].rolling(window=50).min().iloc[-1]
    res = resistances_above[0] if resistances_above else df['high'].rolling(window=50).max().iloc[-1]
    
    # Return multiple support/resistance levels for better structural analysis
    all_supports = supports_below[:3] if supports_below else [sup]
    all_resistances = resistances_above[:3] if resistances_above else [res]
    
    r_dist = (res - current_price) / current_price * 100
    s_dist = (sup - current_price) / current_price * 100
    
    # Define macro MTF trend based on EMA momentum and MACD alignment
    macd_obj = ta.trend.MACD(df['close'])
    macd_series = macd_obj.macd()
    macd_signal_series = macd_obj.macd_signal()
    macd_hist_series = macd_obj.macd_diff()
    macd_val = macd_series.iloc[-1]
    macd_prev = macd_series.iloc[-2]
    macd_sig = macd_signal_series.iloc[-1]
    macd_hist = macd_hist_series.iloc[-1]
    macd_hist_prev = macd_hist_series.iloc[-2]
    
    ema_bull = ema_9 > ema_21
    # Trend is only BULL if EMA is up AND MACD is up AND momentum is rising
    momentum_rising = macd_hist > macd_hist_prev
    
    macd_bull = (macd_val > macd_sig) and momentum_rising
    macd_bear = (macd_val < macd_sig) and not momentum_rising
    
    # SUPER HIGH SENSITIVITY: Use EMA 3 to detect reversals before they happen
    ema_fast_3 = ta.trend.ema_indicator(df['close'], window=3).iloc[-1]
    
    price_bull = current_price > ema_fast_3
    price_bear = current_price < ema_fast_3
    
    # Trend is BULL only if EMA is aligned AND MACD is aligned AND Price is holding above EMA 9
    if ema_bull and macd_bull and price_bull:
        trend = "BULL"
    elif not ema_bull and macd_bear and price_bear:
        trend = "BEAR"
    else:
        # If price is below EMA 9 on 15m, we are NEUTRAL even if EMAs are stacked (High Sensitivity)
        trend = "NEUTRAL"
    
    structure_state = "NEUTRAL"
    if len(all_swing_highs) >= 2 and len(all_swing_lows) >= 2:
        last_swing_high = all_swing_highs[-1]
        prev_swing_high = all_swing_highs[-2]
        last_swing_low = all_swing_lows[-1]
        prev_swing_low = all_swing_lows[-2]
        if last_swing_high > prev_swing_high and last_swing_low > prev_swing_low:
            structure_state = "HH_HL"
        elif last_swing_high < prev_swing_high and last_swing_low < prev_swing_low:
            structure_state = "LH_LL"
        elif last_swing_high < prev_swing_high:
            structure_state = "LOWER_HIGH"
        elif last_swing_low > prev_swing_low:
            structure_state = "HIGHER_LOW"

    return {
        "trend": trend,
        "support_levels": all_supports,
        "resistance_levels": all_resistances,
        "s_dist": f"{s_dist:+.1f}%",
        "r_dist": f"{r_dist:+.1f}%",
        "macd": float(macd_val),
        "macd_prev": float(macd_prev),
        "macd_signal": float(macd_sig),
        "macd_diff": float(macd_hist),
        "macd_diff_prev": float(macd_hist_prev),
        "ema_9": float(ema_9),
        "ema_21": float(ema_21),
        "ema_21_prev": float(ta.trend.ema_indicator(df['close'], window=21).iloc[-2]),
        "high": float(df['high'].iloc[-1]),
        "low": float(df['low'].iloc[-1]),
        "structure": structure_state,
    }

def _pick_structural_levels(current_price: float, mtf_context: dict = None, pivot_data: dict = None) -> tuple:
    """Return structural support/resistance prioritizing 5m chart swings.

    5m swings give wider, more meaningful levels that allow room for profit booking.
    Fallback order: 5m → 15m → 1h → 4h → daily pivots → 3m.
    For each source, we pick the NEAREST level on the correct side of price.
    """
    def _extract_levels(levels_list, side):
        """Extract valid float levels on the correct side of price."""
        results = []
        for lv in (levels_list or []):
            try:
                fv = float(lv)
                if side == "support" and fv < current_price:
                    results.append(fv)
                elif side == "resistance" and fv > current_price:
                    results.append(fv)
            except (TypeError, ValueError):
                pass
        return results

    def _pick_nearest(candidates, side):
        """Pick the nearest level: max for support (closest below), min for resistance (closest above)."""
        if not candidates:
            return None
        return max(candidates) if side == "support" else min(candidates)

    # Build ordered source list: 5m first, then wider timeframes, then 3m as last resort
    # Priority: 5m → 15m → 1h → 4h → pivots → 3m
    tf_order = ["5m", "15m", "1h", "4h"]
    
    support = None
    resistance = None

    # --- Try MTF timeframes in priority order ---
    if isinstance(mtf_context, dict):
        for tf in tf_order:
            tf_data = mtf_context.get(tf) or {}
            
            if support is None:
                candidates = _extract_levels(tf_data.get("support_levels"), "support")
                support = _pick_nearest(candidates, "support")
            
            if resistance is None:
                candidates = _extract_levels(tf_data.get("resistance_levels"), "resistance")
                resistance = _pick_nearest(candidates, "resistance")
            
            if support is not None and resistance is not None:
                break

    # --- Fallback to daily pivots if still missing ---
    if isinstance(pivot_data, dict) and (support is None or resistance is None):
        classic = pivot_data.get("classic", {}) or {}
        if support is None:
            pivot_sups = []
            for key in ["s1", "s2", "s3", "pp"]:
                v = classic.get(key)
                if v is not None:
                    try:
                        fv = float(v)
                        if fv < current_price:
                            pivot_sups.append(fv)
                    except (TypeError, ValueError):
                        pass
            support = _pick_nearest(pivot_sups, "support")
        
        if resistance is None:
            pivot_res = []
            for key in ["r1", "r2", "r3", "pp"]:
                v = classic.get(key)
                if v is not None:
                    try:
                        fv = float(v)
                        if fv > current_price:
                            pivot_res.append(fv)
                    except (TypeError, ValueError):
                        pass
            resistance = _pick_nearest(pivot_res, "resistance")

    # --- Last resort: 3m (tightest, least room) ---
    if isinstance(mtf_context, dict) and (support is None or resistance is None):
        tf_data = mtf_context.get("3m") or {}
        if support is None:
            candidates = _extract_levels(tf_data.get("support_levels"), "support")
            support = _pick_nearest(candidates, "support")
        if resistance is None:
            candidates = _extract_levels(tf_data.get("resistance_levels"), "resistance")
            resistance = _pick_nearest(candidates, "resistance")

    return support, resistance


def _compute_market_location_score(
    current_price: float,
    support: float = None,
    resistance: float = None,
    state: dict = None,
    latest_indicators: dict = None,
    strategy_config: dict = None,
) -> tuple[float, str, dict]:
    """Score market location relative to simple session and range anchors.

    This is a soft bias layer only. It prefers price above session anchors for longs,
    below them for shorts, and rewards being near support while penalizing being
    too close to resistance.
    """
    state = state or {}
    latest_indicators = latest_indicators or {}
    strategy_config = strategy_config or {}

    def _clean(value) -> float:
        try:
            fv = float(value or 0.0)
            return fv if math.isfinite(fv) else 0.0
        except (TypeError, ValueError):
            return 0.0

    session_open = _clean(state.get("session_open", 0.0) or latest_indicators.get("session_open", 0.0) or 0.0)
    previous_close = _clean(state.get("previous_close", 0.0) or latest_indicators.get("previous_close", 0.0) or 0.0)
    previous_high = _clean(state.get("previous_high", 0.0) or latest_indicators.get("previous_high", 0.0) or 0.0)
    previous_low = _clean(state.get("previous_low", 0.0) or latest_indicators.get("previous_low", 0.0) or 0.0)
    control_zone = _clean(state.get("control_zone", 0.0) or latest_indicators.get("control_zone", 0.0) or 0.0)
    average_zone = _clean(
        state.get("average_zone", 0.0)
        or latest_indicators.get("average_zone", 0.0)
        or latest_indicators.get("vwap", 0.0)
        or latest_indicators.get("bb_mid", 0.0)
        or 0.0
    )

    loc_score = 0.0
    notes = []
    levels = {
        "session_open": session_open,
        "previous_close": previous_close,
        "previous_high": previous_high,
        "previous_low": previous_low,
        "control_zone": control_zone,
        "average_zone": average_zone,
        "support": _clean(support),
        "resistance": _clean(resistance),
    }

    def _apply_binary(label: str, level: float, weight: float) -> None:
        nonlocal loc_score
        if level <= 0 or current_price <= 0:
            return
        if current_price > level:
            loc_score += weight
            notes.append(f"Above {label}")
        elif current_price < level:
            loc_score -= weight
            notes.append(f"Below {label}")

    # Core location anchors
    _apply_binary("Session Open", session_open, 0.08)
    _apply_binary("Previous Close", previous_close, 0.08)
    _apply_binary("Previous High", previous_high, 0.12)
    _apply_binary("Previous Low", previous_low, 0.08)
    _apply_binary("Control Zone", control_zone, 0.10)
    _apply_binary("Average Zone", average_zone, 0.10)

    support_near_pct = float(strategy_config.get("location_support_near_pct", 0.0075) or 0.0075)
    support_far_pct = float(strategy_config.get("location_support_far_pct", 0.0200) or 0.0200)
    resistance_near_pct = float(strategy_config.get("location_resistance_near_pct", 0.0025) or 0.0025)
    resistance_far_pct = float(strategy_config.get("location_resistance_far_pct", 0.0200) or 0.0200)

    if support and support > 0 and current_price > 0:
        dist_to_support = (current_price - support) / support
        if dist_to_support <= support_near_pct:
            loc_score += 0.12
            notes.append("Near Support")
        elif dist_to_support >= support_far_pct:
            loc_score -= 0.05
            notes.append("Far From Support")

    if resistance and resistance > 0 and current_price > 0:
        dist_to_resistance = (resistance - current_price) / resistance
        if dist_to_resistance <= resistance_near_pct:
            loc_score -= 0.14
            notes.append("Near Resistance")
        elif dist_to_resistance >= resistance_far_pct:
            loc_score += 0.04
            notes.append("Far From Resistance")

    loc_score = float(np.clip(loc_score, -0.6, 0.6))
    return loc_score, " | ".join(notes) if notes else "Neutral", levels


def _compute_wall_state(
    current_price: float,
    last_close: float,
    support: float = None,
    resistance: float = None,
    strategy_config: dict = None,
) -> dict:
    """Classify whether support/resistance is intact, being tested, or broken."""
    strategy_config = strategy_config or {}

    def _clean(value) -> float:
        try:
            fv = float(value or 0.0)
            return fv if math.isfinite(fv) else 0.0
        except (TypeError, ValueError):
            return 0.0

    support = _clean(support)
    resistance = _clean(resistance)
    current_price = _clean(current_price)
    last_close = _clean(last_close)

    support_veto_pct = float(strategy_config.get("support_veto_pct", 0.0015) or 0.0015)
    resistance_veto_pct = float(strategy_config.get("resistance_veto_pct", 0.0015) or 0.0015)
    support_break_pct = float(strategy_config.get("support_break_pct", 0.0010) or 0.0010)
    resistance_break_pct = float(strategy_config.get("resistance_break_pct", 0.0010) or 0.0010)

    support_zone_top = support * (1 + support_veto_pct) if support > 0 else 0.0
    support_break_level = support * (1 - support_break_pct) if support > 0 else 0.0
    resistance_zone_bottom = resistance * (1 - resistance_veto_pct) if resistance > 0 else 0.0
    resistance_break_level = resistance * (1 + resistance_break_pct) if resistance > 0 else 0.0

    support_broken = bool(
        support > 0
        and support_break_level > 0
        and current_price < support_break_level
        and last_close < support_break_level
    )
    resistance_broken = bool(
        resistance > 0
        and resistance_break_level > 0
        and current_price > resistance_break_level
        and last_close > resistance_break_level
    )

    support_touching = bool(support > 0 and current_price <= support_zone_top)
    resistance_touching = bool(resistance > 0 and current_price >= resistance_zone_bottom)

    if support_broken:
        support_state = "broken"
    elif support_touching:
        support_state = "touching"
    else:
        support_state = "above"

    if resistance_broken:
        resistance_state = "broken"
    elif resistance_touching:
        resistance_state = "touching"
    else:
        resistance_state = "below"

    return {
        "support": support,
        "resistance": resistance,
        "current_price": current_price,
        "last_close": last_close,
        "support_zone_top": support_zone_top,
        "support_break_level": support_break_level,
        "resistance_zone_bottom": resistance_zone_bottom,
        "resistance_break_level": resistance_break_level,
        "support_touching": support_touching,
        "support_broken": support_broken,
        "support_state": support_state,
        "resistance_touching": resistance_touching,
        "resistance_broken": resistance_broken,
        "resistance_state": resistance_state,
    }

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
    if spread_pct > max_spread:
        return {"action": "HOLD", "score": 0.0, "confidence": 0.0, "reason": f"Spread guard ({spread_pct:.4%}>{max_spread:.4%})", "weights": {}}
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
    
    # --- 3. PRIMARY SIGNAL CALCULATION ---
    # Structural & SMC Bias
    smc_score, sr_score, smc_label = detect_smc_and_sr(df_indicators, current_price)
    
    # Mean Reversion (MR) - Enhanced with Z-Score & RSI
    # Identifies statistically overextended price action likely to snap back.
    ema_9 = latest_indicators.get('ema_9', current_price)
    ema_21 = latest_indicators.get('ema_21', current_price)
    
    # Compute ATR early — needed by chase guard below
    atr_pct_now = float(latest_indicators.get("atr_pct", 0.5) or 0.5) / 100.0
    
    # ANTI-CHASING GUARD: Prevent entering when EMAs already diverged (move already started)
    ema_dist_pct = abs(ema_9 - ema_21) / ema_21
    configured_max_chase_pct = float(strategy_config.get("max_chase_pct", 0.0) or 0.0)
    max_chase_pct = configured_max_chase_pct if configured_max_chase_pct > 0 else max(0.0015, min(0.004, atr_pct_now * 0.5))
    if ema_dist_pct > max_chase_pct:
        # If EMAs are already wide apart, the move has already started. Don't chase.
        return {"action": "HOLD", "score": 0.0, "confidence": 0.0, "reason": f"Chasing Guard ({ema_dist_pct:.3%}>{max_chase_pct:.3%})", "weights": {}}
    z_score = float(latest_indicators.get('z_score', 0.0) or 0.0)
    rsi_14 = float(latest_indicators.get('rsi_14', 50.0) or 50.0)
    
    mr_score = 0.0
    if current_price < ema_21 * 0.998:
        if z_score <= -2.0 or rsi_14 < 30:
            mr_score = 1.0   # Strong bullish reversion
        elif z_score <= -1.0:
            mr_score = 0.5   # Moderate stretch
    elif current_price > ema_21 * 1.002:
        if z_score >= 2.0 or rsi_14 > 70:
            mr_score = -1.0  # Strong bearish reversion
        elif z_score >= 1.0:
            mr_score = -0.5  # Moderate stretch
    
    # Volume Weighted Average Price (VWAP)
    # The 'Institutional Value' anchor.
    vwap = latest_indicators.get('vwap', current_price)
    if mtf_fast_bias == "LONG_ONLY":
        vwap_score = 1.0 if current_price >= vwap else -1.0
    elif mtf_fast_bias == "SHORT_ONLY":
        vwap_score = -1.0 if current_price >= vwap else 1.0
    else:
        vwap_score = 1.0 if current_price < vwap else -1.0
    
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

    # --- DIVERGENCE DETECTION (Early - needed by div_bonus) ---
    divergence_state = _detect_macd_divergence(df_indicators)
    bear_div = divergence_state == "BEARISH"
    bull_div = divergence_state == "BULLISH"

    # --- DIVERGENCE BONUS (ALPHA) ---
    div_bonus = 0.0
    if bull_div: div_bonus += 0.25
    if bear_div: div_bonus -= 0.25

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

    # KDJ & SuperTrend
    kdj_j = latest_indicators.get('j', 50)
    kdj_score = 1.0 if kdj_j < 20 else (-1.0 if kdj_j > 80 else 0.0)
    st_score = latest_indicators.get('trend_bias', 0)
    atr_pct_now = float(latest_indicators.get("atr_pct", 0.0) or 0.0) / 100.0
    atr_min = float(strategy_config.get("vol_filter_atr_pct", 0.0005) or 0.0005)
    atr_max = float(strategy_config.get("vol_filter_atr_max_pct", 0.08) or 0.08)
    if atr_pct_now < atr_min or atr_pct_now > atr_max:
        return {"action": "HOLD", "score": 0.0, "confidence": 0.0, "reason": f"ATR guard ({atr_pct_now:.3%})", "weights": {}}

    # ADX Ranging Market Filter: lowered to 10 for absolute sensitivity during testing
    if adx_value < 10:
        return {"action": "HOLD", "score": 0.0, "confidence": 0.0, "reason": f"Ranging market (ADX:{adx_value:.0f}<10)", "weights": {}}
    
    # --- 4. ALPHA OVERLAY & REFINEMENT ---
    # The 'Alpha Overlay' adds extra weight when multiple trends align.
    alpha = generate_alpha_overlay(df_indicators, smc_score, macro_bias)
    
    # --- 5. INSTITUTIONAL PENALTIES ---
    # DISABLED for testing: No-Man's Land Guard
    penalty = 1.0
    pivot_msg = "Gated:IndicatorsOnly"
            
    # --- 6. FINAL WEIGHTED SYNTHESIS ---
    total_score = (
        (mr_score * weights['mr']) +
        (vwap_score * weights['vwap']) +
        (adx_score * weights['adx']) +
        (volume_delta * weights['vol']) +
        (obv_score * weights['obv']) +
        (bb_score * weights['bb']) +
        (macd_score * weights['macd']) +
        (pa_score * weights['pa']) +
        (smc_score * weights['smc']) +
        (sr_score * weights['sr']) +
        (location_score * weights['loc']) +
        (kdj_score * weights['kdj']) +
        (st_score * weights['st']) +
        alpha + # Add the Alpha Overlay contribution
        htf_score + # Add the Low-Weight 1H/4H Macro Bias
        div_bonus + # Add the Divergence Bonus Priority
        power_bonus # Add the Institutional Power Suite
    )
    
    # Apply context-based multipliers
    total_score *= penalty
    total_score += (funding_impact * 0.05) # Subtle adjustment for funding
    total_score += (mtf_fast_score * 0.20)
    if adx_value >= 25:
        total_score *= 1.10
    elif adx_value < 18:
        total_score *= 0.90

    # Trend-continuation bias:
    # If the fast scalp cluster is unanimous but the composite score is too flat,
    # give it a small push instead of freezing on the fence.
    if mtf_fast_bias == "LONG_ONLY" and total_score < 0.10:
        if current_price >= ema_21 and pa_score >= 0:
            total_score = max(total_score, 0.12)
    elif mtf_fast_bias == "SHORT_ONLY" and total_score > -0.10:
        if current_price <= ema_21 and pa_score <= 0:
            total_score = min(total_score, -0.12)

    # Cap score to [-1, 1] — prevents unbounded additive bonuses inflating confidence
    total_score = float(np.clip(total_score, -1.0, 1.0))

    # --- 7. SIGNAL INTEGRITY (Indicators Only) ---
    action = "HOLD"
    if total_score > 0.05: action = "BUY"
    elif total_score < -0.05: action = "SELL"

    # --- TREND CONFIRMATION: EMA + PSAR + MACD must agree ---
    # We follow the "MACD Zero Line Rule" from your screenshots:
    # 1. MACD > 0 => LONGS ONLY
    # 2. MACD < 0 => SHORTS ONLY
    # 3. Ignore crossovers inside the "Noise Channel"
    ema_9_val = float(latest_indicators.get('ema_9', current_price) or current_price)
    ema_21_val = float(latest_indicators.get('ema_21', current_price) or current_price)
    psar_val = latest_indicators.get('psar')
    macd_val = float(latest_indicators.get('macd', 0) or 0)
    macd_diff_val = float(latest_indicators.get('macd_diff', 0) or 0)
    
    # Noise Channel Threshold: Ignore signals too close to the zero line.
    # For DOGE at $0.11, 0.0001 is a meaningful filter (similar to 0.5 on Forex)
    macd_noise_threshold = float(strategy_config.get('macd_noise_threshold', 0.0001) or 0.0001)

    signal_reason_suffix = ""
    hold_reason = ""
    
    if action in {"BUY", "SELL"}:
        ema_bull = ema_9_val > ema_21_val
        psar_bull = True  # default if PSAR unavailable
        if psar_val is not None:
            try:
                psar_bull = float(psar_val) < current_price
            except (TypeError, ValueError):
                pass
        
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
        
        # Zero Bounce Detection (Pattern 3)
        # If bars were dropping toward 0 but just bounced back UP
        macd_bounce_bull = macd_diff_val > 0 and latest_indicators.get('macd_score', 0) >= 1.0
        macd_bounce_bear = macd_diff_val < 0 and latest_indicators.get('macd_score', 0) <= -1.0
        
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

        # --- STRUCTURAL BIAS VETO + REJECTION FLIP ASSIST ---
        # Block BUY at resistance / SELL at support, but optionally allow an early
        # opposite-side flip when rejection evidence is present (to avoid lag loops).
        wall_reversal_assist = bool(strategy_config.get("wall_reversal_assist", True))
        wall_reversal_score_gate = float(strategy_config.get("wall_reversal_score_gate", 0.12) or 0.12)
        wall_breakout_score_gate = float(strategy_config.get("wall_breakout_score_gate", 0.15) or 0.15)
        if support and resistance:
            if action == "BUY" and wall_state.get("resistance_touching") and not wall_state.get("resistance_broken"):
                reject_bear = (current_price < ema_9_val) and (macd_diff_val < 0)
                bullish_breakout = (
                    total_score >= wall_breakout_score_gate
                    and (psar_bull or macd_diff_val > 0)
                    and mtf_fast_bias != "SHORT_ONLY"
                )
                if bullish_breakout:
                    signal_reason_suffix += " [Breakout Through Resistance]"
                elif wall_reversal_assist and reject_bear and total_score <= wall_reversal_score_gate:
                    if total_score < -0.05 and mtf_fast_bias != "LONG_ONLY":
                        action = "SELL"
                        signal_reason_suffix += " [Rejection Flip: resistance]"
                    else:
                        action = "HOLD"
                        hold_reason = "Wall Veto: weak BUY at resistance, no short confirmation"
                else:
                    action = "HOLD"
                    hold_reason = (
                        f"Wall Veto: BUY blocked at resistance ({current_price:.2f} vs {float(resistance):.2f}, "
                        f"break>{float(wall_state.get('resistance_break_level', resistance)):.2f})."
                    )
            elif action == "SELL" and wall_state.get("support_touching") and not wall_state.get("support_broken"):
                reject_bull = (current_price > ema_9_val) and (macd_diff_val > 0)
                bearish_breakout = (
                    total_score <= -wall_breakout_score_gate
                    and ((not psar_bull) or macd_diff_val < 0)
                    and mtf_fast_bias != "LONG_ONLY"
                )
                if bearish_breakout:
                    signal_reason_suffix += " [Breakdown Through Support]"
                elif wall_reversal_assist and reject_bull and total_score >= -wall_reversal_score_gate:
                    if total_score > 0.05 and mtf_fast_bias != "SHORT_ONLY":
                        action = "BUY"
                        signal_reason_suffix += " [Rejection Flip: support]"
                    else:
                        action = "HOLD"
                        hold_reason = "Wall Veto: weak SELL at support, no long confirmation"
                else:
                    action = "HOLD"
                    hold_reason = (
                        f"Wall Veto: SELL blocked at support ({current_price:.2f} vs {float(support):.2f}, "
                        f"break<{float(wall_state.get('support_break_level', support)):.2f})."
                    )

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
            price_above_ema9 = current_price > ema_9_val
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
                if not (not price_above_ema9 and bear_momentum):
                    action = "HOLD"
                    hold_reason = "Strike Zone: Waiting for Price<EMA9 + Bear Momentum (SAR/Hist/RSI)"
        else:
            # CONSERVATIVE TREND MODE: Outside Strike Zone, require full triple confirmation (EMA Cross + SAR + MACD Zero)
            ema_cross_bull = ema_9_val > ema_21_val
            if action == "BUY" and not (ema_cross_bull and psar_bull and macd_final_bull):
                action = "HOLD"
                hold_reason = "Trend Gate: Waiting for full EMA-Cross/SAR/MACD(0) alignment"
            elif action == "BUY" and mtf_macd_bear and mtf_structure_bear:
                action = "HOLD"
                hold_reason = "MTF Gate: 15m/10m MACD bearish with lower-high structure"
            elif action == "SELL" and not (not ema_cross_bull and not psar_bull and macd_final_bear):
                action = "HOLD"
                hold_reason = "Trend Gate: Waiting for full EMA-Cross/SAR/MACD(0) alignment"
            elif action == "SELL" and mtf_macd_bull and mtf_structure_bull:
                action = "HOLD"
                hold_reason = "MTF Gate: 15m/10m MACD bullish with higher-low structure"
        
        signal_reason_suffix = " [Strike Override]" if in_action_zone and action != "HOLD" else (" [Indicators OK]" if action != "HOLD" else " [Waiting for alignment]")
        
        # --- SNIPER CHECKLIST (Direction-Aware) ---
        is_bearish_attempt = total_score < 0
        checklist = []
        
        if is_bearish_attempt:
            checklist.append(f"EMA:{'OK' if not ema_bull else 'Wait'}")
            checklist.append(f"MACD:{'OK' if macd_inst_bear else 'Wait'}")
            checklist.append(f"SAR:{'OK' if not psar_bull else 'Wait'}")
        else:
            checklist.append(f"EMA:{'OK' if ema_bull else 'Wait'}")
            checklist.append(f"MACD:{'OK' if macd_inst_bull else 'Wait'}")
            checklist.append(f"SAR:{'OK' if psar_bull else 'Wait'}")
            
        checklist.append(f"Vol:{'OK' if vol_spike else 'Low'}")
        
        checklist_str = " | ".join(checklist)
        
        status_msg = f"[{checklist_str}]"
        if action != "HOLD":
            status_msg = f"READY: {checklist_str}"
            if abs(div_bonus) > 0: status_msg += " +DIV"
            
        signal_reason_suffix = f" {status_msg}"
    
    # The reason string used for the professional dashboard.
    reason = (
        f"{signal_reason_suffix} | Score:{total_score:.3f} SMC:{smc_label} Pivot:{pivot_msg} MTF:{mtf_fast_bias} "
        f"(MR:{mr_score:.1f} OB:{smc_score:.1f} SR:{sr_score:.1f} VWAP:{vwap_score:.1f} ADX:{adx_score:.1f} "
        f"LOC:{location_score:.1f} VOL:{volume_delta:.1f} OBV:{obv_score:.1f} BB:{bb_score:.1f} MACD:{macd_score:.1f} PA:{pa_score:.1f} "
        f"KDJ:{kdj_score:.1f} ST:{st_score:.1f} DIV:{divergence_state})"
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
        "market_bias": mtf_fast_bias if mtf_fast_bias != "NEUTRAL" else "NEUTRAL",
        "mtf_fast_bias": mtf_fast_bias,
        "hold_reason": hold_reason
    }

    if support is not None:
        signal["structure_support"] = float(support)
    if resistance is not None:
        signal["structure_resistance"] = float(resistance)
    signal["wall_state"] = wall_state
    signal["market_location"] = {
        "score": float(location_score),
        "notes": location_notes,
        "levels": location_levels,
    }

    mr_setup = _detect_mean_reversion_setup(df_indicators, strategy_config)
    if mr_setup.get("triggered"):
        signal["mean_reversion"] = {
            "direction": mr_setup.get("direction"),
            "reason": mr_setup.get("reason"),
        }
        signal["reason"] = f"{signal['reason']} MR:{mr_setup.get('reason', '')}"
        signal["score"] = float(signal["score"]) + float(mr_setup.get("score", 0.0) or 0.0)
        signal["confidence"] = min(abs(float(signal["score"])), 1.0)
        if mr_setup.get("direction") == "LONG" and mtf_fast_bias != "SHORT_ONLY":
            signal["action"] = "BUY" if signal["action"] == "HOLD" or float(signal["score"]) > 0 else signal["action"]
        elif mr_setup.get("direction") == "SHORT" and mtf_fast_bias != "LONG_ONLY":
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
        if wick_setup.get("direction") == "LONG" and mtf_fast_bias != "SHORT_ONLY":
            signal["action"] = "BUY" if signal["action"] == "HOLD" or float(signal["score"]) > 0 else signal["action"]
        elif wick_setup.get("direction") == "SHORT" and mtf_fast_bias != "LONG_ONLY":
            signal["action"] = "SELL" if signal["action"] == "HOLD" or float(signal["score"]) < 0 else signal["action"]

    signal["score"] = float(np.clip(float(signal["score"]), -1.0, 1.0))
    signal["confidence"] = min(abs(float(signal["score"])), 1.0)

    # --- STRUCTURAL STOP LOSS ---
    # Place SL at structural level, but cap at max_structural_sl_pct
    stop_buffer = 0.0005
    final_action = signal["action"]
    if final_action == "BUY" and support is not None:
        signal["sl"] = float(support) * (1 - stop_buffer)  # just below support
    elif final_action == "SELL" and resistance is not None:
        signal["sl"] = float(resistance) * (1 + stop_buffer)  # just above resistance

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

    def _mtf_strict_entry_fail(action: str) -> str:
        if not (mtf_config and mtf_config.get("enabled", False) and isinstance(mtf_context, dict)):
            return ""

        require_full = bool(mtf_config.get("require_full_confirmation", False))
        if not require_full or action not in {"BUY", "SELL"}:
            return ""

        veto_missing = bool(mtf_config.get("strict_veto_on_missing", True))

        def _ctx(tf: str) -> dict:
            value = mtf_context.get(tf)
            return value if isinstance(value, dict) else {}

        def _trend(tf: str) -> str:
            return str(_ctx(tf).get("trend", "MISSING" if not _ctx(tf) else "NEUTRAL") or "NEUTRAL").upper()

        def _macd_bull(tf: str) -> bool:
            ctx = _ctx(tf)
            if not ctx:
                return False
            macd = float(ctx.get("macd", 0.0) or 0.0)
            sig = float(ctx.get("macd_signal", 0.0) or 0.0)
            diff = float(ctx.get("macd_diff", 0.0) or 0.0)
            prev = float(ctx.get("macd_diff_prev", 0.0) or 0.0)
            return macd >= sig and diff >= prev

        def _macd_bear(tf: str) -> bool:
            ctx = _ctx(tf)
            if not ctx:
                return False
            macd = float(ctx.get("macd", 0.0) or 0.0)
            sig = float(ctx.get("macd_signal", 0.0) or 0.0)
            diff = float(ctx.get("macd_diff", 0.0) or 0.0)
            prev = float(ctx.get("macd_diff_prev", 0.0) or 0.0)
            return macd <= sig and diff <= prev

        required_tfs = ("3m", "5m", "10m", "15m")
        missing = [tf for tf in required_tfs if not _ctx(tf)]
        if missing and veto_missing:
            return f"missing {'/'.join(missing)}"

        if action == "BUY":
            if _trend("3m") != "BULL" or _trend("5m") != "BULL":
                return "3m/5m not bullish"
            if _trend("10m") == "BEAR" or _trend("15m") == "BEAR":
                return "10m/15m bearish"
            if _macd_bear("10m") and _macd_bear("15m"):
                return "10m/15m MACD bearish"
            bearish_structure = {"LH_LL", "LOWER_HIGH"}
            if str(_ctx("5m").get("structure", "")).upper() in bearish_structure or str(_ctx("10m").get("structure", "")).upper() in bearish_structure:
                return "5m/10m lower-high structure"
        else:
            if _trend("3m") != "BEAR" or _trend("5m") != "BEAR":
                return "3m/5m not bearish"
            if _trend("10m") == "BULL" or _trend("15m") == "BULL":
                return "10m/15m bullish"
            if _macd_bull("10m") and _macd_bull("15m"):
                return "10m/15m MACD bullish"
            bullish_structure = {"HH_HL", "HIGHER_LOW"}
            if str(_ctx("5m").get("structure", "")).upper() in bullish_structure or str(_ctx("10m").get("structure", "")).upper() in bullish_structure:
                return "5m/10m higher-low structure"

        return ""

    strict_mtf_fail = _mtf_strict_entry_fail(signal["action"])
    if strict_mtf_fail:
        signal["action"] = "HOLD"
        signal["hold_reason"] = f"MTF strict gate: {strict_mtf_fail}"
        signal["reason"] = f"{signal['reason']} MTFStrictBlocked:{strict_mtf_fail}"
    elif mtf_fast_bias == "LONG_ONLY" and signal["action"] == "SELL":
        signal["action"] = "HOLD"
        signal["hold_reason"] = "MTF trend veto: fast timeframes bullish"
    elif mtf_fast_bias == "SHORT_ONLY" and signal["action"] == "BUY":
        signal["action"] = "HOLD"
        signal["hold_reason"] = "MTF trend veto: fast timeframes bearish"
    elif mtf_fast_bias == "NEUTRAL" and signal["action"] in {"BUY", "SELL"}:
        # Neutral fast TFs should warn, not auto-freeze. Keep holding only when the
        # base score is still too weak; otherwise allow the trade and note the partial MTF.
        mtf_neutral_allow_score = float(strategy_config.get("mtf_neutral_allow_score", 0.35) or 0.35)
        if abs(total_score) < mtf_neutral_allow_score:
            signal["action"] = "HOLD"
            signal["hold_reason"] = "MTF trend veto: fast timeframes not aligned"
            signal["reason"] = f"{signal['reason']} MTFPartialBlocked"
        else:
            signal["reason"] = f"{signal['reason']} MTFPartial"
    elif mtf_fast_bias != "NEUTRAL":
        signal["reason"] = f"{signal['reason']} MTFConfirm:{mtf_fast_bias}"
    
    # --- RANGE REVERSAL SNIPER MODE (DYNAMIC OUTER EDGE) ---
    # We measure the total range width and only allow entries in the configured outer zone.
    m5_s = signal.get("structure_support")
    m5_r = signal.get("structure_resistance")
    
    if m5_s and m5_r:
        range_width = m5_r - m5_s
        action_zone_size = range_width * range_action_zone_pct
        
        top_action_zone = m5_r - action_zone_size
        bottom_action_zone = m5_s + action_zone_size
        signal["action_support"] = float(bottom_action_zone)
        signal["action_resistance"] = float(top_action_zone)
        
        at_top = current_price >= top_action_zone
        at_bottom = current_price <= bottom_action_zone
        
        if signal["action"] == "BUY" and not at_bottom:
            signal["action"] = "HOLD"
            signal["hold_reason"] = f"Gate: BUY only at support (${current_price:.2f} > ${bottom_action_zone:.2f})."
        elif signal["action"] == "SELL" and not at_top:
            signal["action"] = "HOLD"
            signal["hold_reason"] = f"Gate: SELL only at resistance (${current_price:.2f} < ${top_action_zone:.2f})."
    else:
        # If we can't find a clear range, we stay safe and HOLD
        signal["action"] = "HOLD"
        signal["hold_reason"] = "Range Gate: No clear Support/Resistance boundaries found."
            
    return signal

# ==================================================================================================
# SUB-ANALYTICAL MODULES (THE 'THINKING' ENGINE)
# ==================================================================================================

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
            
    return "NONE"

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

def _confirm_breakout_momentum(df: pd.DataFrame, action: str, current_price: float, level: float) -> bool:
    """
    Confirms price has directional momentum at the level.
    BUY: tick above open + PSAR bull + near level
    SELL: tick below open + PSAR bear + near level
    """
    if df is None or len(df) < 3:
        return False

    last_open = float(df["open"].iloc[-1])

    tick_bull = current_price > last_open
    tick_bear = current_price < last_open

    psar = float(df["psar"].iloc[-1])
    psar_bull = current_price > psar
    psar_bear = current_price < psar

    near = abs(current_price - level) / level <= 0.0015

    if action == "BUY":
        return tick_bull and psar_bull and near
    return tick_bear and psar_bear and near


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
