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

from safety import sr_wall_escape_ready as _sr_wall_escape_ready

logger = logging.getLogger(__name__)

# Learned weights written by ml/optimizer.py
WEIGHTS_FILE = os.path.join(os.path.dirname(__file__), '..', 'ml', 'weights.json')

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
    'ob': 0.14,    # Order Book Pressure
    'vwap': 0.12,  # Volume Weighted Average Price
    'adx': 0.08,   # Trend Strength
    'vol': 0.08,   # Volume Delta
    'obv': 0.05,   # Accumulation / distribution flow
    'bb': 0.03,    # Bollinger Band Exhaustion
    'macd': 0.18,  # Momentum Authority
    'pa': 0.20,    # Price Action / SAR Authority
    'smc': 0.05,   # Market Structure
    'sr': 0.02,    # Support/Resistance Walls
    'loc': 0.08,   # Market location / session context
    'kdj': 0.03,   # Stochastic Momentum
    'st': 0.22,    # EMA Trend Authority
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
