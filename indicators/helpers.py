import pandas as pd
import numpy as np

from .calc import calculate_base_indicators
from .signals import generate_quant_signal

def _detect_momentum_exhaustion(df: pd.DataFrame) -> str:
    """Detect momentum deceleration before MACD fully flips."""
    if df is None or len(df) < 4:
        return "NONE"

    recent = df.iloc[-4:].copy()
    roc = recent["close"].pct_change()
    if roc.isna().sum() > 1:
        return "NONE"

    roc_vals = roc.iloc[-3:].fillna(0.0).tolist()
    if len(roc_vals) < 3:
        return "NONE"

    closes = recent["close"].tolist()
    price_up = closes[-1] > closes[-3]
    price_down = closes[-1] < closes[-3]

    bull_exhaust = price_up and roc_vals[-1] < roc_vals[-2] < roc_vals[-3]
    bear_exhaust = price_down and roc_vals[-1] > roc_vals[-2] > roc_vals[-3]

    if bull_exhaust:
        return "BULL_EXHAUST"
    if bear_exhaust:
        return "BEAR_EXHAUST"
    return "NONE"

def _body_range_ratio_score(df: pd.DataFrame, lookback: int = 3) -> float:
    """Signed conviction score from candle body-to-range efficiency."""
    if df is None or len(df) < max(lookback, 3):
        return 0.0

    recent = df.iloc[-lookback:].copy()
    ranges = (recent["high"] - recent["low"]).replace(0, np.nan)
    bodies = (recent["close"] - recent["open"]).abs()
    ratios = (bodies / (ranges + 1e-9)).replace([np.inf, -np.inf], np.nan).dropna()
    if ratios.empty:
        return 0.0

    avg_ratio = float(np.clip(ratios.mean(), 0.0, 1.0))
    net_move = float(recent["close"].iloc[-1] - recent["open"].iloc[0])
    if abs(net_move) < 1e-12:
        return 0.0

    direction = 1.0 if net_move > 0 else -1.0
    conviction = max(0.0, (avg_ratio - 0.30) / 0.40)
    if conviction <= 0.0:
        return 0.0
    return float(np.clip(direction * conviction, -1.0, 1.0))

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
