"""Divergence detection: MACD divergence (regular + hidden) and CVD momentum divergence."""

import pandas as pd
from ..location import _detect_cvd_divergence


def _detect_macd_divergence(df: pd.DataFrame) -> str:
    """
    MACD Divergence Engine.
    Detects when Price and Momentum are moving in opposite directions.
    - Bearish Divergence: Higher Highs in Price + Lower Highs in MACD
    - Bullish Divergence: Lower Lows in Price + Higher Lows in MACD
    """
    if len(df) < 50:
        return "NONE"

    data = df.iloc[-40:].copy()

    highs = []
    macd_peaks = []

    for i in range(2, len(data) - 2):
        if data['high'].iloc[i] > data['high'].iloc[i - 1] and data['high'].iloc[i] > data['high'].iloc[i + 1]:
            highs.append((i, data['high'].iloc[i]))
            macd_peaks.append(data['macd'].iloc[i])

    if len(highs) >= 2:
        p1_idx, p1_price = highs[-2]
        p2_idx, p2_price = highs[-1]
        m1 = macd_peaks[-2]
        m2 = macd_peaks[-1]
        if p2_price > p1_price and m2 < m1 and (p2_idx - p1_idx) > 3:
            return "BEARISH"
        if p2_price < p1_price and m2 > m1 and (p2_idx - p1_idx) > 3:
            return "HIDDEN_BEARISH"

    lows = []
    macd_troughs = []

    for i in range(2, len(data) - 2):
        if data['low'].iloc[i] < data['low'].iloc[i - 1] and data['low'].iloc[i] < data['low'].iloc[i + 1]:
            lows.append((i, data['low'].iloc[i]))
            macd_troughs.append(data['macd'].iloc[i])

    if len(lows) >= 2:
        p1_idx, p1_price = lows[-2]
        p2_idx, p2_price = lows[-1]
        m1 = macd_troughs[-2]
        m2 = macd_troughs[-1]
        if p2_price < p1_price and m2 > m1 and (p2_idx - p1_idx) > 3:
            return "BULLISH"
        if p2_price > p1_price and m2 < m1 and (p2_idx - p1_idx) > 3:
            return "HIDDEN_BULLISH"

    return "NONE"


def _detect_cvd_momentum_divergence(df: pd.DataFrame) -> tuple[str, float]:
    """Detect simple price/CVD divergence using OHLCV approximation."""
    return _detect_cvd_divergence(df)