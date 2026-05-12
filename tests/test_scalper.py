"""
Tests for the Ultimate Scalper strategy.
Verifies Volman Buildup, Brooks H2/L2, and Cameron Momentum logic.
"""

import pytest
import pandas as pd
import numpy as np
from indicators.signals.scalper import detect_best_scalper_signal

def _make_scalp_df(n_rows=50, trend="bull"):
    """Creates a synthetic DF optimized for scalping setups."""
    dates = pd.date_range("2025-01-01", periods=n_rows, freq="1min")
    
    if trend == "bull":
        # Price above EMA 50
        base = 100.0
        closes = base + np.linspace(0, 2, n_rows) # Steady uptrend
    else:
        base = 100.0
        closes = base - np.linspace(0, 2, n_rows) # Steady downtrend
        
    df = pd.DataFrame({
        "open": closes - 0.1,
        "high": closes + 0.2,
        "low": closes - 0.2,
        "close": closes,
        "volume": 1000.0,
    }, index=dates)
    
    # Add indicators
    df['ema_50'] = df['close'].rolling(50, min_periods=1).mean()
    df['ema_21'] = df['close'].rolling(21, min_periods=1).mean()
    df['ema_9'] = df['close'].rolling(9, min_periods=1).mean()
    df['atr'] = 0.1
    df['rsi_14'] = 60 if trend == "bull" else 40
    
    return df

def test_scalper_bull_setup():
    df = _make_scalp_df(trend="bull")
    # Simulate a buildup: tight range near EMA 21
    last_idx = df.index[-1]
    ema_21_last = df.loc[last_idx, 'ema_21']
    
    # Set last 5 bars to be very tight and near EMA 21
    for i in range(1, 6):
        idx = df.index[-i]
        df.loc[idx, 'close'] = ema_21_last + 0.01
        df.loc[idx, 'high'] = ema_21_last + 0.02
        df.loc[idx, 'low'] = ema_21_last - 0.01
        df.loc[idx, 'open'] = ema_21_last
        
    # Trigger H2: current close breaks previous high
    prev_high = df.iloc[-2]['high']
    df.loc[df.index[-1], 'close'] = prev_high + 0.05
    df.loc[df.index[-1], 'volume'] = 2000.0 # Vol spike
    
    result = detect_best_scalper_signal(df)
    assert result['triggered'] is True
    assert result['direction'] == "LONG"
    assert "Volman" in result['reason']

def test_scalper_bear_setup():
    df = _make_scalp_df(trend="bear")
    last_idx = df.index[-1]
    ema_21_last = df.loc[last_idx, 'ema_21']
    
    # Set last 5 bars to be very tight and near EMA 21
    for i in range(1, 6):
        idx = df.index[-i]
        df.loc[idx, 'close'] = ema_21_last - 0.01
        df.loc[idx, 'high'] = ema_21_last + 0.01
        df.loc[idx, 'low'] = ema_21_last - 0.02
        df.loc[idx, 'open'] = ema_21_last
        
    # Trigger L2: current close breaks previous low
    prev_low = df.iloc[-2]['low']
    df.loc[df.index[-1], 'close'] = prev_low - 0.05
    df.loc[df.index[-1], 'volume'] = 2000.0 # Vol spike
    
    result = detect_best_scalper_signal(df)
    assert result['triggered'] is True
    assert result['direction'] == "SHORT"
    assert "Brooks L2" in result['reason']
