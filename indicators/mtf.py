import pandas as pd
import ta

def build_mtf_timeframe_context(df: pd.DataFrame) -> dict:
    """
    Analyzes a timeframe for the dashboard.
    Uses swing detection (local highs/lows) to find NEAREST structural S/R
    instead of rolling window absolute min/max which misses recent pivots.
    """
    if df is None or len(df) < 50:
        return {
            "trend": "NEUT",
            "support_levels": [],
            "resistance_levels": [],
            "s_dist": "-",
            "r_dist": "-",
            "rsi_14": 50.0,
            "rsi_14_prev": 50.0,
            "macd": 0.0,
            "macd_prev": 0.0,
            "macd_signal": 0.0,
            "macd_diff": 0.0,
            "macd_diff_prev": 0.0,
            "ema_9": 0.0,
            "ema_21": 0.0,
            "ema_21_prev": 0.0,
            "high": 0.0,
            "low": 0.0,
            "structure": "NEUTRAL",
        }

    ema_9 = ta.trend.ema_indicator(df['close'], window=9).iloc[-1]
    ema_21 = ta.trend.ema_indicator(df['close'], window=21).iloc[-1]
    rsi_series = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
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
        "rsi_14": float(rsi_series.iloc[-1]),
        "rsi_14_prev": float(rsi_series.iloc[-2]),
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
