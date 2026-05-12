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
    ema_200 = ta.trend.ema_indicator(df['close'], window=200).iloc[-1] if len(df) >= 200 else None
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

    price_bull = current_price > ema_9
    price_bear = current_price < ema_9

    # 2-of-3 vote: EMA stack, MACD momentum, price vs EMA9
    bull_votes = int(ema_bull) + int(macd_bull) + int(price_bull)
    bear_votes = int(not ema_bull) + int(macd_bear) + int(price_bear)

    if bull_votes >= 2:
        trend = "BULL"
    elif bear_votes >= 2:
        trend = "BEAR"
    else:
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

    _win20 = min(20, len(df))
    ema_200_bull = ema_200 is not None and current_price > ema_200
    ema_200_pct_dist = (current_price - ema_200) / ema_200 * 100 if ema_200 is not None else 0.0
    return {
        "trend": trend,
        "support_levels": all_supports,
        "resistance_levels": all_resistances,
        "recent_low_20": float(df['low'].rolling(window=_win20).min().iloc[-1]),
        "recent_high_20": float(df['high'].rolling(window=_win20).max().iloc[-1]),
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
        "ema_200": float(ema_200) if ema_200 is not None else 0.0,
        "ema_200_bull": ema_200_bull,
        "ema_200_pct_dist": float(ema_200_pct_dist),
        "high": float(df['high'].iloc[-1]),
        "low": float(df['low'].iloc[-1]),
        "structure": structure_state,
    }

def _tighten_level(current_price: float, confirmed, candidate, is_support: bool, max_sl_pct: float):
    """Return candidate if swing is too far and candidate is closer. Otherwise None."""
    if confirmed is None or candidate is None:
        return None
    try:
        cand = float(candidate)
        if is_support:
            if cand >= current_price:
                return None
            confirmed_d = (current_price - float(confirmed)) / current_price
            cand_d = (current_price - cand) / current_price
        else:
            if cand <= current_price:
                return None
            confirmed_d = (float(confirmed) - current_price) / current_price
            cand_d = (cand - current_price) / current_price
        if confirmed_d > max_sl_pct and 0 < cand_d < confirmed_d:
            return cand
    except (TypeError, ValueError):
        pass
    return None


def _pick_structural_levels(current_price: float, mtf_context: dict = None, pivot_data: dict = None, max_sl_pct: float = 0.012) -> tuple:
    """Return structural support/resistance prioritizing the strongest valid swing levels.

    We keep only levels on the correct side of price, reject stale levels outside the
    structural cap, and then choose the deepest valid support / highest valid resistance.
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

    def _within_cap(level: float) -> bool:
        if level is None or current_price <= 0:
            return False
        try:
            return abs(float(level) - current_price) / current_price <= max_sl_pct
        except (TypeError, ValueError):
            return False

    def _pick_strongest(candidates, side):
        """Pick the strongest valid level: deepest support or highest resistance."""
        valid = [float(v) for v in candidates if _within_cap(v)]
        if not valid:
            return None
        return min(valid) if side == "support" else max(valid)

    # Build a candidate pool from MTF swings, recent extremes, pivots, and 3m fallback.
    tf_order = ["5m", "15m", "1h", "4h"]

    mtf_support_candidates = []
    mtf_resistance_candidates = []
    pivot_support_candidates = []
    pivot_resistance_candidates = []
    fallback_support_candidates = []
    fallback_resistance_candidates = []

    # --- Collect MTF candidates ---
    if isinstance(mtf_context, dict):
        for tf in tf_order:
            tf_data = mtf_context.get(tf) or {}
            mtf_support_candidates.extend(_extract_levels(tf_data.get("support_levels"), "support"))
            mtf_resistance_candidates.extend(_extract_levels(tf_data.get("resistance_levels"), "resistance"))
            mtf_support_candidates.extend(_extract_levels([tf_data.get("recent_low_20")], "support"))
            mtf_resistance_candidates.extend(_extract_levels([tf_data.get("recent_high_20")], "resistance"))

    # --- Add daily pivots if available ---
    if isinstance(pivot_data, dict):
        classic = pivot_data.get("classic", {}) or {}
        for key in ["s1", "s2", "s3", "pp"]:
            pivot_support_candidates.extend(_extract_levels([classic.get(key)], "support"))
        for key in ["r1", "r2", "r3", "pp"]:
            pivot_resistance_candidates.extend(_extract_levels([classic.get(key)], "resistance"))

    # --- Last resort: 3m (tightest, least room) ---
    if isinstance(mtf_context, dict):
        tf_data = mtf_context.get("3m") or {}
        fallback_support_candidates.extend(_extract_levels(tf_data.get("support_levels"), "support"))
        fallback_resistance_candidates.extend(_extract_levels(tf_data.get("resistance_levels"), "resistance"))

    support = _pick_strongest(mtf_support_candidates, "support")
    if support is None:
        support = _pick_strongest(pivot_support_candidates, "support")
    if support is None:
        support = _pick_strongest(fallback_support_candidates, "support")

    resistance = _pick_strongest(mtf_resistance_candidates, "resistance")
    if resistance is None:
        resistance = _pick_strongest(pivot_resistance_candidates, "resistance")
    if resistance is None:
        resistance = _pick_strongest(fallback_resistance_candidates, "resistance")

    if support is None:
        support = current_price * (1 - max_sl_pct * 0.8)

    if resistance is None:
        resistance = current_price * (1 + max_sl_pct * 0.8)

    return support, resistance
