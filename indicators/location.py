import pandas as pd
import numpy as np
import math

from .calc import TUNING

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

def _compute_vpoc(df: pd.DataFrame, lookback: int = 50) -> float:
    """Approximate a VPOC from OHLCV midpoint weighted by volume."""
    if df is None or len(df) < 3:
        return 0.0

    recent = df.iloc[-lookback:].copy()
    midpoints = (recent["high"] + recent["low"]) / 2.0
    volumes = recent["volume"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if float(volumes.sum()) <= 0:
        return float(midpoints.iloc[-1])

    try:
        return float(np.average(midpoints.fillna(method="ffill").fillna(method="bfill"), weights=volumes))
    except Exception:
        return float(midpoints.iloc[-1])

def _compute_anchored_vwap(df: pd.DataFrame) -> float:
    """Approximate a session-anchored VWAP using the current UTC day if available."""
    if df is None or len(df) < 2:
        return 0.0

    recent = df.copy()
    if "timestamp" in recent.columns:
        try:
            ts = pd.to_datetime(recent["timestamp"], utc=True, errors="coerce")
            if ts.notna().any():
                day_start = ts.dt.floor("D").iloc[-1]
                day_mask = ts >= day_start
                day_df = recent.loc[day_mask].copy()
                if len(day_df) >= 2:
                    recent = day_df
        except Exception:
            pass

    typical = (recent["high"] + recent["low"] + recent["close"]) / 3.0
    vol = recent["volume"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    vol_sum = float(vol.sum())
    if vol_sum <= 0:
        return float(typical.iloc[-1])
    try:
        return float(np.average(typical.fillna(method="ffill").fillna(method="bfill"), weights=vol))
    except Exception:
        return float(typical.iloc[-1])

def _compute_cvd(df: pd.DataFrame) -> pd.Series:
    """Approximate cumulative volume delta from OHLCV candles."""
    if df is None or len(df) < 2:
        return pd.Series(dtype=float)

    rng = (df["high"] - df["low"]).replace(0, np.nan)
    buy_vol = (((df["close"] - df["low"]) / (rng + 1e-9)).clip(0.0, 1.0) * df["volume"].fillna(0.0)).fillna(0.0)
    sell_vol = (((df["high"] - df["close"]) / (rng + 1e-9)).clip(0.0, 1.0) * df["volume"].fillna(0.0)).fillna(0.0)
    cvd = (buy_vol - sell_vol).cumsum()
    return cvd

def _detect_cvd_divergence(df: pd.DataFrame) -> tuple[str, float]:
    """Detect simple CVD divergence against price swings."""
    if df is None or len(df) < 10:
        return "NONE", 0.0

    cvd = _compute_cvd(df)
    if cvd.empty or len(cvd) < 6:
        return "NONE", 0.0

    price_now = float(df["close"].iloc[-1])
    price_prev = float(df["close"].iloc[-5])
    cvd_now = float(cvd.iloc[-1])
    cvd_prev = float(cvd.iloc[-5])

    if price_now > price_prev and cvd_now < cvd_prev:
        return "BEARISH", -0.18
    if price_now < price_prev and cvd_now > cvd_prev:
        return "BULLISH", 0.18
    return "NONE", 0.0
def _compute_market_location_score(
    current_price: float,
    support: float = None,
    resistance: float = None,
    vpoc: float = None,
    anchored_vwap: float = None,
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
        "vpoc": _clean(vpoc),
        "anchored_vwap": _clean(anchored_vwap),
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

    vpoc = _clean(vpoc)
    vpoc_near_pct = float(strategy_config.get("vpoc_near_pct", 0.0010) or 0.0010)
    vpoc_break_pct = float(strategy_config.get("vpoc_break_pct", 0.0015) or 0.0015)
    if vpoc > 0 and current_price > 0:
        dist_to_vpoc = abs(current_price - vpoc) / vpoc
        if dist_to_vpoc <= vpoc_near_pct:
            loc_score -= 0.08
            notes.append("Near VPOC")
        elif current_price > vpoc and (current_price - vpoc) / vpoc >= vpoc_break_pct:
            loc_score += 0.05
            notes.append("Above VPOC")
        elif current_price < vpoc and (vpoc - current_price) / vpoc >= vpoc_break_pct:
            loc_score -= 0.05
            notes.append("Below VPOC")

    anchored_vwap = _clean(anchored_vwap)
    avwap_near_pct = float(strategy_config.get("anchored_vwap_near_pct", 0.0010) or 0.0010)
    if anchored_vwap > 0 and current_price > 0:
        dist_to_avwap = abs(current_price - anchored_vwap) / anchored_vwap
        if dist_to_avwap <= avwap_near_pct:
            loc_score -= 0.05
            notes.append("Near Anchored VWAP")
        elif current_price > anchored_vwap:
            loc_score += 0.03
            notes.append("Above Anchored VWAP")
        else:
            loc_score -= 0.03
            notes.append("Below Anchored VWAP")

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
