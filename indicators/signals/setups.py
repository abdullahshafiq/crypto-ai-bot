"""Trade setup detectors: mean reversion, wick sweep, VWAP bounce, ORB breakout."""

import pandas as pd


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


def _detect_vwap_bounce_setup(
    df: pd.DataFrame,
    current_price: float,
    latest_indicators: dict = None,
    strategy_config: dict = None,
    anchored_vwap: float = None,
    mtf_fast_bias: str = "NEUTRAL",
) -> dict:
    """
    Detect a VWAP bounce/reclaim setup aligned with the article:
    - Trend context
    - Pullback to VWAP
    - Rejection candle
    - Volume confirmation
    """
    if df is None or len(df) < 5:
        return {"direction": "NEUTRAL", "score": 0.0, "reason": "", "sl": None, "triggered": False, "mode": "VWAP_BOUNCE"}

    cfg = strategy_config or {}
    if not bool(cfg.get("vwap_bounce_enabled", True)):
        return {"direction": "NEUTRAL", "score": 0.0, "reason": "", "sl": None, "triggered": False, "mode": "VWAP_BOUNCE"}

    last = df.iloc[-1]
    prev = df.iloc[-2]
    latest_indicators = latest_indicators or {}

    vwap_ref = float(anchored_vwap or 0.0)
    if vwap_ref <= 0:
        try:
            vwap_ref = float(last["vwap"]) if "vwap" in df.columns else 0.0
        except (TypeError, ValueError):
            vwap_ref = 0.0
    if vwap_ref <= 0 or current_price <= 0:
        return {"direction": "NEUTRAL", "score": 0.0, "reason": "", "sl": None, "triggered": False, "mode": "VWAP_BOUNCE"}

    near_pct = float(cfg.get("vwap_bounce_near_pct", 0.0015) or 0.0015)
    reclaim_pct = float(cfg.get("vwap_bounce_reclaim_pct", 0.0002) or 0.0002)
    vol_mult = float(cfg.get("vwap_bounce_vol_mult", 1.15) or 1.15)
    slope_ok = True
    try:
        if "ema_9" in df.columns and "ema_21" in df.columns:
            slope_ok = bool(float(last["ema_9"]) > float(last["ema_21"]))
    except Exception:
        slope_ok = True

    current_vol = float(last["volume"]) if "volume" in df.columns else 0.0
    avg_vol = float(df["volume"].rolling(window=min(20, len(df))).mean().iloc[-1]) if "volume" in df.columns else 0.0
    volume_ok = avg_vol > 0 and current_vol >= avg_vol * vol_mult
    candle_bull = float(last["close"]) >= float(last["open"])
    candle_bear = float(last["close"]) <= float(last["open"])
    price_vs_vwap = (current_price - vwap_ref) / vwap_ref
    recent = df.tail(3)

    long_touch = bool((recent["low"] <= vwap_ref * (1 + near_pct)).any()) if "low" in recent.columns else False
    short_touch = bool((recent["high"] >= vwap_ref * (1 - near_pct)).any()) if "high" in recent.columns else False

    long_confirmed = (
        price_vs_vwap >= -near_pct
        and long_touch
        and current_price > vwap_ref * (1 + reclaim_pct)
        and candle_bull
        and volume_ok
        and slope_ok
        and mtf_fast_bias != "SHORT_ONLY"
    )
    short_confirmed = (
        price_vs_vwap <= near_pct
        and short_touch
        and current_price < vwap_ref * (1 - reclaim_pct)
        and candle_bear
        and volume_ok
        and (not slope_ok or mtf_fast_bias != "LONG_ONLY")
    )

    if long_confirmed and not short_confirmed:
        sl = float(vwap_ref) * (1 - max(near_pct, 0.001))
        return {
            "triggered": True,
            "mode": "VWAP_BOUNCE",
            "direction": "LONG",
            "score": 0.30,
            "reason": f"VWAP bounce above {vwap_ref:.5f}",
            "sl": sl,
        }

    if short_confirmed and not long_confirmed:
        sl = float(vwap_ref) * (1 + max(near_pct, 0.001))
        return {
            "triggered": True,
            "mode": "VWAP_BOUNCE",
            "direction": "SHORT",
            "score": -0.30,
            "reason": f"VWAP rejection below {vwap_ref:.5f}",
            "sl": sl,
        }

    return {"direction": "NEUTRAL", "score": 0.0, "reason": "", "sl": None, "triggered": False, "mode": "VWAP_BOUNCE"}


def _detect_orb_breakout_setup(
    df: pd.DataFrame,
    current_price: float,
    latest_indicators: dict = None,
    strategy_config: dict = None,
    mtf_fast_bias: str = "NEUTRAL",
) -> dict:
    """
    Detect a rolling opening-range breakout setup.
    Uses the first N minutes of the current UTC day as the opening range.
    """
    if df is None or len(df) < 10:
        return {"direction": "NEUTRAL", "score": 0.0, "reason": "", "sl": None, "triggered": False, "mode": "ORB_BREAKOUT"}

    cfg = strategy_config or {}
    if not bool(cfg.get("orb_enabled", True)):
        return {"direction": "NEUTRAL", "score": 0.0, "reason": "", "sl": None, "triggered": False, "mode": "ORB_BREAKOUT"}

    if "timestamp" not in df.columns:
        return {"direction": "NEUTRAL", "score": 0.0, "reason": "", "sl": None, "triggered": False, "mode": "ORB_BREAKOUT"}

    latest_indicators = latest_indicators or {}
    last = df.iloc[-1]
    try:
        last_ts = pd.to_datetime(last["timestamp"], utc=True, errors="coerce")
    except Exception:
        last_ts = pd.NaT
    if pd.isna(last_ts):
        return {"direction": "NEUTRAL", "score": 0.0, "reason": "", "sl": None, "triggered": False, "mode": "ORB_BREAKOUT"}

    orb_minutes = int(cfg.get("orb_minutes", 15) or 15)
    orb_minutes = max(5, min(60, orb_minutes))
    breakout_buffer_pct = float(cfg.get("orb_breakout_buffer_pct", 0.0005) or 0.0005)
    volume_mult = float(cfg.get("orb_volume_mult", 1.25) or 1.25)

    day_start = last_ts.floor("D")
    day_mask = pd.to_datetime(df["timestamp"], utc=True, errors="coerce") >= day_start
    day_df = df.loc[day_mask].copy()
    if len(day_df) < 3:
        return {"direction": "NEUTRAL", "score": 0.0, "reason": "", "sl": None, "triggered": False, "mode": "ORB_BREAKOUT"}

    orb_end = day_start + pd.Timedelta(minutes=orb_minutes)
    orb_df = day_df[pd.to_datetime(day_df["timestamp"], utc=True, errors="coerce") < orb_end]
    if len(orb_df) < 2:
        return {"direction": "NEUTRAL", "score": 0.0, "reason": "", "sl": None, "triggered": False, "mode": "ORB_BREAKOUT"}

    orb_high = float(orb_df["high"].max())
    orb_low = float(orb_df["low"].min())
    if orb_high <= 0 or orb_low <= 0 or current_price <= 0:
        return {"direction": "NEUTRAL", "score": 0.0, "reason": "", "sl": None, "triggered": False, "mode": "ORB_BREAKOUT"}

    current_vol = float(last["volume"]) if "volume" in df.columns else 0.0
    avg_vol = float(day_df["volume"].rolling(window=min(20, len(day_df))).mean().iloc[-1]) if "volume" in df.columns else 0.0
    volume_ok = avg_vol > 0 and current_vol >= avg_vol * volume_mult
    candle_bull = float(last["close"]) >= float(last["open"])
    candle_bear = float(last["close"]) <= float(last["open"])
    macd_up = float(latest_indicators.get("macd_diff", 0.0) or 0.0) > 0
    macd_down = float(latest_indicators.get("macd_diff", 0.0) or 0.0) < 0

    breakout_long = (
        current_price >= orb_high * (1 + breakout_buffer_pct)
        and volume_ok
        and candle_bull
        and (mtf_fast_bias != "SHORT_ONLY")
        and (macd_up or float(last.get("ema_9", current_price)) > float(last.get("ema_21", current_price)))
    )
    breakout_short = (
        current_price <= orb_low * (1 - breakout_buffer_pct)
        and volume_ok
        and candle_bear
        and (mtf_fast_bias != "LONG_ONLY")
        and (macd_down or float(last.get("ema_9", current_price)) < float(last.get("ema_21", current_price)))
    )

    if breakout_long and not breakout_short:
        sl = orb_high * (1 - max(breakout_buffer_pct, 0.001))
        return {
            "triggered": True,
            "mode": "ORB_BREAKOUT",
            "direction": "LONG",
            "score": 0.40,
            "reason": f"ORB breakout above {orb_high:.5f}",
            "sl": sl,
        }

    if breakout_short and not breakout_long:
        sl = orb_low * (1 + max(breakout_buffer_pct, 0.001))
        return {
            "triggered": True,
            "mode": "ORB_BREAKOUT",
            "direction": "SHORT",
            "score": -0.40,
            "reason": f"ORB breakdown below {orb_low:.5f}",
            "sl": sl,
        }

    return {"direction": "NEUTRAL", "score": 0.0, "reason": "", "sl": None, "triggered": False, "mode": "ORB_BREAKOUT"}