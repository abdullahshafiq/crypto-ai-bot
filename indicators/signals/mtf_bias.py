"""Multi-timeframe bias computation: fast trend vote (3m/5m/10m/15m) and RSI consensus."""

def compute_mtf_bias(mtf_config: dict, mtf_context: dict, strategy_config: dict) -> dict:
    """
    Compute MTF fast bias (3m/5m/10m/15m trend vote) and MTF RSI bias.

    Returns dict with:
        mtf_fast_score, mtf_fast_bias, mtf_rsi_score, mtf_rsi_bias
    """
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
            tf_ctx = mtf_context.get(tf)
            if not isinstance(tf_ctx, dict):
                continue
            trend = str(tf_ctx.get("trend", "NEUT") or "NEUT").upper()
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

    mtf_rsi_score = 0.0
    mtf_rsi_bias = "NEUTRAL"
    if mtf_config and mtf_config.get("enabled", False) and isinstance(mtf_context, dict):
        rsi_bull_level = float(strategy_config.get("mtf_rsi_bull_level", 55) or 55)
        rsi_bear_level = float(strategy_config.get("mtf_rsi_bear_level", 45) or 45)
        rsi_min_agree = int(strategy_config.get("mtf_rsi_min_agree", 2) or 2)
        rsi_min_agree = max(1, min(4, rsi_min_agree))
        rsi_checks = [
            ("3m", 0.30),
            ("5m", 0.25),
            ("15m", 0.25),
            ("1h", 0.20),
        ]
        bull_votes = 0
        bear_votes = 0
        for tf, weight in rsi_checks:
            tf_ctx = mtf_context.get(tf)
            if not isinstance(tf_ctx, dict):
                continue
            try:
                rsi_val = float(tf_ctx.get("rsi_14", 50.0) or 50.0)
            except (TypeError, ValueError):
                continue
            if rsi_val >= rsi_bull_level:
                mtf_rsi_score += weight
                bull_votes += 1
            elif rsi_val <= rsi_bear_level:
                mtf_rsi_score -= weight
                bear_votes += 1
        if bull_votes >= rsi_min_agree:
            mtf_rsi_bias = "BULLISH"
        elif bear_votes >= rsi_min_agree:
            mtf_rsi_bias = "BEARISH"

    return {
        "mtf_fast_score": mtf_fast_score,
        "mtf_fast_bias": mtf_fast_bias,
        "mtf_rsi_score": mtf_rsi_score,
        "mtf_rsi_bias": mtf_rsi_bias,
    }