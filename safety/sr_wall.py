def sr_wall_escape_ready(
    action: str,
    sr_score: float,
    total_score: float,
    mtf_fast_bias: str,
    macd_diff_val: float,
    psar_bull: bool,
    current_price: float,
    ema_9_val: float,
    breakout_gate: float,
) -> bool:
    """Allow a hard-wall trade only when breakdown/breakout evidence is already strong."""
    action = str(action or "").upper()
    mtf_fast_bias = str(mtf_fast_bias or "").upper()
    breakout_gate = float(breakout_gate or 0.0)

    if action == "SELL" and sr_score >= 1.0:
        return (
            (mtf_fast_bias == "SHORT_ONLY" and total_score <= -breakout_gate)
            or (
                macd_diff_val < 0
                and (not psar_bull or current_price < ema_9_val)
            )
        )

    if action == "BUY" and sr_score <= -1.0:
        return (
            (mtf_fast_bias == "LONG_ONLY" and total_score >= breakout_gate)
            or (
                macd_diff_val > 0
                and (psar_bull or current_price > ema_9_val)
            )
        )

    return False
