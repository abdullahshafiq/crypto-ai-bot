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
    break_confirmed: bool = False,
) -> bool:
    """Allow a hard-wall trade only when breakdown/breakout evidence is already very strong."""
    action = str(action or "").upper()
    mtf_fast_bias = str(mtf_fast_bias or "").upper()
    breakout_gate = float(breakout_gate or 0.15)

    # Require strong score and MTF alignment for any 'escape' through a wall
    strong_score = abs(total_score) >= max(0.25, breakout_gate * 1.5)

    if action == "SELL" and sr_score >= 1.0:
        # SHORT into support: require actual support break + SHORT_ONLY MTF + strong score + momentum
        return (
            break_confirmed
            and mtf_fast_bias == "SHORT_ONLY"
            and strong_score
            and macd_diff_val < 0
            and (not psar_bull or current_price < ema_9_val)
        )

    if action == "BUY" and sr_score <= -1.0:
        # BUY into resistance: require actual resistance break + LONG_ONLY MTF + strong score + momentum
        return (
            break_confirmed
            and mtf_fast_bias == "LONG_ONLY"
            and strong_score
            and macd_diff_val > 0
            and (psar_bull or current_price > ema_9_val)
        )

    return False
