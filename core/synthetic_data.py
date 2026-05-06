import pandas as pd
import time


def _fallback_bootstrap_ohlcv(symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
    """
    Offline demo fallback used only when Binance market data is unreachable.
    Generates a minimal synthetic OHLCV history around a stable anchor so the
    paper executor can warm up and run locally.
    """
    limit = max(20, int(limit or 100))
    anchor = 9.10 if "AVAX" in str(symbol).upper() else 100.0
    if "BTC" in str(symbol).upper():
        anchor = 65000.0
    now = pd.Timestamp.utcnow().floor("min")
    try:
        if str(timeframe).endswith("m"):
            delta = pd.Timedelta(minutes=int(str(timeframe)[:-1] or 1))
        elif str(timeframe).endswith("h"):
            delta = pd.Timedelta(hours=int(str(timeframe)[:-1] or 1))
        else:
            delta = pd.Timedelta(minutes=1)
    except Exception:
        delta = pd.Timedelta(minutes=1)
    rows = []
    price = float(anchor)
    for i in range(limit):
        ts = now - delta * (limit - i)
        drift = ((i % 7) - 3) * (anchor * 0.0002)
        open_p = price
        close_p = max(0.0001, price + drift)
        high_p = max(open_p, close_p) * 1.0008
        low_p = min(open_p, close_p) * 0.9992
        vol = 1000.0 + (i % 10) * 25.0
        rows.append({
            "timestamp": ts,
            "open": float(open_p),
            "high": float(high_p),
            "low": float(low_p),
            "close": float(close_p),
            "volume": float(vol),
        })
        price = close_p
    return pd.DataFrame(rows)
