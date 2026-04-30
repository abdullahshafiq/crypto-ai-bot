import ccxt
import pandas as pd
import time
import logging
import re

logger = logging.getLogger(__name__)


class MarketData:
    def __init__(self, market: str = "usdm"):
        """
        market:
          - "usdm": Binance USDⓈ-M futures (perps) public endpoints (recommended for this bot)
          - "spot": Binance spot public endpoints
        """
        market = str(market or "usdm").strip().lower()

        # Use public endpoints for market data (demo-fapi blocks many public endpoints).
        if market in {"usdm", "futures", "future"}:
            self.exchange = ccxt.binanceusdm({
                'enableRateLimit': True,
                'options': {
                    'adjustForTimeDifference': True,
                }
            })
            self.market = "usdm"
        else:
            self.exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {
                    'adjustForTimeDifference': True,
                }
            })
            self.market = "spot"

        # Internal state for tracking real price ticks
        self.tick_history = []       # [(timestamp, price)]
        self.volume_history = []     # [(timestamp, volume)]
        self.last_price = None
        self.ob_depth_levels = 10    # Aggregate top N levels for imbalance

    def normalize_symbol(self, symbol: str) -> str:
        symbol = str(symbol or "").strip()
        if not symbol:
            return symbol
        if self.market == "usdm" and ":" not in symbol:
            # CCXT futures symbols use the settlement asset suffix, e.g. BTC/USDT:USDT.
            if "/" in symbol:
                base, quote = symbol.split("/", 1)
                if ":" not in quote:
                    quote = quote.split(":")[0]
                return f"{base}/{quote}:USDT"
        return symbol

    def _resample_ohlcv(self, df: pd.DataFrame, rule: str) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()

        d = df.copy()
        d = d.sort_values("timestamp")
        d = d.set_index("timestamp")

        try:
            rs = d.resample(rule, origin="epoch", label="right", closed="right")
        except TypeError:
            # Older pandas without `origin`
            rs = d.resample(rule, label="right", closed="right")

        agg = rs.agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        })
        agg = agg.dropna(subset=["open", "high", "low", "close"])
        agg = agg.reset_index()
        return agg

    def fetch_ohlcv(self, symbol: str = 'BTC/USDT', timeframe: str = '1h', limit: int = 100) -> pd.DataFrame:
        """Fetches historical OHLCV data with error handling and retry."""
        symbol = self.normalize_symbol(symbol)
        timeframe = str(timeframe or "1h").strip()
        limit = int(limit or 100)
        if limit < 10:
            limit = 10

        supported = getattr(self.exchange, "timeframes", None) or {}

        def _fetch(tf: str, lim: int, attempts: int = 3) -> pd.DataFrame:
            for attempt in range(attempts):
                try:
                    ohlcv = self.exchange.fetch_ohlcv(symbol, tf, limit=lim)
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    return df
                except ccxt.RateLimitExceeded as e:
                    logger.warning(f"Rate limited fetching OHLCV ({tf}), backing off: {e}")
                    time.sleep(2 + attempt * 2)
                except ccxt.NetworkError as e:
                    logger.error(f"Network error fetching OHLCV ({tf}): {e}")
                    time.sleep(1 + attempt * 2)
                except ccxt.ExchangeError as e:
                    logger.error(f"Exchange error fetching OHLCV ({tf}): {e}")
                    break
                except Exception as e:
                    logger.error(f"Unexpected error fetching OHLCV ({tf}): {e}")
                    break
            return pd.DataFrame()

        # 1) Direct fetch for supported or unknown timeframes
        if (not supported) or (timeframe in supported):
            df = _fetch(timeframe, limit)
            if not df.empty:
                return df
            # If the exchange claims it supports it but we still couldn't fetch, don't try to resample.
            if supported and timeframe in supported:
                return pd.DataFrame()

        # 2) Local resample for custom intervals (e.g., 3h)
        m = re.fullmatch(r"(\d+)\s*([mhdwM])", timeframe)
        if not m:
            logger.error(f"Unsupported timeframe format: {timeframe}")
            return pd.DataFrame()

        n = int(m.group(1))
        unit = m.group(2)
        if n <= 1:
            logger.error(f"Unsupported timeframe: {timeframe}")
            return pd.DataFrame()

        if unit == "h":
            base_tf = "1h"
            if supported and base_tf not in supported:
                logger.error(f"Cannot resample {timeframe}: base {base_tf} not supported")
                return pd.DataFrame()
            fetch_limit = min(1000, (limit * n) + 10)
            base = _fetch(base_tf, fetch_limit)
            if base.empty:
                return pd.DataFrame()
            res = self._resample_ohlcv(base, f"{n}h")
            return res.tail(limit).reset_index(drop=True)

        if unit == "m":
            base_tf = "1m"
            if supported and base_tf not in supported:
                logger.error(f"Cannot resample {timeframe}: base {base_tf} not supported")
                return pd.DataFrame()
            fetch_limit = min(1500, (limit * n) + 30)
            base = _fetch(base_tf, fetch_limit)
            if base.empty:
                return pd.DataFrame()
            res = self._resample_ohlcv(base, f"{n}min")
            return res.tail(limit).reset_index(drop=True)

        logger.error(f"Unsupported timeframe (no resample strategy): {timeframe}")
        return pd.DataFrame()

    def fetch_funding_rate(self, symbol: str = 'BTC/USDT') -> float:
        """Fetches the funding rate from binance futures."""
        try:
            symbol = self.normalize_symbol(symbol)
            fapi = ccxt.binanceusdm()
            fr = fapi.fetch_funding_rate(symbol)
            return float(fr.get('fundingRate', 0.0))
        except Exception as e:
            logger.error(f"Failed to fetch funding rate: {e}")
            return 0.0

    def fetch_order_book_and_ticks(self, symbol: str = 'BTC/USDT') -> dict | None:
        """
        Fetches the order book and tracks real price ticks for micro-timeframe
        return calculations. Returns None on API failure.
        """
        try:
            symbol = self.normalize_symbol(symbol)
            ob = self.exchange.fetch_order_book(symbol, limit=self.ob_depth_levels)
            ticker = self.exchange.fetch_ticker(symbol)
        except ccxt.RateLimitExceeded as e:
            logger.warning(f"Rate limited, backing off: {e}")
            time.sleep(5)
            return None
        except ccxt.NetworkError as e:
            logger.error(f"Network error: {e}")
            return None
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching market state: {e}")
            return None

        current_price = ticker['last']
        current_time = time.time()

        # Best bid/ask from level 1
        bid = ob['bids'][0][0] if ob['bids'] else current_price
        ask = ob['asks'][0][0] if ob['asks'] else current_price
        spread_pct = (ask - bid) / bid if bid > 0 else 0

        # FIX #2: Aggregate top N levels of depth for stable imbalance
        bid_vol = sum(level[1] for level in ob['bids'][:self.ob_depth_levels]) if ob['bids'] else 0.0
        ask_vol = sum(level[1] for level in ob['asks'][:self.ob_depth_levels]) if ob['asks'] else 0.0

        # FIX #8: Use actual exchange price, not random noise
        # Track real ticker prices for accurate ret_5s / ret_30s
        self.tick_history.append((current_time, current_price))

        # Track volume from ticker (base volume in last trade)
        tick_volume = ticker.get('quoteVolume', 0) or 0
        self.volume_history.append((current_time, tick_volume))

        # Prune history older than 65 seconds (keep a small buffer)
        self.tick_history = [(t, p) for t, p in self.tick_history if current_time - t <= 65]
        self.volume_history = [(t, v) for t, v in self.volume_history if current_time - t <= 65]

        # FIX #9: Calculate ret_5s, ret_30s with proper warmup handling
        price_5s_ago = None
        price_30s_ago = None

        for t, p in reversed(self.tick_history):
            age = current_time - t
            if age >= 5 and price_5s_ago is None:
                price_5s_ago = p
            if age >= 30 and price_30s_ago is None:
                price_30s_ago = p

        # If we don't have enough history, returns are None (warmup)
        ret_5s = (current_price - price_5s_ago) / price_5s_ago if price_5s_ago else None
        ret_30s = (current_price - price_30s_ago) / price_30s_ago if price_30s_ago else None

        # FIX #10: Detect volume spike relative to recent average, not magic number
        vol_values = [v for _, v in self.volume_history]
        if len(vol_values) >= 5:
            avg_vol = sum(vol_values[:-1]) / (len(vol_values) - 1)
            latest_vol = vol_values[-1]
            volume_state = "spike" if avg_vol > 0 and latest_vol > avg_vol * 3 else "normal"
        else:
            volume_state = "normal"  # Not enough data to judge

        self.last_price = current_price

        return {
            "price": current_price,
            "bid": bid,
            "ask": ask,
            "bid_vol": bid_vol,
            "ask_vol": ask_vol,
            "spread_pct": spread_pct,
            "ret_5s": ret_5s,      # None during warmup
            "ret_30s": ret_30s,    # None during warmup
            "vol_60s": sum(vol_values),
            "volume_state": volume_state,
            "warmup": price_5s_ago is None or price_30s_ago is None
        }
