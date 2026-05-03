import os
import logging

import requests

logger = logging.getLogger(__name__)

_CRYPTOPANIC_URL = "https://cryptopanic.com/api/v1/posts/"
_SYMBOL_MAP = {
    "SOL": "SOL",
    "SOL/USDC": "SOL",
    "SOL/USDC:USDC": "SOL",
    "BTC": "BTC",
    "BTC/USDT": "BTC",
    "BTC/USDT:USDT": "BTC",
    "ETH": "ETH",
    "ETH/USDT": "ETH",
    "ETH/USDT:USDT": "ETH",
}

_MOCK_HEADLINES = [
    "Bitcoin adoption grows as major bank announces custody services.",
    "Regulatory concerns in Europe cause uncertainty in crypto markets.",
    "New technological upgrade goes live on major blockchain, promising lower fees.",
    "Federal Reserve hints at interest rate cuts, boosting risk-on assets.",
    "Whale movement detected: large amounts of BTC transferred to exchanges.",
    "Major crypto exchange faces technical outage during high volatility.",
    "Institutional investors increase exposure to digital assets.",
    "Inflation data comes in hotter than expected, crypto markets react negatively.",
    "Rumors of ETF approval for alternative cryptocurrencies circulate.",
]


class NewsData:
    def __init__(self):
        self._api_key = (
            os.getenv("CRYPTOPANIC_API_KEY")
            or os.getenv("NEWS_API_KEY")
            or ""
        ).strip()
        self._allow_mock_news = os.getenv("ALLOW_MOCK_NEWS", "").strip() == "1"

    def fetch_latest_news(self, symbol: str = "BTC", limit: int = 8) -> list:
        base_asset = _SYMBOL_MAP.get(str(symbol or "").upper(), str(symbol or "").split("/")[0].upper())

        if self._api_key:
            headlines = self._fetch_cryptopanic(base_asset, limit=limit)
            if headlines:
                return headlines

        if self._allow_mock_news:
            return self._fetch_mock(limit=limit)

        logger.info("No news API configured; returning empty headline set.")
        return []

    def _fetch_cryptopanic(self, currency: str, limit: int = 8) -> list:
        try:
            resp = requests.get(
                _CRYPTOPANIC_URL,
                params={
                    "auth_token": self._api_key,
                    "currencies": currency,
                    "kind": "news",
                    "public": "true",
                },
                timeout=10,
            )
            if resp.status_code != 200:
                logger.warning(f"CryptoPanic returned status {resp.status_code}")
                return []

            data = resp.json()
            results = data.get("results", [])
            if not results:
                return []

            headlines = []
            for post in results[:limit]:
                title = (post.get("title") or "").strip()
                if title:
                    headlines.append(title)

            if headlines:
                logger.info(f"Fetched {len(headlines)} news headlines from CryptoPanic for {currency}")
                return headlines

            return []
        except Exception as e:
            logger.warning(f"CryptoPanic fetch failed: {e}")
            return []

    def _fetch_mock(self, limit: int = 3) -> list:
        import random

        k = min(max(1, int(limit)), len(_MOCK_HEADLINES))
        return random.sample(_MOCK_HEADLINES, k)
