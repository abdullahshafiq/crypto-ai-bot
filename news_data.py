import random

class NewsData:
    def __init__(self):
        # In a real scenario, this would connect to NewsAPI, CryptoPanic, or Twitter.
        # For demo purposes, we will return some simulated recent news headlines.
        self.mock_news = [
            "Bitcoin adoption grows as major bank announces custody services.",
            "Regulatory concerns in Europe cause uncertainty in crypto markets.",
            "New technological upgrade goes live on major blockchain, promising lower fees.",
            "Federal Reserve hints at interest rate cuts, boosting risk-on assets.",
            "Whale movement detected: large amounts of BTC transferred to exchanges.",
            "Major crypto exchange faces technical outage during high volatility.",
            "Institutional investors increase exposure to digital assets.",
            "Inflation data comes in hotter than expected, crypto markets react negatively.",
            "Rumors of ETF approval for alternative cryptocurrencies circulate."
        ]
        
    def fetch_latest_news(self, symbol: str = 'BTC') -> list:
        """
        Returns a list of simulated news headlines for the given symbol.
        """
        # Randomly select 3 headlines to simulate current news context
        return random.sample(self.mock_news, 3)
