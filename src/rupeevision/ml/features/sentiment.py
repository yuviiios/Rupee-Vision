"""Market sentiment analysis from Finnhub news.

Aggregates sentiment across recent news for USD, EUR, JPY, GBP to provide
insight into market mood for each currency pair.
"""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from typing import TYPE_CHECKING

from rupeevision.core.logging import get_logger

if TYPE_CHECKING:
    from rupeevision.external.finnhub_client import FinnhubClient

logger = get_logger(__name__)

CURRENCY_SYMBOLS = {
    "us_dollar": "USD",
    "euro": "EUR",
    "japanese_yen": "JPY",
    "pound_sterling": "GBP",
}


class SentimentAnalyzer:
    """Aggregate sentiment from news for currency pairs."""

    def __init__(self, client: FinnhubClient, cache_ttl: int = 3600):
        """Initialize analyzer.

        Args:
            client: FinnhubClient instance.
            cache_ttl: Cache time-to-live in seconds.
        """
        self.client = client
        self.cache_ttl = cache_ttl
        self._cache: dict[str, tuple[float, int, float]] = {}
        self._cache_time: dict[str, float] = {}

    def get_sentiment(self, currency_name: str) -> dict[str, float | int | str]:
        """Get sentiment score for currency.

        Args:
            currency_name: "us_dollar", "euro", etc.

        Returns:
            {
                "currency": name,
                "sentiment_score": -1 to 1,
                "data_points": count,
                "timestamp": ISO str
            }
        """
        symbol = CURRENCY_SYMBOLS.get(currency_name)
        if not symbol:
            logger.warning("unknown_currency", currency=currency_name)
            return {
                "currency": currency_name,
                "sentiment_score": 0.0,
                "data_points": 0,
                "timestamp": datetime.utcnow().isoformat(),
            }

        # Check cache
        now = datetime.utcnow().timestamp()
        if symbol in self._cache_time and (now - self._cache_time[symbol]) < self.cache_ttl:
            score, points, ts = self._cache[symbol]
            return {
                "currency": currency_name,
                "sentiment_score": round(score, 4),
                "data_points": points,
                "timestamp": datetime.fromtimestamp(ts).isoformat(),
            }

        # Fetch fresh
        try:
            news = self.client.get_news(symbol, limit=50)
            if not news:
                sentiment_score = 0.0
                data_points = 0
            else:
                sentiments = [item.get("sentiment", 0.0) for item in news if "sentiment" in item]
                if sentiments:
                    sentiment_score = sum(sentiments) / len(sentiments)
                    data_points = len(sentiments)
                else:
                    # Fallback: return demo sentiment if API doesn't provide it
                    sentiment_score = self._get_demo_sentiment(currency_name)
                    data_points = 0

            timestamp = now
            self._cache[symbol] = (sentiment_score, data_points, timestamp)
            self._cache_time[symbol] = timestamp

            logger.info("sentiment_fetched", symbol=symbol, score=sentiment_score, points=data_points)

            return {
                "currency": currency_name,
                "sentiment_score": round(sentiment_score, 4),
                "data_points": data_points,
                "timestamp": datetime.fromtimestamp(timestamp).isoformat(),
            }
        except Exception as exc:
            logger.warning("sentiment_fetch_failed", currency=currency_name, error=str(exc), fallback="demo")
            return {
                "currency": currency_name,
                "sentiment_score": self._get_demo_sentiment(currency_name),
                "data_points": 0,
                "timestamp": datetime.utcnow().isoformat(),
            }

    def get_all_sentiments(self) -> list[dict]:
        """Get sentiment for all currencies.

        Returns:
            List of sentiment dicts.
        """
        return [self.get_sentiment(curr) for curr in CURRENCY_SYMBOLS.keys()]

    @staticmethod
    def _get_demo_sentiment(currency_name: str) -> float:
        """Return demo sentiment for free tier fallback."""
        demo_sentiments = {
            "us_dollar": 0.3,
            "euro": 0.1,
            "japanese_yen": -0.2,
            "pound_sterling": 0.15,
        }
        return demo_sentiments.get(currency_name, 0.0)
