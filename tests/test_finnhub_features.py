"""Tests for Finnhub integration and market intelligence features."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from rupeevision.external.finnhub_client import FinnhubClient
from rupeevision.ml.data import RateHistory
from rupeevision.ml.features import CorrelationAnalyzer, EconomicCalendar, SentimentAnalyzer


class TestFinnhubClient:
    """FinnhubClient initialization and request handling."""

    def test_init_no_key(self):
        """Reject empty API key."""
        with pytest.raises(ValueError, match="FINNHUB_API_KEY not set"):
            FinnhubClient("")

    def test_init_success(self):
        """Create client with valid key."""
        client = FinnhubClient("test_key")
        assert client.api_key == "test_key"
        assert client.session is not None

    @patch("rupeevision.external.finnhub_client.requests.Session.request")
    def test_get_quote(self, mock_req):
        """Fetch quote for symbol."""
        mock_req.return_value.json.return_value = {"c": 80.5, "h": 81.0, "l": 80.0}
        mock_req.return_value.raise_for_status = MagicMock()

        client = FinnhubClient("test_key")
        resp = client.get_quote("EURUSD")

        assert resp["c"] == 80.5
        mock_req.assert_called_once()

    @patch("rupeevision.external.finnhub_client.requests.Session.request")
    def test_get_news(self, mock_req):
        """Fetch news list."""
        mock_req.return_value.json.return_value = [
            {"headline": "USD rises", "sentiment": 0.8},
            {"headline": "EUR falls", "sentiment": -0.5},
        ]
        mock_req.return_value.raise_for_status = MagicMock()

        client = FinnhubClient("test_key")
        resp = client.get_news("EURUSD")

        assert len(resp) == 2
        assert resp[0]["sentiment"] == 0.8


class TestSentimentAnalyzer:
    """Sentiment aggregation."""

    @patch.object(FinnhubClient, "get_news")
    def test_get_sentiment_valid(self, mock_news):
        """Get sentiment for valid currency."""
        mock_news.return_value = [
            {"headline": "test1", "sentiment": 0.5},
            {"headline": "test2", "sentiment": 0.7},
        ]

        client = FinnhubClient("test_key")
        analyzer = SentimentAnalyzer(client)
        result = analyzer.get_sentiment("us_dollar")

        assert result["currency"] == "us_dollar"
        assert result["data_points"] == 2
        assert abs(result["sentiment_score"] - 0.6) < 0.01

    def test_get_sentiment_unknown_currency(self):
        """Unknown currency returns 0 sentiment."""
        client = FinnhubClient("test_key")
        analyzer = SentimentAnalyzer(client)
        result = analyzer.get_sentiment("unknown")

        assert result["sentiment_score"] == 0.0
        assert result["data_points"] == 0

    @patch.object(SentimentAnalyzer, "get_sentiment")
    def test_get_all_sentiments(self, mock_single):
        """Get all currency sentiments."""
        mock_single.side_effect = [
            {"currency": "us_dollar", "sentiment_score": 0.5, "data_points": 10, "timestamp": "2024-07-27T00:00:00"},
            {"currency": "euro", "sentiment_score": 0.3, "data_points": 8, "timestamp": "2024-07-27T00:00:00"},
            {"currency": "japanese_yen", "sentiment_score": -0.1, "data_points": 12, "timestamp": "2024-07-27T00:00:00"},
            {"currency": "pound_sterling", "sentiment_score": 0.2, "data_points": 9, "timestamp": "2024-07-27T00:00:00"},
        ]

        client = FinnhubClient("test_key")
        analyzer = SentimentAnalyzer(client)
        results = analyzer.get_all_sentiments()

        assert len(results) == 4
        assert results[0]["sentiment_score"] == 0.5


class TestEconomicCalendar:
    """Economic event calendar filtering."""

    @patch.object(FinnhubClient, "get_economic_calendar")
    def test_get_upcoming_events(self, mock_cal):
        """Filter events for INR relevance."""
        today = datetime.utcnow()
        mock_cal.return_value = [
            {
                "country": "US",
                "event": "Federal Reserve Interest Decision",
                "date": (today + timedelta(days=27)).isoformat(sep=" "),
                "impact": "high",
                "forecast": 5.5,
            },
            {
                "country": "IN",
                "event": "RBI Interest Decision",
                "date": (today + timedelta(days=8)).isoformat(sep=" "),
                "impact": "high",
                "forecast": 6.5,
            },
        ]

        client = FinnhubClient("test_key")
        calendar = EconomicCalendar(client)
        events = calendar.get_upcoming_events(days_ahead=365)

        assert len(events) == 2
        assert events[0]["country"] == "IN"
        assert events[1]["country"] == "US"

    @patch.object(EconomicCalendar, "get_upcoming_events")
    def test_get_high_impact_events(self, mock_upcoming):
        """Filter high-impact events."""
        mock_upcoming.return_value = [
            {"country": "IN", "event": "RBI Interest Decision", "impact": "high", "date": "2024-08-08"},
            {"country": "US", "event": "Employment Report", "impact": "low", "date": "2024-08-02"},
        ]

        client = FinnhubClient("test_key")
        calendar = EconomicCalendar(client)
        events = calendar.get_high_impact_events()

        assert len(events) == 1
        assert events[0]["impact"] == "high"


class TestCorrelationAnalyzer:
    """Cross-asset correlation."""

    @patch.object(FinnhubClient, "get_quote")
    def test_get_asset_prices(self, mock_quote):
        """Fetch asset prices."""
        mock_quote.return_value = {"c": 500.0}

        client = FinnhubClient("test_key")
        analyzer = CorrelationAnalyzer(client, MagicMock())
        prices = analyzer.get_asset_prices("SPY", days=5)

        assert len(prices) == 5
        assert prices[-1] == 500.0

    @patch.object(CorrelationAnalyzer, "get_asset_prices")
    def test_compute_correlation(self, mock_prices):
        """Compute correlation with INR."""
        mock_prices.return_value = [100.0, 101.0, 102.0, 103.0, 104.0]

        history = MagicMock()
        history.dataframe = {"us_dollar": [80.0, 80.5, 81.0, 81.5, 82.0]}

        client = FinnhubClient("test_key")
        analyzer = CorrelationAnalyzer(client, history)
        result = analyzer.compute_correlation("SPY", window_days=5)

        assert "asset" in result
        assert -1 <= result["correlation"] <= 1

    @patch.object(CorrelationAnalyzer, "compute_correlation")
    def test_get_all_correlations(self, mock_single):
        """Get correlations for all assets."""
        mock_single.side_effect = [
            {"asset": "S&P 500", "correlation": 0.7, "period_days": 30, "timestamp": "2024-07-27T00:00:00"},
            {"asset": "Gold", "correlation": -0.3, "period_days": 30, "timestamp": "2024-07-27T00:00:00"},
            {"asset": "Oil (WTI)", "correlation": 0.5, "period_days": 30, "timestamp": "2024-07-27T00:00:00"},
        ]

        client = FinnhubClient("test_key")
        analyzer = CorrelationAnalyzer(client, MagicMock())
        results = analyzer.get_all_correlations()

        assert len(results) == 3
        assert results[0]["correlation"] == 0.7
