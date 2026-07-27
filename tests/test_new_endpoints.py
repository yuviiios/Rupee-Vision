"""Tests for new Finnhub endpoints."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from rupeevision.api.app import Services, create_app
from rupeevision.core.config import Settings


@pytest.fixture
def mock_services_with_finnhub():
    """Create mock services with Finnhub enabled."""
    settings = Settings(finnhub_api_key="test_key")
    sentiment = MagicMock()
    sentiment.get_all_sentiments.return_value = [
        {"currency": "us_dollar", "sentiment_score": 0.5, "data_points": 10, "timestamp": "2024-07-27T00:00:00"},
        {"currency": "euro", "sentiment_score": -0.2, "data_points": 8, "timestamp": "2024-07-27T00:00:00"},
    ]

    calendar = MagicMock()
    calendar.get_upcoming_events.return_value = [
        {"country": "IN", "event": "RBI Decision", "date": "2024-08-08", "impact": "high"},
    ]

    correlation = MagicMock()
    correlation.get_all_correlations.return_value = [
        {"asset": "S&P 500", "correlation": 0.7, "period_days": 30, "timestamp": "2024-07-27T00:00:00"},
    ]

    services = Services(
        settings=settings,
        history=MagicMock(),
        registry=MagicMock(),
        forecaster=MagicMock(),
        finnhub_client=MagicMock(),
        sentiment_analyzer=sentiment,
        calendar=calendar,
        correlation_analyzer=correlation,
    )
    return services


@pytest.fixture
def mock_services_no_finnhub():
    """Create mock services without Finnhub."""
    settings = Settings(finnhub_api_key="")
    services = Services(
        settings=settings,
        history=MagicMock(),
        registry=MagicMock(),
        forecaster=MagicMock(),
        finnhub_client=None,
        sentiment_analyzer=None,
        calendar=None,
        correlation_analyzer=None,
    )
    return services


class TestSentimentEndpoint:
    """GET /sentiment endpoint."""

    def test_sentiment_enabled(self, mock_services_with_finnhub):
        """Return sentiments when Finnhub enabled."""
        app = create_app(settings=mock_services_with_finnhub.settings, services=mock_services_with_finnhub)
        client = app.test_client()

        resp = client.get("/sentiment")

        assert resp.status_code == 200
        data = resp.get_json()
        assert "sentiments" in data
        assert len(data["sentiments"]) == 2

    def test_sentiment_disabled(self, mock_services_no_finnhub):
        """Return 503 when Finnhub disabled."""
        app = create_app(settings=mock_services_no_finnhub.settings, services=mock_services_no_finnhub)
        client = app.test_client()

        resp = client.get("/sentiment")

        assert resp.status_code == 503
        data = resp.get_json()
        assert "error" in data


class TestCalendarEndpoint:
    """GET /economic-calendar endpoint."""

    def test_calendar_enabled(self, mock_services_with_finnhub):
        """Return events when Finnhub enabled."""
        app = create_app(settings=mock_services_with_finnhub.settings, services=mock_services_with_finnhub)
        client = app.test_client()

        resp = client.get("/economic-calendar?days=30")

        assert resp.status_code == 200
        data = resp.get_json()
        assert "events" in data
        assert len(data["events"]) == 1

    def test_calendar_disabled(self, mock_services_no_finnhub):
        """Return 503 when Finnhub disabled."""
        app = create_app(settings=mock_services_no_finnhub.settings, services=mock_services_no_finnhub)
        client = app.test_client()

        resp = client.get("/economic-calendar")

        assert resp.status_code == 503


class TestCorrelationEndpoint:
    """GET /correlation endpoint."""

    def test_correlation_enabled(self, mock_services_with_finnhub):
        """Return correlations when Finnhub enabled."""
        app = create_app(settings=mock_services_with_finnhub.settings, services=mock_services_with_finnhub)
        client = app.test_client()

        resp = client.get("/correlation")

        assert resp.status_code == 200
        data = resp.get_json()
        assert "correlations" in data
        assert len(data["correlations"]) == 1

    def test_correlation_disabled(self, mock_services_no_finnhub):
        """Return 503 when Finnhub disabled."""
        app = create_app(settings=mock_services_no_finnhub.settings, services=mock_services_no_finnhub)
        client = app.test_client()

        resp = client.get("/correlation")

        assert resp.status_code == 503
