"""Finnhub API client wrapper.

Handles auth (X-Finnhub-Token header), rate limiting, retries, error translation.
"""

from __future__ import annotations

import time
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from rupeevision.core.logging import get_logger

logger = get_logger(__name__)


class FinnhubClient:
    """Finnhub API client with retry + rate-limit backoff."""

    BASE_URL = "https://finnhub.io/api/v1"
    MAX_RETRIES = 3
    BACKOFF_FACTOR = 0.5

    def __init__(self, api_key: str):
        """Initialize client with API key.

        Args:
            api_key: Finnhub API token from settings.

        Raises:
            ValueError: If api_key is empty or None.
        """
        if not api_key:
            raise ValueError("FINNHUB_API_KEY not set in environment or .env")
        self.api_key = api_key
        self.session = self._build_session()

    def _build_session(self) -> requests.Session:
        """Create requests session with retry strategy."""
        session = requests.Session()
        retry = Retry(
            total=self.MAX_RETRIES,
            backoff_factor=self.BACKOFF_FACTOR,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.headers.update({"X-Finnhub-Token": self.api_key})
        return session

    def _request(self, method: str, endpoint: str, **kwargs: Any) -> dict[str, Any]:
        """Generic request handler.

        Args:
            method: HTTP method (GET, POST).
            endpoint: Path relative to BASE_URL, e.g. "/quote".
            **kwargs: Passed to requests (params, json, etc).

        Returns:
            Parsed JSON response.

        Raises:
            RuntimeError: On HTTP error or timeout.
        """
        url = f"{self.BASE_URL}{endpoint}"
        try:
            resp = self.session.request(method, url, timeout=10, **kwargs)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as exc:
            logger.error("finnhub_request_failed", endpoint=endpoint, error=str(exc))
            raise RuntimeError(f"Finnhub API failed: {exc}") from exc

    def get_quote(self, symbol: str) -> dict[str, Any]:
        """Fetch current quote for a symbol.

        Args:
            symbol: Stock ticker, e.g. "AAPL", "TCS.NS".

        Returns:
            {c: current_price, h: high, l: low, o: open, pc: previous_close, ...}
        """
        return self._request("GET", "/quote", params={"symbol": symbol})

    def get_news(self, symbol: str, limit: int = 50) -> list[dict[str, Any]]:
        """Fetch recent news for a symbol.

        Args:
            symbol: Stock ticker or currency pair.
            limit: Max articles (default 50, max 100).

        Returns:
            List of {headline, source, url, sentiment, datetime, ...}
        """
        resp = self._request("GET", "/news", params={"symbol": symbol, "limit": limit})
        return resp if isinstance(resp, list) else []

    def get_economic_calendar(self) -> list[dict[str, Any]]:
        """Fetch economic calendar events.

        Returns:
            List of {country, event, impact, date, actual, forecast, previous, ...}
        """
        resp = self._request("GET", "/economic-calendar")
        return resp if isinstance(resp, list) else []

    def get_company_news_sentiment(self, symbol: str) -> dict[str, Any]:
        """Fetch aggregated sentiment for a symbol.

        Returns:
            {symbol, sentiment: -1 to 1, change: daily change, ...}
        """
        return self._request("GET", "/news-sentiment", params={"symbol": symbol})
