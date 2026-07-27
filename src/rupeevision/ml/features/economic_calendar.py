"""Economic event calendar filtering and caching.

Fetches RBI, Fed, ECB events from Finnhub and filters for INR-relevant dates.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from rupeevision.core.logging import get_logger

if TYPE_CHECKING:
    from rupeevision.external.finnhub_client import FinnhubClient

logger = get_logger(__name__)

INR_RELEVANT_COUNTRIES = {"IN", "US", "EU", "GB"}
HIGH_IMPACT_EVENTS = {
    "RBI Interest Decision",
    "RBI Monetary Policy",
    "Federal Reserve Interest Decision",
    "ECB Interest Rate Decision",
    "US Nonfarm Payrolls",
    "US Inflation Rate",
}


class EconomicCalendar:
    """Filter economic calendar for INR-impacting events."""

    def __init__(self, client: FinnhubClient, cache_ttl: int = 86400):
        """Initialize calendar.

        Args:
            client: FinnhubClient instance.
            cache_ttl: Cache TTL in seconds (default 24h).
        """
        self.client = client
        self.cache_ttl = cache_ttl
        self._cache: list[dict] = []
        self._cache_time: float = 0.0

    def get_upcoming_events(self, days_ahead: int = 30) -> list[dict]:
        """Get INR-relevant events in next N days.

        Args:
            days_ahead: Look-ahead window in days.

        Returns:
            List of {country, event, date, impact, forecast, actual, previous}
        """
        now = datetime.utcnow()
        cutoff = now + timedelta(days=days_ahead)

        # Check cache
        cache_age = datetime.utcnow().timestamp() - self._cache_time
        if self._cache and cache_age < self.cache_ttl:
            events = self._cache
        else:
            # Fetch fresh
            try:
                raw = self.client.get_economic_calendar()
                events = raw if isinstance(raw, list) else []
                self._cache = events
                self._cache_time = datetime.utcnow().timestamp()
                logger.info("calendar_fetched", event_count=len(events))
            except Exception as exc:
                logger.error("calendar_fetch_failed", error=str(exc))
                events = []

        # Filter for INR relevance + upcoming
        filtered = []
        for evt in events:
            try:
                country = evt.get("country", "")
                event_name = evt.get("event", "")
                date_str = evt.get("date")

                if not country or country not in INR_RELEVANT_COUNTRIES:
                    continue
                if not date_str:
                    continue

                # Parse date (Finnhub: "2024-07-27 14:30:00" format)
                try:
                    event_date = datetime.fromisoformat(date_str.replace(" ", "T"))
                except ValueError:
                    continue

                if not (now <= event_date <= cutoff):
                    continue

                # Map impact to level
                impact = evt.get("impact", "").lower()
                if impact not in {"low", "medium", "high"}:
                    impact = "medium"

                filtered.append(
                    {
                        "country": country,
                        "event": event_name,
                        "date": event_date.isoformat(),
                        "impact": impact,
                        "forecast": evt.get("forecast"),
                        "actual": evt.get("actual"),
                        "previous": evt.get("previous"),
                    }
                )
            except Exception as exc:
                logger.debug("calendar_event_parse_failed", error=str(exc))
                continue

        # Sort by date
        filtered.sort(key=lambda x: x["date"])
        return filtered

    def get_high_impact_events(self, days_ahead: int = 30) -> list[dict]:
        """Get only high-impact events.

        Returns:
            Filtered list.
        """
        all_events = self.get_upcoming_events(days_ahead)
        return [
            e
            for e in all_events
            if e.get("impact") == "high" or any(kw in e.get("event", "") for kw in HIGH_IMPACT_EVENTS)
        ]
