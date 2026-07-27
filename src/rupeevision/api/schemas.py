"""Request schemas for the prediction API.

Validation is declarative and centralised. The original endpoint called
``int(request.args.get("days", 7))`` inside a bare ``try``, so a non-numeric
``days`` raised ``ValueError`` and was reported as HTTP 500 rather than 400, and a
POST without a JSON content type dereferenced ``None``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from rupeevision.core.constants import DEFAULT_FORECAST_DAYS, MAX_FORECAST_DAYS


class PredictRequest(BaseModel):
    """Parameters for a forecast request.

    Attributes:
        days: Forecast horizon. Bounded because recursive forecasting compounds
            error and because each step costs a forward pass per currency.
    """

    model_config = ConfigDict(extra="forbid")

    days: int = Field(
        default=DEFAULT_FORECAST_DAYS,
        ge=1,
        le=MAX_FORECAST_DAYS,
        description=f"Number of days to forecast (1-{MAX_FORECAST_DAYS}).",
    )

    @classmethod
    def from_request(cls, *, args: Any, json_body: Any) -> PredictRequest:
        """Build a request model from query parameters or a JSON body.

        Query parameters take precedence when present, which keeps the endpoint
        usable from a browser address bar for debugging.

        Args:
            args: Flask ``request.args``-like mapping.
            json_body: Parsed JSON body, or ``None`` when absent or non-JSON.

        Returns:
            A validated :class:`PredictRequest`.

        Raises:
            pydantic.ValidationError: If the payload is malformed.
        """
        if args and "days" in args:
            return cls.model_validate({"days": args["days"]})
        if isinstance(json_body, dict):
            return cls.model_validate(json_body)
        # No parameters supplied at all: fall back to the documented default
        # rather than failing, matching the previous public behaviour.
        return cls()


class NewsItem(BaseModel):
    """News article from Finnhub."""

    headline: str
    source: str
    url: str
    sentiment: float = Field(ge=-1, le=1)
    datetime: int


class SentimentResponse(BaseModel):
    """Market sentiment aggregation per currency."""

    currency: str
    sentiment_score: float = Field(ge=-1, le=1, description="Aggregate sentiment -1 (bearish) to +1 (bullish)")
    data_points: int = Field(ge=0, description="Number of news items aggregated")
    timestamp: datetime


class EconomicEvent(BaseModel):
    """Economic calendar event."""

    country: str
    event: str
    date: datetime
    impact: str = Field(description="low, medium, high")
    forecast: float | None = None
    actual: float | None = None
    previous: float | None = None


class CorrelationData(BaseModel):
    """Cross-asset correlation coefficient."""

    asset: str = Field(description="S&P500, Gold, Oil, etc")
    correlation: float = Field(ge=-1, le=1)
    period_days: int = Field(description="Rolling window days")
    timestamp: datetime
