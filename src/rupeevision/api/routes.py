"""HTTP endpoints.

Routes stay thin: parse and validate, delegate to the ML layer, serialise. All
error translation happens in :mod:`rupeevision.api.errors`.
"""

from __future__ import annotations

from datetime import date
from http import HTTPStatus
from typing import Any

from flask import Blueprint, current_app, jsonify, request

from rupeevision.api.schemas import PredictRequest
from rupeevision.core.constants import CURRENCY_CODES, MAX_FORECAST_DAYS
from rupeevision.core.exceptions import RupeeVisionError
from rupeevision.core.logging import get_logger

logger = get_logger(__name__)

bp = Blueprint("api", __name__)


def _services() -> Any:
    """Return the service container attached by the app factory."""
    return current_app.extensions["rupeevision"]


@bp.get("/health")
def health() -> tuple[Any, int]:
    """Liveness probe.

    Returns 200 whenever the process is able to serve traffic. Deliberately does
    not touch models or data so that a slow artifact load cannot cause an
    orchestrator to kill an otherwise healthy container.
    """
    return jsonify({"status": "ok"}), HTTPStatus.OK


@bp.get("/ready")
def ready() -> tuple[Any, int]:
    """Readiness probe.

    Verifies that every model artifact can be loaded and that the dataset is not
    stale beyond the configured threshold. Returns 503 when either check fails so
    a load balancer stops routing to this instance.
    """
    services = _services()
    settings = services.settings
    checks: dict[str, Any] = {}
    healthy = True

    try:
        services.registry.warmup()
        checks["models"] = {"status": "ok", "count": len(services.registry.currencies)}
    except RupeeVisionError as exc:
        healthy = False
        checks["models"] = {"status": "error", "reason": exc.message}

    as_of = services.history.as_of
    staleness = (date.today() - as_of).days
    limit = settings.max_data_staleness_days
    data_ok = staleness <= limit
    healthy = healthy and data_ok
    checks["data"] = {
        "status": "ok" if data_ok else "stale",
        "as_of": as_of.isoformat(),
        "age_days": staleness,
        "max_age_days": limit,
    }

    status = HTTPStatus.OK if healthy else HTTPStatus.SERVICE_UNAVAILABLE
    return jsonify({"status": "ready" if healthy else "not_ready", "checks": checks}), status


@bp.get("/currencies")
def currencies() -> tuple[Any, int]:
    """List the supported currencies and the latest observed rate for each."""
    services = _services()
    history = services.history
    payload = [
        {
            "name": name,
            "code": CURRENCY_CODES.get(name, name),
            "latest_rate": round(history.latest(name), 4),
        }
        for name in services.registry.currencies
    ]
    return (
        jsonify(
            {
                "as_of": history.as_of.isoformat(),
                "max_forecast_days": MAX_FORECAST_DAYS,
                "currencies": payload,
            }
        ),
        HTTPStatus.OK,
    )


@bp.route("/predict", methods=["GET", "POST"])
def predict() -> tuple[Any, int]:
    """Forecast exchange rates for the requested horizon.

    Accepts ``?days=N`` on GET or ``{"days": N}`` as a JSON body on POST.
    ``silent=True`` on the body parse means a missing or non-JSON body yields
    ``None`` and is handled by the schema instead of raising a 400 from Werkzeug
    before validation runs.
    """
    payload = request.get_json(silent=True) if request.method == "POST" else None
    parsed = PredictRequest.from_request(args=request.args, json_body=payload)

    forecast = _services().forecaster.forecast(parsed.days)
    logger.info("prediction_served", horizon_days=parsed.days, as_of=str(forecast.as_of))
    return jsonify(forecast.to_payload()), HTTPStatus.OK


@bp.get("/sentiment")
def sentiment() -> tuple[Any, int]:
    """Fetch market sentiment for all currencies.

    Returns aggregated sentiment scores from Finnhub news (-1 bearish, +1 bullish).

    Returns:
        {sentiments: [{currency, sentiment_score, data_points, timestamp}, ...]}
        or {error: "..."} if Finnhub unavailable.
    """
    services = _services()
    if not services.sentiment_analyzer:
        return (
            jsonify({"error": "Finnhub API key not configured. Set FINNHUB_API_KEY environment variable."}),
            HTTPStatus.SERVICE_UNAVAILABLE,
        )

    sentiments = services.sentiment_analyzer.get_all_sentiments()
    logger.info("sentiment_served", count=len(sentiments))
    return jsonify({"sentiments": sentiments}), HTTPStatus.OK


@bp.get("/economic-calendar")
def economic_calendar() -> tuple[Any, int]:
    """Fetch upcoming economic events relevant to INR.

    Returns RBI, Fed, ECB events filtered for next 30 days.

    Query params:
        days: Look-ahead window (default 30).

    Returns:
        {events: [{country, event, date, impact, ...}, ...]}
    """
    services = _services()
    if not services.calendar:
        return (
            jsonify({"error": "Finnhub API key not configured."}),
            HTTPStatus.SERVICE_UNAVAILABLE,
        )

    days = request.args.get("days", 30, type=int)
    events = services.calendar.get_upcoming_events(days_ahead=min(days, 365))
    logger.info("calendar_served", event_count=len(events))
    return jsonify({"events": events}), HTTPStatus.OK


@bp.get("/correlation")
def correlation() -> tuple[Any, int]:
    """Fetch cross-asset correlation with INR rates.

    Returns rolling correlation for S&P 500, Gold, Oil.

    Returns:
        {correlations: [{asset, correlation, period_days, timestamp}, ...]}
    """
    services = _services()
    if not services.correlation_analyzer:
        return (
            jsonify({"error": "Finnhub API key not configured."}),
            HTTPStatus.SERVICE_UNAVAILABLE,
        )

    correlations = services.correlation_analyzer.get_all_correlations()
    logger.info("correlation_served", asset_count=len(correlations))
    return jsonify({"correlations": correlations}), HTTPStatus.OK


def register_routes(app: Any) -> None:
    """Register the API blueprint on ``app``."""
    app.register_blueprint(bp)
