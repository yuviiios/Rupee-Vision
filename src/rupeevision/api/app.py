"""Flask application factory.

Replaces the import-time side effects of the original ``flaskapp-1.py``, where
reading the dataset and deserialising four Keras models happened at module scope.
That made the module impossible to import in a test, and impossible to configure
without editing source. Everything is now constructed inside :func:`create_app`
and reachable through ``app.extensions["rupeevision"]``.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import structlog
from flask import Flask, Response, g, request
from flask_cors import CORS

from rupeevision.api.errors import register_error_handlers
from rupeevision.api.routes import register_routes
from rupeevision.core.config import Settings, get_settings
from rupeevision.core.logging import configure_logging, get_logger
from rupeevision.ml.data import load_rate_history
from rupeevision.ml.forecaster import Forecaster
from rupeevision.ml.registry import ModelRegistry

if TYPE_CHECKING:  # pragma: no cover
    from rupeevision.ml.data import RateHistory

logger = get_logger(__name__)

#: Header clients may set to propagate their own correlation id.
REQUEST_ID_HEADER = "X-Request-ID"


@dataclass(frozen=True, slots=True)
class Services:
    """Container for the long-lived collaborators shared by all requests.

    Passing this explicitly (rather than relying on module globals) is what makes
    the app testable: a test can build one with stub models and no TensorFlow.
    """

    settings: Settings
    history: RateHistory
    registry: ModelRegistry
    forecaster: Forecaster


def build_services(settings: Settings) -> Services:
    """Construct the service container from configuration."""
    history = load_rate_history(settings.data_path)
    registry = ModelRegistry(settings.model_dir)
    forecaster = Forecaster(
        registry,
        history,
        cache_enabled=settings.forecast_cache_size > 0,
    )
    return Services(
        settings=settings,
        history=history,
        registry=registry,
        forecaster=forecaster,
    )


def create_app(
    settings: Settings | None = None,
    *,
    services: Services | None = None,
) -> Flask:
    """Create and configure a Flask application.

    Args:
        settings: Configuration override. Defaults to the environment-derived
            singleton from :func:`get_settings`.
        services: Pre-built collaborators. Supplying these skips dataset and model
            loading entirely, which is how tests avoid importing TensorFlow.

    Returns:
        A configured :class:`flask.Flask` instance.
    """
    resolved = settings or get_settings()
    configure_logging(level=resolved.log_level, json_output=resolved.log_json)

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = resolved.max_content_length
    # Return 404 rather than a 308 redirect for a missing trailing slash, which is
    # friendlier to CORS preflight and to API clients.
    app.url_map.strict_slashes = False

    # Explicit allowlist replaces the previous wildcard `CORS(app)`.
    CORS(
        app,
        resources={r"/*": {"origins": resolved.allowed_origins}},
        methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", REQUEST_ID_HEADER],
        max_age=600,
    )

    # Rebind the container's settings to the ones this app was created with.
    # A caller-supplied container carries whatever settings it was built from, and
    # routes reading `services.settings` would otherwise silently observe stale
    # configuration that differs from the app's own.
    container = build_services(resolved) if services is None else services
    if container.settings is not resolved:
        container = replace(container, settings=resolved)
    app.extensions["rupeevision"] = container

    _register_rate_limiter(app, resolved)
    _register_request_logging(app)
    _register_security_headers(app)
    register_error_handlers(app)
    register_routes(app)

    logger.info(
        "app_created",
        debug=resolved.debug,
        allowed_origins=resolved.allowed_origins,
        rate_limit=resolved.rate_limit,
    )
    return app


def _register_rate_limiter(app: Flask, settings: Settings) -> None:
    """Apply a per-client request cap.

    One ``/predict`` call costs up to ``MAX_FORECAST_DAYS`` LSTM forward passes per
    currency, so an unthrottled endpoint is a cheap CPU-amplification target. The
    in-memory backend is adequate for a single instance; a multi-instance
    deployment should point ``storage_uri`` at Redis.
    """
    try:
        from flask_limiter import Limiter
        from flask_limiter.util import get_remote_address
    except ImportError:  # pragma: no cover - optional dependency
        logger.warning("rate_limiting_unavailable", reason="flask-limiter not installed")
        return

    limiter = Limiter(
        get_remote_address,
        app=app,
        default_limits=[settings.rate_limit],
        # Health and readiness probes are polled frequently by orchestrators and
        # must never be throttled.
        default_limits_exempt_when=lambda: request.path in {"/health", "/ready"},
        storage_uri="memory://",
        strategy="fixed-window",
    )
    app.extensions["limiter"] = limiter


def _register_request_logging(app: Flask) -> None:
    """Bind a correlation id to each request and log its outcome and latency."""

    @app.before_request
    def _start_request() -> None:
        incoming = request.headers.get(REQUEST_ID_HEADER)
        g.request_id = incoming or uuid.uuid4().hex[:16]
        g.request_start = time.perf_counter()
        structlog.contextvars.bind_contextvars(request_id=g.request_id)

    @app.after_request
    def _finish_request(response: Response) -> Response:
        started = getattr(g, "request_start", None)
        duration_ms = round((time.perf_counter() - started) * 1000, 2) if started else None
        request_id = getattr(g, "request_id", None)
        if request_id:
            response.headers[REQUEST_ID_HEADER] = request_id
        logger.info(
            "request_completed",
            method=request.method,
            path=request.path,
            status=response.status_code,
            duration_ms=duration_ms,
        )
        return response

    @app.teardown_request
    def _clear_context(_: BaseException | None) -> None:
        # Context vars are bound per worker thread and must be unbound, or a
        # recycled thread would leak the previous request's id into new logs.
        structlog.contextvars.clear_contextvars()


def _register_security_headers(app: Flask) -> None:
    """Attach baseline hardening headers to every response."""

    @app.after_request
    def _headers(response: Response) -> Response:
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("Cache-Control", "no-store")
        return response

    return None


def main() -> None:
    """Entry point for local development only.

    Production serving goes through a real WSGI server; see ``gunicorn.conf.py``.
    ``debug`` defaults to ``False`` and must be opted into via the environment,
    because the Werkzeug debugger permits arbitrary code execution.
    """
    settings = get_settings()
    app = create_app(settings)
    app.run(host=settings.host, port=settings.port, debug=settings.debug)


if __name__ == "__main__":  # pragma: no cover
    main()
