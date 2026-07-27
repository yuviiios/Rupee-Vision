"""Centralised error handling.

Every failure path converges on :func:`_error_response`, producing a consistent
body:

.. code-block:: json

    {"error": {"code": "validation_error", "message": "...", "request_id": "..."}}

Unexpected exceptions are logged in full server-side and reported to the client
as a generic message plus the request id. The original handler returned
``str(exc)`` directly, disclosing filesystem paths and dependency internals.
"""

from __future__ import annotations

from http import HTTPStatus
from typing import Any

from flask import Flask, current_app, g, jsonify
from pydantic import ValidationError as PydanticValidationError
from werkzeug.exceptions import HTTPException

from rupeevision.core.exceptions import RupeeVisionError, ValidationError
from rupeevision.core.logging import get_logger

logger = get_logger(__name__)


def _error_response(
    *,
    code: str,
    message: str,
    status: int,
    details: Any | None = None,
) -> tuple[Any, int]:
    """Build a uniform JSON error response."""
    body: dict[str, Any] = {"code": code, "message": message}
    request_id = getattr(g, "request_id", None)
    if request_id:
        body["request_id"] = request_id
    if details is not None:
        body["details"] = details
    return jsonify({"error": body}), status


def _describe_pydantic_error(exc: PydanticValidationError) -> list[dict[str, Any]]:
    """Convert pydantic's error list into a compact, client-safe form."""
    return [
        {
            "field": ".".join(str(part) for part in error["loc"]) or "body",
            "message": error["msg"],
        }
        for error in exc.errors()
    ]


def register_error_handlers(app: Flask) -> None:
    """Attach the application's error handlers to ``app``."""

    @app.errorhandler(PydanticValidationError)
    def _handle_pydantic(exc: PydanticValidationError) -> tuple[Any, int]:
        details = _describe_pydantic_error(exc)
        logger.info("request_validation_failed", details=details)
        return _error_response(
            code="validation_error",
            message="Request parameters are invalid.",
            status=HTTPStatus.BAD_REQUEST,
            details=details,
        )

    @app.errorhandler(ValidationError)
    def _handle_validation(exc: ValidationError) -> tuple[Any, int]:
        logger.info("request_rejected", reason=exc.message)
        return _error_response(
            code=exc.code,
            message=exc.message,
            status=exc.status_code,
            details=exc.details,
        )

    @app.errorhandler(RupeeVisionError)
    def _handle_domain(exc: RupeeVisionError) -> tuple[Any, int]:
        # Domain errors carry client-safe messages by construction, but are logged
        # with a traceback because they represent a real service fault.
        logger.error("domain_error", code=exc.code, reason=exc.message, exc_info=True)
        return _error_response(code=exc.code, message=exc.message, status=exc.status_code)

    @app.errorhandler(HTTPStatus.TOO_MANY_REQUESTS)
    def _handle_rate_limit(exc: HTTPException) -> tuple[Any, int]:
        # Flask-Limiter puts the raw limit expression in `description`, which reads
        # as a cryptic "1 per 1 minute" to a client. Report the limit as data.
        logger.warning("rate_limit_exceeded", limit=str(exc.description))
        return _error_response(
            code="rate_limit_exceeded",
            message="Too many requests. Retry after the limit window elapses.",
            status=HTTPStatus.TOO_MANY_REQUESTS,
            details={"limit": str(exc.description)},
        )

    @app.errorhandler(HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
    def _handle_too_large(_: HTTPException) -> tuple[Any, int]:
        limit = app.config.get("MAX_CONTENT_LENGTH")
        return _error_response(
            code="payload_too_large",
            message="Request body exceeds the maximum accepted size.",
            status=HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            details={"max_bytes": limit},
        )

    @app.errorhandler(HTTPException)
    def _handle_http(exc: HTTPException) -> tuple[Any, int]:
        status = exc.code or HTTPStatus.INTERNAL_SERVER_ERROR
        return _error_response(
            code=(exc.name or "http_error").lower().replace(" ", "_"),
            message=exc.description or exc.name or "Request failed.",
            status=status,
        )

    @app.errorhandler(Exception)
    def _handle_unexpected(exc: Exception) -> tuple[Any, int]:
        # Full detail to the logs, nothing but a correlation id to the client.
        logger.exception("unhandled_exception", exc_type=type(exc).__name__)
        if current_app.config.get("PROPAGATE_UNHANDLED"):  # pragma: no cover
            raise exc
        return _error_response(
            code="internal_error",
            message="An unexpected error occurred. Quote the request id when reporting it.",
            status=HTTPStatus.INTERNAL_SERVER_ERROR,
        )
