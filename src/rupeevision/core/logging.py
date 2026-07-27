"""Structured logging setup.

The service previously emitted no application logs at all; the only signal was
Werkzeug's access line. This module configures ``structlog`` on top of the stdlib
``logging`` module so that:

* every application event is a machine-parseable JSON object carrying a timestamp,
  level, logger name and — inside a request — the correlation id bound by the API
  middleware;
* third-party loggers (Werkzeug, TensorFlow, urllib3) are rendered through the
  same formatter, so a log aggregator sees one consistent schema rather than a
  mixture of JSON and free text.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

_configured = False


def configure_logging(*, level: str = "INFO", json_output: bool = True) -> None:
    """Configure ``structlog`` and the stdlib root logger.

    Idempotent: repeated calls (for example one per test that builds the app
    factory) are ignored, so handlers are not stacked and lines are not duplicated.

    Args:
        level: Minimum level name, e.g. ``"INFO"``.
        json_output: ``True`` for one JSON object per line (production), ``False``
            for human-readable console output (local development).
    """
    global _configured
    if _configured:
        return

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Processors applied to events from both structlog and stdlib loggers.
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *shared_processors,
            # Hand off to ProcessorFormatter, which performs the final render on
            # the stdlib side. This is what allows one formatter to serve both.
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    renderer: Any = (
        structlog.processors.JSONRenderer()
        if json_output
        else structlog.dev.ConsoleRenderer(colors=False)
    )
    formatter = structlog.stdlib.ProcessorFormatter(
        # Strip structlog's internal bookkeeping key before rendering.
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.format_exc_info,
            renderer,
        ],
    )

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    # force-replace: Anaconda, TensorFlow and pytest all install handlers of their
    # own, which would otherwise duplicate every line in a different format.
    root.handlers = [handler]
    root.setLevel(numeric_level)

    # TensorFlow is extremely chatty at INFO and drowns out application events.
    logging.getLogger("tensorflow").setLevel(logging.WARNING)

    _configured = True


def reset_logging() -> None:
    """Clear the configured flag so a test can reconfigure logging.

    Only intended for use in tests; production configures logging exactly once.
    """
    global _configured
    _configured = False


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a lazily-bound logger for ``name``.

    Deliberately does *not* call :func:`configure_logging`. Modules bind their
    logger at import time, which happens before the application factory has read
    its settings; configuring here would lock in default settings and make
    ``RUPEEVISION_LOG_JSON`` and ``RUPEEVISION_LOG_LEVEL`` silently ineffective.

    ``structlog.get_logger`` returns a proxy that resolves configuration on its
    first emitted event, so entry points only need to call
    :func:`configure_logging` before serving traffic.
    """
    return structlog.get_logger(name)  # type: ignore[no-any-return]
