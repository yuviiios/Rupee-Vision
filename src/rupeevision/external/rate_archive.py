"""On-demand snapshots of the published RBI/FBIL reference-rate archive."""

from __future__ import annotations

from dataclasses import replace
from io import BytesIO
from threading import Lock
from time import monotonic
from typing import TYPE_CHECKING

import pandas as pd
import requests

from rupeevision.core.logging import get_logger
from rupeevision.ml.data import RateHistory, archive_history
from rupeevision.ml.features import CorrelationAnalyzer
from rupeevision.ml.session_models import build_forecaster

if TYPE_CHECKING:
    from flask import Flask

    from rupeevision.api.app import Services

ARCHIVE_URL = (
    "https://raw.githubusercontent.com/the-solipsist/rbi-forex-reference-rates/"
    "main/rbi_forex_reference_rates_long.csv"
)
logger = get_logger(__name__)


def fetch_history() -> RateHistory:
    """Load the four supported series in their published quote units."""
    response = requests.get(ARCHIVE_URL, timeout=(4, 12))
    response.raise_for_status()
    rows = pd.read_csv(BytesIO(response.content), usecols=["date", "currency", "rate", "unit"])
    return archive_history(rows)


class ArchiveRefresh:
    """Serialize refreshes per worker while ordinary forecasts keep their snapshot."""

    def __init__(self) -> None:
        self.lock = Lock()
        self.next_check = 0.0
        self.status = "unavailable"

    def check(self, app: Flask) -> tuple[Services, str]:
        """Return the active snapshot and whether the latest check succeeded."""
        with self.lock:
            services = app.extensions["rupeevision"]
            if monotonic() < self.next_check:
                return services, self.status
            try:
                history = fetch_history()
                if history.as_of < services.history.as_of:
                    raise ValueError("Archive is behind the active dataset")
                if not history.frame.equals(services.history.frame):
                    correlation = services.correlation_analyzer
                    if correlation is not None:
                        correlation = CorrelationAnalyzer(
                            services.finnhub_client, history,
                            cache_ttl=services.settings.feature_cache_ttl_seconds,
                        )
                    services = replace(
                        services, history=history, correlation_analyzer=correlation,
                        forecaster=build_forecaster(
                            services.registry, history,
                            cache_enabled=services.settings.forecast_cache_size > 0,
                        ),
                    )
                    app.extensions["rupeevision"] = services
                self.status = "verified"
                self.next_check = monotonic() + 300
            except (requests.RequestException, ValueError, TypeError) as exc:
                logger.warning("archive_refresh_failed", reason=str(exc))
                self.status = "unavailable"
                self.next_check = monotonic() + 30
            return services, self.status
