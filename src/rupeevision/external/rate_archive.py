"""On-demand snapshots of the published RBI/FBIL reference-rate archive."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from io import BytesIO
from threading import Lock
from time import monotonic
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import requests

from rupeevision.core.constants import CURRENCY_CODES, SEQ_LENGTH
from rupeevision.core.logging import get_logger
from rupeevision.ml.data import RateHistory
from rupeevision.ml.features import CorrelationAnalyzer
from rupeevision.ml.forecaster import Forecaster

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
    names = {code: name for name, code in CURRENCY_CODES.items()}
    rows = rows.loc[rows["currency"].isin(names)].copy()
    expected_units = np.where(rows["currency"] == "JPY", 100, 1)
    if not (rows["unit"] == expected_units).all():
        raise ValueError("Unexpected quote unit")
    rows["date"] = pd.to_datetime(rows["date"], format="%Y-%m-%d", errors="raise")
    if rows["date"].isna().any() or not np.isfinite(rows["rate"]).all() or (rows["rate"] <= 0).any():
        raise ValueError("Invalid observation")
    frame = rows.pivot(index="date", columns="currency", values="rate").sort_index()
    frame = frame.rename(columns=names).reindex(columns=list(CURRENCY_CODES))
    if frame.empty or frame.iloc[-1].isna().any() or frame.count().min() < SEQ_LENGTH:
        raise ValueError("Incomplete rate history")
    as_of = frame.index[-1].date()
    if as_of > date.today():
        raise ValueError("Future observation")
    return RateHistory(frame=frame.ffill(), as_of=as_of)


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
                        forecaster=Forecaster(
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
