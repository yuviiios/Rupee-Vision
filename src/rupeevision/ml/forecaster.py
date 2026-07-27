"""Recursive multi-step exchange-rate forecasting.

Each currency has its own two-layer LSTM trained on a 30-step window of scaled
prices. A forecast is produced autoregressively: predict one step, append it to
the window, drop the oldest observation, repeat.

Performance and correctness fixes over the original implementation:

* **``model(x)`` instead of ``model.predict(x)``.** ``predict()`` carries the full
  Keras dispatch path and re-traces its inner function; calling it in a loop is a
  documented source of unbounded memory growth. A direct call on a constant-shape
  tensor traces once.
* **Compute the maximum horizon once, then slice.** Forecast steps are a prefix
  chain — the first 7 values of a 30-day forecast are exactly the 7-day forecast.
  Combined with the cache this means a warm process answers any horizon without a
  single forward pass.
* **Explicit window shift.** The original used ``np.roll(window, -1)`` with no
  ``axis``, which flattens before rolling. It happened to be correct only because
  the array was ``(30, 1)``; with two or more features it would have shuffled
  values across columns and silently corrupted the window.
* **Forecast dates are anchored to the last observation**, not to
  ``datetime.now()``. The old frontend labelled forecasts with today's date
  regardless of how stale the underlying data was.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import date, timedelta
from typing import TYPE_CHECKING, Any

import numpy as np

from rupeevision.core.constants import CURRENCY_CODES, MAX_FORECAST_DAYS
from rupeevision.core.exceptions import InferenceError
from rupeevision.core.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from rupeevision.ml.data import RateHistory
    from rupeevision.ml.registry import CurrencyModel, ModelRegistry

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Forecast:
    """A completed multi-currency forecast.

    Attributes:
        dates: Forecast dates, ascending, starting the day after ``as_of``.
        series: Currency name to predicted rates, aligned with ``dates``.
        as_of: Date of the newest observation the forecast was conditioned on.
    """

    dates: tuple[date, ...]
    series: dict[str, tuple[float, ...]]
    as_of: date

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-serialisable API response body."""
        return {
            "as_of": self.as_of.isoformat(),
            "horizon_days": len(self.dates),
            "dates": [day.isoformat() for day in self.dates],
            "predictions": {
                currency: {
                    "code": CURRENCY_CODES.get(currency, currency),
                    "values": [round(value, 4) for value in values],
                }
                for currency, values in self.series.items()
            },
        }


class Forecaster:
    """Produces cached, thread-safe recursive forecasts.

    The cache is keyed only on the dataset's ``as_of`` date because model weights
    and input data are otherwise immutable for the lifetime of the process. When
    the dataset is refreshed the key changes and the stale entry is discarded.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        history: RateHistory,
        *,
        max_days: int = MAX_FORECAST_DAYS,
        cache_enabled: bool = True,
    ) -> None:
        self._registry = registry
        self._history = history
        self._max_days = max_days
        self._cache_enabled = cache_enabled
        self._cache: Forecast | None = None
        self._cache_lock = threading.Lock()

    def forecast(self, days: int) -> Forecast:
        """Return a forecast for the next ``days`` sessions.

        Args:
            days: Horizon in days, ``1 <= days <= max_days``.

        Returns:
            A :class:`Forecast` truncated to ``days`` steps.

        Raises:
            InferenceError: If ``days`` is out of range or a forward pass fails.
        """
        if not 1 <= days <= self._max_days:
            msg = f"days must be between 1 and {self._max_days}"
            raise InferenceError(msg)

        full = self._full_horizon()
        if days == len(full.dates):
            return full
        return Forecast(
            dates=full.dates[:days],
            series={currency: values[:days] for currency, values in full.series.items()},
            as_of=full.as_of,
        )

    def invalidate(self) -> None:
        """Drop the cached forecast, forcing recomputation on the next call."""
        with self._cache_lock:
            self._cache = None

    def _full_horizon(self) -> Forecast:
        """Return the cached maximum-horizon forecast, computing it if needed."""
        if self._cache_enabled:
            cached = self._cache
            if cached is not None and cached.as_of == self._history.as_of:
                return cached

        with self._cache_lock:
            cached = self._cache
            if self._cache_enabled and cached is not None and cached.as_of == self._history.as_of:
                return cached
            computed = self._compute(self._max_days)
            if self._cache_enabled:
                self._cache = computed
            return computed

    def _compute(self, days: int) -> Forecast:
        """Run the recursive forecast for every currency."""
        series: dict[str, tuple[float, ...]] = {}
        for currency in self._registry.currencies:
            model = self._registry.get(currency)
            series[currency] = self._forecast_currency(model, days)

        forecast = Forecast(
            dates=self._forecast_dates(days),
            series=series,
            as_of=self._history.as_of,
        )
        logger.info(
            "forecast_computed",
            horizon_days=days,
            currencies=len(series),
            as_of=str(self._history.as_of),
        )
        return forecast

    def _forecast_currency(self, model: CurrencyModel, days: int) -> tuple[float, ...]:
        """Recursively forecast ``days`` steps for a single currency.

        The scaled window has shape ``(seq_length, 1)``. Each iteration reshapes it
        to ``(1, seq_length, 1)``, takes one forward pass, appends the prediction
        and discards the oldest step.

        Raises:
            InferenceError: If the model emits a non-finite value, which would
                otherwise propagate as ``NaN`` through the rest of the chain.
        """
        window = self._history.window(model.currency, model.seq_length)
        scaled = _scale(model, window).astype("float32")

        scaled_predictions: list[float] = []
        # Serialise forward passes: Keras models are not safe to call concurrently.
        with model.lock:
            for step in range(days):
                batch = scaled.reshape(1, model.seq_length, 1)
                raw = model.model(batch, training=False)
                value = float(np.asarray(raw).reshape(-1)[0])
                if not np.isfinite(value):
                    msg = (
                        f"{model.currency} model produced a non-finite value at "
                        f"step {step + 1}"
                    )
                    raise InferenceError(msg)
                scaled_predictions.append(value)
                # Explicit shift along the time axis; see module docstring on np.roll.
                scaled = np.vstack([scaled[1:], [[value]]]).astype("float32")

        # One vectorised inverse transform instead of one call per step.
        column = np.asarray(scaled_predictions, dtype="float64").reshape(-1, 1)
        prices = _inverse_scale(model, column).reshape(-1)
        return tuple(float(price) for price in prices)

    def _forecast_dates(self, days: int) -> tuple[date, ...]:
        """Return ``days`` consecutive dates following the last observation."""
        start = self._history.as_of
        return tuple(start + timedelta(days=offset) for offset in range(1, days + 1))


def _feature_frame(model: CurrencyModel, values: np.ndarray) -> Any:
    """Wrap ``values`` in a DataFrame matching the scaler's fitted feature names.

    The scalers were fitted on a single-column pandas DataFrame, so they carry
    ``feature_names_in_``. Passing a bare ndarray triggers a scikit-learn
    ``UserWarning`` on every call ("X does not have valid feature names"). Naming
    the column restores the contract instead of suppressing the warning.
    """
    import pandas as pd

    names = getattr(model.scaler, "feature_names_in_", None)
    if names is None:
        return values
    return pd.DataFrame(values, columns=list(names))


def _scale(model: CurrencyModel, values: np.ndarray) -> np.ndarray:
    """Scale raw prices into the model's input range."""
    return np.asarray(model.scaler.transform(_feature_frame(model, values)))


def _inverse_scale(model: CurrencyModel, values: np.ndarray) -> np.ndarray:
    """Map scaled model outputs back to prices in INR."""
    return np.asarray(model.scaler.inverse_transform(_feature_frame(model, values)))
