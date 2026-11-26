"""Direct session forecasts selected by the walk-forward retraining job."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from rupeevision.core.constants import CURRENCY_CODES, MAX_FORECAST_DAYS
from rupeevision.ml.forecaster import Forecast, Forecaster

WINDOW = 30
STEPS = np.arange(1, MAX_FORECAST_DAYS + 1)


def return_windows(values):
    """Past 30 daily log returns, in percentage points, ending at each origin."""
    return np.lib.stride_tricks.sliding_window_view(100 * np.diff(np.log(values)), WINDOW)


def ridge_features(windows):
    return np.column_stack(
        [windows[:, -n:].sum(axis=1) for n in (1, 2, 5, 10, 20)] +
        [windows[:, -n:].std(axis=1) for n in (5, 20)]
    )


def levels_from_returns(anchor, scaled_returns):
    return np.asarray(anchor).reshape(-1, 1) * np.exp(scaled_returns * np.sqrt(STEPS) / 100)


class SessionModels:
    """Load the evaluated release once and share it across data refreshes."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.release = json.loads((directory / "release.json").read_text())
        self.models = {}
        for code in CURRENCY_CODES.values():
            choice = self.release["selected"][code]
            if choice == "LSTM":
                import keras
                self.models[code] = keras.models.load_model(directory / f"{code}.keras", compile=False)
            elif choice.startswith("Ridge"):
                import joblib
                self.models[code] = joblib.load(directory / f"{code}-{choice}.joblib")
            elif choice != "Last rate":
                raise ValueError(f"Unknown model for {code}: {choice}")

    @property
    def currencies(self):
        return tuple(CURRENCY_CODES)

    def warmup(self):
        # All selected artifacts are loaded at construction.
        return None

    def forecast(self, history, days):
        if not 1 <= days <= MAX_FORECAST_DAYS:
            raise ValueError("Invalid session horizon")
        series = {}
        for name, code in CURRENCY_CODES.items():
            recent = history.window(name, WINDOW + 1).ravel()
            anchor = recent[-1]
            choice = self.release["selected"][code]
            if choice == "Last rate":
                values = np.repeat(anchor, days)
            else:
                windows = return_windows(recent)
                model = self.models[code]
                if choice == "LSTM":
                    offsets = model(windows[:, :, None], training=False).numpy()
                else:
                    offsets = model.predict(ridge_features(windows))
                values = levels_from_returns(anchor, offsets)[0, :days]
            if not np.isfinite(values).all() or (values <= 0).any():
                raise ValueError("Invalid forecast output")
            series[name] = tuple(map(float, values))
        # Weekdays approximate future fixing dates. The scoring target is session number.
        dates = pd.bdate_range(pd.Timestamp(history.as_of) + pd.offsets.BDay(), periods=days)
        return Forecast(tuple(dates.date), series, history.as_of, "estimated_weekdays", self.release["selected"])

class SessionForecaster:
    def __init__(self, models, history):
        self.models = models
        self.history = history

    def forecast(self, days):
        return self.models.forecast(self.history, days)


def build_forecaster(registry, history, *, cache_enabled):
    if isinstance(registry, SessionModels):
        return SessionForecaster(registry, history)
    return Forecaster(registry, history, cache_enabled=cache_enabled)
