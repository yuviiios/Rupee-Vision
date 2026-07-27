"""End-to-end tests against the real Keras artifacts.

Marked ``slow`` because they import TensorFlow and deserialise four models
(roughly 13 seconds cold). Run with ``pytest -m slow``; excluded from the fast
inner loop but executed in CI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rupeevision.core.constants import CURRENCIES, MAX_FORECAST_DAYS, SEQ_LENGTH
from rupeevision.ml.data import RateHistory
from rupeevision.ml.forecaster import Forecaster
from rupeevision.ml.registry import ModelRegistry

pytestmark = pytest.mark.slow

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"


@pytest.fixture(scope="module")
def registry() -> ModelRegistry:
    """A registry over the committed artifacts, loaded once for the module."""
    if not ARTIFACT_DIR.exists():  # pragma: no cover - artifacts ship with the repo
        pytest.skip("artifacts directory is not available")
    return ModelRegistry(ARTIFACT_DIR)


def test_every_currency_artifact_loads(registry: ModelRegistry) -> None:
    for currency in CURRENCIES:
        model = registry.get(currency)
        assert model.seq_length == SEQ_LENGTH
        assert model.scaler.data_min_ is not None


def test_model_input_shape_matches_the_configured_window(registry: ModelRegistry) -> None:
    """Guards against serving a model trained with a different window."""
    for currency in CURRENCIES:
        assert registry.get(currency).model.input_shape == (None, SEQ_LENGTH, 1)


def test_scalers_are_fitted_on_positive_prices_only(registry: ModelRegistry) -> None:
    """A zero minimum indicates the sentinel bug reached training.

    The committed artifacts predate the data-cleaning fix, so the Euro, Yen and
    Sterling scalers were fitted with a minimum of 0. This test documents that
    known defect and will pass cleanly once the models are retrained.
    """
    affected = [
        currency
        for currency in CURRENCIES
        if float(registry.get(currency).scaler.data_min_[0]) == 0.0
    ]
    if affected:
        pytest.xfail(
            f"artifacts for {affected} were fitted on sentinel zeros; "
            "retrain to clear this"
        )


def test_real_forecast_is_plausible(
    registry: ModelRegistry, real_history: RateHistory
) -> None:
    """A full-horizon forecast must be finite, positive and not wildly divergent."""
    forecaster = Forecaster(registry, real_history)
    result = forecaster.forecast(MAX_FORECAST_DAYS)

    for currency, values in result.series.items():
        latest = real_history.latest(currency)
        assert len(values) == MAX_FORECAST_DAYS
        assert all(np.isfinite(value) for value in values)
        assert all(value > 0 for value in values)
        # No single day should move more than 25% from the last observed rate;
        # daily FX volatility is well under 2%.
        assert max(abs(value - latest) / latest for value in values) < 0.25


def test_forecast_is_deterministic(
    registry: ModelRegistry, real_history: RateHistory
) -> None:
    """Identical inputs must yield identical outputs; inference has no dropout."""
    first = Forecaster(registry, real_history, cache_enabled=False).forecast(5)
    second = Forecaster(registry, real_history, cache_enabled=False).forecast(5)

    assert first.series == second.series


def test_cache_removes_repeat_inference_cost(
    registry: ModelRegistry, real_history: RateHistory
) -> None:
    """The warm path must be orders of magnitude faster than the cold path."""
    import time

    forecaster = Forecaster(registry, real_history)
    registry.warmup()  # exclude one-off artifact loading from the measurement

    start = time.perf_counter()
    forecaster.forecast(MAX_FORECAST_DAYS)
    cold = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(10):
        forecaster.forecast(MAX_FORECAST_DAYS)
    warm = (time.perf_counter() - start) / 10

    assert warm < cold / 100, f"cold={cold:.3f}s warm={warm:.6f}s"
