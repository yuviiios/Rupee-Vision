"""Tests for recursive forecasting, caching and thread safety."""

from __future__ import annotations

import threading
from datetime import timedelta

import numpy as np
import pytest

from rupeevision.core.constants import CURRENCIES, MAX_FORECAST_DAYS, SEQ_LENGTH
from rupeevision.core.exceptions import InferenceError
from rupeevision.ml.data import RateHistory
from rupeevision.ml.forecaster import Forecaster
from tests.conftest import FakeRegistry


def test_forecast_returns_requested_horizon(forecaster: Forecaster) -> None:
    result = forecaster.forecast(7)

    assert len(result.dates) == 7
    assert set(result.series) == set(CURRENCIES)
    for values in result.series.values():
        assert len(values) == 7


def test_forecast_dates_follow_the_last_observation_not_today(
    forecaster: Forecaster, history: RateHistory
) -> None:
    """Regression test for forecasts labelled with ``datetime.now()``.

    The dataset is stale by design, so anchoring to the current date silently
    mislabelled every prediction.
    """
    result = forecaster.forecast(3)

    assert result.as_of == history.as_of
    assert result.dates[0] == history.as_of + timedelta(days=1)
    assert result.dates[-1] == history.as_of + timedelta(days=3)
    assert list(result.dates) == sorted(result.dates)


@pytest.mark.parametrize("days", [0, -1, MAX_FORECAST_DAYS + 1, 999])
def test_out_of_range_horizon_is_rejected(forecaster: Forecaster, days: int) -> None:
    with pytest.raises(InferenceError, match="days must be between"):
        forecaster.forecast(days)


@pytest.mark.parametrize("days", [1, 7, 14, MAX_FORECAST_DAYS])
def test_all_valid_horizons_produce_finite_values(forecaster: Forecaster, days: int) -> None:
    result = forecaster.forecast(days)

    for values in result.series.values():
        assert len(values) == days
        assert all(np.isfinite(value) for value in values)
        assert all(value > 0 for value in values)


def test_shorter_horizon_is_a_prefix_of_a_longer_one(forecaster: Forecaster) -> None:
    """Recursive steps form a chain, so slicing must equal recomputing."""
    short = forecaster.forecast(5)
    long = forecaster.forecast(MAX_FORECAST_DAYS)

    for currency in CURRENCIES:
        assert long.series[currency][:5] == short.series[currency]


def test_maximum_horizon_is_computed_once_and_reused(
    forecaster: Forecaster, registry: FakeRegistry
) -> None:
    """The cache must eliminate forward passes for repeat requests."""
    forecaster.forecast(7)
    after_first = registry.total_calls()
    assert after_first == MAX_FORECAST_DAYS * len(CURRENCIES)

    for days in (1, 7, 14, MAX_FORECAST_DAYS):
        forecaster.forecast(days)

    assert registry.total_calls() == after_first, "cached forecasts must not re-run the model"


def test_invalidate_forces_recomputation(
    forecaster: Forecaster, registry: FakeRegistry
) -> None:
    forecaster.forecast(7)
    baseline = registry.total_calls()

    forecaster.invalidate()
    forecaster.forecast(7)

    assert registry.total_calls() == baseline * 2


def test_cache_can_be_disabled(registry: FakeRegistry, history: RateHistory) -> None:
    forecaster = Forecaster(registry, history, cache_enabled=False)  # type: ignore[arg-type]

    forecaster.forecast(1)
    first = registry.total_calls()
    forecaster.forecast(1)

    assert registry.total_calls() > first


def test_window_shift_preserves_length_and_drops_oldest(
    registry: FakeRegistry, history: RateHistory
) -> None:
    """Verify the window update, replacing the axis-less ``np.roll``.

    The fake model returns the window mean, so with a strictly increasing input
    window each successive forecast must also increase: the oldest (smallest)
    value is discarded and a larger prediction appended. A window corrupted by
    flattening would not preserve this ordering.
    """
    forecaster = Forecaster(registry, history)  # type: ignore[arg-type]
    values = forecaster.forecast(10).series["US Dollar"]

    assert list(values) == sorted(values)
    model = registry.get("US Dollar")
    assert model.model.input_shape == (None, SEQ_LENGTH, 1)


def test_non_finite_prediction_raises_instead_of_propagating_nan(
    registry: FakeRegistry, history: RateHistory
) -> None:
    """A NaN from the model must fail loudly, not flow into the response."""

    class NaNModel:
        input_shape = (None, SEQ_LENGTH, 1)

        def __call__(self, batch: object, training: bool = False) -> np.ndarray:  # noqa: ARG002
            return np.array([[np.nan]])

    original = registry.get("US Dollar")
    registry._models["US Dollar"] = type(original)(  # type: ignore[index]
        currency=original.currency,
        model=NaNModel(),
        scaler=original.scaler,
        lock=threading.Lock(),
        seq_length=original.seq_length,
    )
    forecaster = Forecaster(registry, history)  # type: ignore[arg-type]

    with pytest.raises(InferenceError, match="non-finite"):
        forecaster.forecast(3)


def test_concurrent_forecasts_are_consistent(
    registry: FakeRegistry, history: RateHistory
) -> None:
    """Parallel requests must all observe identical, complete results.

    Keras models are not re-entrant; before the per-model lock, interleaved calls
    could return inconsistent output.
    """
    forecaster = Forecaster(registry, history, cache_enabled=False)  # type: ignore[arg-type]
    collected: list[tuple[float, ...]] = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(6)

    def worker() -> None:
        try:
            barrier.wait(timeout=10)
            collected.append(forecaster.forecast(5).series["US Dollar"])
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors, f"concurrent forecasting raised: {errors}"
    assert len(collected) == 6
    assert len(set(collected)) == 1, "concurrent forecasts diverged"


def test_payload_shape_is_stable(forecaster: Forecaster) -> None:
    payload = forecaster.forecast(2).to_payload()

    assert set(payload) == {"as_of", "horizon_days", "dates", "predictions"}
    assert payload["horizon_days"] == 2
    assert len(payload["dates"]) == 2
    usd = payload["predictions"]["US Dollar"]
    assert usd["code"] == "USD"
    assert len(usd["values"]) == 2
    # Rounded to four decimals: the model has no meaningful precision beyond that.
    assert all(round(value, 4) == value for value in usd["values"])
