"""Shared fixtures.

The fixtures here deliberately avoid importing TensorFlow. A fake model exposes
the only surface :class:`~rupeevision.ml.forecaster.Forecaster` uses — a callable
returning an array — which keeps the unit suite fast enough to run on every save
and makes the API tests deterministic. Tests that exercise the real artifacts are
marked ``slow`` and live in ``test_artifacts.py``.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler

from rupeevision.api.app import Services, create_app
from rupeevision.core.config import Settings
from rupeevision.core.constants import CURRENCIES, SEQ_LENGTH
from rupeevision.ml.data import RateHistory, load_rate_history
from rupeevision.ml.forecaster import Forecaster
from rupeevision.ml.registry import CurrencyModel

#: Deterministic base level per currency, so assertions can be exact.
_BASE_RATES = {
    "US Dollar": 80.0,
    "Euro": 90.0,
    "Japanese Yen": 55.0,
    "Pound Sterling": 105.0,
}


class FakeKerasModel:
    """Minimal stand-in for a Keras model.

    Returns the mean of the input window, which makes recursive forecasts
    analytically predictable while still exercising the real window-shift logic.

    Attributes:
        calls: Number of forward passes, used to assert caching behaviour.
        input_shape: Mimics the attribute the registry validates.
    """

    def __init__(self, seq_length: int = SEQ_LENGTH) -> None:
        self.calls = 0
        self.input_shape = (None, seq_length, 1)
        self._lock = threading.Lock()

    def __call__(self, batch: Any, training: bool = False) -> np.ndarray:
        """Return the per-sample mean of ``batch`` as shape ``(n, 1)``."""
        assert training is False, "inference must never run in training mode"
        with self._lock:
            self.calls += 1
        array = np.asarray(batch, dtype="float64")
        return array.mean(axis=1).reshape(-1, 1)


@pytest.fixture
def rate_frame() -> pd.DataFrame:
    """A clean 400-row wide frame with a mild upward drift per currency."""
    index = pd.date_range("2024-01-01", periods=400, freq="D")
    drift = np.linspace(0.0, 2.0, len(index))
    return pd.DataFrame(
        {currency: base + drift for currency, base in _BASE_RATES.items()},
        index=index,
    )


@pytest.fixture
def history(rate_frame: pd.DataFrame) -> RateHistory:
    """A validated :class:`RateHistory` built from :func:`rate_frame`."""
    return RateHistory(frame=rate_frame, as_of=rate_frame.index[-1].date())


class FakeRegistry:
    """In-memory registry serving :class:`FakeKerasModel` instances."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self._models: dict[str, CurrencyModel] = {}
        for currency in CURRENCIES:
            scaler = MinMaxScaler()
            scaler.fit(frame[[currency]])
            self._models[currency] = CurrencyModel(
                currency=currency,
                model=FakeKerasModel(),
                scaler=scaler,
                lock=threading.Lock(),
                seq_length=SEQ_LENGTH,
            )

    @property
    def currencies(self) -> tuple[str, ...]:
        return CURRENCIES

    @property
    def metadata(self) -> dict[str, Any]:
        return {"seq_length": SEQ_LENGTH, "git_sha": "test"}

    def get(self, currency: str) -> CurrencyModel:
        return self._models[currency]

    def warmup(self) -> None:
        return None

    def is_loaded(self) -> bool:
        return True

    def total_calls(self) -> int:
        """Total forward passes across every fake model."""
        return sum(model.model.calls for model in self._models.values())


@pytest.fixture
def registry(rate_frame: pd.DataFrame) -> FakeRegistry:
    """A :class:`FakeRegistry` fitted to the test frame."""
    return FakeRegistry(rate_frame)


@pytest.fixture
def forecaster(registry: FakeRegistry, history: RateHistory) -> Forecaster:
    """A forecaster wired to fake models, with caching enabled."""
    return Forecaster(registry, history)  # type: ignore[arg-type]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings pointing at a temporary directory, with limits relaxed."""
    return Settings(
        data_path=tmp_path / "data.csv",
        model_dir=tmp_path / "artifacts",
        log_level="WARNING",
        log_json=True,
        rate_limit="10000 per minute",
        max_data_staleness_days=10_000,
    )


@pytest.fixture
def services(
    settings: Settings,
    history: RateHistory,
    registry: FakeRegistry,
    forecaster: Forecaster,
) -> Services:
    """A service container assembled entirely from fakes."""
    return Services(
        settings=settings,
        history=history,
        registry=registry,  # type: ignore[arg-type]
        forecaster=forecaster,
    )


@pytest.fixture
def app(settings: Settings, services: Services) -> Iterator[Any]:
    """A Flask app built from fake services, so no artifacts are loaded."""
    application = create_app(settings, services=services)
    application.config.update(TESTING=True)
    yield application


@pytest.fixture
def client(app: Any) -> Any:
    """A Flask test client."""
    return app.test_client()


@pytest.fixture
def csv_path(tmp_path: Path) -> Path:
    """Path for a temporary long-format CSV written by data tests."""
    return tmp_path / "rates.csv"


def write_long_csv(path: Path, rows: list[tuple[str, str, float]]) -> Path:
    """Write ``(date, currency_name, value)`` rows in the source CSV format.

    Args:
        path: Destination file.
        rows: Rows to write; dates must use ``DD-MM-YYYY``.

    Returns:
        ``path``, for convenient chaining.
    """
    frame = pd.DataFrame(rows, columns=["date", "currency_name", "value"])
    frame["unit"] = "value in indian rupees per unit of foreign currency"
    frame["note"] = ""
    frame.to_csv(path, index=False)
    return path


def full_history_rows(
    days: int,
    *,
    start: str = "2024-01-01",
    zero_euro_days: int = 0,
) -> list[tuple[str, str, float]]:
    """Build synthetic long-format rows for every currency.

    Args:
        days: Number of consecutive dates to generate.
        start: First date, ISO formatted.
        zero_euro_days: Number of leading dates where the Euro is recorded as
            ``0.0``, reproducing the pre-1999 sentinel in the real dataset.

    Returns:
        Rows suitable for :func:`write_long_csv`.
    """
    dates = pd.date_range(start, periods=days, freq="D")
    rows: list[tuple[str, str, float]] = []
    for offset, timestamp in enumerate(dates):
        stamp = timestamp.strftime("%d-%m-%Y")
        for currency, base in _BASE_RATES.items():
            value = base + offset * 0.01
            if currency == "Euro" and offset < zero_euro_days:
                value = 0.0
            rows.append((stamp, currency, value))
    return rows


@pytest.fixture
def real_history() -> RateHistory:
    """The genuine project dataset, for integration-level assertions."""
    path = Path(__file__).resolve().parents[1] / "data" / "newdataset.csv"
    if not path.exists():  # pragma: no cover - dataset ships with the repo
        pytest.skip("project dataset is not available")
    return load_rate_history(path)
