"""Tests for dataset loading, cleaning and integrity validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from rupeevision.core.constants import CURRENCIES, SEQ_LENGTH
from rupeevision.core.exceptions import DataError
from rupeevision.ml.data import RateHistory, load_rate_history
from tests.conftest import full_history_rows, write_long_csv


def test_loads_valid_dataset(csv_path: Path) -> None:
    write_long_csv(csv_path, full_history_rows(60))
    history = load_rate_history(csv_path)

    assert len(history.frame) == 60
    assert list(history.frame.columns) == list(CURRENCIES)
    assert history.as_of.isoformat() == "2024-02-29"


def test_masks_zero_sentinels_instead_of_treating_them_as_prices(csv_path: Path) -> None:
    """Zeros are placeholders, not rates.

    Regression test for the original ``bfill``-only cleaning, which left the
    pre-1999 Euro zeros in place and pinned the scaler minimum at 0.
    """
    write_long_csv(csv_path, full_history_rows(60, zero_euro_days=10))
    history = load_rate_history(csv_path)

    euro = history.frame["Euro"]
    assert not (euro.dropna() <= 0).any()
    # The leading sentinel window is NaN, not 0, and not back-filled.
    assert euro.iloc[:10].isna().all()
    assert euro.iloc[10:].notna().all()
    assert history.frame["Euro"].min() > 0


def test_leading_gap_does_not_shorten_other_currencies(csv_path: Path) -> None:
    """A currency that did not exist yet must not truncate the others."""
    write_long_csv(csv_path, full_history_rows(60, zero_euro_days=10))
    history = load_rate_history(csv_path)

    assert len(history.window("US Dollar", SEQ_LENGTH)) == SEQ_LENGTH
    assert len(history.window("Euro", SEQ_LENGTH)) == SEQ_LENGTH


def test_drops_market_holiday_rows(csv_path: Path) -> None:
    """A date where every currency is zero is a closed market, not data."""
    rows = full_history_rows(40)
    rows += [("01-01-2025", currency, 0.0) for currency in CURRENCIES]
    write_long_csv(csv_path, rows)

    history = load_rate_history(csv_path)
    assert pd.Timestamp("2025-01-01") not in history.frame.index


def test_forward_fills_interior_gaps_without_lookahead(csv_path: Path) -> None:
    """Interior gaps carry the last known price forward, never a future one."""
    rows = [row for row in full_history_rows(40) if not (row[0] == "10-01-2024" and row[1] == "Euro")]
    write_long_csv(csv_path, rows)

    history = load_rate_history(csv_path)
    euro = history.frame["Euro"]
    previous = euro.loc[pd.Timestamp("2024-01-09")]
    filled = euro.loc[pd.Timestamp("2024-01-10")]
    assert filled == previous


def test_sorts_unordered_input(csv_path: Path) -> None:
    """The real dataset is stored newest-first; the loader must sort it."""
    rows = list(reversed(full_history_rows(40)))
    write_long_csv(csv_path, rows)

    history = load_rate_history(csv_path)
    assert history.frame.index.is_monotonic_increasing


def test_deduplicates_repeated_observations(csv_path: Path) -> None:
    """Later rows are corrections and win."""
    rows = full_history_rows(40)
    rows.append(("01-01-2024", "US Dollar", 99.0))
    write_long_csv(csv_path, rows)

    history = load_rate_history(csv_path)
    assert history.frame.loc[pd.Timestamp("2024-01-01"), "US Dollar"] == 99.0


def test_missing_file_raises_data_error(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="not found"):
        load_rate_history(tmp_path / "absent.csv")


def test_missing_currency_raises_data_error(csv_path: Path) -> None:
    rows = [row for row in full_history_rows(40) if row[1] != "Euro"]
    write_long_csv(csv_path, rows)

    with pytest.raises(DataError, match="missing required currencies"):
        load_rate_history(csv_path)


def test_all_zero_currency_raises_data_error(csv_path: Path) -> None:
    """A currency that is entirely sentinel values cannot be served."""
    rows = [
        (date, currency, 0.0 if currency == "Euro" else value)
        for date, currency, value in full_history_rows(40)
    ]
    write_long_csv(csv_path, rows)

    with pytest.raises(DataError, match="Euro"):
        load_rate_history(csv_path)


def test_unparseable_dates_are_dropped(csv_path: Path) -> None:
    rows = full_history_rows(40)
    rows += [("not-a-date", currency, 50.0) for currency in CURRENCIES]
    write_long_csv(csv_path, rows)

    history = load_rate_history(csv_path)
    assert len(history.frame) == 40


def test_window_returns_trailing_rows_in_order(history: RateHistory) -> None:
    window = history.window("US Dollar", SEQ_LENGTH)

    assert window.shape == (SEQ_LENGTH, 1)
    expected = history.frame["US Dollar"].to_numpy()[-SEQ_LENGTH:]
    assert window.ravel() == pytest.approx(expected, rel=1e-6)


def test_window_rejects_unknown_currency(history: RateHistory) -> None:
    with pytest.raises(DataError, match="not present"):
        history.window("Swiss Franc", SEQ_LENGTH)


def test_window_rejects_insufficient_history(csv_path: Path) -> None:
    write_long_csv(csv_path, full_history_rows(SEQ_LENGTH - 1))
    history = load_rate_history(csv_path)

    with pytest.raises(DataError, match="required to seed a forecast"):
        history.window("US Dollar", SEQ_LENGTH)


def test_latest_returns_most_recent_observation(history: RateHistory) -> None:
    assert history.latest("US Dollar") == pytest.approx(82.0)


class TestRealDataset:
    """Assertions against the dataset that actually ships with the repository."""

    def test_known_shape_and_range(self, real_history: RateHistory) -> None:
        assert real_history.as_of.isoformat() == "2025-03-28"
        assert len(real_history.frame) == 5527

    def test_no_sentinel_values_survive_cleaning(self, real_history: RateHistory) -> None:
        """The 83 zeros in the raw CSV must not reach the model."""
        assert int((real_history.frame <= 0).sum().sum()) == 0

    def test_euro_history_starts_after_introduction(self, real_history: RateHistory) -> None:
        """The Euro did not exist before 1999; its zeros must not become prices."""
        first_valid = real_history.frame["Euro"].first_valid_index()
        assert first_valid is not None
        assert first_valid.year == 1999

    def test_every_currency_has_enough_history_to_forecast(
        self, real_history: RateHistory
    ) -> None:
        for currency in CURRENCIES:
            assert real_history.window(currency, SEQ_LENGTH).shape == (SEQ_LENGTH, 1)
