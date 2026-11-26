"""Dataset loading, cleaning and integrity validation.

The raw ``newdataset.csv`` is a long-format table of daily RBI reference rates
(INR per unit of foreign currency) covering 1998-08-25 to 2025-03-28.

Two data-quality defects in the original pipeline are corrected here:

1. **Sentinel zeros were treated as real prices.** The Euro is recorded as
   ``0.0`` for 81 sessions from 1998-08-25 to 1999-01-01 because the currency did
   not exist yet, and 1999-01-01 is ``0.0`` for the Yen and Sterling as well (a
   holiday row). The old code called ``df.bfill()``, which only replaces ``NaN``
   and therefore replaced nothing. Those zeros then reached ``MinMaxScaler``,
   pinning the feature minimum at 0 and wasting roughly a third of the scaled
   range on a value that can never recur.
2. **No integrity contract.** Nothing asserted a sorted index, positive prices or
   the presence of every expected currency, so a malformed CSV would surface as a
   confusing error deep inside inference instead of at load time.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from rupeevision.core.constants import CURRENCIES, CURRENCY_CODES, SEQ_LENGTH
from rupeevision.core.exceptions import DataError
from rupeevision.core.logging import get_logger

logger = get_logger(__name__)

#: Source CSV date format (day-first, as published).
_DATE_FORMAT = "%d-%m-%Y"

#: Columns actually needed; ``unit`` is a constant string and ``note`` is empty.
_USED_COLUMNS = ["date", "currency_name", "value"]


@dataclass(frozen=True, slots=True)
class RateHistory:
    """An immutable, validated wide-format view of the exchange-rate history.

    Attributes:
        frame: Index is a sorted ``DatetimeIndex``; one float column per currency
            in :data:`CURRENCIES`. Guaranteed free of ``NaN`` and non-positive
            values.
        as_of: Date of the most recent observation. Returned to API clients so a
            stale dataset is visible rather than implied to be current.
    """

    frame: pd.DataFrame
    as_of: date

    def window(self, currency: str, length: int) -> np.ndarray:
        """Return the trailing ``length`` observations for ``currency``.

        Args:
            currency: Dataset currency name, e.g. ``"US Dollar"``.
            length: Number of trailing rows to return.

        Returns:
            Array of shape ``(length, 1)`` and dtype ``float32``.

        Raises:
            DataError: If the currency is absent or has fewer than ``length`` rows.
        """
        if currency not in self.frame.columns:
            msg = f"currency {currency!r} is not present in the dataset"
            raise DataError(msg)
        series = self.frame[currency].dropna()
        if len(series) < length:
            msg = (
                f"{currency} has {len(series)} observations, "
                f"but {length} are required to seed a forecast"
            )
            raise DataError(msg)
        return series.to_numpy(dtype="float32")[-length:].reshape(length, 1)

    def latest(self, currency: str) -> float:
        """Return the most recent observed rate for ``currency``."""
        return float(self.frame[currency].dropna().iloc[-1])


def load_rate_history(
    path: Path,
    *,
    currencies: tuple[str, ...] = CURRENCIES,
) -> RateHistory:
    """Load, clean and validate the exchange-rate dataset.

    Pipeline: read only the needed columns → parse day-first dates → pivot to wide
    format → replace sentinel zeros with ``NaN`` → drop all-missing (holiday) rows
    → forward-fill genuine intra-series gaps → validate the result.

    Forward-fill is used rather than the original backward-fill: ``bfill``
    propagates a *future* price into a past gap, which is look-ahead leakage in a
    time series. ``ffill`` carries the last known price forward, which is what a
    market participant would actually have observed.

    Args:
        path: Path to the long-format CSV.
        currencies: Currencies required to be present.

    Returns:
        A validated :class:`RateHistory`.

    Raises:
        DataError: If the file is missing, malformed, or fails validation.
    """
    if not path.exists():
        msg = f"dataset not found at the configured path: {path.name}"
        raise DataError(msg)

    try:
        columns = pd.read_csv(path, nrows=0).columns
        if {"currency", "rate", "unit", "date"}.issubset(columns):
            return archive_history(pd.read_csv(path, usecols=["currency", "rate", "unit", "date"]))
        raw = pd.read_csv(path, usecols=_USED_COLUMNS, dtype={"currency_name": "string"})
    except (OSError, ValueError) as exc:
        logger.exception("dataset_read_failed", path=str(path))
        msg = f"dataset at {path.name} could not be parsed"
        raise DataError(msg) from exc

    raw["date"] = pd.to_datetime(raw["date"], format=_DATE_FORMAT, errors="coerce")
    bad_dates = int(raw["date"].isna().sum())
    if bad_dates:
        logger.warning("dropping_unparseable_dates", count=bad_dates)
        raw = raw.dropna(subset=["date"])

    duplicates = int(raw.duplicated(subset=["date", "currency_name"]).sum())
    if duplicates:
        # Keep the last occurrence: later rows in the source represent corrections.
        logger.warning("dropping_duplicate_observations", count=duplicates)
        raw = raw.drop_duplicates(subset=["date", "currency_name"], keep="last")

    wide = raw.pivot(index="date", columns="currency_name", values="value")
    wide = wide.sort_index()

    missing = [currency for currency in currencies if currency not in wide.columns]
    if missing:
        msg = f"dataset is missing required currencies: {', '.join(missing)}"
        raise DataError(msg)

    wide = wide.loc[:, list(currencies)].astype("float64")
    wide.columns.name = None

    cleaned = _clean(wide)
    _validate(cleaned, currencies)

    as_of = cleaned.index[-1].date()
    logger.info(
        "dataset_loaded",
        rows=len(cleaned),
        currencies=list(currencies),
        first_observation=str(cleaned.index[0].date()),
        as_of=str(as_of),
    )
    return RateHistory(frame=cleaned, as_of=as_of)


def _clean(wide: pd.DataFrame) -> pd.DataFrame:
    """Replace sentinel values, drop empty rows and forward-fill real gaps."""
    # Non-positive rates are impossible; they are placeholders for "not yet
    # issued" (pre-1999 Euro) or "market closed" (1999-01-01).
    sentinels = int((wide <= 0).sum().sum())
    if sentinels:
        logger.info("masking_non_positive_sentinels", count=sentinels)
    frame = wide.mask(wide <= 0)

    empty_rows = int(frame.isna().all(axis=1).sum())
    if empty_rows:
        logger.info("dropping_market_holiday_rows", count=empty_rows)
    frame = frame.dropna(how="all")

    # Leading NaNs (a currency that did not exist yet) are intentionally left in
    # place: forward-fill cannot invent a price, and RateHistory.window() drops
    # them per-currency. Only interior gaps are filled.
    return frame.ffill()


def _validate(frame: pd.DataFrame, currencies: tuple[str, ...]) -> None:
    """Assert the integrity contract, raising :class:`DataError` on violation."""
    if frame.empty:
        msg = "dataset contains no usable observations after cleaning"
        raise DataError(msg)

    if not frame.index.is_monotonic_increasing:
        msg = "dataset index is not chronologically sorted"
        raise DataError(msg)

    if frame.index.has_duplicates:
        msg = "dataset contains duplicate dates after pivoting"
        raise DataError(msg)

    for currency in currencies:
        column = frame[currency]
        series = column.dropna()
        if series.empty:
            msg = f"{currency} has no valid observations after cleaning"
            raise DataError(msg)
        if (series <= 0).any():
            msg = f"{currency} still contains non-positive rates after cleaning"
            raise DataError(msg)
        # After forward-fill, any remaining NaNs must form a leading prefix (the
        # period before the currency existed). A gap after the first real
        # observation means ffill did not apply and the series is unusable.
        missing = column.isna().to_numpy()
        first_valid = int(missing.argmin())
        if missing[first_valid:].any():
            msg = f"{currency} contains interior gaps after forward-fill"
            raise DataError(msg)


def archive_history(rows: pd.DataFrame) -> RateHistory:
    """Validate archived fixing units and keep jointly observed trading sessions."""
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
    frame = frame.dropna()
    if len(frame) < SEQ_LENGTH + 1:
        raise ValueError("Too few complete trading sessions")
    return RateHistory(frame=frame, as_of=as_of)
