"""Single source of truth for domain constants shared by training and serving.

Both the training pipeline (``train/``) and the inference API (``rupeevision.api``)
import from this module. Previously the currency list, sequence length and
artifact-naming logic were duplicated across ``train.py`` and ``flaskapp-1.py``,
which meant retraining with a different window silently desynchronised the two
sides (the server kept slicing 30 rows regardless of what the model expected).
"""

from __future__ import annotations

from typing import Final

#: Currencies quoted in INR per unit of foreign currency, exactly as they appear
#: in the ``currency_name`` column of the source dataset.
CURRENCIES: Final[tuple[str, ...]] = (
    "US Dollar",
    "Euro",
    "Japanese Yen",
    "Pound Sterling",
)

#: Number of trailing observations fed to the LSTM to produce one forecast step.
#: Persisted into artifact metadata and asserted at load time so a retrain with a
#: different window cannot be served by a mismatched API.
SEQ_LENGTH: Final[int] = 30

#: Hard upper bound on a single forecast request. Recursive forecasting compounds
#: error, so anything beyond this is noise, and the bound also caps CPU per request.
MAX_FORECAST_DAYS: Final[int] = 30

#: Default horizon when the caller does not specify one.
DEFAULT_FORECAST_DAYS: Final[int] = 7

#: ISO-4217 codes, used for API responses and display. Keyed by dataset name.
CURRENCY_CODES: Final[dict[str, str]] = {
    "US Dollar": "USD",
    "Euro": "EUR",
    "Japanese Yen": "JPY",
    "Pound Sterling": "GBP",
}


def artifact_slug(currency: str) -> str:
    """Return the filesystem-safe slug used in model and scaler filenames.

    Filenames must not contain spaces: the original ``scaler_Japanese Yen.pkl``
    was hostile to shell globbing, URLs and Docker COPY patterns.

    >>> artifact_slug("Japanese Yen")
    'japanese_yen'
    """
    return currency.lower().replace(" ", "_")


def model_filename(currency: str) -> str:
    """Return the canonical model filename for ``currency``."""
    return f"lstm_{artifact_slug(currency)}.keras"


def legacy_model_filename(currency: str) -> str:
    """Return the deprecated Keras HDF5 filename, kept for backward compatibility."""
    return f"lstm_{artifact_slug(currency)}.h5"


def scaler_filename(currency: str) -> str:
    """Return the canonical scaler filename for ``currency``."""
    return f"scaler_{artifact_slug(currency)}.pkl"


def legacy_scaler_filename(currency: str) -> str:
    """Return the original space-containing scaler filename."""
    return f"scaler_{currency}.pkl"
