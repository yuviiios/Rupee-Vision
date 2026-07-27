"""Thread-safe, lazily-initialised registry for models and scalers.

Three problems in the original design are addressed:

1. **Module-import side effects.** ``flaskapp-1.py`` read the CSV and called
   ``load_model`` four times at import time, so importing the app cost several
   seconds and required artifacts on disk — which made the module effectively
   untestable.
2. **No thread safety.** Keras models were shared mutable globals invoked from
   Flask's threaded server. Concurrent calls into the same model can interleave
   and produce nondeterministic output. Inference is now serialised per currency.
3. **No artifact provenance.** Nothing recorded which data, code revision or
   sequence length produced a set of weights, so a retrain with a different
   window would be served silently by a mismatched API.
"""

from __future__ import annotations

import json
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rupeevision.core.constants import (
    CURRENCIES,
    SEQ_LENGTH,
    legacy_model_filename,
    legacy_scaler_filename,
    model_filename,
    scaler_filename,
)
from rupeevision.core.exceptions import ModelMismatchError, ModelNotFoundError
from rupeevision.core.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - import-time only for type checkers
    from sklearn.preprocessing import MinMaxScaler

logger = get_logger(__name__)

#: Filename of the optional sibling metadata document written by the trainer.
METADATA_FILENAME = "metadata.json"


@dataclass(frozen=True, slots=True)
class CurrencyModel:
    """A loaded model, its scaler, and a lock guarding inference.

    Attributes:
        currency: Dataset currency name.
        model: Compiled-free Keras model used for forward passes only.
        scaler: Fitted ``MinMaxScaler`` mapping prices to the model's input range.
        lock: Serialises calls into ``model``; Keras models are not re-entrant.
        seq_length: Input window the model was trained with.
    """

    currency: str
    model: Any
    scaler: MinMaxScaler
    lock: threading.Lock
    seq_length: int


class ModelRegistry:
    """Loads and caches one :class:`CurrencyModel` per currency on first use.

    Loading is guarded by a double-checked lock so that concurrent first requests
    do not each pay the (multi-second) cost of deserialising the same artifact.
    """

    def __init__(
        self,
        model_dir: Path,
        *,
        currencies: tuple[str, ...] = CURRENCIES,
        seq_length: int = SEQ_LENGTH,
    ) -> None:
        self._model_dir = model_dir
        self._currencies = currencies
        self._seq_length = seq_length
        self._cache: dict[str, CurrencyModel] = {}
        self._load_lock = threading.Lock()
        self._metadata = self._read_metadata()

    @property
    def currencies(self) -> tuple[str, ...]:
        """Currencies this registry can serve."""
        return self._currencies

    @property
    def metadata(self) -> dict[str, Any]:
        """Artifact provenance, or an empty mapping if none was written."""
        return self._metadata

    def get(self, currency: str) -> CurrencyModel:
        """Return the :class:`CurrencyModel` for ``currency``, loading if needed.

        Raises:
            ModelNotFoundError: If the model or scaler file is absent.
            ModelMismatchError: If artifact metadata conflicts with this build.
        """
        cached = self._cache.get(currency)
        if cached is not None:
            return cached

        with self._load_lock:
            # Re-check: another thread may have loaded it while we waited.
            cached = self._cache.get(currency)
            if cached is not None:
                return cached
            loaded = self._load(currency)
            self._cache[currency] = loaded
            return loaded

    def warmup(self) -> None:
        """Eagerly load every currency.

        Called from the readiness probe and (optionally) at startup so that the
        first real request does not absorb deserialisation latency.
        """
        for currency in self._currencies:
            self.get(currency)

    def is_loaded(self) -> bool:
        """Whether every configured currency is resident in the cache."""
        return all(currency in self._cache for currency in self._currencies)

    def _load(self, currency: str) -> CurrencyModel:
        """Deserialise one model and scaler pair and verify its contract."""
        # Imported lazily: TensorFlow costs seconds to import, and unit tests that
        # stub the registry should not pay for it.
        import keras

        model_path = self._resolve(model_filename(currency), legacy_model_filename(currency))
        scaler_path = self._resolve(scaler_filename(currency), legacy_scaler_filename(currency))

        logger.info(
            "loading_artifacts",
            currency=currency,
            model=model_path.name,
            scaler=scaler_path.name,
        )
        # compile=False skips rebuilding an optimizer that inference never uses:
        # it is faster and avoids spurious "compiled metrics" warnings.
        model = keras.models.load_model(model_path, compile=False)
        scaler = self._load_scaler(scaler_path, currency)

        seq_length = int(self._metadata.get("seq_length", self._seq_length))
        if seq_length != self._seq_length:
            msg = (
                "artifact was trained with a different input window than this "
                "build expects; retrain or deploy matching artifacts"
            )
            raise ModelMismatchError(msg)
        self._verify_input_shape(model, currency)

        return CurrencyModel(
            currency=currency,
            model=model,
            scaler=scaler,
            lock=threading.Lock(),
            seq_length=seq_length,
        )

    def _load_scaler(self, path: Path, currency: str) -> MinMaxScaler:
        """Deserialise a pickled scaler, surfacing version skew as a log event.

        ``joblib.load`` of a scikit-learn estimator built by a different library
        version emits ``InconsistentVersionWarning`` to stderr, which is invisible
        in a JSON log pipeline. Because a silently mis-behaving scaler corrupts
        every prediction, the skew is captured and logged as structured data with
        both versions recorded.
        """
        import joblib
        import sklearn
        from sklearn.exceptions import InconsistentVersionWarning

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", InconsistentVersionWarning)
            scaler: MinMaxScaler = joblib.load(path)

        for entry in caught:
            if issubclass(entry.category, InconsistentVersionWarning):
                logger.warning(
                    "scaler_version_skew",
                    currency=currency,
                    artifact=path.name,
                    fitted_with=getattr(entry.message, "original_sklearn_version", "unknown"),
                    running=sklearn.__version__,
                    detail=(
                        "Scaler was pickled by a different scikit-learn version. "
                        "Retrain to guarantee identical scaling behaviour."
                    ),
                )
        return scaler

    def _verify_input_shape(self, model: Any, currency: str) -> None:
        """Assert the model accepts ``(batch, seq_length, 1)`` inputs.

        Catches a window mismatch even when no metadata document exists, which is
        the case for artifacts produced before provenance was recorded.
        """
        shape = getattr(model, "input_shape", None)
        if not shape or len(shape) != 3:
            logger.warning("model_input_shape_unknown", currency=currency, shape=shape)
            return
        _, timesteps, features = shape
        if timesteps not in (None, self._seq_length) or features not in (None, 1):
            msg = (
                f"{currency} model expects input shape {shape}, but this build "
                f"supplies (None, {self._seq_length}, 1)"
            )
            raise ModelMismatchError(msg)

    def _resolve(self, preferred: str, legacy: str) -> Path:
        """Return the artifact path, preferring the modern filename.

        Raises:
            ModelNotFoundError: If neither filename exists.
        """
        candidate = self._model_dir / preferred
        if candidate.exists():
            return candidate
        fallback = self._model_dir / legacy
        if fallback.exists():
            logger.warning("using_legacy_artifact_filename", filename=legacy)
            return fallback
        msg = f"required artifact {preferred!r} was not found in the artifact directory"
        raise ModelNotFoundError(msg)

    def _read_metadata(self) -> dict[str, Any]:
        """Read ``metadata.json`` from the artifact directory if present."""
        path = self._model_dir / METADATA_FILENAME
        if not path.exists():
            logger.warning("artifact_metadata_missing", path=str(path.name))
            return {}
        try:
            with path.open(encoding="utf-8") as handle:
                data: dict[str, Any] = json.load(handle)
        except (OSError, json.JSONDecodeError):
            logger.exception("artifact_metadata_unreadable", path=str(path.name))
            return {}
        logger.info(
            "artifact_metadata_loaded",
            trained_at=data.get("trained_at"),
            git_sha=data.get("git_sha"),
        )
        return data
