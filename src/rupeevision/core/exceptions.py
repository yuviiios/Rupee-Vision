"""Domain exception hierarchy.

Every error the API can surface derives from :class:`RupeeVisionError`, carrying
an HTTP status and a stable machine-readable ``code``. This lets the API register
a single set of handlers (see :mod:`rupeevision.api.errors`) instead of the
previous bare ``except Exception`` that turned validation mistakes into HTTP 500
and leaked internal exception text to the client.
"""

from __future__ import annotations

from http import HTTPStatus


class RupeeVisionError(Exception):
    """Base class for all application errors.

    Attributes:
        status_code: HTTP status to return for this error.
        code: Stable, machine-readable identifier for API clients.
        message: Client-safe description. Must never contain filesystem paths,
            stack traces or dependency internals.
    """

    status_code: int = HTTPStatus.INTERNAL_SERVER_ERROR
    code: str = "internal_error"

    def __init__(self, message: str = "An unexpected error occurred.") -> None:
        super().__init__(message)
        self.message = message


class ValidationError(RupeeVisionError):
    """Client supplied a malformed or out-of-range request."""

    status_code = HTTPStatus.BAD_REQUEST
    code = "validation_error"

    def __init__(self, message: str, *, details: object | None = None) -> None:
        super().__init__(message)
        self.details = details


class DataError(RupeeVisionError):
    """The source dataset is missing, unreadable or fails its integrity contract."""

    status_code = HTTPStatus.SERVICE_UNAVAILABLE
    code = "data_unavailable"


class ModelNotFoundError(RupeeVisionError):
    """A required model or scaler artifact is absent from the artifact directory."""

    status_code = HTTPStatus.SERVICE_UNAVAILABLE
    code = "model_unavailable"


class ModelMismatchError(RupeeVisionError):
    """Artifact metadata is incompatible with the running code.

    Raised when, for example, a model was trained with a different sequence
    length than :data:`rupeevision.core.constants.SEQ_LENGTH`. Serving such a
    model would produce silently wrong numbers.
    """

    status_code = HTTPStatus.INTERNAL_SERVER_ERROR
    code = "model_mismatch"


class InferenceError(RupeeVisionError):
    """Forecast computation failed."""

    status_code = HTTPStatus.INTERNAL_SERVER_ERROR
    code = "inference_failed"
