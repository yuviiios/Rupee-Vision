"""Environment-driven application settings.

Every value that differs between a laptop, CI and a deployed container lives
here and is read from the environment (or a local ``.env``). The previous code
hardcoded the debug flag, the CORS policy and the artifact paths, which made the
service impossible to deploy without editing source.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

#: Repository root, resolved from this file's location (``src/rupeevision/core``).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Runtime configuration, populated from environment variables.

    All fields may be overridden with a ``RUPEEVISION_``-prefixed environment
    variable, e.g. ``RUPEEVISION_LOG_LEVEL=DEBUG``.
    """

    model_config = SettingsConfigDict(
        env_prefix="RUPEEVISION_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -- Server ---------------------------------------------------------------
    host: str = Field(default="127.0.0.1", description="Bind address for the dev server.")
    port: int = Field(default=5000, ge=1, le=65535)
    debug: bool = Field(
        default=False,
        description=(
            "Enables the Werkzeug interactive debugger. Never enable on a "
            "network-reachable deployment: the debugger allows arbitrary code "
            "execution."
        ),
    )

    # -- Observability --------------------------------------------------------
    log_level: str = Field(default="INFO")
    log_json: bool = Field(
        default=True,
        description="Emit JSON logs (production) instead of coloured console logs.",
    )

    # -- Security -------------------------------------------------------------
    # NoDecode disables pydantic-settings' eager JSON parsing of complex types.
    # Without it, a plain "a,b" environment value raises SettingsError before any
    # validator runs, so the comma-separated form documented in .env.example would
    # crash the process at startup.
    allowed_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"],
        description=(
            "CORS allowlist. Replaces the previous wildcard policy. Accepts a "
            "comma-separated string or a JSON array."
        ),
    )
    rate_limit: str = Field(
        default="60 per minute",
        description="Flask-Limiter expression applied to prediction endpoints.",
    )
    max_content_length: int = Field(
        default=16 * 1024,
        ge=1,
        description="Maximum accepted request body size in bytes.",
    )

    # -- Data and artifacts ---------------------------------------------------
    data_path: Path = Field(default=PROJECT_ROOT / "data" / "newdataset.csv")
    model_dir: Path = Field(default=PROJECT_ROOT / "artifacts")

    # -- Performance ----------------------------------------------------------
    forecast_cache_size: int = Field(
        default=8,
        ge=0,
        description=(
            "Number of distinct forecast horizons to memoise. Inputs are fixed "
            "until the dataset changes, so repeat requests are served for free."
        ),
    )
    max_data_staleness_days: int = Field(
        default=7,
        ge=1,
        description="Readiness probe fails past this age of the newest observation.",
    )

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        """Parse the CORS allowlist from either a JSON array or ``a,b``.

        Because :class:`NoDecode` suppresses automatic JSON decoding, this
        validator owns both forms.
        """
        if not isinstance(value, str):
            return value
        text = value.strip()
        if text.startswith("["):
            import json

            try:
                return json.loads(text)
            except json.JSONDecodeError as exc:
                msg = f"allowed_origins is not valid JSON: {text!r}"
                raise ValueError(msg) from exc
        return [origin.strip() for origin in text.split(",") if origin.strip()]

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        """Normalise and reject unknown log level names early, at startup."""
        level = value.upper()
        valid = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}
        if level not in valid:
            msg = f"log_level must be one of {sorted(valid)}, got {value!r}"
            raise ValueError(msg)
        return level


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached so that the ``.env`` file is parsed once and every module observes
    identical configuration.
    """
    return Settings()
