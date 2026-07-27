"""Tests for environment-driven configuration."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rupeevision.core.config import Settings, get_settings


def test_defaults_are_safe() -> None:
    """Debug must default to off: the Werkzeug debugger permits code execution."""
    settings = Settings()

    assert settings.debug is False
    assert settings.host == "127.0.0.1"
    assert "*" not in settings.allowed_origins


def test_environment_overrides_are_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUPEEVISION_PORT", "8080")
    monkeypatch.setenv("RUPEEVISION_LOG_LEVEL", "debug")
    monkeypatch.setenv("RUPEEVISION_DEBUG", "true")

    settings = Settings()

    assert settings.port == 8080
    assert settings.log_level == "DEBUG"
    assert settings.debug is True


def test_origins_accept_a_comma_separated_string(monkeypatch: pytest.MonkeyPatch) -> None:
    """Comma-separated is the ergonomic form for a shell or Docker env var."""
    monkeypatch.setenv(
        "RUPEEVISION_ALLOWED_ORIGINS",
        "https://a.example, https://b.example",
    )

    assert Settings().allowed_origins == ["https://a.example", "https://b.example"]


def test_origins_accept_a_json_array(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUPEEVISION_ALLOWED_ORIGINS", '["https://a.example"]')

    assert Settings().allowed_origins == ["https://a.example"]


def test_invalid_log_level_fails_at_startup() -> None:
    """Misconfiguration should crash on boot, not at the first log call."""
    with pytest.raises(ValidationError, match="log_level"):
        Settings(log_level="LOUD")


@pytest.mark.parametrize("port", [0, -1, 70_000])
def test_invalid_port_is_rejected(port: int) -> None:
    with pytest.raises(ValidationError):
        Settings(port=port)


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()
