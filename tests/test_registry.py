"""Tests for artifact resolution, metadata handling and thread-safe loading."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from rupeevision.core.constants import (
    SEQ_LENGTH,
    artifact_slug,
    legacy_model_filename,
    model_filename,
    scaler_filename,
)
from rupeevision.core.exceptions import ModelNotFoundError
from rupeevision.ml.registry import ModelRegistry


@pytest.mark.parametrize(
    ("currency", "expected"),
    [
        ("US Dollar", "us_dollar"),
        ("Japanese Yen", "japanese_yen"),
        ("Euro", "euro"),
        ("Pound Sterling", "pound_sterling"),
    ],
)
def test_artifact_slug_is_filesystem_safe(currency: str, expected: str) -> None:
    """Slugs must not contain spaces; the original filenames did."""
    slug = artifact_slug(currency)

    assert slug == expected
    assert " " not in slug


def test_canonical_filenames_use_the_modern_keras_format() -> None:
    assert model_filename("US Dollar") == "lstm_us_dollar.keras"
    assert legacy_model_filename("US Dollar") == "lstm_us_dollar.h5"
    assert scaler_filename("Japanese Yen") == "scaler_japanese_yen.pkl"


def test_missing_artifact_raises_a_clear_error(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path)

    with pytest.raises(ModelNotFoundError, match="was not found"):
        registry.get("US Dollar")


def test_error_message_does_not_leak_absolute_paths(tmp_path: Path) -> None:
    """Client-facing messages must not disclose the server filesystem layout."""
    registry = ModelRegistry(tmp_path)

    with pytest.raises(ModelNotFoundError) as excinfo:
        registry.get("US Dollar")

    assert str(tmp_path) not in excinfo.value.message


def test_metadata_is_read_when_present(tmp_path: Path) -> None:
    payload = {"seq_length": SEQ_LENGTH, "git_sha": "abc123", "trained_at": "2026-01-01"}
    (tmp_path / "metadata.json").write_text(json.dumps(payload), encoding="utf-8")

    registry = ModelRegistry(tmp_path)

    assert registry.metadata["git_sha"] == "abc123"


def test_corrupt_metadata_is_tolerated(tmp_path: Path) -> None:
    """Unreadable provenance must not prevent the service from starting."""
    (tmp_path / "metadata.json").write_text("{not json", encoding="utf-8")

    assert ModelRegistry(tmp_path).metadata == {}


def test_missing_metadata_is_tolerated(tmp_path: Path) -> None:
    assert ModelRegistry(tmp_path).metadata == {}


def test_is_loaded_is_false_before_any_get(tmp_path: Path) -> None:
    assert ModelRegistry(tmp_path).is_loaded() is False


def test_concurrent_first_load_happens_once(tmp_path: Path) -> None:
    """The double-checked lock must not deserialise the same artifact twice.

    Loading a real model costs seconds, so a thundering herd of first requests
    would otherwise each pay that cost.
    """
    load_count = 0
    barrier = threading.Barrier(5)

    class CountingRegistry(ModelRegistry):
        def _load(self, currency: str) -> object:  # type: ignore[override]
            nonlocal load_count
            load_count += 1
            return object()

    registry = CountingRegistry(tmp_path)
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            barrier.wait(timeout=10)
            registry.get("US Dollar")
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)

    assert not errors
    assert load_count == 1
