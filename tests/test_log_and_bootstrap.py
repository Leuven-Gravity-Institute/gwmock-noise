"""Tests for logging utilities and package bootstrap entry points."""

from __future__ import annotations

import importlib
import importlib.metadata
import logging
import runpy
from pathlib import Path

import pytest

import gwmock_noise.version as version_module
from gwmock_noise.utils import log as log_utils


def _reset_gwmock_logger() -> logging.Logger:
    """Reset logger handlers between tests."""
    logger = logging.getLogger("gwmock-noise")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.propagate = False
    return logger


def test_setup_logger_rejects_unknown_log_level() -> None:
    """setup_logger rejects invalid string log levels."""
    _reset_gwmock_logger()
    with pytest.raises(ValueError, match="not understood"):
        log_utils.setup_logger(log_level="not-a-level")


def test_get_version_information_returns_version_string() -> None:
    """get_version_information returns the package version string."""
    assert isinstance(log_utils.get_version_information(), str)
    assert log_utils.get_version_information()


def test_setup_logger_adds_stream_and_file_handlers_once(tmp_path: Path) -> None:
    """setup_logger creates stream/file handlers and does not duplicate them."""
    logger = _reset_gwmock_logger()

    log_utils.setup_logger(outdir=str(tmp_path), label="run", log_level="INFO")
    log_utils.setup_logger(outdir=str(tmp_path), label="run", log_level="DEBUG")

    stream_handlers = [
        handler
        for handler in logger.handlers
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler)
    ]
    file_handlers = [handler for handler in logger.handlers if isinstance(handler, logging.FileHandler)]

    assert len(stream_handlers) == 1
    assert len(file_handlers) == 1
    assert (tmp_path / "run.log").exists()
    assert all(handler.level == logging.DEBUG for handler in logger.handlers)


def test_setup_logger_prints_version_with_numeric_level(monkeypatch: pytest.MonkeyPatch) -> None:
    """setup_logger supports numeric levels and emits version log output."""
    logger = _reset_gwmock_logger()
    emitted: list[str] = []

    class CollectHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            emitted.append(record.getMessage())

    logger.addHandler(CollectHandler())
    monkeypatch.setattr(log_utils, "get_version_information", lambda: "9.9.9")

    log_utils.setup_logger(log_level=logging.INFO, print_version=True)

    assert any("9.9.9" in message for message in emitted)


def test_main_module_calls_setup_logger(monkeypatch: pytest.MonkeyPatch) -> None:
    """Running gwmock_noise.__main__ calls setup_logger(print_version=True)."""
    called: list[bool] = []

    def _fake_setup_logger(*, print_version: bool = False, **_: object) -> None:
        called.append(print_version)

    monkeypatch.setattr("gwmock_noise.utils.log.setup_logger", _fake_setup_logger)
    runpy.run_module("gwmock_noise.__main__", run_name="__main__")

    assert called == [True]


def test_version_module_uses_fallback_when_package_metadata_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Version module falls back when package metadata is unavailable."""
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda _: (_ for _ in ()).throw(importlib.metadata.PackageNotFoundError()),
    )
    reloaded = importlib.reload(version_module)
    assert reloaded.__version__ == "0+unknown"
