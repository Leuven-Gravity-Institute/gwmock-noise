"""Tests for optional output adapters."""

from __future__ import annotations

import pytest

from gwmock_noise import output


def test_output_module_rejects_unknown_export() -> None:
    """Unknown lazy exports raise AttributeError."""
    with pytest.raises(AttributeError, match="has no attribute"):
        _ = output.DoesNotExist
