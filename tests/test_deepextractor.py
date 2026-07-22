"""Tests for the DeepExtractor glitch-reconstruction model."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from gwmock_noise import DeepExtractorGlitch, LogNormalAmplitudeDistribution
from gwmock_noise.glitches import deepextractor as deepextractor_module
from gwmock_noise.glitches.deepextractor import (
    GLITCH_CLASS_NAMES,
    LABEL_ORDER_FILENAME,
    LABELS_FILENAME,
    SAMPLES_FILENAME,
)
from gwmock_noise.glitches.models import normalize_glitch_models
from gwmock_noise.simulators import InjectGlitches
from gwmock_noise.simulators.glitches import _ZeroNoiseSimulator

NATIVE_ROW_LENGTH = 256


def _write_flat_psd(path: Path) -> None:
    """Write a strictly positive flat PSD covering the full test band."""
    frequencies = np.linspace(0.0, 4096.0, 129)
    psd = np.ones_like(frequencies)
    np.savetxt(path, np.column_stack((frequencies, psd)))


def _write_dataset(directory: Path, *, zero_classes: tuple[str, ...] = ()) -> None:
    """Write a small synthetic DeepExtractor dataset (two rows per class)."""
    rng = np.random.default_rng(1234)
    label_order = np.array(list(reversed(GLITCH_CLASS_NAMES)))
    n_rows = 2 * label_order.size

    samples = rng.normal(size=(n_rows, NATIVE_ROW_LENGTH))
    labels = np.zeros((n_rows, label_order.size))
    for row in range(n_rows):
        column = row % label_order.size
        labels[row, column] = 1.0
        if str(label_order[column]) in zero_classes:
            samples[row] = 0.0

    np.save(directory / SAMPLES_FILENAME, samples)
    np.save(directory / LABELS_FILENAME, labels)
    np.save(directory / LABEL_ORDER_FILENAME, label_order)


@pytest.fixture
def hf_stub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Patch the huggingface_hub loader to serve synthetic files from tmp_path."""
    _write_dataset(tmp_path)
    downloads: list[str] = []

    def fake_download(*, repo_id: str, filename: str, repo_type: str, local_files_only: bool = False) -> str:
        assert repo_type == "dataset"
        assert repo_id
        downloads.append(filename)
        return str(tmp_path / filename)

    monkeypatch.setattr(
        deepextractor_module,
        "_load_hf_hub",
        lambda: SimpleNamespace(hf_hub_download=fake_download),
    )
    return downloads


def _make_model(psd_file: Path, **overrides: Any) -> DeepExtractorGlitch:
    """Construct a DeepExtractor model with test-friendly defaults."""
    parameters: dict[str, Any] = {
        "rate": 0.5,
        "amplitude_distribution": LogNormalAmplitudeDistribution(mean=1.0, std=0.0),
        "psd_file": psd_file,
        "snr": 10.0,
    }
    parameters.update(overrides)
    return DeepExtractorGlitch(**parameters)


def _optimal_snr(waveform: np.ndarray, psd_file: Path, sampling_frequency: float) -> float:
    """Recompute the optimal SNR of a waveform against the raw PSD table."""
    table = np.loadtxt(psd_file)
    frequencies = np.fft.rfftfreq(waveform.size, d=1.0 / sampling_frequency)
    psd = np.interp(frequencies, table[:, 0], table[:, 1], left=0.0, right=0.0)
    valid = (frequencies >= 2.0) & (frequencies <= sampling_frequency / 2.0) & (psd > 0.0)
    waveform_fd = np.fft.rfft(waveform) / sampling_frequency
    delta_frequency = sampling_frequency / waveform.size
    return float(np.sqrt(4.0 * delta_frequency * np.sum(np.abs(waveform_fd[valid]) ** 2 / psd[valid])))


def test_rejects_unknown_glitch_class(tmp_path: Path) -> None:
    """Configuration validation rejects class names outside the dataset."""
    psd_file = tmp_path / "psd.txt"
    _write_flat_psd(psd_file)

    with pytest.raises(ValueError, match="Unknown glitch classes"):
        _make_model(psd_file, glitch_classes=["Blip", "Wandering Line"])
    with pytest.raises(ValueError, match="at least one class"):
        _make_model(psd_file, glitch_classes=[])
    with pytest.raises(ValueError, match="duplicates"):
        _make_model(psd_file, glitch_classes=["Blip", "Blip"])


def test_rejects_invalid_snr_configuration(tmp_path: Path) -> None:
    """SNR validation covers scalars and per-class mappings."""
    psd_file = tmp_path / "psd.txt"
    _write_flat_psd(psd_file)

    with pytest.raises(ValueError, match="finite and greater than zero"):
        _make_model(psd_file, snr=0.0)
    with pytest.raises(ValueError, match="missing glitch classes"):
        _make_model(psd_file, glitch_classes=["Blip", "Tomte"], snr={"Blip": 8.0})
    with pytest.raises(ValueError, match="unconfigured glitch classes"):
        _make_model(psd_file, glitch_classes=["Blip"], snr={"Blip": 8.0, "Tomte": 9.0})
    with pytest.raises(ValueError, match="finite and greater than zero"):
        _make_model(psd_file, glitch_classes=["Blip"], snr={"Blip": -3.0})


def test_rejects_invalid_rate_mapping(tmp_path: Path) -> None:
    """Per-class rate mappings are validated like SNR mappings."""
    psd_file = tmp_path / "psd.txt"
    _write_flat_psd(psd_file)

    with pytest.raises(ValueError, match="missing glitch classes"):
        _make_model(psd_file, glitch_classes=["Blip", "Tomte"], rate={"Blip": 0.5})
    with pytest.raises(ValueError, match="unconfigured glitch classes"):
        _make_model(psd_file, glitch_classes=["Blip"], rate={"Blip": 0.5, "Tomte": 0.5})
    with pytest.raises(ValueError, match="finite and non-negative"):
        _make_model(psd_file, glitch_classes=["Blip"], rate={"Blip": -0.5})


def test_per_class_rate_sums_to_total_and_weights_draws(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rate mapping sets the total Poisson rate and weights class draws."""
    _write_dataset(tmp_path, zero_classes=tuple(name for name in GLITCH_CLASS_NAMES if name != "Tomte"))
    monkeypatch.setattr(
        deepextractor_module,
        "_load_hf_hub",
        lambda: SimpleNamespace(
            hf_hub_download=lambda *, repo_id, filename, repo_type, local_files_only=False: str(tmp_path / filename)
        ),
    )
    psd_file = tmp_path / "psd.txt"
    _write_flat_psd(psd_file)

    model = _make_model(
        psd_file,
        glitch_classes=["Tomte", "Whistle"],
        rate={"Tomte": 0.3, "Whistle": 0.0},
        snr={"Tomte": 5.0, "Whistle": 15.0},
    )

    assert model.rate == pytest.approx(0.3)
    # Whistle rows are zeroed, so a zero-rate class must never be drawn: any
    # Whistle draw would raise on its empty frequency content.
    for seed in range(8):
        waveform = model.generate_waveform(4096.0, rng=np.random.default_rng(seed))
        assert _optimal_snr(waveform, psd_file, 4096.0) == pytest.approx(5.0, rel=1e-8)


def test_serialize_reports_per_class_rate(tmp_path: Path) -> None:
    """Serialize echoes the configured per-class rate mapping."""
    psd_file = tmp_path / "psd.txt"
    _write_flat_psd(psd_file)
    model = _make_model(
        psd_file,
        glitch_classes=["Blip", "Tomte"],
        rate={"Blip": 0.2, "Tomte": 0.1},
    )

    serialized = model.serialize()
    assert serialized["rate"] == {"Blip": 0.2, "Tomte": 0.1}


def test_rejects_invalid_frequency_cutoffs(tmp_path: Path) -> None:
    """Cutoff ordering is validated at construction time."""
    psd_file = tmp_path / "psd.txt"
    _write_flat_psd(psd_file)

    with pytest.raises(ValueError, match="high_frequency_cutoff"):
        _make_model(psd_file, low_frequency_cutoff=64.0, high_frequency_cutoff=32.0)


def test_local_files_only_is_forwarded_to_downloader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The local_files_only flag reaches every hf_hub_download call."""
    _write_dataset(tmp_path)
    captured: list[bool] = []

    def fake_download(*, repo_id: str, filename: str, repo_type: str, local_files_only: bool) -> str:
        captured.append(local_files_only)
        return str(tmp_path / filename)

    monkeypatch.setattr(
        deepextractor_module,
        "_load_hf_hub",
        lambda: SimpleNamespace(hf_hub_download=fake_download),
    )
    psd_file = tmp_path / "psd.txt"
    _write_flat_psd(psd_file)
    model = _make_model(psd_file, local_files_only=True)

    model.generate_waveform(4096.0, rng=np.random.default_rng(0))

    assert captured == [True, True, True]


class _StubLocalEntryNotFoundError(FileNotFoundError):
    """Stand-in for huggingface_hub's LocalEntryNotFoundError."""


def _offline_hf_hub(tmp_path: Path, *, cached: bool) -> SimpleNamespace:
    """Build an hf_hub stub whose online download fails as if offline."""

    def fake_download(*, repo_id: str, filename: str, repo_type: str, local_files_only: bool) -> str:
        if not local_files_only:
            raise _StubLocalEntryNotFoundError("Hub unreachable")
        if not cached:
            raise _StubLocalEntryNotFoundError("not in cache")
        return str(tmp_path / filename)

    return SimpleNamespace(
        hf_hub_download=fake_download,
        errors=SimpleNamespace(LocalEntryNotFoundError=_StubLocalEntryNotFoundError),
    )


def test_offline_falls_back_to_cache_with_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unreachable Hub warns about the skipped ETag check and reuses the cache."""
    _write_dataset(tmp_path)
    monkeypatch.setattr(deepextractor_module, "_load_hf_hub", lambda: _offline_hf_hub(tmp_path, cached=True))
    psd_file = tmp_path / "psd.txt"
    _write_flat_psd(psd_file)
    model = _make_model(psd_file)

    with caplog.at_level(logging.WARNING, logger="gwmock-noise"):
        waveform = model.generate_waveform(4096.0, rng=np.random.default_rng(0))

    assert "skipping the ETag check" in caplog.text
    assert np.max(np.abs(waveform)) > 0.0


def test_offline_without_cache_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unreachable Hub with no cached files surfaces the not-found error."""
    _write_dataset(tmp_path)
    monkeypatch.setattr(deepextractor_module, "_load_hf_hub", lambda: _offline_hf_hub(tmp_path, cached=False))
    psd_file = tmp_path / "psd.txt"
    _write_flat_psd(psd_file)
    model = _make_model(psd_file)

    with caplog.at_level(logging.WARNING, logger="gwmock-noise"), pytest.raises(_StubLocalEntryNotFoundError):
        model.generate_waveform(4096.0, rng=np.random.default_rng(0))
    assert "skipping the ETag check" in caplog.text


def test_generate_waveform_requires_optional_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The model raises a focused error when huggingface_hub is unavailable."""
    psd_file = tmp_path / "psd.txt"
    _write_flat_psd(psd_file)
    model = _make_model(psd_file)

    monkeypatch.setattr(
        deepextractor_module.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ModuleNotFoundError(name=name)),
    )

    with pytest.raises(ImportError, match="requires the optional dependency 'huggingface_hub'"):
        model.generate_waveform(4096.0, rng=np.random.default_rng(2))


def test_dataset_download_is_lazy(tmp_path: Path, hf_stub: list[str]) -> None:
    """Construction and serialization never download; generation does once."""
    psd_file = tmp_path / "psd.txt"
    _write_flat_psd(psd_file)
    model = _make_model(psd_file)

    model.serialize()
    assert hf_stub == []

    model.generate_waveform(4096.0, rng=np.random.default_rng(0))
    assert sorted(hf_stub) == sorted([SAMPLES_FILENAME, LABELS_FILENAME, LABEL_ORDER_FILENAME])

    model.generate_waveform(4096.0, rng=np.random.default_rng(1))
    assert len(hf_stub) == 3


def test_generated_waveform_matches_target_snr(tmp_path: Path, hf_stub: list[str]) -> None:
    """The colored waveform reproduces the configured optimal SNR exactly."""
    psd_file = tmp_path / "psd.txt"
    _write_flat_psd(psd_file)
    model = _make_model(psd_file, snr=10.0)

    waveform = model.generate_waveform(4096.0, rng=np.random.default_rng(5))

    assert waveform.shape == (NATIVE_ROW_LENGTH,)
    assert _optimal_snr(waveform, psd_file, 4096.0) == pytest.approx(10.0, rel=1e-8)


def test_per_class_snr_and_class_selection(tmp_path: Path, hf_stub: list[str]) -> None:
    """Per-class SNR mappings apply to draws restricted to that class."""
    psd_file = tmp_path / "psd.txt"
    _write_flat_psd(psd_file)
    model = _make_model(psd_file, glitch_classes=["Tomte"], snr={"Tomte": 5.0})

    for seed in range(4):
        waveform = model.generate_waveform(4096.0, rng=np.random.default_rng(seed))
        assert _optimal_snr(waveform, psd_file, 4096.0) == pytest.approx(5.0, rel=1e-8)


def test_class_selection_respects_label_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Draws honor the dataset's label order rather than a fixed class order."""
    _write_dataset(tmp_path, zero_classes=tuple(name for name in GLITCH_CLASS_NAMES if name != "Whistle"))

    monkeypatch.setattr(
        deepextractor_module,
        "_load_hf_hub",
        lambda: SimpleNamespace(
            hf_hub_download=lambda *, repo_id, filename, repo_type, local_files_only=False: str(tmp_path / filename)
        ),
    )
    psd_file = tmp_path / "psd.txt"
    _write_flat_psd(psd_file)
    model = _make_model(psd_file, glitch_classes=["Whistle"], snr=7.0)

    for seed in range(4):
        waveform = model.generate_waveform(4096.0, rng=np.random.default_rng(seed))
        assert np.max(np.abs(waveform)) > 0.0


def test_amplitude_distribution_scales_on_top_of_snr(tmp_path: Path, hf_stub: list[str]) -> None:
    """A non-unit amplitude mean rescales the SNR-calibrated waveform."""
    psd_file = tmp_path / "psd.txt"
    _write_flat_psd(psd_file)
    model = _make_model(
        psd_file,
        snr=10.0,
        amplitude_distribution=LogNormalAmplitudeDistribution(mean=2.0, std=0.0),
    )

    waveform = model.generate_waveform(4096.0, rng=np.random.default_rng(7))

    assert _optimal_snr(waveform, psd_file, 4096.0) == pytest.approx(20.0, rel=1e-8)


def test_resampling_matches_simulation_rate(tmp_path: Path, hf_stub: list[str]) -> None:
    """Waveforms are resampled onto the requested sample grid."""
    psd_file = tmp_path / "psd.txt"
    _write_flat_psd(psd_file)
    model = _make_model(psd_file)

    assert model.generate_waveform(4096.0, rng=np.random.default_rng(0)).size == NATIVE_ROW_LENGTH
    assert model.generate_waveform(2048.0, rng=np.random.default_rng(0)).size == NATIVE_ROW_LENGTH // 2
    assert model.generate_waveform(8192.0, rng=np.random.default_rng(0)).size == NATIVE_ROW_LENGTH * 2


def test_generate_waveform_is_deterministic_for_a_seed(tmp_path: Path, hf_stub: list[str]) -> None:
    """Identical seeds reproduce identical waveforms."""
    psd_file = tmp_path / "psd.txt"
    _write_flat_psd(psd_file)
    model = _make_model(psd_file)

    first = model.generate_waveform(4096.0, rng=np.random.default_rng(11))
    second = model.generate_waveform(4096.0, rng=np.random.default_rng(11))

    np.testing.assert_array_equal(first, second)


def test_serialize_reports_full_configuration(tmp_path: Path) -> None:
    """Serialize includes the DeepExtractor configuration without dataset access."""
    psd_file = tmp_path / "psd.txt"
    _write_flat_psd(psd_file)
    model = _make_model(psd_file, snr=dict.fromkeys(GLITCH_CLASS_NAMES, 8.0))

    assert model.serialize() == {
        "kind": "deepextractor",
        "rate": 0.5,
        "amplitude_distribution": {"distribution": "lognormal", "mean": 1.0, "std": 0.0},
        "psd_file": str(psd_file),
        "snr": dict.fromkeys(GLITCH_CLASS_NAMES, 8.0),
        "glitch_classes": list(GLITCH_CLASS_NAMES),
        "low_frequency_cutoff": 2.0,
        "high_frequency_cutoff": None,
        "repo_id": "tomdooney/deepextractor-glitch-reconstructions",
        "local_files_only": False,
    }


def test_config_round_trip_and_injection(tmp_path: Path, hf_stub: list[str]) -> None:
    """Dict-form config normalizes and injects nonzero strain."""
    psd_file = tmp_path / "psd.txt"
    _write_flat_psd(psd_file)

    models = normalize_glitch_models(
        [
            {
                "kind": "deepextractor",
                "rate": {"Blip": 3.0, "Koi_Fish": 1.0},
                "amplitude_distribution": {"mean": 1.0, "std": 0.0},
                "psd_file": str(psd_file),
                "snr": {"Blip": 8.0, "Koi_Fish": 12.0},
                "glitch_classes": ["Blip", "Koi_Fish"],
            }
        ]
    )

    assert len(models) == 1
    assert isinstance(models[0], DeepExtractorGlitch)
    assert models[0].rate == 4.0

    base = _ZeroNoiseSimulator(detectors=["E1"], duration=4.0, sampling_frequency=4096.0, seed=1)
    simulator = InjectGlitches(base, models)
    result = simulator.generate(4.0, 4096.0, ["E1"], seed=1)

    assert np.max(np.abs(result["E1"])) > 0.0
    counts = simulator.metadata["glitches"]["counts"]
    assert counts[0]["kind"] == "deepextractor"
    assert counts[0]["count"] > 0
