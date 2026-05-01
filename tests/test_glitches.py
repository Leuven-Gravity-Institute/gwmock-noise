"""Tests for glitch injection simulators."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from gwmock_noise.config import (
    BlipGlitch,
    LogNormalAmplitudeDistribution,
    NoiseConfig,
    OutputConfig,
    ScatteredLightGlitch,
)
from gwmock_noise.simulators import DefaultNoiseSimulator, InjectGlitches


class ZeroNoiseSimulator:
    """Minimal protocol-compatible base simulator for additive glitch tests."""

    def __init__(self) -> None:
        """Initialize zero-valued base-simulator state."""
        self.duration = 4.0
        self.sampling_frequency = 256.0
        self.detectors = ["H1", "L1"]
        self.seed: int | None = None

    def reset(self) -> None:
        """Reset the placeholder base state."""
        return None

    def generate(
        self,
        duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Return zero strain for each requested detector."""
        self.duration = duration
        self.sampling_frequency = sampling_frequency
        self.detectors = list(detectors)
        self.seed = seed
        n_samples = round(duration * sampling_frequency)
        return {detector: np.zeros(n_samples, dtype=float) for detector in detectors}

    @property
    def metadata(self) -> dict[str, Any]:
        """Return representative metadata for the wrapper tests."""
        return {"implementation": "zero"}


def _smoothed_absolute_envelope(strain: np.ndarray, *, window: int = 33) -> np.ndarray:
    """Estimate the absolute-value envelope with a short moving average."""
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(np.abs(strain), kernel, mode="same")


def _fwhm_duration(strain: np.ndarray, *, sampling_frequency: float) -> float:
    """Estimate the waveform full width at half maximum from its smoothed envelope."""
    envelope = _smoothed_absolute_envelope(strain)
    half_max = 0.5 * float(np.max(envelope))
    support = np.flatnonzero(envelope >= half_max)
    return float((support[-1] - support[0] + 1) / sampling_frequency)


def test_blip_glitch_duration_tracks_requested_width() -> None:
    """Blip glitches preserve the configured duration scale within tolerance."""
    model = BlipGlitch(
        rate=0.25,
        amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.0, std=0.0),
        width=0.02,
    )

    waveform = model.generate_waveform(4096.0, rng=np.random.default_rng(12))

    assert _fwhm_duration(waveform, sampling_frequency=4096.0) == pytest.approx(0.02, rel=0.2)


def test_scattered_light_glitch_peaks_near_segment_center() -> None:
    """Scattered-light glitches are localized around the Gaussian-envelope center."""
    model = ScatteredLightGlitch(
        rate=0.1,
        amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.0, std=0.0),
        duration=0.5,
        peak_frequency=32.0,
        arch_exponent=1.5,
    )

    waveform = model.generate_waveform(1024.0, rng=np.random.default_rng(8))
    peak_index = int(np.argmax(np.abs(_smoothed_absolute_envelope(waveform))))
    midpoint = waveform.size // 2

    assert peak_index == pytest.approx(midpoint, abs=waveform.size * 0.1)


def test_glitch_injection_rate_matches_poisson_uncertainty_over_long_run() -> None:
    """Injected glitch counts remain consistent with the configured Poisson rate."""
    model = BlipGlitch(
        rate=0.25,
        amplitude_distribution=LogNormalAmplitudeDistribution(mean=0.5, std=0.0),
        width=0.01,
    )
    simulator = InjectGlitches(ZeroNoiseSimulator(), [model])

    simulator.generate(duration=1000.0, sampling_frequency=256.0, detectors=["H1"], seed=17)

    count = simulator.metadata["glitches"]["counts"][0]["count"]
    expected = model.rate * 1000.0
    sigma = np.sqrt(expected)
    assert abs(count - expected) <= (2.0 * sigma)


def test_inject_glitches_is_deterministic_given_same_seed() -> None:
    """InjectGlitches reproduces the same realization when restarted from one seed."""
    glitch_models = [
        BlipGlitch(
            rate=0.5,
            amplitude_distribution=LogNormalAmplitudeDistribution(mean=0.75, std=0.0),
            width=0.015,
        ),
        ScatteredLightGlitch(
            rate=0.2,
            amplitude_distribution=LogNormalAmplitudeDistribution(mean=0.5, std=0.0),
            duration=0.25,
            peak_frequency=24.0,
            arch_exponent=1.2,
        ),
    ]
    first = InjectGlitches(ZeroNoiseSimulator(), glitch_models)
    second = InjectGlitches(ZeroNoiseSimulator(), glitch_models)

    first_realization = first.generate(duration=8.0, sampling_frequency=512.0, detectors=["H1", "L1"], seed=1234)
    second_realization = second.generate(duration=8.0, sampling_frequency=512.0, detectors=["H1", "L1"], seed=1234)

    np.testing.assert_allclose(first_realization["H1"], second_realization["H1"])
    np.testing.assert_allclose(first_realization["L1"], second_realization["L1"])
    assert first.metadata["glitches"]["counts"] == second.metadata["glitches"]["counts"]


def test_default_simulator_reports_glitch_metadata(tmp_path: Path) -> None:
    """DefaultNoiseSimulator dispatches glitch injection from config models."""
    out_dir = tmp_path / "output"
    config = NoiseConfig(
        detectors=["H1"],
        duration=8.0,
        sampling_frequency=256.0,
        output=OutputConfig(directory=out_dir, prefix="glitches"),
        seed=19,
        glitches=[
            {
                "kind": "blip",
                "rate": 0.25,
                "width": 0.01,
                "amplitude_distribution": {
                    "distribution": "lognormal",
                    "mean": 0.5,
                    "std": 0.0,
                },
            }
        ],
    )

    simulator = DefaultNoiseSimulator()
    simulator.run(config)

    metadata = json.loads((out_dir / "glitches_H1.json").read_text())
    assert metadata["implementation"] == "inject_glitches"
    assert metadata["base_implementation"] == "zero"
    assert metadata["glitches"]["models"] == [
        {
            "kind": "blip",
            "rate": 0.25,
            "width": 0.01,
            "amplitude_distribution": {
                "distribution": "lognormal",
                "mean": 0.5,
                "std": 0.0,
            },
        }
    ]
