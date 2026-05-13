"""Tests for glitch injection simulators."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from gwmock_noise.config import NoiseConfig, OutputConfig
from gwmock_noise.glitches import BlipGlitch, LogNormalAmplitudeDistribution, ScatteredLightGlitch
from gwmock_noise.simulators import DefaultNoiseSimulator, InjectGlitches
from gwmock_noise.simulators.glitches import _ZeroNoiseSimulator


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


class NoResetNoiseSimulator:
    """Protocol-compatible base simulator variant without reset()."""

    def __init__(self) -> None:
        """Initialize zero-valued base-simulator state."""
        self.duration = 4.0
        self.sampling_frequency = 256.0
        self.detectors = ["H1"]
        self.seed: int | None = None

    def generate(
        self,
        duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Generate zero strain for each requested detector.

        Args:
            duration: The duration of the simulation in seconds.
            sampling_frequency: The sampling frequency in Hz.
            detectors: The list of detectors to simulate.
            seed: The random seed to use for the simulation.

        Returns:
            Dictionary containing the zero strain for each requested detector.

        """
        self.duration = duration
        self.sampling_frequency = sampling_frequency
        self.detectors = list(detectors)
        self.seed = seed
        n_samples = round(duration * sampling_frequency)
        return {detector: np.zeros(n_samples, dtype=float) for detector in detectors}

    @property
    def metadata(self) -> dict[str, Any]:
        """Return representative metadata for the wrapper tests.

        Returns:
            Dictionary containing the implementation name.

        """
        return {"implementation": "no_reset"}


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
        components=[
            {
                "simulator": "glitches",
                "models": [
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


def test_default_simulator_wraps_existing_simulator_when_glitches_enabled(tmp_path: Path) -> None:
    """Colored and glitch components compose additively."""
    psd_path = tmp_path / "flat_psd.txt"
    frequencies = np.linspace(0.0, 128.0, 129)
    np.savetxt(psd_path, np.column_stack((frequencies, np.full_like(frequencies, 2.0e-3))))

    out_dir = tmp_path / "output"
    config = NoiseConfig(
        detectors=["H1"],
        duration=4.0,
        sampling_frequency=256.0,
        output=OutputConfig(directory=out_dir, prefix="colored_glitches"),
        seed=5,
        components=[
            {"simulator": "colored", "psd_file": psd_path},
            {
                "simulator": "glitches",
                "models": [
                    {
                        "kind": "blip",
                        "rate": 0.2,
                        "width": 0.01,
                        "amplitude_distribution": {"distribution": "lognormal", "mean": 0.5, "std": 0.0},
                    }
                ],
            },
        ],
    )
    DefaultNoiseSimulator().run(config)
    metadata = json.loads((out_dir / "colored_glitches_H1.json").read_text())
    assert metadata["implementation"] == "composed"
    assert [component["simulator"] for component in metadata["components"]] == ["colored", "glitches"]


def test_zero_noise_simulator_reset_returns_none() -> None:
    """Zero-noise helper reset is a no-op."""
    simulator = _ZeroNoiseSimulator(detectors=["H1"], duration=1.0, sampling_frequency=16.0, seed=None)
    assert simulator.reset() is None


def test_inject_glitches_rejects_empty_model_list() -> None:
    """InjectGlitches requires at least one model."""
    with pytest.raises(ValueError, match="must contain at least one glitch model"):
        InjectGlitches(ZeroNoiseSimulator(), [])


def test_draw_interarrival_handles_zero_rate_and_uninitialized_rng() -> None:
    """Interarrival logic handles zero rate and missing RNG state."""
    model = BlipGlitch(
        rate=0.1,
        amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.0, std=0.0),
        width=0.01,
    )
    simulator = InjectGlitches(ZeroNoiseSimulator(), [model])
    assert np.isinf(simulator._draw_interarrival(0.0))
    simulator._rng = None
    with pytest.raises(RuntimeError, match="not initialized"):
        simulator._draw_interarrival(0.1)


def test_inject_glitches_reset_without_base_reset_still_succeeds() -> None:
    """Wrapper reset tolerates bases without a callable reset."""
    model = BlipGlitch(
        rate=0.1,
        amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.0, std=0.0),
        width=0.01,
    )
    simulator = InjectGlitches(NoResetNoiseSimulator(), [model])
    simulator.reset()
    assert simulator.metadata["implementation"] == "inject_glitches"


def test_inject_glitches_rejects_missing_or_bad_base_output() -> None:
    """Wrapper validates base detector keys and output shape."""

    class MissingDetectorBase(ZeroNoiseSimulator):
        def generate(
            self,
            duration: float,
            sampling_frequency: float,
            detectors: list[str],
            seed: int | None = None,
        ) -> dict[str, np.ndarray]:
            _ = (seed,)
            n_samples = round(duration * sampling_frequency)
            return {"H1": np.zeros(n_samples)}

    class BadShapeBase(ZeroNoiseSimulator):
        def generate(
            self,
            duration: float,
            sampling_frequency: float,
            detectors: list[str],
            seed: int | None = None,
        ) -> dict[str, np.ndarray]:
            _ = (duration, detectors, seed)
            return {"H1": np.zeros(round(duration * sampling_frequency) + 1)}

    model = BlipGlitch(
        rate=0.1,
        amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.0, std=0.0),
        width=0.01,
    )
    with pytest.raises(KeyError, match="did not return detector"):
        InjectGlitches(MissingDetectorBase(), [model]).generate(1.0, 16.0, ["H1", "L1"], seed=1)
    with pytest.raises(ValueError, match="output shape must match"):
        InjectGlitches(BadShapeBase(), [model]).generate(1.0, 16.0, ["H1"], seed=1)


def test_inject_glitches_reset_calls_base_reset_when_available() -> None:
    """Wrapper reset delegates to base reset when present."""

    class ResetTrackingBase(ZeroNoiseSimulator):
        def __init__(self) -> None:
            super().__init__()
            self.reset_calls = 0

        def reset(self) -> None:
            self.reset_calls += 1

    model = BlipGlitch(
        rate=0.1,
        amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.0, std=0.0),
        width=0.01,
    )
    base = ResetTrackingBase()
    simulator = InjectGlitches(base, [model])
    simulator.reset()
    assert base.reset_calls == 1


def test_inject_glitches_reuses_rng_state_when_seed_is_none() -> None:
    """generate() skips process reinit when seed is None and RNG exists."""
    model = BlipGlitch(
        rate=0.2,
        amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.0, std=0.0),
        width=0.01,
    )
    simulator = InjectGlitches(ZeroNoiseSimulator(), [model])
    simulator.generate(duration=1.0, sampling_frequency=64.0, detectors=["H1"], seed=5)
    first_elapsed = simulator.metadata["glitches"]["elapsed_time_seconds"]
    simulator.generate(duration=1.0, sampling_frequency=64.0, detectors=["H1"], seed=None)
    second_elapsed = simulator.metadata["glitches"]["elapsed_time_seconds"]
    assert second_elapsed > first_elapsed


def test_inject_glitches_raises_when_rng_is_missing_after_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defensive branch raises when process init fails to provide RNG."""
    model = BlipGlitch(
        rate=0.2,
        amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.0, std=0.0),
        width=0.01,
    )
    simulator = InjectGlitches(ZeroNoiseSimulator(), [model])

    def _broken_initialize(_: int | None) -> None:
        simulator._rng = None

    monkeypatch.setattr(simulator, "_initialize_process", _broken_initialize)
    with pytest.raises(RuntimeError, match="not initialized"):
        simulator.generate(duration=1.0, sampling_frequency=64.0, detectors=["H1"], seed=1)


def test_inject_glitches_handles_empty_waveform_events() -> None:
    """Events with empty waveforms do not increment counts."""

    class EmptyWaveformBlip(BlipGlitch):
        def generate_waveform(
            self,
            sampling_frequency: float,
            rng: np.random.Generator | None = None,
        ) -> np.ndarray:
            _ = (sampling_frequency, rng)
            return np.array([], dtype=float)

    model = EmptyWaveformBlip(
        rate=1.0,
        amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.0, std=0.0),
        width=0.01,
    )
    simulator = InjectGlitches(ZeroNoiseSimulator(), [model])
    simulator.generate(duration=5.0, sampling_frequency=64.0, detectors=["H1"], seed=3)
    assert simulator.metadata["glitches"]["counts"][0]["count"] == 0
