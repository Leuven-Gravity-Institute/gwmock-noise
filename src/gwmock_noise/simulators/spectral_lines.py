"""Spectral-line simulators and wrappers."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import numpy as np

from gwmock_noise.gaussian import SpectralLine, normalize_spectral_lines
from gwmock_noise.simulators.base import ConfigurableNoiseSimulator
from gwmock_noise.simulators.protocol import NoiseSimulator

if TYPE_CHECKING:
    from gwmock_noise.config.models import NoiseComponentConfig, NoiseConfig

TWO_PI = 2.0 * np.pi


def _serialize_line(line: SpectralLine) -> dict[str, float | None]:
    """Convert a line definition into metadata-friendly scalars."""
    return {
        "frequency": line.frequency,
        "amplitude": line.amplitude,
        "phase": line.phase,
        "drift_rate": line.drift_rate,
    }


class SpectralLineSimulator(ConfigurableNoiseSimulator):
    """Generate additive spectral lines directly in the time domain."""

    simulator_name = "spectral_lines"

    def __init__(
        self,
        *,
        lines: list[SpectralLine],
        detectors: list[str] | None = None,
        sampling_frequency: float = 4096.0,
        duration: float = 4.0,
        seed: int | None = None,
    ) -> None:
        """Initialize the line generator."""
        if not lines:
            raise ValueError("lines must contain at least one spectral line.")

        self.lines = list(lines)
        self.detectors = list(detectors) if detectors is not None else ["H1", "L1"]
        self.duration = duration
        self.sampling_frequency = sampling_frequency
        self.seed = seed

        self._elapsed_time = 0.0
        self._initial_phases: np.ndarray | None = None
        self._phase_state = np.zeros(len(self.lines), dtype=float)

        self._validate_runtime(duration=duration, sampling_frequency=sampling_frequency, detectors=self.detectors)

    @classmethod
    def from_component(cls, component: NoiseComponentConfig, config: NoiseConfig) -> SpectralLineSimulator:
        """Construct a spectral-line simulator from one component definition."""
        options = dict(component.options)
        lines = normalize_spectral_lines(options.pop("lines", []))
        if not lines:
            raise ValueError("SpectralLineSimulator requires at least one spectral line.")
        return cls(
            lines=lines,
            detectors=config.detectors,
            duration=config.duration,
            sampling_frequency=config.sampling_frequency,
            seed=config.seed,
            **options,
        )

    def _validate_runtime(
        self,
        *,
        duration: float,
        sampling_frequency: float,
        detectors: list[str],
    ) -> None:
        """Validate shared runtime parameters."""
        if duration <= 0:
            raise ValueError("duration must be greater than zero.")
        if sampling_frequency <= 0:
            raise ValueError("sampling_frequency must be greater than zero.")
        if not detectors:
            raise ValueError("detectors must contain at least one detector.")
        if len(set(detectors)) != len(detectors):
            raise ValueError("detectors must not contain duplicate names.")

    def _validate_line_band(self, *, start_time: float, end_time: float, sampling_frequency: float) -> None:
        """Ensure line frequencies stay in-band for the generated segment."""
        nyquist = sampling_frequency / 2.0
        for line in self.lines:
            start_frequency = line.frequency + (line.drift_rate * start_time)
            end_frequency = line.frequency + (line.drift_rate * end_time)
            minimum_frequency = min(start_frequency, end_frequency)
            maximum_frequency = max(start_frequency, end_frequency)
            if minimum_frequency < 0.0 or maximum_frequency > nyquist:
                raise ValueError("spectral line frequencies must remain within [0, Nyquist] over the segment.")

    def _initialize_phases(self) -> None:
        """Resolve fixed and random initial phases."""
        rng = np.random.default_rng(self.seed)
        self._initial_phases = np.array(
            [float(rng.uniform(0.0, TWO_PI)) if line.phase is None else float(line.phase) for line in self.lines],
            dtype=float,
        )

    def _generate_lines(self, *, duration: float, sampling_frequency: float) -> np.ndarray:
        """Generate the summed line signal for one detector."""
        n_samples = round(duration * sampling_frequency)
        if n_samples < 1:
            raise ValueError("duration and sampling_frequency must produce at least one sample.")

        start_time = self._elapsed_time
        sample_times = start_time + (np.arange(n_samples, dtype=float) / sampling_frequency)
        end_time = sample_times[-1]
        self._validate_line_band(
            start_time=start_time,
            end_time=end_time,
            sampling_frequency=sampling_frequency,
        )

        if self._initial_phases is None:
            self._initialize_phases()

        strain = np.zeros(n_samples, dtype=float)
        amplitude_scale = np.sqrt(sampling_frequency / 2.0)
        for index, line in enumerate(self.lines):
            phase = (
                TWO_PI * (line.frequency * sample_times + 0.5 * line.drift_rate * np.square(sample_times))
            ) + self._initial_phases[index]
            strain += line.amplitude * amplitude_scale * np.sin(phase)
            self._phase_state[index] = float(phase[-1])

        self._elapsed_time += n_samples / sampling_frequency
        return strain

    def reset(self) -> None:
        """Clear continuity state and phase initialization."""
        self._elapsed_time = 0.0
        self._initial_phases = None
        self._phase_state = np.zeros(len(self.lines), dtype=float)

    def generate(
        self,
        duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Generate line-only strain for all requested detectors."""
        runtime_detectors = list(detectors)
        self._validate_runtime(
            duration=duration,
            sampling_frequency=sampling_frequency,
            detectors=runtime_detectors,
        )

        self.duration = duration
        self.sampling_frequency = sampling_frequency
        self.detectors = runtime_detectors

        if seed is not None:
            self.seed = seed
            self.reset()

        line_signal = self._generate_lines(duration=duration, sampling_frequency=sampling_frequency)
        return {detector: line_signal.copy() for detector in runtime_detectors}

    def generate_stream(
        self,
        chunk_duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> Iterator[dict[str, np.ndarray]]:
        """Yield spectral-line chunks lazily."""
        while True:
            yield self.generate(chunk_duration, sampling_frequency, detectors, seed)
            seed = None

    @property
    def metadata(self) -> dict[str, Any]:
        """Return simulator metadata."""
        return {
            "implementation": "spectral_lines",
            "duration": self.duration,
            "sampling_frequency": self.sampling_frequency,
            "detectors": list(self.detectors),
            "seed": self.seed,
            "spectral_lines": {
                "lines": [_serialize_line(line) for line in self.lines],
                "elapsed_time_seconds": self._elapsed_time,
            },
        }


class AddLines:
    """Wrap a base simulator and add spectral lines to its output."""

    def __init__(self, base: NoiseSimulator, lines: list[SpectralLine]) -> None:
        """Initialize the additive wrapper."""
        if not lines:
            raise ValueError("lines must contain at least one spectral line.")

        self.base = base
        self._line_simulator = SpectralLineSimulator(
            lines=lines,
            detectors=list(base.detectors),
            duration=base.duration,
            sampling_frequency=base.sampling_frequency,
            seed=base.seed,
        )

        self.duration = base.duration
        self.sampling_frequency = base.sampling_frequency
        self.detectors = list(base.detectors)
        self.seed = base.seed

    def reset(self) -> None:
        """Reset the additive line state and any resettable base state."""
        self._line_simulator.reset()
        if hasattr(self.base, "reset"):
            self.base.reset()

    def generate(
        self,
        duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Generate base noise and add the configured spectral lines."""
        runtime_detectors = list(detectors)
        base_result = self.base.generate(duration, sampling_frequency, runtime_detectors, seed=seed)
        line_result = self._line_simulator.generate(duration, sampling_frequency, runtime_detectors, seed=seed)

        self.duration = duration
        self.sampling_frequency = sampling_frequency
        self.detectors = runtime_detectors
        self.seed = seed if seed is not None else self.base.seed

        combined: dict[str, np.ndarray] = {}
        for detector in runtime_detectors:
            if detector not in base_result:
                raise KeyError(f"Base simulator did not return detector '{detector}'.")
            base_strain = np.asarray(base_result[detector], dtype=float)
            line_strain = line_result[detector]
            if base_strain.shape != line_strain.shape:
                raise ValueError("Base simulator output shape must match the generated spectral-line shape.")
            combined[detector] = base_strain + line_strain
        return combined

    def generate_stream(
        self,
        chunk_duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> Iterator[dict[str, np.ndarray]]:
        """Yield additive spectral-line chunks lazily."""
        while True:
            yield self.generate(chunk_duration, sampling_frequency, detectors, seed)
            seed = None

    @property
    def metadata(self) -> dict[str, Any]:
        """Return additive-wrapper metadata."""
        base_metadata = dict(self.base.metadata)
        return base_metadata | {
            "implementation": "add_lines",
            "base_implementation": base_metadata.get("implementation"),
            "spectral_lines": self._line_simulator.metadata["spectral_lines"],
        }
