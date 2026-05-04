"""Gengli-backed glitch models and population-file helpers."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np

from gwmock_noise.config.models import GlitchModel
from gwmock_noise.simulators._spectral import load_spectral_series
from gwmock_noise.simulators.colored import _tukey_window

POPULATION_SNR_DATASET = "snr"
UINT32_EXCLUSIVE_MAX = 2**32


def _load_gengli() -> Any:
    """Import the optional gengli dependency with a focused error."""
    try:
        return importlib.import_module("gengli")
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised via tests
        if exc.name != "gengli":
            raise
        raise ImportError(
            "GengliBlipGlitch requires the optional dependency 'gengli'. Install gwmock-noise[gengli]."
        ) from exc


def read_blip_population_file(path: str | Path) -> np.ndarray:
    """Load the supported blip-population schema from disk."""
    population_path = Path(path)
    if not population_path.exists():
        raise FileNotFoundError(f"Population file not found: {population_path}")

    with h5py.File(population_path, "r") as handle:
        if POPULATION_SNR_DATASET not in handle:
            raise ValueError(f"Population file must contain the '{POPULATION_SNR_DATASET}' dataset.")
        snr_samples = np.asarray(handle[POPULATION_SNR_DATASET][...], dtype=float)

    if snr_samples.ndim != 1:
        raise ValueError(f"Population dataset '{POPULATION_SNR_DATASET}' must be one-dimensional.")
    if snr_samples.size == 0:
        raise ValueError("Population file must contain at least one SNR sample.")
    if not np.all(np.isfinite(snr_samples)):
        raise ValueError("Population file contains non-finite SNR samples.")
    if np.any(snr_samples <= 0.0):
        raise ValueError("Population file SNR samples must be greater than zero.")
    return snr_samples


def write_blip_population_file(
    path: str | Path,
    *,
    snr_samples: np.ndarray,
    metadata: dict[str, str | float] | None = None,
) -> Path:
    """Write the population-file schema consumed by ``GengliBlipGlitch``."""
    population_path = Path(path)
    population_path.parent.mkdir(parents=True, exist_ok=True)

    values = np.asarray(snr_samples, dtype=float)
    if values.ndim != 1:
        raise ValueError("snr_samples must be one-dimensional.")
    if values.size == 0:
        raise ValueError("snr_samples must contain at least one value.")
    if not np.all(np.isfinite(values)):
        raise ValueError("snr_samples must be finite.")
    if np.any(values <= 0.0):
        raise ValueError("snr_samples must be greater than zero.")

    with h5py.File(population_path, "w") as handle:
        handle.create_dataset(POPULATION_SNR_DATASET, data=values)
        if metadata is not None:
            for key, value in metadata.items():
                handle.attrs[key] = value

    return population_path


@dataclass(slots=True)
class GengliBlipGlitch(GlitchModel):
    """File-backed gengli blip generator colored against a target PSD."""

    population_file: Path
    psd_file: Path
    gengli_detector: str = "L1"
    low_frequency_cutoff: float = 2.0
    high_frequency_cutoff: float | None = None
    kind: Literal["gengli_blip"] = field(init=False, default="gengli_blip")
    _population_snrs: np.ndarray = field(init=False, repr=False)
    _psd_frequencies: np.ndarray = field(init=False, repr=False)
    _psd_values: np.ndarray = field(init=False, repr=False)
    _generator: Any = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        """Validate configured files and preload the population/PSD tables."""
        GlitchModel.__post_init__(self)
        self.population_file = Path(self.population_file)
        self.psd_file = Path(self.psd_file)
        if not self.gengli_detector:
            raise ValueError("gengli_detector must be a non-empty string.")
        if self.low_frequency_cutoff < 0.0:
            raise ValueError("low_frequency_cutoff must be non-negative.")
        if self.high_frequency_cutoff is not None and self.high_frequency_cutoff <= self.low_frequency_cutoff:
            raise ValueError("high_frequency_cutoff must be greater than low_frequency_cutoff.")

        self._population_snrs = read_blip_population_file(self.population_file)
        self._psd_frequencies, self._psd_values = load_spectral_series(self.psd_file, kind="PSD")
        if not np.all(np.isfinite(self._psd_values)):
            raise ValueError("PSD file contains non-finite values.")
        if np.any(self._psd_values < 0.0):
            raise ValueError("PSD file contains negative values.")

    @classmethod
    def from_population_file(
        cls,
        population_file: str | Path,
        **kwargs: Any,
    ) -> GengliBlipGlitch:
        """Construct a gengli glitch model from a population file path."""
        return cls(population_file=Path(population_file), **kwargs)

    def _get_generator(self) -> Any:
        """Create and cache the gengli generator for the configured detector."""
        if self._generator is None:
            self._generator = _load_gengli().glitch_generator(self.gengli_detector)
        return self._generator

    def _draw_snr(self, rng: np.random.Generator) -> float:
        """Sample one target SNR from the preloaded population table."""
        index = int(rng.integers(0, self._population_snrs.size))
        return float(self._population_snrs[index])

    def _color_glitch(self, white_glitch: np.ndarray, *, sampling_frequency: float) -> np.ndarray:
        """Color a whitened gengli waveform using the configured PSD."""
        n_samples = int(white_glitch.size)
        nyquist = sampling_frequency / 2.0
        high_frequency_cutoff = nyquist if self.high_frequency_cutoff is None else self.high_frequency_cutoff
        if high_frequency_cutoff > nyquist:
            raise ValueError("high_frequency_cutoff must not exceed the Nyquist frequency.")

        frequencies = np.fft.rfftfreq(n_samples, d=1.0 / sampling_frequency)
        frequency_mask = (frequencies >= self.low_frequency_cutoff) & (frequencies <= high_frequency_cutoff)
        if not np.any(frequency_mask):
            raise ValueError("The requested frequency range contains no gengli simulation bins.")

        psd = np.zeros_like(frequencies, dtype=float)
        masked_frequencies = frequencies[frequency_mask]
        psd[frequency_mask] = np.interp(
            masked_frequencies,
            self._psd_frequencies,
            self._psd_values,
            left=0.0,
            right=0.0,
        )
        psd[frequency_mask] = np.clip(psd[frequency_mask], a_min=0.0, a_max=None)
        psd[frequency_mask] *= _tukey_window(masked_frequencies.size)

        white_glitch_fd = np.fft.rfft(white_glitch) / sampling_frequency
        colored_glitch_fd = np.zeros_like(white_glitch_fd, dtype=np.complex128)
        colored_glitch_fd[frequency_mask] = white_glitch_fd[frequency_mask] * np.sqrt(psd[frequency_mask])
        return np.fft.irfft(colored_glitch_fd, n=n_samples) * sampling_frequency

    def generate_waveform(
        self,
        sampling_frequency: float,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Generate one colored gengli blip waveform."""
        if sampling_frequency <= 0.0:
            raise ValueError("sampling_frequency must be greater than zero.")

        generator = np.random.default_rng() if rng is None else rng
        raw_glitch = self._get_generator().get_glitch(
            seed=int(generator.integers(0, UINT32_EXCLUSIVE_MAX)),
            snr=self._draw_snr(generator),
            srate=sampling_frequency,
            glitch_type="Blip",
        )
        white_glitch = np.asarray(raw_glitch, dtype=float).reshape(-1)
        if white_glitch.size == 0:
            return white_glitch

        amplitude = self.amplitude_distribution.sample(generator)
        colored_glitch = self._color_glitch(white_glitch, sampling_frequency=sampling_frequency)
        return amplitude * colored_glitch

    def serialize(self) -> dict[str, Any]:
        """Return metadata-friendly model parameters."""
        return GlitchModel.serialize(self) | {
            "population_file": str(self.population_file),
            "psd_file": str(self.psd_file),
            "gengli_detector": self.gengli_detector,
            "low_frequency_cutoff": self.low_frequency_cutoff,
            "high_frequency_cutoff": self.high_frequency_cutoff,
            "population_size": int(self._population_snrs.size),
        }
