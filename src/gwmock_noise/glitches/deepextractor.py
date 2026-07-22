"""DeepExtractor glitch-reconstruction model backed by a HuggingFace dataset."""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from gwmock_noise.glitches._coloring import color_whitened_waveform, optimal_snr
from gwmock_noise.glitches.models import GlitchModel
from gwmock_noise.simulators._spectral import load_spectral_series
from gwmock_noise.utils.log import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)

DEEPEXTRACTOR_REPO_ID = "tomdooney/deepextractor-glitch-reconstructions"
SAMPLES_FILENAME = "glitch_GAN_samples_scaled_balanced.npy"
LABELS_FILENAME = "glitch_GAN_labels_balanced.npy"
LABEL_ORDER_FILENAME = "glitch_GAN_label_order.npy"
NATIVE_SAMPLING_FREQUENCY = 4096.0
_TABLE_NDIM = 2
GLITCH_CLASS_NAMES = (
    "Blip",
    "Fast_Scattering",
    "Koi_Fish",
    "Low_Frequency_Burst",
    "Scattered_Light",
    "Tomte",
    "Whistle",
)


def _connection_error_types(hf_hub: Any) -> tuple[type[BaseException], ...]:
    """Resolve the exception types that signal an unreachable Hub.

    Collected dynamically so the module keeps working across huggingface_hub
    versions and whether or not ``requests`` is importable. ``requests`` errors
    cover raw connectivity failures; ``LocalEntryNotFoundError`` is what
    huggingface_hub raises when it can neither reach the Hub nor find the file
    in the cache.
    """
    types: list[type[BaseException]] = []
    errors_module = getattr(hf_hub, "errors", None)
    if errors_module is None:
        try:
            errors_module = importlib.import_module("huggingface_hub.errors")
        except ModuleNotFoundError:  # pragma: no cover - errors submodule always ships with the hub
            errors_module = None
    for name in ("LocalEntryNotFoundError", "OfflineModeIsEnabled"):
        candidate = getattr(errors_module, name, None)
        if isinstance(candidate, type) and issubclass(candidate, BaseException):
            types.append(candidate)
    try:
        requests = importlib.import_module("requests")
    except ModuleNotFoundError:  # pragma: no cover - requests ships with huggingface_hub
        pass
    else:
        types.append(requests.exceptions.ConnectionError)
        types.append(requests.exceptions.Timeout)
    return tuple(types) or (OSError,)


def _load_hf_hub() -> Any:
    """Import the optional huggingface_hub dependency with a focused error."""
    try:
        return importlib.import_module("huggingface_hub")
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised via tests
        if exc.name != "huggingface_hub":
            raise
        raise ImportError(
            "DeepExtractorGlitch requires the optional dependency 'huggingface_hub'. "
            "Install gwmock-noise[deepextractor]."
        ) from exc


@dataclass(slots=True)
class DeepExtractorGlitch(GlitchModel):
    """Real O3 glitch reconstructions colored against a target PSD.

    Waveforms come from the DeepExtractor glitch-reconstruction dataset:
    whitened, amplitude-normalized 2-second time series sampled at 4096 Hz,
    covering seven Gravity Spy classes. Each drawn waveform is resampled to
    the simulation rate, colored against the configured PSD, and rescaled so
    its optimal SNR ``sqrt(4 df sum(|h(f)|^2 / S(f)))`` against that PSD
    matches the configured target. The dataset (~2.3 GB) is downloaded lazily
    on first use and cached by huggingface_hub; later runs reuse the cached
    files. On each run the Hub is contacted first to validate the cached
    files' ETags and download anything missing. If the Hub is unreachable, a
    warning notes that the ETag check was skipped and the cached files are used
    instead; if they are not cached, the first ``generate_waveform`` raises
    huggingface_hub's ``LocalEntryNotFoundError`` (a ``FileNotFoundError``
    subclass). Set ``local_files_only=True`` to skip the network unconditionally
    and read straight from the cache.

    ``rate`` accepts either a single number — the total Poisson rate shared by
    all configured classes, drawn uniformly — or a mapping from class name to
    per-class Poisson rate, in which case the total rate is their sum and each
    event's class is drawn proportionally to its rate.

    Resampling uses linear interpolation without an anti-aliasing filter, so
    sampling frequencies below 4096 Hz alias high-frequency content; the SNR
    calibration itself is unaffected because it is computed after resampling.
    """

    rate: float | dict[str, float]
    psd_file: str | Path
    snr: float | dict[str, float]
    glitch_classes: list[str] | None = None
    low_frequency_cutoff: float = 2.0
    high_frequency_cutoff: float | None = None
    repo_id: str = DEEPEXTRACTOR_REPO_ID
    local_files_only: bool = False
    kind: Literal["deepextractor"] = field(init=False, default="deepextractor")
    _psd_frequencies: np.ndarray = field(init=False, repr=False)
    _psd_values: np.ndarray = field(init=False, repr=False)
    _samples: Any = field(init=False, default=None, repr=False)
    _class_indices: dict[str, np.ndarray] | None = field(init=False, default=None, repr=False)
    _class_rates: dict[str, float] | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        """Validate the configuration and preload the PSD table."""
        if self.glitch_classes is None:
            self.glitch_classes = list(GLITCH_CLASS_NAMES)
        else:
            self.glitch_classes = list(self.glitch_classes)
        if not self.glitch_classes:
            raise ValueError("glitch_classes must contain at least one class name.")
        unknown = sorted(set(self.glitch_classes) - set(GLITCH_CLASS_NAMES))
        if unknown:
            raise ValueError(
                f"Unknown glitch classes {', '.join(unknown)}; supported classes are {', '.join(GLITCH_CLASS_NAMES)}."
            )
        if len(set(self.glitch_classes)) != len(self.glitch_classes):
            raise ValueError("glitch_classes must not contain duplicates.")

        self._normalize_rate()
        GlitchModel.__post_init__(self)
        self._validate_snr()

        if not self.repo_id:
            raise ValueError("repo_id must be a non-empty string.")
        if self.low_frequency_cutoff < 0.0:
            raise ValueError("low_frequency_cutoff must be non-negative.")
        if self.high_frequency_cutoff is not None and self.high_frequency_cutoff <= self.low_frequency_cutoff:
            raise ValueError("high_frequency_cutoff must be greater than low_frequency_cutoff.")

        self._psd_frequencies, self._psd_values = load_spectral_series(self.psd_file, kind="PSD")
        if not np.all(np.isfinite(self._psd_values)):
            raise ValueError("PSD file contains non-finite values.")
        if np.any(self._psd_values < 0.0):
            raise ValueError("PSD file contains negative values.")

    def _check_mapping_covers_classes(self, mapping: dict[str, float], parameter: str) -> None:
        """Ensure a per-class mapping matches the configured glitch classes exactly."""
        configured = set(self.glitch_classes or ())
        missing = sorted(configured - set(mapping))
        if missing:
            raise ValueError(f"{parameter} mapping is missing glitch classes: {', '.join(missing)}.")
        unknown = sorted(set(mapping) - configured)
        if unknown:
            raise ValueError(f"{parameter} mapping contains unconfigured glitch classes: {', '.join(unknown)}.")

    def _normalize_rate(self) -> None:
        """Fold a per-class rate mapping into the total Poisson rate."""
        self._class_rates = None
        if not isinstance(self.rate, dict):
            return
        self._check_mapping_covers_classes(self.rate, "rate")
        for value in self.rate.values():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError("rate values must be numbers.")
            if not np.isfinite(value) or value < 0.0:
                raise ValueError("rate values must be finite and non-negative.")
        self._class_rates = {name: float(value) for name, value in self.rate.items()}
        self.rate = float(sum(self._class_rates.values()))

    def _validate_snr(self) -> None:
        """Validate the scalar or per-class target-SNR configuration."""
        if isinstance(self.snr, dict):
            self._check_mapping_covers_classes(self.snr, "snr")
            values = list(self.snr.values())
        else:
            values = [self.snr]
        for value in values:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError("snr values must be numbers.")
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError("snr values must be finite and greater than zero.")

    def _target_snr(self, glitch_class: str) -> float:
        """Return the target optimal SNR for one glitch class."""
        if isinstance(self.snr, dict):
            return float(self.snr[glitch_class])
        return float(self.snr)

    def _draw_class(self, rng: np.random.Generator) -> str:
        """Draw one glitch class, weighted by per-class rates when configured.

        Falls back to a uniform draw for scalar-rate models and for the direct
        ``generate_waveform`` call on a model whose per-class rates are all
        zero (the injector itself never fires events for a zero total rate).
        """
        configured_classes = self.glitch_classes or []
        if self._class_rates is not None:
            weights = np.array([self._class_rates[name] for name in configured_classes], dtype=float)
            total = float(weights.sum())
            if total > 0.0:
                return configured_classes[int(rng.choice(len(configured_classes), p=weights / total))]
        return configured_classes[int(rng.integers(0, len(configured_classes)))]

    def _download_dataset_file(self, hf_hub: Any, filename: str) -> str:
        """Fetch one dataset file, preferring the network but tolerating outages.

        When ``local_files_only`` is set the network is skipped entirely. Otherwise
        the online path runs first so a cached file's ETag is validated against the
        Hub. If the Hub is unreachable, a warning notes the skipped ETag check and
        the cached copy is used; a genuinely missing cache then raises
        huggingface_hub's ``LocalEntryNotFoundError``.
        """
        kwargs = {"repo_id": self.repo_id, "filename": filename, "repo_type": "dataset"}
        if self.local_files_only:
            return hf_hub.hf_hub_download(**kwargs, local_files_only=True)
        try:
            return hf_hub.hf_hub_download(**kwargs, local_files_only=False)
        except _connection_error_types(hf_hub) as exc:
            logger.warning(
                "Could not reach the Hugging Face Hub to validate '%s' (%s); "
                "skipping the ETag check and falling back to the local cache.",
                filename,
                exc,
            )
            return hf_hub.hf_hub_download(**kwargs, local_files_only=True)

    def _get_dataset(self) -> tuple[Any, dict[str, np.ndarray]]:
        """Download (if needed) and memory-map the reconstruction dataset."""
        if self._samples is None or self._class_indices is None:
            hf_hub = _load_hf_hub()
            samples_path = self._download_dataset_file(hf_hub, SAMPLES_FILENAME)
            labels_path = self._download_dataset_file(hf_hub, LABELS_FILENAME)
            label_order_path = self._download_dataset_file(hf_hub, LABEL_ORDER_FILENAME)

            samples = np.load(samples_path, mmap_mode="r")
            labels = np.asarray(np.load(labels_path), dtype=float)
            label_order = [str(name) for name in np.load(label_order_path)]

            if samples.ndim != _TABLE_NDIM:
                raise ValueError("DeepExtractor samples dataset must be two-dimensional.")
            if labels.ndim != _TABLE_NDIM or labels.shape[0] != samples.shape[0]:
                raise ValueError("DeepExtractor labels must match the samples row count.")
            if labels.shape[1] != len(label_order):
                raise ValueError("DeepExtractor label order must match the labels column count.")

            class_columns = np.argmax(labels, axis=1)
            class_indices: dict[str, np.ndarray] = {}
            for glitch_class in self.glitch_classes or ():
                if glitch_class not in label_order:
                    raise ValueError(f"Glitch class '{glitch_class}' is not present in the dataset labels.")
                indices = np.flatnonzero(class_columns == label_order.index(glitch_class))
                if indices.size == 0:
                    raise ValueError(f"Glitch class '{glitch_class}' has no samples in the dataset.")
                class_indices[glitch_class] = indices

            self._samples = samples
            self._class_indices = class_indices
        return self._samples, self._class_indices

    @staticmethod
    def _resample(white_waveform: np.ndarray, sampling_frequency: float) -> np.ndarray:
        """Resample a native-rate waveform onto the simulation sample grid."""
        if sampling_frequency == NATIVE_SAMPLING_FREQUENCY:
            return white_waveform
        native_times = np.arange(white_waveform.size, dtype=float) / NATIVE_SAMPLING_FREQUENCY
        n_resampled = max(1, round(white_waveform.size * sampling_frequency / NATIVE_SAMPLING_FREQUENCY))
        resampled_times = np.arange(n_resampled, dtype=float) / sampling_frequency
        return np.interp(resampled_times, native_times, white_waveform)

    def generate_waveform(
        self,
        sampling_frequency: float,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Generate one colored, SNR-calibrated DeepExtractor glitch."""
        if sampling_frequency <= 0.0:
            raise ValueError("sampling_frequency must be greater than zero.")

        generator = np.random.default_rng() if rng is None else rng
        samples, class_indices = self._get_dataset()

        glitch_class = self._draw_class(generator)
        pool = class_indices[glitch_class]
        sample_index = int(pool[int(generator.integers(0, pool.size))])
        white_waveform = np.asarray(samples[sample_index], dtype=float)

        white_waveform = self._resample(white_waveform, sampling_frequency)
        colored = color_whitened_waveform(
            white_waveform,
            sampling_frequency=sampling_frequency,
            psd_frequencies=self._psd_frequencies,
            psd_values=self._psd_values,
            low_frequency_cutoff=self.low_frequency_cutoff,
            high_frequency_cutoff=self.high_frequency_cutoff,
        )
        achieved_snr = optimal_snr(colored, sampling_frequency=sampling_frequency)
        if achieved_snr <= 0.0:
            raise ValueError("The drawn glitch has no power in the requested frequency band.")

        amplitude = self.amplitude_distribution.sample(generator)
        return amplitude * (self._target_snr(glitch_class) / achieved_snr) * colored.time_series

    def serialize(self) -> dict[str, Any]:
        """Return metadata-friendly model parameters."""
        serialized = GlitchModel.serialize(self)
        if self._class_rates is not None:
            serialized["rate"] = dict(self._class_rates)
        return serialized | {
            "psd_file": str(self.psd_file),
            "snr": dict(self.snr) if isinstance(self.snr, dict) else self.snr,
            "glitch_classes": list(self.glitch_classes or ()),
            "low_frequency_cutoff": self.low_frequency_cutoff,
            "high_frequency_cutoff": self.high_frequency_cutoff,
            "repo_id": self.repo_id,
            "local_files_only": self.local_files_only,
        }
