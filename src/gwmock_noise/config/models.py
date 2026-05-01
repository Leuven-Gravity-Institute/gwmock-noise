"""Pydantic models for noise simulation configuration."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, model_validator

DETECTOR_PAIR_SIZE = 2


class OutputConfig(BaseModel):
    """Configuration for simulation output."""

    directory: Path = Field(default=Path("."), description="Output directory for generated data.")
    prefix: str = Field(default="noise", description="Prefix for output filenames.")


class NoiseConfig(BaseModel):
    """Configuration for gravitational wave detector noise simulation.

    This model is designed to be imported and composed into larger configuration
    structures by upstream packages (e.g., gwmock).
    """

    detectors: list[str] = Field(
        default=["H1", "L1"],
        description="List of detector names to simulate.",
        min_length=1,
    )
    duration: float = Field(
        default=4.0,
        gt=0,
        description="Duration of the noise realization in seconds.",
    )
    sampling_frequency: float = Field(
        default=4096.0,
        gt=0,
        description="Sampling frequency in Hz.",
    )
    output: OutputConfig = Field(
        default_factory=OutputConfig,
        description="Output configuration.",
    )
    seed: int | None = Field(
        default=None,
        description="Random seed for reproducibility. If None, use system entropy.",
    )
    psd_file: Path | None = Field(
        default=None,
        description="Optional PSD file for FFT-based colored-noise simulation.",
    )
    psd_schedule: list[tuple[float, Path]] | None = Field(
        default=None,
        description="Optional time-ordered list of (gps_offset_seconds, PSD file) anchors for time-varying colored-noise simulation.",
        min_length=1,
    )
    psd_files: dict[str, Path] | None = Field(
        default=None,
        description="Optional per-detector PSD files for correlated-noise simulation.",
    )
    csd_files: dict[str, Path] | None = Field(
        default=None,
        description="Optional pairwise CSD files keyed as 'DET1-DET2' for correlated-noise simulation.",
    )
    low_frequency_cutoff: float = Field(
        default=2.0,
        ge=0,
        description="Lower frequency cutoff applied during colored-noise generation.",
    )
    high_frequency_cutoff: float | None = Field(
        default=None,
        gt=0,
        description="Upper frequency cutoff applied during colored-noise generation.",
    )

    def _validate_frequency_bounds(self, *, nyquist: float) -> None:
        """Validate low/high cutoff values against Nyquist."""
        low = self.low_frequency_cutoff
        high = self.high_frequency_cutoff

        if low < 0:
            raise ValueError("low_frequency_cutoff must be >= 0.")
        if low > nyquist:
            raise ValueError(
                f"low_frequency_cutoff must be <= Nyquist ({nyquist} Hz) for sampling_frequency={self.sampling_frequency} Hz."
            )

        if high is None:
            return

        if high < 0:
            raise ValueError("high_frequency_cutoff must be >= 0.")
        if high > nyquist:
            raise ValueError(
                f"high_frequency_cutoff must be <= Nyquist ({nyquist} Hz) for sampling_frequency={self.sampling_frequency} Hz."
            )
        if high <= low:
            raise ValueError("high_frequency_cutoff must be greater than low_frequency_cutoff.")

    def _validate_psd_inputs(self) -> None:
        """Validate PSD configuration choices."""
        if self.psd_file is not None and self.psd_files is not None:
            raise ValueError("psd_file and psd_files are mutually exclusive.")
        if self.psd_file is not None and self.psd_schedule is not None:
            raise ValueError("psd_file and psd_schedule are mutually exclusive.")
        if self.psd_schedule is not None and self.psd_files is not None:
            raise ValueError("psd_schedule and psd_files are mutually exclusive.")

        if self.psd_files is not None and set(self.psd_files) != set(self.detectors):
            raise ValueError("psd_files keys must exactly match detectors.")
        if self.psd_schedule is None:
            return

        offsets = [offset for offset, _ in self.psd_schedule]
        if offsets != sorted(offsets):
            raise ValueError("psd_schedule entries must be sorted by GPS offset.")
        if len(offsets) != len(set(offsets)):
            raise ValueError("psd_schedule entries must use distinct GPS offsets.")

    def _validate_csd_inputs(self) -> None:
        """Validate pairwise CSD configuration keys."""
        if self.csd_files is None:
            return

        if self.psd_files is None:
            raise ValueError("csd_files requires psd_files to be configured.")

        seen_pairs: set[tuple[str, str]] = set()
        for pair_key in self.csd_files:
            detectors = pair_key.split("-")
            if len(detectors) != DETECTOR_PAIR_SIZE or not all(detectors):
                raise ValueError("csd_files keys must use the 'DET1-DET2' format.")

            detector_a, detector_b = tuple(sorted(detectors))
            if detector_a == detector_b:
                raise ValueError("csd_files keys must reference two distinct detectors.")
            if detector_a not in self.detectors or detector_b not in self.detectors:
                raise ValueError("csd_files keys must reference configured detectors.")

            normalized_pair = (detector_a, detector_b)
            if normalized_pair in seen_pairs:
                raise ValueError("csd_files contains duplicate detector pairs.")
            seen_pairs.add(normalized_pair)

    @model_validator(mode="after")
    def validate_frequency_cutoffs(self) -> NoiseConfig:
        """Validate cutoff ordering and Nyquist limits."""
        nyquist = self.sampling_frequency / 2
        self._validate_frequency_bounds(nyquist=nyquist)
        self._validate_psd_inputs()
        self._validate_csd_inputs()

        return self

    model_config = {"frozen": False, "extra": "ignore"}
