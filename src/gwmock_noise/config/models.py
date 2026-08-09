"""Pydantic configuration schema for noise simulation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, Field, field_validator, model_validator


class NoiseComponentConfig(BaseModel):
    """One configurable noise component in a composed simulation."""

    simulator: str = Field(description="Registered simulator/component name.")
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Simulator-specific options passed to the component builder.",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_component_definition(cls, value: Any) -> Any:
        """Accept string, flat mapping, or explicit ``{simulator, options}`` input."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return {"simulator": value, "options": {}}
        if not isinstance(value, dict):
            raise TypeError("components entries must be strings, mappings, or NoiseComponentConfig instances.")

        if "simulator" not in value:
            raise ValueError("components entries must define a simulator name.")

        normalized = dict(value)
        simulator = normalized.pop("simulator")
        declared_options = normalized.pop("options", None)
        if declared_options is None:
            options = normalized
        else:
            if not isinstance(declared_options, dict):
                raise ValueError("component options must be a mapping when provided explicitly.")
            overlap = set(normalized) & set(declared_options)
            if overlap:
                duplicated = ", ".join(sorted(overlap))
                raise ValueError(f"component options duplicate explicit fields: {duplicated}.")
            options = dict(declared_options)
            options.update(normalized)
        return {"simulator": simulator, "options": options}

    @model_validator(mode="after")
    def validate_component(self) -> Self:
        """Validate the normalized component entry."""
        if not self.simulator.strip():
            raise ValueError("component simulator names must be non-empty.")
        return self


#: Characters that cannot appear in a detector or channel name without breaking an artifact.
#:
#: `/` is the worst of them and the reason this exists: HDF5 treats it as a group separator, so a channel
#: like `MOCK/NOISE` silently produced a nested group instead of a dataset, and GWpy then failed to read
#: the file the writer had just reported as written. The failure was invisible -- a path came back, the
#: file existed, and only reading it showed the damage.
#:
#: `\\` and `:` follow because these names reach file names: a colon opens an alternate data stream on
#: NTFS, and a backslash is a path separator there.
#:
#: Rejecting at the boundary rather than escaping deeper: an escape has to be maintained for whatever
#: character bites next, and the last two attempts here each fixed one character and broke another.
_UNSAFE_NAME_CHARACTERS = ("/", "\\", ":")


def _reject_unsafe(value: str, *, field: str) -> str:
    """Return *value* unchanged, or raise if it cannot survive being part of an artifact's identity.

    Args:
        value: The detector or channel name.
        field: The field being validated, for the message.

    Returns:
        The value, unchanged.

    Raises:
        ValueError: If the value contains a character that would break the HDF5 layout or the file name.
    """
    # A channel may legitimately contain one colon, as `DETECTOR:CHANNEL`; it is the *file name* that
    # cannot, and the writer no longer puts the channel there. Detectors get the stricter rule because
    # they do become file names.
    forbidden = ("/", "\\") if field == "channel" else _UNSAFE_NAME_CHARACTERS
    found = [character for character in forbidden if character in value]
    if found:
        raise ValueError(
            f"{field} {value!r} contains {', '.join(repr(character) for character in found)}, which "
            f"cannot appear in an artifact name: '/' is an HDF5 group separator and '\\' and ':' are "
            f"path syntax on Windows. Rename it, or the artifact would be written somewhere other than "
            f"where it is reported."
        )
    return value


class OutputConfig(BaseModel):
    """Configuration for simulation output."""

    directory: Path = Field(default=Path("."), description="Output directory for generated data.")
    prefix: str = Field(default="noise", description="Prefix for output filenames.")
    format: str = Field(
        default="npy",
        description="Artifact format written by BaseNoiseSimulator.run(): 'npy', 'gwf', or 'hdf5'.",
        pattern="^(npy|gwf|hdf5)$",
    )
    gps_start: float = Field(
        default=0.0,
        description="GPS start time used for timestamped output formats such as GWF.",
    )

    channel: str = Field(
        default="MOCK_NOISE",
        description="Channel name suffix for GWF frame output. Assembled as {detector}:{channel}.",
    )
    channels: dict[str, str] | None = Field(
        default=None,
        description=(
            "Per-detector full channel names, e.g. {'H1': 'H1:STRAIN_NOISE'}. When set, takes precedence over channel."
        ),
    )

    @model_validator(mode="after")
    def _validate_channel_names(self) -> Self:
        """Reject channel names that the selected format cannot represent.

        Only for the formats that use the channel. `npy` writes a bare array and never reads it, so
        rejecting `MOCK/NOISE` there would turn a configuration that worked into a hard failure on
        upgrade for no benefit -- both reviewers caught that, and it is why this is not a field
        validator: a field validator cannot see `format`.

        Returns:
            The validated model.

        Raises:
            ValueError: If a channel the format will use contains characters it cannot carry.
        """
        if self.format not in {"gwf", "hdf5"}:
            return self
        _reject_unsafe(self.channel, field="channel")
        for detector, channel in (self.channels or {}).items():
            _reject_unsafe(channel, field="channel")
            # The key is a detector name and gets the detector rule. Such an entry can never match a
            # validated detector, so it is inert rather than dangerous -- but an inert override is
            # almost certainly a typo, and saying so beats silently ignoring it.
            _reject_unsafe(detector, field="detector")
        return self


def _default_components() -> list[NoiseComponentConfig]:
    """Return the legacy default of one white-noise component."""
    return [NoiseComponentConfig(simulator="white")]


class NoiseConfig(BaseModel):
    """Generic configuration for composed detector-noise simulations."""

    @field_validator("detectors")
    @classmethod
    def _validate_detectors(cls, value: list[str]) -> list[str]:
        """Reject a detector name that would put path syntax into a file name."""
        for detector in value:
            _reject_unsafe(detector, field="detector")
        return value

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
    components: list[NoiseComponentConfig] = Field(
        default_factory=_default_components,
        description="Ordered list of noise components to generate and add together.",
    )

    model_config = {"frozen": False, "extra": "ignore"}
