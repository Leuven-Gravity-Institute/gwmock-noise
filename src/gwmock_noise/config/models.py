"""Pydantic configuration schema for noise simulation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, Field, field_validator, model_validator

from gwmock_noise.naming import reject_repeated, reject_unsafe


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

    @field_validator("prefix")
    @classmethod
    def _validate_prefix(cls, value: str) -> str:
        """Reject a prefix that would put path syntax into a file name.

        Every format prepends the prefix and none writes it inside the artifact, so unlike the channel
        this applies to `npy` as well, and a field validator suffices because the rule does not depend on
        `format`. This was the one name component nobody checked, and not only on the bypass path: a
        fully validated `OutputConfig(prefix="sub/run")` wrote `sub/run_H1.npy`, below the directory the
        caller named. A reviewer found it in round 9, after eight rounds spent on the other two names.

        Returns:
            The validated prefix.

        Raises:
            ValueError: If the prefix contains path syntax.
        """
        return reject_unsafe(value, field="prefix")

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
        reject_unsafe(self.channel, field="channel")
        for detector, channel in (self.channels or {}).items():
            reject_unsafe(channel, field="channel")
            # The key is a detector name and gets the detector rule. Such an entry can never match a
            # validated detector, so it is inert rather than dangerous -- but an inert override is
            # almost certainly a typo, and saying so beats silently ignoring it.
            reject_unsafe(detector, field="detector")
        return self


def _default_components() -> list[NoiseComponentConfig]:
    """Return the legacy default of one white-noise component."""
    return [NoiseComponentConfig(simulator="white")]


class NoiseConfig(BaseModel):
    """Generic configuration for composed detector-noise simulations."""

    @field_validator("detectors")
    @classmethod
    def _validate_detectors(cls, value: list[str]) -> list[str]:
        """Reject a detector name that would put path syntax into a file name, or a repeated one.

        A repeat is rejected here rather than downstream because the writers key their output by detector,
        so a duplicate collapses into a single entry and the run reports one artifact for two requested
        detectors -- no error, no second file, and nothing to tell the caller their list was not honoured.

        Returns:
            The validated detectors.

        Raises:
            ValueError: If a name carries path syntax, or the same detector appears twice.
        """
        for detector in value:
            reject_unsafe(detector, field="detector")
        return reject_repeated(value)

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

    @model_validator(mode="after")
    def _validate_times_the_name_will_carry(self) -> Self:
        """Reject a fractional epoch or duration for the formats whose artifact names carry them.

        `format_time_token` refuses these too and is the guarantee; this is the convenience, so a caller
        learns at the config boundary rather than part-way through a run. Both layers, as every other
        name rule here has -- `model_construct` skips validators and this repo's own tests use it.

        Only `gwf` and `hdf5`. An `npy` artifact's name carries no time, so a fractional duration there
        collides with nothing, and refusing it would break configurations that work today for no benefit
        -- the over-rejection this package has already walked back twice.

        Returns:
            The validated model.

        Raises:
            ValueError: If the epoch or the duration is not a whole number of seconds.
        """
        if self.output.format not in {"gwf", "hdf5"}:
            return self
        for name, value in (("gps_start", self.output.gps_start), ("duration", self.duration)):
            if not float(value).is_integer():
                raise ValueError(
                    f"{name} {value!r} is not a whole number of seconds, and a {self.output.format} "
                    f"artifact name carries it: times are written as integers, so two runs a fraction "
                    f"apart would compose one name and the second would overwrite the first. Use a "
                    f"whole second, or write `npy`, whose name carries no time."
                )
        return self
