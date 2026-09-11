"""Composable glitch-only component simulator."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gwmock_noise.glitches.models import normalize_glitch_models
from gwmock_noise.simulators.base import ConfigurableNoiseSimulator
from gwmock_noise.simulators.glitches import InjectGlitches, _ZeroNoiseSimulator

if TYPE_CHECKING:
    from gwmock_noise.config.models import NoiseComponentConfig, NoiseConfig


class GlitchNoiseSimulator(ConfigurableNoiseSimulator):
    """Generate transient glitches as an additive standalone component."""

    simulator_name = "glitches"

    @classmethod
    def from_component(cls, component: NoiseComponentConfig, config: NoiseConfig) -> InjectGlitches:
        """Construct a glitch-only component from one component definition."""
        options = dict(component.options)
        glitch_models = normalize_glitch_models(options.pop("models", []))
        if not glitch_models:
            raise ValueError("glitches component requires at least one glitch model in 'models'.")
        if options:
            unexpected = ", ".join(sorted(options))
            raise ValueError(f"glitches component received unexpected options: {unexpected}.")
        return InjectGlitches(
            _ZeroNoiseSimulator(
                detectors=config.detectors,
                duration=config.duration,
                sampling_frequency=config.sampling_frequency,
                seed=config.seed,
            ),
            glitch_models,
            # The run's epoch, so the truth catalogue reads in the same GPS time the
            # artifact names and the HDF5 `x0` attribute carry. Without it the catalogue
            # would time every event from zero while the strain sat at a real epoch.
            gps_start=config.output.gps_start,
        )
