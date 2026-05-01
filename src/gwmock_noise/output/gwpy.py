"""GWpy output adapter."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from gwmock_noise.simulators.protocol import NoiseSimulator

if TYPE_CHECKING:
    from gwpy.timeseries import TimeSeries

_GWPY_IMPORT_ERROR = "gwpy is required to use GWpyAdapter. Install it with `pip install gwmock-noise[gwpy]`."


def _load_timeseries() -> type[TimeSeries]:
    """Import and return gwpy.TimeSeries on demand."""
    try:
        module = import_module("gwpy.timeseries")
    except ImportError as exc:
        raise ImportError(_GWPY_IMPORT_ERROR) from exc
    return module.TimeSeries


class GWpyAdapter:
    """Wrap a NoiseSimulator and return gwpy TimeSeries outputs."""

    def __init__(self, base: NoiseSimulator, gps_start: float = 0.0) -> None:
        """Initialize the adapter and verify gwpy is available."""
        _load_timeseries()
        self.base = base
        self.gps_start = gps_start

    def generate(
        self,
        duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> dict[str, TimeSeries]:
        """Generate gwpy TimeSeries objects for each detector."""
        timeseries = _load_timeseries()
        arrays = self.base.generate(
            duration=duration,
            sampling_frequency=sampling_frequency,
            detectors=detectors,
            seed=seed,
        )
        segment_start = self.gps_start
        wrapped = {
            detector: timeseries(
                data,
                t0=segment_start,
                sample_rate=sampling_frequency,
                channel=detector,
            )
            for detector, data in arrays.items()
        }
        self.gps_start = segment_start + duration
        return wrapped

    @property
    def metadata(self) -> dict[str, Any]:
        """Return adapter metadata layered on top of the base simulator."""
        return self.base.metadata | {"output_adapter": "gwpy", "gps_start": self.gps_start}
