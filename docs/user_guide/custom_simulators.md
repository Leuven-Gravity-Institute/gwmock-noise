# Advanced: Custom simulators

For minimal usage snippets see [Minimal usage](minimal_usage.md).

`gwmock-noise` exposes a public structural protocol,
`gwmock_noise.NoiseSimulator`, for simulator implementations that want to plug
into the package without subclassing internal base classes.

## Required surface

A protocol-conformant simulator provides:

- `duration`, `sampling_frequency`, `detectors`, and `seed` attributes
- `generate(...)` for one-shot realizations
- `generate_stream(...)` for stateful chunk iteration
- `metadata` for descriptive runtime metadata

The public `open_stream(...)` helper consumes any object that satisfies that
protocol. The helper does not reach into package internals; it simply validates
the public contract and returns the simulator's stream iterator.

## Minimal example

```python
from collections.abc import Iterator
from typing import Any

import numpy as np

from gwmock_noise import NoiseSimulator, open_stream


class RampNoiseSimulator:
    def __init__(self) -> None:
        self.duration = 1.0
        self.sampling_frequency = 8.0
        self.detectors = ["H1"]
        self.seed = None
        self._offset = 0

    def generate(
        self,
        duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> dict[str, np.ndarray]:
        n_samples = round(duration * sampling_frequency)
        self.duration = duration
        self.sampling_frequency = sampling_frequency
        self.detectors = list(detectors)
        self.seed = seed
        return {detector: np.arange(n_samples, dtype=float) for detector in detectors}

    def generate_stream(
        self,
        chunk_duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> Iterator[dict[str, np.ndarray]]:
        n_samples = round(chunk_duration * sampling_frequency)
        while True:
            start = self._offset
            stop = start + n_samples
            self._offset = stop
            yield {detector: np.arange(start, stop, dtype=float) for detector in detectors}

    @property
    def metadata(self) -> dict[str, Any]:
        return {"implementation": "ramp"}


simulator: NoiseSimulator = RampNoiseSimulator()
stream = open_stream(
    simulator,
    chunk_duration=0.5,
    sampling_frequency=8.0,
    detectors=["H1", "L1"],
    seed=7,
)
first_chunk = next(stream)
```

## Continuation contract

If your simulator advertises stateful continuation, consecutive chunks from
`generate_stream(...)` should be the same realization a caller would obtain by
asking the same simulator for one seeded single-shot run over the combined
duration. For overlap-add or filter-memory models, keep that state inside the
iterator rather than serializing opaque tokens between calls.
