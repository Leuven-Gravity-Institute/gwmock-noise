# Advanced: Joint strain and witness channels

For minimal usage snippets see [Minimal usage](minimal_usage.md). For the legacy
strain-only protocol see [Custom simulators](custom_simulators.md).

`gwmock-noise` exposes a second, optional structural protocol,
`gwmock_noise.JointStrainWitnessSimulator`, for backends that generate
simultaneous strain **and** auxiliary/witness channels with a documented
statistical relationship between them. It is strictly additive: nothing about
the legacy `NoiseSimulator` protocol changes, and a `NoiseSimulator`-only
consumer is unaffected by its existence. A class may implement both protocols,
or only this one.

## Required surface

A protocol-conformant joint simulator provides:

- `duration`, `sampling_frequency`, `detectors`, `witnesses`, and `seed`
  attributes
- `generate_joint(...)` for one-shot simultaneous realizations, returning a
  `JointRealization`
- `generate_joint_stream(...)` for stateful chunk iteration, yielding
  `JointRealization` objects
- `covariance()` returning the joint one-sided cross-spectral density as a
  `JointCovariance`
- `metadata` for descriptive runtime metadata, in the same role as
  `NoiseSimulator.metadata`

`JointRealization` carries `strain` and `witness` dictionaries (each
`dict[str, numpy.ndarray]`, the same per-channel array convention
`NoiseSimulator.generate` uses), `channel_metadata`
(`dict[str, ChannelMetadata]`, typed per-channel unit/domain/kind/dtype metadata
for every channel in both dictionaries), and `provenance` (an
implementation-defined mapping describing how the realization was produced).

## Fourier convention

Wherever this protocol exposes a frequency-domain quantity (`covariance()`), the
frequency grid is the one-sided, non-negative grid produced by
`numpy.fft.rfftfreq`, and the reported density is the _one-sided_ spectral
density -- the convention already used by `MultichannelNoiseSimulator`.

## Continuation contract

The same continuation contract `NoiseSimulator.generate_stream` makes for
strain-only output applies here to both strain and witness channels at once:
consecutive chunks from `generate_joint_stream(...)` should equal the same
realization a caller would obtain from one seeded `generate_joint(...)` call
over the combined duration.

## The public dummy backend

`gwmock_noise.JointDummyCorrelatedSimulator` is a public, physics-free reference
implementation: every channel is temporally white with a fixed equicorrelation
structure, so every strain/witness pair is genuinely correlated by construction
and `covariance()` is exact and known in closed form rather than fitted.

```python
from gwmock_noise import JointDummyCorrelatedSimulator

simulator = JointDummyCorrelatedSimulator(
    detectors=["H1", "L1"],
    witnesses=["SEIS1"],
    sampling_frequency=256.0,
    coupling=0.3,
)
realization = simulator.generate_joint(4.0, 256.0, ["H1", "L1"], ["SEIS1"], seed=7)
realization.strain["H1"]
realization.witness["SEIS1"]
```

## External discovery without a private import

A backend -- including one that lives in a private package this repository never
imports -- becomes discoverable by registering itself under the
`gwmock_noise.joint_backends` entry-point group in its own package metadata:

```toml
[project.entry-points."gwmock_noise.joint_backends"]
my_backend = "my_package.module:MyBackendClass"
```

`gwmock_noise.discover_joint_backends()` and
`gwmock_noise.load_joint_backend(name)` find and import a registered backend
purely through `importlib.metadata.entry_points`, with no reference to any
specific package in the discovery code itself:

```python
from gwmock_noise import available_joint_backend_names, load_joint_backend

available_joint_backend_names()  # e.g. ("dummy_correlated",)
backend_class = load_joint_backend("dummy_correlated")
```
