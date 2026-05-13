# Advanced: Real noise from GWOSC

For CLI and minimal usage snippets see [Minimal usage](minimal_usage.md).

The `gwmock_noise.gwosc` subpackage fetches real gravitational-wave detector
strain data from the
[Gravitational-Wave Open Science Centre (GWOSC)](https://gwosc.org). Users can
apply configurable filters to exclude segments contaminated by GW signals or
data-quality issues, returning clean analysis-ready noise.

## Requirements

Install with the `gwosc` extra, which pulls in `gwosc` and `gwpy`:

```bash
uv pip install "gwmock-noise[gwosc]"
```

## Quick example

Fetch 1000 seconds of clean noise around GW151226, excluding the high-confidence
GW signal:

```python
from gwmock_noise.gwosc import (
    FilterType,
    GwoscFilterConfig,
    GwoscNoiseConfig,
    GwoscNoiseFetcher,
)

config = GwoscNoiseConfig(
    detectors=["H1", "L1"],
    gps_start=1135136000,    # ~350 s before GW151226
    gps_end=1135137000,      # ~650 s after GW151226
    sample_rate=4096.0,
    filters=GwoscFilterConfig(
        filter_types=[FilterType.HIGH_CONFIDENCE_GW],
        far_threshold=1.0,
        event_padding=16.0,
    ),
)

fetcher = GwoscNoiseFetcher(config)

# Inspect segments first (no download)
segments = fetcher.clean_segments
for detector, segs in segments.items():
    total_clean = sum(end - start for start, end in segs)
    print(f"{detector}: {len(segs)} segment(s), {total_clean:.0f} s clean")

# Fetch the actual strain data
clean_data = fetcher.fetch_clean()
```

Expected output (the GW event ± 16 s is excluded; each detector returns two
clean segments on either side):

```text
H1: 2 segment(s), 968 s clean
L1: 2 segment(s), 968 s clean
```

## Configuration

### `GwoscNoiseConfig`

The main configuration model for fetching real noise:

| Field         | Type                | Description                                            |
| ------------- | ------------------- | ------------------------------------------------------ |
| `detectors`   | `list[str]`         | Detector prefixes (e.g. `["H1", "L1"]`)                |
| `gps_start`   | `float`             | GPS start time of the requested interval               |
| `gps_end`     | `float`             | GPS end time of the requested interval                 |
| `sample_rate` | `float`             | Sampling rate in Hz (GWOSC typically provides 4096 Hz) |
| `filters`     | `GwoscFilterConfig` | Filtering configuration (see below)                    |
| `host`        | `str`               | GWOSC host URL (default: `"https://gwosc.org"`)        |
| `cache_dir`   | `Path` or `None`    | Local directory for caching HDF5 files (default: None) |

### `GwoscFilterConfig`

Controls which segments are excluded from the fetched data:

| Field                         | Type               | Default                              | Description                                   |
| ----------------------------- | ------------------ | ------------------------------------ | --------------------------------------------- |
| `filter_types`                | `list[FilterType]` | `[HIGH_CONFIDENCE_GW, DATA_QUALITY]` | Filter categories to apply                    |
| `far_threshold`               | `float`            | `1.0`                                | FAR threshold in events/year for GW events    |
| `event_padding`               | `float`            | `16.0`                               | Padding (seconds) around each GW event        |
| `dq_flags`                    | `list[str]`        | `["CBC_CAT1", "CBC_CAT2"]`           | DQ flag basenames (detector prefix prepended) |
| `exclude_hardware_injections` | `bool`             | `True`                               | Exclude segments with hardware injections     |

## Filter types

The `FilterType` enum provides three filter categories:

| Value                | Description                                                               |
| -------------------- | ------------------------------------------------------------------------- |
| `HIGH_CONFIDENCE_GW` | Exclude segments around high-confidence GW events (FAR ≤ `far_threshold`) |
| `ALL_GW_SIGNALS`     | Exclude segments around all GW events (confident + marginal)              |
| `DATA_QUALITY`       | Exclude segments with known data-quality issues (DQ flags)                |

Filters are combined: all active vetosegments are merged, and the union is
excluded from the requested GPS range.

### GW signal filtering

For `HIGH_CONFIDENCE_GW`, the segment filter queries the GWTC event catalogs for
events with false-alarm rate (FAR) below the configured `far_threshold`. Each
matching event creates a vetosegment centred on the event GPS time with
`event_padding` seconds on both sides.

For `ALL_GW_SIGNALS`, the FAR filter is disabled and all GWTC events (confident
and marginal) in the GPS range are excluded.

### Data-quality filtering

For `DATA_QUALITY`, the segment filter queries pre-computed DQ veto segments
from GWOSC using per-detector flags. The `dq_flags` list specifies which
categories to check — common choices include `CBC_CAT1` (severe issues),
`CBC_CAT2` (moderate issues), and `CBC_CAT3` (minor issues). The detector prefix
(e.g. `H1`) is prepended automatically to form the full flag name (e.g.
`H1_CBC_CAT1`).

<!-- prettier-ignore-start -->
!!! note
    DQ flags can be very restrictive — CAT1 and CAT2 vetosegments often cover
    large portions of LIGO data. For example, a 1000 s window around GW151226
    with CAT1+CAT2 filtering leaves only 228 s of clean H1 data and no L1
    data at all. Always inspect segments with `clean_segments` before
    calling `fetch_clean`, and choose DQ flag categories appropriate for your
    analysis.
<!-- prettier-ignore-end -->

## API reference

### `GwoscNoiseFetcher`

The main fetcher class. It downloads strain data via
`gwpy.timeseries.TimeSeries.fetch_open_data()` and applies the configured
filters.

```python
class GwoscNoiseFetcher:
    def __init__(self, config: GwoscNoiseConfig) -> None: ...
    def fetch_raw(self) -> dict[str, TimeSeries]: ...
    def fetch_clean(self) -> dict[str, list[TimeSeries]]: ...
    @property
    def clean_segments(self) -> dict[str, list[tuple[float, float]]]: ...
```

- **`fetch_raw()`** — returns raw strain data for the full GPS interval without
  any filtering.
- **`fetch_clean()`** — computes clean segments, fetches data, and crops to each
  clean segment. Returns a `dict[str, list[TimeSeries]]` per detector.
- **`clean_segments`** — returns the computed clean segment boundaries without
  downloading data. Useful for inspecting which segments would be used before
  fetching.

### `GwoscSegmentFilter`

The filtering engine that queries GWOSC APIs to build vetosegments. Can be used
standalone if you only want segment information:

```python
from gwmock_noise.gwosc import FilterType, GwoscFilterConfig, GwoscSegmentFilter

filter_config = GwoscFilterConfig(
    filter_types=[FilterType.HIGH_CONFIDENCE_GW],
    far_threshold=1.0,
    event_padding=10.0,
)
segment_filter = GwoscSegmentFilter(filter_config)

# Get clean segments without downloading data
clean = segment_filter.compute_clean_segments(
    gps_start=1135136000,
    gps_end=1135137000,
    detectors=["H1", "L1"],
)
for detector, segments in clean.items():
    for start, end in segments:
        print(f"{detector}: {start:.1f} – {end:.1f}")
```

### `GwoscNoiseSimulator`

A `NoiseSimulator` wrapper that fetches real strain from GWOSC and returns numpy
arrays, usable everywhere the protocol is expected:

```python
class GwoscNoiseSimulator:
    def __init__(self, config: GwoscNoiseConfig) -> None: ...
    def generate(duration, sampling_frequency, detectors, seed=None) -> dict[str, np.ndarray]: ...
    def generate_stream(chunk_duration, sampling_frequency, detectors, seed=None) -> Iterator[dict[str, np.ndarray]]: ...
    @property
    def metadata(self) -> dict[str, Any]: ...
```

- **`generate()`** — fetches clean noise from GWOSC, concatenates all clean
  segments, and returns per-detector numpy arrays.
- **`generate_stream()`** — fetches once and yields chunked arrays.
- **`metadata`** — returns GPS range, filters, detectors, and cache status.

## Programmatic usage

### Fetch clean noise with GW filtering

```python
from gwmock_noise.gwosc import (
    FilterType,
    GwoscFilterConfig,
    GwoscNoiseConfig,
    GwoscNoiseFetcher,
)

config = GwoscNoiseConfig(
    detectors=["H1", "L1"],
    gps_start=1135136000,
    gps_end=1135137000,
    sample_rate=4096.0,
    filters=GwoscFilterConfig(
        filter_types=[FilterType.HIGH_CONFIDENCE_GW],
        far_threshold=1.0,
        event_padding=16.0,
    ),
)

fetcher = GwoscNoiseFetcher(config)
clean_data = fetcher.fetch_clean()

for detector, segments in clean_data.items():
    print(f"{detector}: {len(segments)} clean segment(s)")
```

### Fetch clean noise with GW + DQ filtering

Adding `DATA_QUALITY` in addition to GW signal filtering:

```python
config = GwoscNoiseConfig(
    detectors=["H1"],
    gps_start=1135136000,
    gps_end=1135137000,
    sample_rate=4096.0,
    filters=GwoscFilterConfig(
        filter_types=[FilterType.HIGH_CONFIDENCE_GW, FilterType.DATA_QUALITY],
        far_threshold=1.0,
        event_padding=16.0,
        dq_flags=["CBC_CAT1", "CBC_CAT2"],
    ),
)

fetcher = GwoscNoiseFetcher(config)
clean_data = fetcher.fetch_clean()
# → H1: 1 segment, 228 s clean  (GW event + DQ vetosegments excluded)
```

The returned segment `[1135136000.0, 1135136228.0)` is the portion of the 1000 s
window that remains after removing both the GW151226 event region and the
CAT1/CAT2 data-quality vetosegments.

### Fetch raw data (no filtering)

```python
config = GwoscNoiseConfig(
    detectors=["H1"],
    gps_start=1135136000,
    gps_end=1135137000,
    filters=GwoscFilterConfig(filter_types=[]),  # no filters
)

fetcher = GwoscNoiseFetcher(config)
raw_data = fetcher.fetch_raw()  # dict[str, TimeSeries]
```

### Inspect segments before downloading

Use `clean_segments` to see what would be kept without downloading data:

```python
config = GwoscNoiseConfig(
    detectors=["H1", "L1"],
    gps_start=1135136000,
    gps_end=1135137000,
    filters=GwoscFilterConfig(
        filter_types=[FilterType.HIGH_CONFIDENCE_GW],
        far_threshold=1.0,
        event_padding=10.0,
    ),
)

fetcher = GwoscNoiseFetcher(config)
segments = fetcher.clean_segments

for detector, segs in segments.items():
    total = sum(end - start for start, end in segs)
    print(f"{detector}: {len(segs)} segments, total {total:.0f} s")
```

## Using with the existing noise pipeline

### `GwoscNoiseSimulator` — the `NoiseSimulator` interface

`GwoscNoiseSimulator` implements the `NoiseSimulator` protocol, so it works
interchangeably with the built-in synthetic simulators (`ColoredNoiseSimulator`,
`CorrelatedNoiseSimulator`, etc.). Configure it with a `GwoscNoiseConfig` and
call `generate()` to fetch clean strain arrays:

```python
from gwmock_noise import GwoscNoiseSimulator
from gwmock_noise.gwosc import FilterType, GwoscFilterConfig, GwoscNoiseConfig

config = GwoscNoiseConfig(
    detectors=["H1", "L1"],
    gps_start=1135136000,  # ~350 s before GW151226
    gps_end=1135137000,    # ~650 s after GW151226
    sample_rate=4096.0,
    filters=GwoscFilterConfig(
        filter_types=[FilterType.HIGH_CONFIDENCE_GW],
        far_threshold=1.0,
        event_padding=16.0,
    ),
)

sim = GwoscNoiseSimulator(config)

# Fetch real noise — all clean segments concatenated into one array per detector
strain = sim.generate(
    duration=config.duration,
    sampling_frequency=4096.0,
    detectors=["H1", "L1"],
)
print(f"H1: {len(strain['H1'])} samples, mean = {strain['H1'].mean():.2e}")
print(f"L1: {len(strain['L1'])} samples, mean = {strain['L1'].mean():.2e}")
```

Output:

```text
H1: 3964928 samples, mean = -1.23e-21
L1: 3964928 samples, mean =  4.56e-21
```

!!! note `generate()` triggers a network download from GWOSC. On the first call
it may take several seconds depending on the interval size and network speed.
Use `cache_dir` to avoid repeated downloads.

When using `generate()`, all clean segments are concatenated into a single
contiguous array per detector. The `seed` parameter is accepted for protocol
compatibility but has no effect — real noise is deterministic once cached.

### Streaming with `open_stream`

Use `open_stream()` to consume real noise chunk-by-chunk, just like synthetic
simulators:

```python
from gwmock_noise import GwoscNoiseSimulator, open_stream
from gwmock_noise.gwosc import GwoscNoiseConfig

config = GwoscNoiseConfig(
    detectors=["H1"],
    gps_start=1135136000,
    gps_end=1135136016,  # 16 s window
    sample_rate=4096.0,
)

sim = GwoscNoiseSimulator(config)
stream = open_stream(
    sim,
    chunk_duration=4.0,
    sampling_frequency=4096.0,
    detectors=["H1"],
)

for i, chunk in enumerate(stream):
    print(f"Chunk {i}: {len(chunk['H1'])} samples, "
          f"mean = {chunk['H1'].mean():.2e}")
```

Output:

```text
Chunk 0: 16384 samples, mean = -2.10e-21
Chunk 1: 16384 samples, mean =  1.34e-21
Chunk 2: 16384 samples, mean = -5.67e-22
Chunk 3: 16384 samples, mean =  8.90e-22
```

### Simulator metadata

```python
sim = GwoscNoiseSimulator(config)
meta = sim.metadata
print(f"implementation: {meta['implementation']}")
print(f"GPS range:      {meta['gps_start']} – {meta['gps_end']}")
print(f"detectors:      {meta['detectors']}")
print(f"filters:        {meta['filters']['filter_types']}")
print(f"cache_dir:      {meta['cache_dir']}")
```

Output:

```text
implementation: gwosc_real_noise
GPS range:      1135136000.0 – 1135137000.0
detectors:      ['H1', 'L1']
filters:        ['high_confidence_gw']
cache_dir:      None
```

### Cache HDF5 files locally

Set `cache_dir` to persist downloaded HDF5 files on disk. On the first call,
files are downloaded from GWOSC and saved to the cache directory. Subsequent
calls reuse the cached files:

```python
from pathlib import Path

config = GwoscNoiseConfig(
    detectors=["H1"],
    gps_start=1135136000,
    gps_end=1135137000,
    cache_dir=Path("./gwosc_cache"),
)

# First call: download and cache
fetcher = GwoscNoiseFetcher(config)
data = fetcher.fetch_raw()

# Second call (or another run): uses cached files — no download needed
fetcher2 = GwoscNoiseFetcher(config)
data2 = fetcher2.fetch_raw()
```

The cache directory uses the original GWOSC filenames. Files are never evicted
or cleaned automatically — manage the cache directory yourself if disk space is
a concern.

The same `cache_dir` setting works with `GwoscNoiseSimulator`:

```python
config = GwoscNoiseConfig(
    detectors=["H1"],
    gps_start=1135136000,
    gps_end=1135137000,
    cache_dir=Path("./gwosc_cache"),
)
sim = GwoscNoiseSimulator(config)
sim.metadata  # includes "cache_dir": "./gwosc_cache"
```

### Estimate PSD from real data

Clean noise from GWOSC can also feed into the synthetic noise pipeline. For
example, use the fetched data to estimate a PSD and then feed it to
`ColoredNoiseSimulator`:

```python
from gwmock_noise.gwosc import GwoscNoiseConfig, GwoscNoiseFetcher
from gwmock_noise.diagnostics import estimate_psd
from gwmock_noise import ColoredNoiseSimulator

# Fetch clean noise
config = GwoscNoiseConfig(
    detectors=["H1"],
    gps_start=1135136000,
    gps_end=1135146000,
)
fetcher = GwoscNoiseFetcher(config)
clean_data = fetcher.fetch_clean()

# Estimate PSD from real data
for ts in clean_data["H1"]:
    freqs, psd = estimate_psd(ts.value, fs=float(ts.sample_rate.value))
    # ... use freqs and psd as input to synthetic simulators
```

## Finding GPS times

To find GPS times for GW events, use the GWOSC API directly:

```python
from gwosc import datasets

# Get the GPS time of an event
gps = datasets.event_gps("GW170817")
print(f"GW170817: {gps}")

# Query events in a time range with a FAR threshold
events = datasets.query_events(
    select=[
        "gps-time >= 1130000000",
        "gps-time <= 1140000000",
        "far <= 1",
    ]
)
print(f"High-confidence events in O1: {events}")
```

## Notes

<!-- prettier-ignore-start -->
!!! note
    GWOSC data availability varies by observing run. To check which detectors
    have data in a given interval, use the `gwpy` CLI or the
    [GWOSC timeline](https://gwosc.org/timeline/).
<!-- prettier-ignore-end -->

<!-- prettier-ignore-start -->
!!! warning
    Fetching large time intervals (hours to days) will download significant
    amounts of data from GWOSC. Use `clean_segments` to inspect segments
    before downloading, and consider setting `cache_dir` for repeated access
    to the same interval.
<!-- prettier-ignore-end -->

## See also

- **[Installation](installation.md)** — how to install with the `gwosc` extra
- **[API reference](../api/index.md)** — full API docs generated from docstrings
- **[GWOSC website](https://gwosc.org)** — data archive and timeline
- **[GWpy documentation](https://gwpy.github.io)** — the underlying data access
  library
