# Real noise from GWOSC

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

Fetch 1000 seconds of clean noise around the GW150914 event for the LIGO Hanford
detector, excluding the GW signal and known data-quality issues:

```python
import matplotlib.pyplot as plt

from gwmock_noise.gwosc import (
    FilterType,
    GwoscFilterConfig,
    GwoscNoiseConfig,
    GwoscNoiseFetcher,
)

config = GwoscNoiseConfig(
    detectors=["H1"],
    gps_start=1126259362,   # 100 s before GW150914
    gps_end=1126260362,     # 1000 s later
    sample_rate=4096.0,
    filters=GwoscFilterConfig(
        filter_types=[
            FilterType.HIGH_CONFIDENCE_GW,
            FilterType.DATA_QUALITY,
        ],
        far_threshold=1.0,
        event_padding=16.0,
    ),
)

fetcher = GwoscNoiseFetcher(config)
clean_segments = fetcher.fetch_clean()

for detector, segments in clean_segments.items():
    print(f"{detector}: {len(segments)} clean segment(s)")
    for i, ts in enumerate(segments):
        print(f"  segment {i}: {float(ts.t0.value):.1f} → " f"{float(ts.t0.value) + float(ts.duration.value):.1f}")
        ts.plot()
        plt.show()
```

## Configuration

### `GwoscNoiseConfig`

The main configuration model for fetching real noise:

| Field         | Type                | Description                                                |
| ------------- | ------------------- | ---------------------------------------------------------- |
| `detectors`   | `list[str]`         | Detector prefixes (e.g. `["H1", "L1"]`)                    |
| `gps_start`   | `float`             | GPS start time of the requested interval                   |
| `gps_end`     | `float`             | GPS end time of the requested interval                     |
| `sample_rate` | `float`             | Sampling rate in Hz (GWOSC typically provides 4096 Hz)     |
| `filters`     | `GwoscFilterConfig` | Filtering configuration (see below)                        |
| `host`        | `str`               | GWOSC host URL (default: `"https://gwosc.org"`)            |
| `cache`       | `bool`              | Whether to cache downloaded files locally (default: False) |

### `GwoscFilterConfig`

Controls which segments are excluded from the fetched data:

| Field                         | Type               | Default                              | Description                                         |
| ----------------------------- | ------------------ | ------------------------------------ | --------------------------------------------------- |
| `filter_types`                | `list[FilterType]` | `[HIGH_CONFIDENCE_GW, DATA_QUALITY]` | Filter categories to apply                          |
| `far_threshold`               | `float`            | `1.0`                                | FAR threshold in events/year for GW event filtering |
| `event_padding`               | `float`            | `16.0`                               | Padding (seconds) around each GW event              |
| `dq_flags`                    | `list[str]`        | `["CBC_CAT1", "CBC_CAT2"]`           | DQ flag basenames (detector prefix prepended)       |
| `exclude_hardware_injections` | `bool`             | `True`                               | Exclude segments with hardware injections           |

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
events with false-alarm rate (FAR) below the configured `far_threshold`. Only
events in the requested GPS range are considered. Each matching event creates a
vetosegment centred on the event GPS time with `event_padding` seconds on both
sides.

For `ALL_GW_SIGNALS`, the FAR filter is disabled and all GWTC events (confident
and marginal) in the GPS range are excluded.

### Data-quality filtering

For `DATA_QUALITY`, the segment filter queries pre-computed DQ veto segments
from GWOSC using per-detector flags. The `dq_flags` list specifies which
categories to check (common choices: `CBC_CAT1`, `CBC_CAT2`, `CBC_CAT3`). The
detector prefix (e.g. `H1`) is prepended automatically to form the full flag
name (e.g. `H1_CBC_CAT1`).

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
    gps_start=1126259362,
    gps_end=1126260362,
    detectors=["H1", "L1"],
)
for detector, segments in clean.items():
    for start, end in segments:
        print(f"{detector}: {start:.1f} – {end:.1f}")
```

## Programmatic usage

### Fetch clean noise

The simplest workflow: configure, fetch, inspect:

```python
from gwmock_noise.gwosc import (
    FilterType,
    GwoscFilterConfig,
    GwoscNoiseConfig,
    GwoscNoiseFetcher,
)

config = GwoscNoiseConfig(
    detectors=["H1", "L1"],
    gps_start=1261875618,
    gps_end=1261877618,
    sample_rate=4096.0,
    filters=GwoscFilterConfig(
        filter_types=[FilterType.HIGH_CONFIDENCE_GW, FilterType.DATA_QUALITY],
        far_threshold=1.0,
        event_padding=16.0,
        dq_flags=["CBC_CAT1", "CBC_CAT2"],
    ),
)

fetcher = GwoscNoiseFetcher(config)

# Fetch clean segments
clean_data = fetcher.fetch_clean()
for detector, segments in clean_data.items():
    print(f"{detector}: {len(segments)} clean segment(s)")
    for i, ts in enumerate(segments):
        print(f"  segment {i}: duration = {float(ts.duration.value):.1f} s")
```

### Fetch raw data (no filtering)

If you want all data without any filtering:

```python
config = GwoscNoiseConfig(
    detectors=["H1"],
    gps_start=1126259000,
    gps_end=1126260000,
    filters=GwoscFilterConfig(filter_types=[]),  # no filters
)

fetcher = GwoscNoiseFetcher(config)
raw_data = fetcher.fetch_raw()  # dict[str, TimeSeries]
```

### Inspect segments before downloading

Use `clean_segments` to see what would be kept without downloading:

```python
config = GwoscNoiseConfig(
    detectors=["H1", "L1"],
    gps_start=1126259362,
    gps_end=1126260362,
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

Clean noise from GWOSC can serve as input to the synthetic noise pipeline. For
example, you can use the fetched data to estimate a PSD and then feed it to
`ColoredNoiseSimulator`:

```python
from gwmock_noise.gwosc import GwoscNoiseConfig, GwoscNoiseFetcher
from gwmock_noise.diagnostics import estimate_psd
from gwmock_noise import ColoredNoiseSimulator

# Fetch clean noise
config = GwoscNoiseConfig(
    detectors=["H1"],
    gps_start=1126259362,
    gps_end=1126269362,
)
fetcher = GwoscNoiseFetcher(config)
clean_data = fetcher.fetch_clean()

# Estimate PSD from real data
for ts in clean_data["H1"]:
    freqs, psd = estimate_psd(ts.value, fs=float(ts.sample_rate.value))
    # ... use freqs and psd as input to synthetic simulators
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
    before downloading, and consider enabling `cache=True` for repeated access
    to the same interval.
<!-- prettier-ignore-end -->

## See also

- **[Installation](installation.md)** — how to install with the `gwosc` extra
- **[API reference](../api/index.md)** — full API docs generated from docstrings
- **[GWOSC website](https://gwosc.org)** — data archive and timeline
- **[GWpy documentation](https://gwpy.github.io)** — the underlying data access
  library
