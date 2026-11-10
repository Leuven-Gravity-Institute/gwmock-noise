# Duty cycle

A detector does not observe all the time. It loses lock and relocks, and it
stops for planned maintenance. `gwmock_noise.duty_cycle` draws when each
interferometer of a network observes, so a mock data set can say which
interferometers were observing at any time.

```python
from gwmock_noise.duty_cycle import DAY, DutyCycleModel, MaintenanceSchedule, simulate_duty_cycle

model = DutyCycleModel(maintenance=MaintenanceSchedule(strategy="rotating"))
realization = simulate_duty_cycle(model, ["E1", "E2", "E3"], start=1_000_000_000, duration=30 * DAY, seed=1)

realization.observing["E1"]  # (n, 2) array of [start, end) observing intervals, GPS seconds
realization.fraction_with_n_observing()  # time with 0, 1, 2, 3 interferometers observing
realization.state_vector("E1", start=1_000_000_000, sampling_frequency=1.0, n_samples=2048)
```

The realization does not change the strain. It is a record laid over it: segment
lists, flags in an injection catalogue, or a state-vector channel.

The state vector's bits follow LIGO's calibrated state vector, so a pipeline
that selects analysable LIGO data by its first bits selects observing time here
too:

| Bit | Value | Set while                            | LIGO meaning (LIGO name; PyCBC name)                     |
| --- | ----- | ------------------------------------ | -------------------------------------------------------- |
| 0   | 1     | the interferometer observes          | h(t) was successfully computed (`HOFT_OK`; `HOFT_OK`)    |
| 1   | 2     | the interferometer observes          | observation-intent mode (`OBS_INTENT`; `SCIENCE_INTENT`) |
| 2   | 4     | the interferometer observes          | observation-ready mode (`OBS_READY`; `SCIENCE_QUALITY`)  |
| 16  | 65536 | the interferometer is in maintenance | not in use in LIGO's O2 vector; outside PyCBC's list     |

`OBSERVING_MASK` (`0b111`) and `MAINTENANCE_BIT` name them. The LIGO meanings
are those of Table III of Viets et al., _Reconstructing the calibrated strain
signal in the Advanced LIGO detectors_, Class. Quantum Grav. 35, 095015 (2018),
[arXiv:1710.09973](https://arxiv.org/abs/1710.09973), for O2. PyCBC's names are
in
[`pycbc/frame/__init__.py`](https://github.com/gwastro/pycbc/blob/master/pycbc/frame/__init__.py);
they come from a later revision of the vector, and the two lists agree only up
to bit 8 (bit 9 is `NO_GAP` in O2 and `KAPPA_A_OK` in PyCBC). Nothing above bit
8 is used for that reason, and bits 3 to 8 describe the calibration and hardware
injections, which the realization does not model.

That paper advises analysing data where bits 0 and 1 are both set. Bit 2 is set
too, so that a pipeline that also requires it (PyCBC's `SCIENCE_QUALITY`) still
selects observing time. PyCBC's low-latency search, for example, selects it with
`--analyze-flags <ifo>:HOFT_OK,SCIENCE_INTENT`; see below.

Only these bits are ever set. In LIGO's vector bit 0 is the logical AND of bits
2 to 4, 11, 13, 17 and 25. Of those this realization sets only bit 2; the rest
record how the calibration was produced, which it does not model, so bit 0 here
marks observing time and does not vouch for a calibration. Likewise the
realization sets no data-quality veto, gap or hardware-injection flag, so a mask
that also requires one of LIGO's other bits, such as `NO_GAP` or the
`NO_*_HW_INJ` bits, selects nothing: ask only for bits 0 to 2. Bit 2 says the
interferometer is ready to observe, not that its data are free of glitches. The
layout is LIGO's; Virgo's state vector is laid out differently.

The samples are `uint32`, wide enough for bit 16 (gwpy's `StateVector` offers as
many bits as the type has). A sample outside the realization's span is 0.

## Reading the state vector in a pipeline

A pipeline that reads a state-vector channel selects observing time with the
LIGO flags above. These examples are run by this package's tests with each
pipeline's current release, so a change on either side is caught there. They
build the state vector from a realization; in a gwmock run it comes from frame
files, with the segment files beside it: see gwmock's
[Analysing Data with a Duty Cycle](https://leuven-gravity-institute.github.io/gwmock/user-guide/analysis-pipelines/).

**gwpy** names the bits and turns each into segments. Those of `HOFT_OK` and
`OBS_INTENT` together are the observing time, to within one sample:

<!-- example: gwpy -->

```python
import numpy as np
from gwpy.timeseries import StateVector

from gwmock_noise.duty_cycle import DAY, DutyCycleModel, simulate_duty_cycle

start, rate = 1_000_000_000, 16.0
realization = simulate_duty_cycle(DutyCycleModel(), ["E1", "E2", "E3"], start=start, duration=2 * DAY, seed=1)
samples = realization.state_vector("E1", start=start, sampling_frequency=rate, n_samples=int(2 * DAY * rate))

bits = ["HOFT_OK", "OBS_INTENT", "OBS_READY"] + [None] * 13 + ["MAINTENANCE"]
state = StateVector(samples, t0=start, sample_rate=rate, bits=bits)
flags = state.to_dqflags()
observing = (flags["HOFT_OK"] & flags["OBS_INTENT"]).active

exact = realization.observing["E1"]
assert len(observing) == len(exact)
for (a, b), (c, d) in zip(observing, exact):  # within one sample: gwpy's segments are sample-quantized
    assert abs(float(a) - c) <= 1 / rate and abs(float(b) - d) <= 1 / rate
```

**PyCBC**'s low-latency search reads the state vector from the channel
`--state-channel` names, and the flags `--analyze-flags` names, as in

<!-- example: pycbc-live -->

```bash
pycbc_live --state-channel E1:STATE_VECTOR --analyze-flags E1:HOFT_OK,SCIENCE_INTENT
```

It needs both: with `--analyze-flags` alone it reads no state vector. The flags
of one interferometer go in one comma-separated value, because repeating the
interferometer keeps only the last. It turns the flags into a mask and analyses
a block of data only when every sample of the block has all of the mask's bits
set:

<!-- example: pycbc -->

```python
import numpy as np
from pycbc.frame import flag_names_to_bitmask

from gwmock_noise.duty_cycle import DAY, DutyCycleModel, simulate_duty_cycle

start, rate = 1_000_000_000, 16.0
realization = simulate_duty_cycle(DutyCycleModel(), ["E1", "E2", "E3"], start=start, duration=2 * DAY, seed=1)
samples = realization.state_vector("E1", start=start, sampling_frequency=rate, n_samples=int(2 * DAY * rate))

mask = flag_names_to_bitmask("HOFT_OK,SCIENCE_INTENT".split(","))
analysable = np.bitwise_and(samples.astype(np.int32), mask) == mask  # PyCBC reads the channel as int32
times = start + np.arange(samples.size) / rate
assert np.array_equal(analysable, realization.is_observing("E1", times))
```

## The model

The model follows Negri et al., _Impact of the Einstein Telescope's duty cycle
on the estimation of binary black holes parameters_,
[arXiv:2606.19201](https://arxiv.org/abs/2606.19201).

**Unplanned unlocks** form a continuous-time Markov chain (the paper's §II.1 and
Appendix B).

- An observing interferometer unlocks at rate `q_u = 1 / mean_uptime`.
- An unlocked one relocks at rate `q_d = 1 / mean_downtime`.
- Lock and gap durations are therefore exponential.
- While two or more interferometers observe, a correlated unlock takes every
  observing interferometer down at once, at rate `q_c`. Each interferometer's
  total unlock rate stays `q_u`.

`rate_matrix` is the paper's Eq. 4. `stationary_distribution` gives the long-run
fraction of time with each number of interferometers observing.

**Planned maintenance** is a schedule laid over the chain. During its window an
interferometer does not observe and its chain pauses. When the window closes it
observes again. There are two strategies:

- `coincident`: every interferometer is maintained in the same window.
- `rotating`: the interferometers are maintained one after another, so the
  others observe meanwhile. Their windows must fit in one period.

`strategy="none"` turns maintenance off.

The simulation is exact, one event at a time. The lock states at the start are
drawn from the stationary distribution of the chain without maintenance, so a
span that starts inside or just after a window begins with a transient lasting
about the mean downtime. Correlated unlocks couple the interferometers, so a
realization depends on the whole network and on its order. The same seed,
detectors and order give the same realization. Maintenance windows are fixed in
GPS time and do not depend on the span.

## Parameters and their defaults

Every parameter is free. The defaults are the paper's, except the maintenance
period, phase and strategy, which the paper does not give.

| Parameter                                     | Default           | Source                                                                                                                                                                                                                                                                                  |
| --------------------------------------------- | ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `DutyCycleModel.mean_uptime`                  | 5.4 h             | The paper, fitted to LIGO's O4a observing segments                                                                                                                                                                                                                                      |
| `DutyCycleModel.mean_downtime`                | 1.6 h             | The same fit                                                                                                                                                                                                                                                                            |
| `DutyCycleModel.correlated_downtime_fraction` | 0.02              | The paper: weather-induced simultaneous unlocks cause 2 % of the downtime. `q_c` is solved from it for the network at hand.                                                                                                                                                             |
| `MaintenanceSchedule.duration`                | 16.8 h            | 10 % of the period: the paper's planned maintenance, read as 10 % of the time (below)                                                                                                                                                                                                   |
| `MaintenanceSchedule.period`                  | 1 week            | LVK practice. Every LIGO and Virgo detector is maintained on Tuesday mornings, local time, in a window scheduled at 4 h ([LIGO-L1800079](https://dcc-llo.ligo.org/public/0151/L1800079/004/O3CoincidentRunningStrategy.pdf); [arXiv:2607.26872](https://arxiv.org/abs/2607.26872), §8). |
| `MaintenanceSchedule.offset`                  | Tuesday 07:00 UTC | Virgo's O3 window, Tuesday 07:00-11:00 UTC (LIGO-L1800079-v3, §2.1). The offset is in GPS seconds modulo the period; GPS 0 is a Sunday at 00:00 UTC, and the leap-second offset is ignored.                                                                                             |
| `MaintenanceSchedule.strategy`                | `rotating`        | One of the paper's two for the triangle; it uses `coincident` for the 2L                                                                                                                                                                                                                |

With these defaults each interferometer observes about 70 % of the time, the
paper's Appendix C figure.

**The maintenance fraction.** The paper states its planned maintenance two ways:
as 10 % of the time (the Fig. 1 caption and Appendix C), and as 10 % of the
downtime (§II.1.3). At 76 % observing, 10 % of the downtime would be 2.4 % of
the time. This implementation reads it as 10 % of the time, which reproduces the
paper's network fractions below.

The LVK's scheduled windows are shorter than the paper's 10 %. LIGO's 4 h of
weekly maintenance plus 6 h of commissioning is 6 % of a week (LIGO-L1800079).
Virgo spent 10 % of O4b and O4c in planned downtime, counting maintenance,
calibration and commissioning (arXiv:2607.26872, Table 3). To model the 4 h
maintenance window alone, set `duration=4 * HOUR`: with a weekly period that is
2.4 % of the time.

## How well it reproduces the paper

The tests check three things against the paper:

- the rate matrix, against its Eq. 4;
- with no correlated unlocks, the stationary distribution, against its binomial
  Eq. 1;
- a long simulation, against the stationary distribution.

They also compare the network fractions the paper reports (§III.1) at a
single-interferometer uptime of 76 %. That uptime is read as the fraction of
time observing, so the chain's own uptime is 0.76 / 0.9 with 10 % maintenance.

| Network, maintenance | Measure               | Paper | This model |
| -------------------- | --------------------- | ----- | ---------- |
| Triangle, rotating   | two or more observing | 85 %  | 86.9 %     |
| Triangle, rotating   | all three observing   | 41 %  | 42.4 %     |
| Triangle, coincident | two or more observing | 82 %  | 84.1 %     |
| Triangle, coincident | all three observing   | 53 %  | 54.6 %     |
| 2L, coincident       | both observing        | 64 %  | 64.4 %     |

The paper also reports 3 % of the time with no triangle interferometer observing
under rotating maintenance, against 1.1 % here. Its other figures are reproduced
to within 2.1 points, so the difference lies in that state alone. With this
implementation's default schedule, raising the correlated share does not resolve
it. At 40 % of the downtime correlated, no interferometer observes 3.4 % of the
time, but all three then observe 45.3 % (paper 41 %) and exactly one 10.9 %
(paper about 12 %). Under that schedule the paper's printed fractions are not
all reachable with its stated 2 %. The paper does not give the length, period or
phase of its windows, and a different schedule may account for the difference.
