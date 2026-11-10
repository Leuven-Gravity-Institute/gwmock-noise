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

The state vector is `uint32`, and its bits follow LIGO's calibrated state
vector, so a pipeline that selects analysable data from that channel selects
observing time here too:

| Bit | Value | Set while                            | LIGO meaning (PyCBC name)           |
| --- | ----- | ------------------------------------ | ----------------------------------- |
| 0   | 1     | the interferometer observes          | h(t) good (`HOFT_OK`)               |
| 1   | 2     | the interferometer observes          | observing intent (`SCIENCE_INTENT`) |
| 2   | 4     | the interferometer observes          | science quality (`SCIENCE_QUALITY`) |
| 16  | 65536 | the interferometer is in maintenance | none                                |

`OBSERVING_MASK` (`0b111`) and `MAINTENANCE_BIT` name them. PyCBC's low-latency
search, for example, analyses the data when given
`--analyze-flags <ifo>:HOFT_OK <ifo>:SCIENCE_INTENT`. Bit 16 is outside every
flag PyCBC names (bits 0 to 14). A sample outside the realization's span is 0.

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
