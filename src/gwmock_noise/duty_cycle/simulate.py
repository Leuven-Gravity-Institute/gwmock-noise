"""Draw one realization of a network's duty cycle."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from gwmock_noise.duty_cycle.model import CORRELATED_MINIMUM, DutyCycleModel, stationary_distribution

#: Bit of a state-vector sample set while the interferometer observes.
OBSERVING_BIT = 1
#: Bit of a state-vector sample set while the interferometer is in planned maintenance.
MAINTENANCE_BIT = 2


@dataclass(frozen=True, eq=False)
class DutyCycleRealization:
    """When each interferometer observed, over one span.

    Two realizations are equal when every field is, comparing the interval arrays by value.

    Attributes:
        detectors: The interferometers, in the order the model was drawn for.
        start: Start of the span, in GPS seconds.
        end: End of the span, in GPS seconds.
        observing: Per interferometer, an ``(n, 2)`` array of the ``[start, end)`` intervals it observed,
            sorted and disjoint.
        maintenance: Per interferometer, the ``[start, end)`` intervals of its planned maintenance.
        model: The model drawn from.
        seed: The seed drawn with.
    """

    detectors: tuple[str, ...]
    start: float
    end: float
    observing: dict[str, np.ndarray] = field(repr=False)
    maintenance: dict[str, np.ndarray] = field(repr=False)
    model: DutyCycleModel
    seed: int | None

    def __eq__(self, other: object) -> bool:
        """Compare every field, the interval arrays by value."""
        if not isinstance(other, DutyCycleRealization):
            return NotImplemented
        return (
            (self.detectors, self.start, self.end, self.model, self.seed)
            == (other.detectors, other.start, other.end, other.model, other.seed)
            and all(np.array_equal(self.observing[d], other.observing[d]) for d in self.detectors)
            and all(np.array_equal(self.maintenance[d], other.maintenance[d]) for d in self.detectors)
        )

    __hash__ = None  # type: ignore[assignment]  # holds arrays, like any mutable container

    def observing_fraction(self, detector: str) -> float:
        """Fraction of the span the interferometer observed."""
        segments = self.observing[detector]
        return float(np.sum(segments[:, 1] - segments[:, 0]) / (self.end - self.start))

    def is_observing(self, detector: str, times: np.ndarray) -> np.ndarray:
        """Whether the interferometer observed at each time.

        Args:
            detector: The interferometer.
            times: GPS times.

        Returns:
            A boolean array the shape of ``times``.
        """
        return _inside(self.observing[detector], np.asarray(times, dtype=float))

    def state_vector(self, detector: str, start: float, sampling_frequency: float, n_samples: int) -> np.ndarray:
        """The interferometer's state at each sample of a uniform grid.

        Bit 0 (:data:`OBSERVING_BIT`) is set while it observes; bit 1 (:data:`MAINTENANCE_BIT`) while it is
        in planned maintenance. A sample takes the state at its own time. A sample outside the
        realization's span is ``0``, the same value as an interferometer that is down and not in
        maintenance: the realization says nothing about it.

        Args:
            detector: The interferometer.
            start: GPS time of the first sample.
            sampling_frequency: Samples per second.
            n_samples: Number of samples.

        Returns:
            A ``uint8`` array of ``n_samples`` states.

        Raises:
            ValueError: If the sampling frequency is not positive and finite.
        """
        if not (math.isfinite(sampling_frequency) and sampling_frequency > 0.0):
            raise ValueError(f"sampling_frequency must be positive and finite, not {sampling_frequency!r}.")
        times = start + np.arange(n_samples) / sampling_frequency
        state = np.where(_inside(self.observing[detector], times), OBSERVING_BIT, 0)
        state |= np.where(_inside(self.maintenance[detector], times), MAINTENANCE_BIT, 0)
        return state.astype(np.uint8)

    def fraction_with_n_observing(self) -> np.ndarray:
        """Fraction of the span with ``0, ..., N`` interferometers observing at once."""
        edges = np.unique(
            np.concatenate([[self.start, self.end], *[segments.ravel() for segments in self.observing.values()]])
        )
        middles = 0.5 * (edges[:-1] + edges[1:])
        counts = sum(_inside(self.observing[detector], middles).astype(int) for detector in self.detectors)
        fractions = np.zeros(len(self.detectors) + 1)
        np.add.at(fractions, counts, np.diff(edges))
        return fractions / (self.end - self.start)


def _inside(intervals: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Whether each time lies in one of the sorted, disjoint ``[start, end)`` intervals."""
    if intervals.size == 0:
        return np.zeros(times.shape, dtype=bool)
    index = np.searchsorted(intervals[:, 0], times, side="right") - 1
    valid = index >= 0
    result = np.zeros(times.shape, dtype=bool)
    result[valid] = times[valid] < intervals[index[valid], 1]
    return result


def simulate_duty_cycle(
    model: DutyCycleModel,
    detectors: Sequence[str],
    start: float,
    duration: float,
    seed: int | None = None,
) -> DutyCycleRealization:
    """Draw when each interferometer observes over ``[start, start + duration)``.

    The chain is simulated exactly, one event at a time, and its lock states are drawn at ``start`` from
    the stationary distribution of the chain without maintenance. With maintenance that is the
    stationary state only away from a window; the transient it leaves lasts about the mean downtime. Correlated unlocks couple the interferometers, so the draw depends
    on the whole network and on its order: the same seed gives the same realization for the same
    detectors in the same order. Maintenance windows are fixed in GPS time, independent of the span.

    Args:
        model: The duty-cycle model.
        detectors: The interferometers, in order.
        start: Start of the span, in GPS seconds.
        duration: Length of the span, in seconds.
        seed: Seed of the random stream.

    Returns:
        The realization.

    Raises:
        ValueError: If there are no detectors, a name repeats, the span is not positive and finite, or
            a rotating schedule does not fit the network.
    """
    names = tuple(detectors)
    if not names:
        raise ValueError("a duty cycle needs at least one interferometer.")
    if len(set(names)) != len(names):
        raise ValueError(f"interferometer names must be distinct, not {names}.")
    if not (math.isfinite(start) and math.isfinite(duration) and duration > 0.0):
        raise ValueError(
            f"the span must be finite with a positive duration, not start={start!r}, duration={duration!r}."
        )
    n = len(names)
    model.maintenance.check_network(n)
    end = start + duration
    if not math.isfinite(end):
        raise ValueError(f"start={start!r} plus duration={duration!r} overflows to {end!r}.")
    if end <= start:
        raise ValueError(f"a duration of {duration!r} s does not move the end past start={start!r} in floating point.")
    rng = np.random.default_rng(seed)
    maintenance = {name: model.maintenance.windows(index, start, end) for index, name in enumerate(names)}
    chain = _Chain(model, n, rng)
    chain.start(start, [_inside(maintenance[name], np.array([start]))[0] for name in names])
    boundaries = sorted(
        (time, index, entering)
        for index, name in enumerate(names)
        for window in maintenance[name]
        for time, entering in ((window[0], True), (window[1], False))
    )
    next_boundary = 0
    now = start
    while now < end:
        boundary_time = boundaries[next_boundary][0] if next_boundary < len(boundaries) else end
        stop = min(boundary_time, end)
        # An event closer than one floating-point step would leave time where it is, and the
        # loop would never end; it happens one step later instead.
        event_time = chain.next_event(now)
        if event_time <= now:
            event_time = float(np.nextafter(now, math.inf))
        if event_time < stop:
            now = event_time
            chain.apply_event(now)
            continue
        now = stop
        if now >= end:
            break
        # Every boundary at this instant: a window opens, or closes and the interferometer relocks.
        while next_boundary < len(boundaries) and boundaries[next_boundary][0] <= now:
            _, index, entering = boundaries[next_boundary]
            chain.set_maintenance(now, index, entering)
            next_boundary += 1
    segments = dict(zip(names, chain.segments(end), strict=True))
    return DutyCycleRealization(names, float(start), float(end), segments, maintenance, model, seed)


class _Chain:
    """The interferometers' lock states, advanced one event at a time, and the observing transitions."""

    def __init__(self, model: DutyCycleModel, n: int, rng: np.random.Generator) -> None:
        self.q_u, self.q_d = model.unlock_rate, model.relock_rate
        self.q_c = model.correlated_rate(n)
        self.n = n
        self.rng = rng
        self.locked = np.zeros(n, dtype=bool)
        self.in_maintenance = np.zeros(n, dtype=bool)
        self.observing = np.zeros(n, dtype=bool)
        self.transitions: list[list[float]] = [[] for _ in range(n)]
        self._pending: tuple[float, float, float, float] | None = None

    def start(self, now: float, in_maintenance: list[bool]) -> None:
        """Draw the lock states from the stationary distribution and record who observes at ``now``."""
        pi = stationary_distribution(self.n, self.q_u, self.q_d, self.q_c)
        n_locked = self.rng.choice(self.n + 1, p=pi / pi.sum())
        self.locked[self.rng.choice(self.n, size=n_locked, replace=False)] = True
        self.in_maintenance[:] = in_maintenance
        self._record(now)

    def _rates(self) -> tuple[np.ndarray, np.ndarray, float, float]:
        active = ~self.in_maintenance
        up = self.locked & active
        down = ~self.locked & active
        coupled = int(up.sum()) >= CORRELATED_MINIMUM
        independent = self.q_u - self.q_c if coupled else self.q_u
        correlated = self.q_c if coupled else 0.0
        return up, down, independent, correlated

    def next_event(self, now: float) -> float:
        """Time of the chain's next event after ``now`` if nothing else intervenes."""
        up, down, independent, correlated = self._rates()
        total = int(up.sum()) * independent + int(down.sum()) * self.q_d + correlated
        if total <= 0.0:
            return math.inf
        return now + self.rng.exponential(1.0 / total)

    def apply_event(self, now: float) -> None:
        """Apply one event of the chain, chosen in proportion to its rate."""
        up, down, independent, correlated = self._rates()
        up_index, down_index = np.flatnonzero(up), np.flatnonzero(down)
        unlocks = up_index.size * independent
        u = self.rng.random() * (correlated + unlocks + down_index.size * self.q_d)
        if u < correlated:
            self.locked[up_index] = False
        elif u < correlated + unlocks:
            self.locked[up_index[min(int((u - correlated) // independent), up_index.size - 1)]] = False
        else:
            self.locked[down_index[min(int((u - correlated - unlocks) // self.q_d), down_index.size - 1)]] = True
        self._record(now)

    def set_maintenance(self, now: float, index: int, entering: bool) -> None:
        """Open or close one interferometer's window; it relocks when its window closes."""
        self.in_maintenance[index] = entering
        if not entering:
            self.locked[index] = True
        self._record(now)

    def _record(self, now: float) -> None:
        new = self.locked & ~self.in_maintenance
        for index in np.flatnonzero(new != self.observing):
            self.transitions[index].append(now)
        self.observing[:] = new

    def segments(self, end: float) -> list[np.ndarray]:
        """Each interferometer's observing intervals, closed at ``end``."""
        result = []
        for index in range(self.n):
            edges = self.transitions[index] + ([end] if self.observing[index] else [])
            pairs = np.asarray(edges, dtype=float).reshape(-1, 2)
            result.append(pairs[pairs[:, 1] > pairs[:, 0]])
        return result
