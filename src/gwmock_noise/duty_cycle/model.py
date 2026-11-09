"""The duty-cycle model: when each interferometer of a network observes.

Each interferometer is either locked and observing, or not. The model follows Negri et al.,
"Impact of the Einstein Telescope's duty cycle on the estimation of binary black holes parameters",
arXiv:2606.19201, and has two parts.

**Unplanned unlocks** form a continuous-time Markov chain (the paper's §II.1 and Appendix B). An
observing interferometer unlocks at rate ``q_u = 1 / mean_uptime`` and an unlocked one relocks at
rate ``q_d = 1 / mean_downtime``, independently of the others, so lock and gap durations are
exponential. While two or more interferometers observe, a correlated unlock -- weather, in the paper --
takes every observing interferometer down at once, at rate ``q_c``. Each interferometer's total unlock
rate stays ``q_u``: the correlated rate is taken out of the independent one, which is
``q_u - q_c`` while another interferometer observes.

**Planned maintenance** is a weekly schedule laid over the chain. During its window an interferometer
does not observe and the chain pauses for it; it observes again when the window closes. All
interferometers are maintained in the same window (``coincident``), or one after another (``rotating``),
so that the others observe meanwhile.

The defaults are the paper's: mean uptime 5.4 h and mean downtime 1.6 h, fitted to LIGO's O4a
observing segments; correlated unlocks causing 2 % of the downtime; and 10 % of the time in planned
maintenance. The maintenance period is a week, the LVK's practice: every LIGO and Virgo detector is
maintained on Tuesday mornings in local time, in a window scheduled at 4 h (LIGO-L1800079;
arXiv:2607.26872, §8). The default window opens on Tuesdays at 07:00 UTC, the hour the European
detector's weekly maintenance has opened.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

#: Correlated unlocks need two or more interferometers observing.
CORRELATED_MINIMUM = 2

#: Seconds in an hour, a day and a week.
HOUR = 3600.0
DAY = 24.0 * HOUR
WEEK = 7.0 * DAY

#: The paper's mean uptime and downtime of one interferometer, fitted to LIGO's O4a observing segments.
DEFAULT_MEAN_UPTIME = 5.4 * HOUR
DEFAULT_MEAN_DOWNTIME = 1.6 * HOUR

#: The paper's share of downtime caused by correlated unlocks.
DEFAULT_CORRELATED_DOWNTIME_FRACTION = 0.02

#: The paper's fraction of time in planned maintenance.
DEFAULT_MAINTENANCE_FRACTION = 0.10

#: The default window opens on Tuesdays at 07:00 UTC. GPS time 0 is Sunday 1980-01-06 00:00 UTC; GPS
#: and UTC differ by leap seconds (18 s since 2017), which this offset ignores.
DEFAULT_MAINTENANCE_OFFSET = 2.0 * DAY + 7.0 * HOUR

MaintenanceStrategy = Literal["coincident", "rotating", "none"]


@dataclass(frozen=True)
class MaintenanceSchedule:
    """Planned maintenance windows, repeating with a fixed period.

    Window ``k`` of interferometer ``j`` (in the order the network lists them) opens at GPS time
    ``k * period + offset``, plus ``j * duration`` for the ``rotating`` strategy, and lasts ``duration``.

    Attributes:
        strategy: ``coincident`` maintains every interferometer in the same window; ``rotating`` maintains
            them one after another within each period, so the others observe meanwhile; ``none`` has no
            planned maintenance.
        period: Time between the starts of successive windows of one interferometer, in seconds.
        duration: Length of one interferometer's window, in seconds.
        offset: Opening time of the windows, as GPS seconds modulo ``period``.
    """

    strategy: MaintenanceStrategy = "rotating"
    period: float = WEEK
    duration: float = DEFAULT_MAINTENANCE_FRACTION * WEEK
    offset: float = DEFAULT_MAINTENANCE_OFFSET

    def __post_init__(self) -> None:
        """Validate the schedule.

        Raises:
            ValueError: If the strategy is unknown, the period or duration is not positive and
                finite, the duration exceeds the period, or the offset is not finite.
        """
        if self.strategy not in ("coincident", "rotating", "none"):
            raise ValueError(f"maintenance strategy must be 'coincident', 'rotating' or 'none', not {self.strategy!r}.")
        if not (math.isfinite(self.period) and self.period > 0.0):
            raise ValueError(f"maintenance period must be positive and finite, not {self.period!r}.")
        if not (math.isfinite(self.duration) and 0.0 < self.duration <= self.period):
            raise ValueError(
                f"maintenance duration must be positive and at most the period ({self.period} s), not {self.duration!r}."
            )
        if not math.isfinite(self.offset):
            raise ValueError(f"maintenance offset must be finite, not {self.offset!r}.")

    @property
    def fraction(self) -> float:
        """Fraction of time each interferometer spends in maintenance."""
        return 0.0 if self.strategy == "none" else self.duration / self.period

    def check_network(self, n_interferometers: int) -> None:
        """Refuse a rotating schedule whose windows do not fit in one period.

        Args:
            n_interferometers: Interferometers in the network.

        Raises:
            ValueError: If ``n_interferometers`` rotating windows are longer than the period.
        """
        if self.strategy == "rotating" and n_interferometers * self.duration > self.period:
            raise ValueError(
                f"a rotating schedule maintains {n_interferometers} interferometers one after another, which needs "
                f"{n_interferometers} x {self.duration} s within a period of {self.period} s."
            )

    def windows(self, index: int, start: float, end: float) -> np.ndarray:
        """Return the windows of one interferometer that overlap ``[start, end)``.

        Args:
            index: The interferometer's position in the network.
            start: Start of the span, in GPS seconds.
            end: End of the span, in GPS seconds.

        Returns:
            An ``(n, 2)`` array of window starts and ends, clipped to the span.
        """
        if self.strategy == "none" or end <= start:
            return np.empty((0, 2))
        shift = self.offset + (index * self.duration if self.strategy == "rotating" else 0.0)
        first = math.floor((start - shift - self.duration) / self.period)
        last = math.ceil((end - shift) / self.period)
        opens = shift + self.period * np.arange(first, last + 1, dtype=float)
        windows = np.column_stack((opens, opens + self.duration))
        windows = windows[(windows[:, 1] > start) & (windows[:, 0] < end)]
        return np.clip(windows, start, end)


@dataclass(frozen=True)
class DutyCycleModel:
    """When each interferometer of a network observes; see the module docstring for the model.

    Attributes:
        mean_uptime: Mean length of an unplanned lock, ``1 / q_u``, in seconds. The paper's 5.4 h, fitted
            to LIGO's O4a observing segments.
        mean_downtime: Mean length of an unplanned gap, ``1 / q_d``, in seconds. The paper's 1.6 h.
        correlated_downtime_fraction: Share of unplanned downtime caused by correlated unlocks, from which
            the correlated rate ``q_c`` is solved for the network at hand. The paper's 2 %. Zero makes
            the interferometers independent.
        maintenance: The planned-maintenance schedule. The default rotates weekly windows covering the
            paper's 10 % of the time.
    """

    mean_uptime: float = DEFAULT_MEAN_UPTIME
    mean_downtime: float = DEFAULT_MEAN_DOWNTIME
    correlated_downtime_fraction: float = DEFAULT_CORRELATED_DOWNTIME_FRACTION
    maintenance: MaintenanceSchedule = MaintenanceSchedule()

    def __post_init__(self) -> None:
        """Validate the model.

        Raises:
            ValueError: If a mean time is not positive and finite, or the correlated fraction is not
                in ``[0, 1)``.
        """
        for name in ("mean_uptime", "mean_downtime"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be positive and finite, not {value!r}.")
        if not (math.isfinite(self.correlated_downtime_fraction) and 0.0 <= self.correlated_downtime_fraction < 1.0):
            raise ValueError(
                f"correlated_downtime_fraction must lie in [0, 1), not {self.correlated_downtime_fraction!r}."
            )

    @property
    def unlock_rate(self) -> float:
        """``q_u``, per second."""
        return 1.0 / self.mean_uptime

    @property
    def relock_rate(self) -> float:
        """``q_d``, per second."""
        return 1.0 / self.mean_downtime

    def correlated_rate(self, n_interferometers: int) -> float:
        """``q_c`` for a network of ``n_interferometers``, per second.

        Solved so that, in the stationary chain without maintenance, correlated unlocks cause
        ``correlated_downtime_fraction`` of all unplanned downtime. Every unlock is followed by a gap of
        the same mean length, so that share is the share of interferometer unlocks that are correlated.

        Args:
            n_interferometers: Interferometers in the network.

        Returns:
            The correlated rate, ``0`` for a single interferometer or a zero fraction.

        Raises:
            ValueError: If the fraction cannot be reached with ``q_c < q_u``.
        """
        target = self.correlated_downtime_fraction
        if target == 0.0 or n_interferometers < CORRELATED_MINIMUM:
            return 0.0
        q_u = self.unlock_rate

        def excess(q_c: float) -> float:
            return correlated_unlock_share(self, n_interferometers, q_c) - target

        upper = q_u * (1.0 - 1e-12)
        if excess(upper) < 0.0:
            raise ValueError(
                f"correlated unlocks cannot cause {target} of the downtime of {n_interferometers} interferometers "
                f"with these rates; the largest reachable share is {correlated_unlock_share(self, n_interferometers, upper)}."
            )
        lower = 0.0
        for _ in range(200):
            middle = 0.5 * (lower + upper)
            if excess(middle) < 0.0:
                lower = middle
            else:
                upper = middle
        return 0.5 * (lower + upper)


def rate_matrix(n_interferometers: int, unlock_rate: float, relock_rate: float, correlated_rate: float) -> np.ndarray:
    """The paper's rate matrix ``Q`` on the number of observing interferometers (its Eq. 4).

    Args:
        n_interferometers: ``N``.
        unlock_rate: ``q_u``.
        relock_rate: ``q_d``.
        correlated_rate: ``q_c``.

    Returns:
        The ``(N + 1, N + 1)`` generator: ``Q[i, j]`` is the rate from ``i`` observing to ``j`` observing,
        and each row sums to zero.
    """
    n = n_interferometers
    q = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        if i < n:
            q[i, i + 1] = (n - i) * relock_rate
        if i == 1:
            q[1, 0] = unlock_rate
        elif i >= CORRELATED_MINIMUM:
            q[i, i - 1] = i * (unlock_rate - correlated_rate)
            q[i, 0] += correlated_rate
        q[i, i] = -q[i].sum()
    return q


def stationary_distribution(
    n_interferometers: int, unlock_rate: float, relock_rate: float, correlated_rate: float
) -> np.ndarray:
    """Long-run fraction of time with ``0, ..., N`` interferometers observing, without maintenance.

    Args:
        n_interferometers: ``N``.
        unlock_rate: ``q_u``.
        relock_rate: ``q_d``.
        correlated_rate: ``q_c``.

    Returns:
        ``pi`` with ``pi @ Q = 0`` and ``sum(pi) = 1``.
    """
    q = rate_matrix(n_interferometers, unlock_rate, relock_rate, correlated_rate)
    # pi Q = 0 has one redundant equation; replace it with the normalisation.
    system = q.T.copy()
    system[-1] = 1.0
    rhs = np.zeros(n_interferometers + 1)
    rhs[-1] = 1.0
    return np.linalg.solve(system, rhs)


def correlated_unlock_share(model: DutyCycleModel, n_interferometers: int, correlated_rate: float) -> float:
    """Share of interferometer unlocks that are correlated, in the stationary chain without maintenance.

    In state ``i`` the interferometers unlock at a total rate ``i * q_u``, of which ``i * q_c`` comes
    from correlated unlocks when ``i >= 2``.

    Args:
        model: Supplies ``q_u`` and ``q_d``.
        n_interferometers: ``N``.
        correlated_rate: ``q_c``.

    Returns:
        The share.
    """
    pi = stationary_distribution(n_interferometers, model.unlock_rate, model.relock_rate, correlated_rate)
    i = np.arange(n_interferometers + 1)
    total = float(np.sum(pi * i * model.unlock_rate))
    correlated = float(np.sum(pi[CORRELATED_MINIMUM:] * i[CORRELATED_MINIMUM:] * correlated_rate))
    return correlated / total
