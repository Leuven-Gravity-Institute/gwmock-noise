"""Tests for the duty-cycle model and its simulation.

The anchors are the paper the model follows, Negri et al., arXiv:2606.19201: its rate matrix
(Appendix B, Eq. 4), its stationary distribution for independent interferometers (Eq. 1), and the
network fractions it reports for the ET designs at a single-interferometer uptime of 76 % (§III.1).
"""

from __future__ import annotations

import math
import os
import subprocess
import sys

import numpy as np
import pytest

from gwmock_noise.duty_cycle import (
    DEFAULT_MAINTENANCE_OFFSET,
    MAINTENANCE_BIT,
    OBSERVING_MASK,
    DutyCycleModel,
    MaintenanceSchedule,
    correlated_unlock_share,
    rate_matrix,
    simulate_duty_cycle,
    stationary_distribution,
)
from gwmock_noise.duty_cycle.model import DAY, HOUR, WEEK

START = 1_000_000_000.0
TRIANGLE = ("E1", "E2", "E3")
NO_MAINTENANCE = MaintenanceSchedule(strategy="none")


# ---------------------------------------------------------------------------------------
# The chain
# ---------------------------------------------------------------------------------------


def test_the_rate_matrix_is_the_papers() -> None:
    """Eq. 4 for N = 3, entry by entry, with rows summing to zero."""
    q_u, q_d, q_c = 0.5, 2.0, 0.1
    q = rate_matrix(3, q_u, q_d, q_c)
    expected = np.array(
        [
            [-3 * q_d, 3 * q_d, 0.0, 0.0],
            [q_u, -(2 * q_d + q_u), 2 * q_d, 0.0],
            [q_c, 2 * (q_u - q_c), -(q_d + 2 * q_u - q_c), q_d],
            [q_c, 0.0, 3 * (q_u - q_c), -(3 * q_u - 2 * q_c)],
        ]
    )
    np.testing.assert_allclose(q, expected, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(q.sum(axis=1), 0.0, rtol=0.0, atol=1e-15)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_independent_interferometers_have_the_papers_binomial_distribution(n: int) -> None:
    """Eq. 1: pi_i = C(N, i) alpha^i / (1 + alpha)^N with alpha = q_d / q_u."""
    model = DutyCycleModel(correlated_downtime_fraction=0.0)
    alpha = model.relock_rate / model.unlock_rate
    expected = [math.comb(n, i) * alpha**i / (1.0 + alpha) ** n for i in range(n + 1)]
    pi = stationary_distribution(n, model.unlock_rate, model.relock_rate, 0.0)
    np.testing.assert_allclose(pi, expected, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("n", [2, 3])
def test_the_correlated_rate_causes_the_stated_share_of_downtime(n: int) -> None:
    """Solved q_c makes correlated unlocks 2 % of all unlocks, so of the downtime, and stays below q_u."""
    model = DutyCycleModel()
    q_c = model.correlated_rate(n)
    assert 0.0 < q_c < model.unlock_rate
    assert correlated_unlock_share(model, n, q_c) == pytest.approx(0.02, rel=1e-9)


def test_one_interferometer_or_no_correlation_has_no_correlated_rate() -> None:
    """Correlated unlocks need two interferometers and a nonzero share."""
    assert DutyCycleModel().correlated_rate(1) == 0.0
    assert DutyCycleModel(correlated_downtime_fraction=0.0).correlated_rate(3) == 0.0


def test_the_defaults_are_the_papers() -> None:
    """5.4 h up, 1.6 h down, 2 % correlated, 10 % maintenance, weekly from Tuesday 07:00 UTC."""
    model = DutyCycleModel()
    assert model.mean_uptime == 5.4 * HOUR
    assert model.mean_downtime == 1.6 * HOUR
    assert model.correlated_downtime_fraction == 0.02
    assert model.maintenance.period == WEEK
    assert model.maintenance.fraction == pytest.approx(0.10, rel=1e-12)
    # GPS 0 is Sunday 00:00 UTC, so Tuesday 07:00 UTC is 2 days and 7 hours later.
    assert DEFAULT_MAINTENANCE_OFFSET == 2 * DAY + 7 * HOUR


# ---------------------------------------------------------------------------------------
# The simulation against the chain and the paper
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("fraction", [0.02, 0.3])
@pytest.mark.parametrize("n", [2, 3])
def test_a_long_realization_has_the_stationary_distribution(n: int, fraction: float) -> None:
    """Without maintenance, the time with i interferometers observing is pi_i (2000 days, about 7000 cycles each).

    The 30 % case makes the correlated term large enough that a simulator splitting the rates
    differently from Eq. 4 would miss it by more than the tolerance.
    """
    model = DutyCycleModel(correlated_downtime_fraction=fraction, maintenance=NO_MAINTENANCE)
    realization = simulate_duty_cycle(model, TRIANGLE[:n], START, 2000 * DAY, seed=1)
    pi = stationary_distribution(n, model.unlock_rate, model.relock_rate, model.correlated_rate(n))
    np.testing.assert_allclose(realization.fraction_with_n_observing(), pi, rtol=0.0, atol=0.01)


@pytest.mark.parametrize(
    ("n", "strategy", "at_least_two", "all_observing"),
    [
        (3, "rotating", 0.85, 0.41),
        (3, "coincident", 0.82, 0.53),
        (2, "coincident", 0.64, 0.64),
    ],
)
def test_the_papers_network_fractions_at_76_percent_uptime(
    n: int, strategy: str, at_least_two: float, all_observing: float
) -> None:
    """§III.1 at a single-interferometer uptime of 76 %, read as the fraction of time observing.

    The paper's planned maintenance is 10 % of the time, so the chain's own uptime is 0.76 / 0.9. It
    reports two or more observing for 85 % (rotating) and 82 % (coincident) of the time, and all three
    for 41 % and 53 %; for ET-2L, both for 64 %. The model reproduces each to within 2.5 points, which
    the paper's rounding and its unstated maintenance schedule allow. The seed makes the simulated
    fractions exact (86.9, 42.4, 84.1, 54.6 and 64.4 %), so the margin is the distance to the paper,
    not a statistical allowance.
    """
    uptime = 0.76 / 0.9
    model = DutyCycleModel(
        mean_downtime=5.4 * HOUR * (1.0 - uptime) / uptime, maintenance=MaintenanceSchedule(strategy=strategy)
    )
    realization = simulate_duty_cycle(model, TRIANGLE[:n], START, 3000 * DAY, seed=7)
    fractions = realization.fraction_with_n_observing()
    assert np.mean([realization.observing_fraction(name) for name in realization.detectors]) == pytest.approx(
        0.76, abs=0.01
    )
    assert fractions[2:].sum() == pytest.approx(at_least_two, abs=0.025)
    assert fractions[-1] == pytest.approx(all_observing, abs=0.025)


def test_the_default_interferometer_observes_70_percent_of_the_time() -> None:
    """The paper's Appendix C: 70 % observing with 10 % planned maintenance."""
    realization = simulate_duty_cycle(DutyCycleModel(), TRIANGLE, START, 1000 * DAY, seed=3)
    for name in TRIANGLE:
        assert realization.observing_fraction(name) == pytest.approx(0.9 * 5.4 / 7.0, abs=0.015)


# ---------------------------------------------------------------------------------------
# Maintenance
# ---------------------------------------------------------------------------------------


def test_coincident_windows_coincide_and_rotating_ones_follow_each_other() -> None:
    """Coincident windows are shared; rotating ones open as the previous interferometer's close."""
    span = (START, START + 4 * WEEK)
    coincident = MaintenanceSchedule(strategy="coincident")
    rotating = MaintenanceSchedule(strategy="rotating")
    for index in range(1, 3):
        np.testing.assert_array_equal(coincident.windows(index, *span), coincident.windows(0, *span))
    first, second = rotating.windows(0, *span), rotating.windows(1, *span)
    closes = first[(first[:, 0] > span[0]) & (first[:, 1] + rotating.duration < span[1]), 1]
    assert closes.size
    for close in closes:
        assert np.isclose(second[:, 0], close, rtol=0.0, atol=1e-6).sum() == 1
    assert not np.any((second[:, 0, None] < first[None, :, 1]) & (first[None, :, 0] < second[:, 1, None]))


def test_a_window_opens_on_tuesday_at_seven_utc_and_lasts_a_tenth_of_a_week() -> None:
    """The default schedule, at its documented time and length."""
    windows = MaintenanceSchedule(strategy="coincident").windows(0, START, START + 3 * WEEK)
    interior = windows[(windows[:, 0] > START) & (windows[:, 1] < START + 3 * WEEK)]
    assert interior.size
    np.testing.assert_allclose(np.mod(interior[:, 0], WEEK), 2 * DAY + 7 * HOUR, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(np.diff(interior, axis=1).ravel(), 16.8 * HOUR, rtol=1e-12, atol=0.0)


def test_windows_are_clipped_to_the_span() -> None:
    """A window straddling either end of the span is cut there; one outside it is dropped."""
    schedule = MaintenanceSchedule(strategy="coincident", period=100.0, duration=40.0, offset=0.0)
    np.testing.assert_array_equal(schedule.windows(0, 20.0, 230.0), [[20.0, 40.0], [100.0, 140.0], [200.0, 230.0]])
    assert schedule.windows(0, 50.0, 90.0).shape == (0, 2)


@pytest.mark.parametrize("strategy", ["coincident", "rotating"])
def test_no_interferometer_observes_during_its_maintenance_and_each_relocks_after(strategy: str) -> None:
    """Maintenance overrides the chain, and the interferometer observes again when its window closes."""
    model = DutyCycleModel(maintenance=MaintenanceSchedule(strategy=strategy))
    realization = simulate_duty_cycle(model, TRIANGLE, START, 6 * WEEK, seed=5)
    for name in TRIANGLE:
        windows = realization.maintenance[name]
        inside = np.concatenate([np.linspace(a, b, 50, endpoint=False) for a, b in windows])
        assert not realization.is_observing(name, inside).any()
        closing = windows[:, 1][windows[:, 1] < realization.end]
        assert realization.is_observing(name, closing).all()


def test_rotating_maintenance_keeps_the_others_observable() -> None:
    """While one interferometer is maintained, the others are not."""
    model = DutyCycleModel(maintenance=MaintenanceSchedule(strategy="rotating"))
    realization = simulate_duty_cycle(model, TRIANGLE, START, 4 * WEEK, seed=6)
    times = START + np.arange(0.0, 4 * WEEK, 60.0)
    states = np.array([realization.state_vector(name, START, 1 / 60.0, times.size) for name in TRIANGLE])
    assert ((states & MAINTENANCE_BIT) > 0).sum(axis=0).max() == 1


def test_an_interferometer_in_maintenance_does_not_count_as_observing() -> None:
    """With half-week rotating windows, one interferometer is outside maintenance at any time.

    So correlated unlocks never apply, even at 30 % of the downtime, and each interferometer is the
    single-interferometer chain, locked when its window closes: outside its window it observes for
    ``pi + (1 - pi) (1 - exp(-L T)) / (L T)`` of the time, with ``L = q_u + q_d``, ``T`` half a week and
    ``pi = q_d / L``. Counting a maintained interferometer as observing would let correlated unlocks take
    the other down.
    """
    schedule = MaintenanceSchedule(strategy="rotating", duration=0.5 * WEEK)
    model = DutyCycleModel(correlated_downtime_fraction=0.3, maintenance=schedule)
    realization = simulate_duty_cycle(model, TRIANGLE[:2], START, 2000 * DAY, seed=13)
    rate = model.unlock_rate + model.relock_rate
    pi = model.relock_rate / rate
    span = rate * 0.5 * WEEK
    expected = 0.5 * (pi + (1.0 - pi) * (1.0 - math.exp(-span)) / span)
    for name in TRIANGLE[:2]:
        assert realization.observing_fraction(name) == pytest.approx(expected, abs=0.01)


def test_a_rotating_schedule_that_does_not_fit_is_refused() -> None:
    """Three windows of 0.4 week cannot follow each other within a week."""
    model = DutyCycleModel(maintenance=MaintenanceSchedule(strategy="rotating", duration=0.4 * WEEK))
    with pytest.raises(ValueError, match="rotating schedule"):
        simulate_duty_cycle(model, TRIANGLE, START, WEEK, seed=0)


# ---------------------------------------------------------------------------------------
# The realization
# ---------------------------------------------------------------------------------------


def test_a_seed_reproduces_the_realization_and_another_changes_it() -> None:
    """The draw is a function of the seed."""
    model = DutyCycleModel()
    first = simulate_duty_cycle(model, TRIANGLE, START, 2 * WEEK, seed=11)
    again = simulate_duty_cycle(model, TRIANGLE, START, 2 * WEEK, seed=11)
    other = simulate_duty_cycle(model, TRIANGLE, START, 2 * WEEK, seed=12)
    for name in TRIANGLE:
        np.testing.assert_array_equal(first.observing[name], again.observing[name])
    assert any(not np.array_equal(first.observing[n], other.observing[n]) for n in TRIANGLE)


def test_segments_are_sorted_disjoint_and_inside_the_span() -> None:
    """Observing segments are well formed."""
    realization = simulate_duty_cycle(DutyCycleModel(), TRIANGLE, START, 3 * WEEK, seed=4)
    for segments in realization.observing.values():
        assert segments.size
        assert np.all(segments[:, 1] > segments[:, 0])
        assert np.all(segments[1:, 0] > segments[:-1, 1])
        assert segments[0, 0] >= realization.start
        assert segments[-1, 1] <= realization.end


def test_the_state_vector_bits_are_the_ones_a_ligo_pipeline_reads() -> None:
    """The layout is a contract with other software, so its values are pinned, not only its names.

    LIGO's calibrated state vector sets bits 0 (``HOFT_OK``), 1 (``SCIENCE_INTENT``) and 2
    (``SCIENCE_QUALITY``) while the data can be analysed; PyCBC names its flags from bit 0 to bit 14.
    """
    assert OBSERVING_MASK == 0b111
    assert MAINTENANCE_BIT == 1 << 16


def test_the_state_vector_carries_its_bits_at_each_samples_own_time() -> None:
    """Every sample is observing (bits 0 to 2), maintenance (bit 16) or neither, at its own time."""
    realization = simulate_duty_cycle(DutyCycleModel(), TRIANGLE, START, 2 * WEEK, seed=8)
    sampling_frequency, n_samples = 1.0, int(2 * WEEK)
    times = START + np.arange(n_samples) / sampling_frequency
    for name in TRIANGLE:
        state = realization.state_vector(name, START, sampling_frequency, n_samples)
        assert state.dtype == np.uint32
        assert set(np.unique(state).tolist()) == {0, OBSERVING_MASK, MAINTENANCE_BIT}
        observing = realization.is_observing(name, times)
        np.testing.assert_array_equal(state == OBSERVING_MASK, observing)
        # What a pipeline that requires HOFT_OK and SCIENCE_INTENT (mask 3) would select.
        np.testing.assert_array_equal((state & 0b11) == 0b11, observing)
        np.testing.assert_array_equal(state == MAINTENANCE_BIT, _inside_any(realization.maintenance[name], times))


def _inside_any(intervals: np.ndarray, times: np.ndarray) -> np.ndarray:
    return np.array([bool(((intervals[:, 0] <= t) & (t < intervals[:, 1])).any()) for t in times])


def test_the_fraction_with_n_observing_adds_up() -> None:
    """The fractions sum to one, and their mean count is the sum of the single fractions."""
    realization = simulate_duty_cycle(DutyCycleModel(), TRIANGLE, START, 3 * WEEK, seed=9)
    fractions = realization.fraction_with_n_observing()
    assert fractions.sum() == pytest.approx(1.0, abs=1e-12)
    mean_count = np.dot(np.arange(4), fractions)
    assert mean_count == pytest.approx(sum(realization.observing_fraction(n) for n in TRIANGLE), abs=1e-9)


# ---------------------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"mean_uptime": 0.0}, "mean_uptime"),
        ({"mean_downtime": math.inf}, "mean_downtime"),
        ({"correlated_downtime_fraction": 1.0}, "correlated_downtime_fraction"),
        ({"correlated_downtime_fraction": -0.1}, "correlated_downtime_fraction"),
    ],
)
def test_an_invalid_model_is_refused(kwargs: dict, message: str) -> None:
    """Each invalid model parameter is named in the error."""
    with pytest.raises(ValueError, match=message):
        DutyCycleModel(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"strategy": "staggered"}, "strategy"),
        ({"period": 0.0}, "period"),
        ({"duration": 2 * WEEK}, "duration"),
        ({"offset": math.nan}, "offset"),
    ],
)
def test_an_invalid_schedule_is_refused(kwargs: dict, message: str) -> None:
    """Each invalid schedule parameter is named in the error."""
    with pytest.raises(ValueError, match=message):
        MaintenanceSchedule(**kwargs)


@pytest.mark.parametrize(
    ("detectors", "duration", "message"),
    [((), WEEK, "at least one"), (("E1", "E1"), WEEK, "distinct"), (TRIANGLE, 0.0, "positive duration")],
)
def test_an_invalid_draw_is_refused(detectors: tuple, duration: float, message: str) -> None:
    """An empty or repeated network, or an empty span, is refused."""
    with pytest.raises(ValueError, match=message):
        simulate_duty_cycle(DutyCycleModel(), detectors, START, duration, seed=0)


def test_realizations_compare_by_value() -> None:
    """The same seed gives an equal realization; another seed an unequal one; neither is hashable."""
    model = DutyCycleModel()
    first = simulate_duty_cycle(model, TRIANGLE, START, WEEK, seed=21)
    assert first == simulate_duty_cycle(model, TRIANGLE, START, WEEK, seed=21)
    assert first != simulate_duty_cycle(model, TRIANGLE, START, WEEK, seed=22)
    assert "observing" not in repr(first)
    with pytest.raises(TypeError):
        hash(first)


@pytest.mark.parametrize("sampling_frequency", [0.0, -1.0, math.inf, math.nan])
def test_the_state_vector_refuses_a_bad_sampling_frequency(sampling_frequency: float) -> None:
    """A sampling frequency that is not positive and finite is refused."""
    realization = simulate_duty_cycle(DutyCycleModel(), TRIANGLE, START, DAY, seed=0)
    with pytest.raises(ValueError, match="sampling_frequency"):
        realization.state_vector("E1", START, sampling_frequency, 10)


def test_an_unreachable_correlated_share_is_refused() -> None:
    """Three interferometers cannot have 95 % of their downtime correlated with q_c < q_u."""
    with pytest.raises(ValueError, match=r"cannot cause 0\.95"):
        DutyCycleModel(correlated_downtime_fraction=0.95).correlated_rate(3)


def test_a_span_inside_a_coincident_window_observes_nothing() -> None:
    """Every interferometer is maintained throughout, so none observes."""
    schedule = MaintenanceSchedule(strategy="coincident", period=WEEK, duration=DAY, offset=0.0)
    realization = simulate_duty_cycle(DutyCycleModel(maintenance=schedule), TRIANGLE, 0.0, 0.5 * DAY, seed=0)
    for name in TRIANGLE:
        assert realization.observing[name].shape == (0, 2)
        assert realization.observing_fraction(name) == 0.0
    np.testing.assert_array_equal(realization.fraction_with_n_observing(), [1.0, 0.0, 0.0, 0.0])


def test_without_maintenance_the_maintenance_bit_is_never_set() -> None:
    """``strategy="none"`` has no windows, whatever its unused duration."""
    model = DutyCycleModel(maintenance=MaintenanceSchedule(strategy="none", duration=2 * WEEK))
    realization = simulate_duty_cycle(model, TRIANGLE, START, WEEK, seed=0)
    state = realization.state_vector("E1", START, 1.0 / 60.0, int(WEEK / 60.0))
    assert not np.any(state & MAINTENANCE_BIT)
    assert realization.maintenance["E1"].shape == (0, 2)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ((2, 0.01, 0.1, 0.03), "must not exceed unlock_rate"),
        ((2, -0.01, 0.1, 0.0), "unlock_rate must be finite"),
        ((2, 0.01, math.nan, 0.0), "relock_rate must be finite"),
        ((0, 0.01, 0.1, 0.0), "at least one interferometer"),
    ],
)
def test_the_rate_matrix_refuses_rates_that_would_make_it_invalid(arguments: tuple, message: str) -> None:
    """A correlated rate above the unlock rate would give a negative transition rate."""
    with pytest.raises(ValueError, match=message):
        rate_matrix(*arguments)
    with pytest.raises(ValueError, match=message):
        stationary_distribution(*arguments)


def test_a_duration_lost_to_rounding_is_refused() -> None:
    """At GPS 1e9 a duration of 1e-8 s leaves the end equal to the start."""
    with pytest.raises(ValueError, match="does not move the end"):
        simulate_duty_cycle(DutyCycleModel(), TRIANGLE, START, 1e-8, seed=0)


def test_a_span_whose_end_overflows_is_refused() -> None:
    """Finite inputs whose sum is infinite are refused before the event loop."""
    with pytest.raises(ValueError, match="overflows"):
        simulate_duty_cycle(DutyCycleModel(), TRIANGLE, 1e308, 1e308, seed=0)


def test_a_span_coarser_than_the_event_times_still_terminates() -> None:
    """At GPS 1e25 one floating-point step is about 2e9 s, far longer than any lock or gap.

    Every drawn wait then rounds to no time at all; each event moves time on by one step instead, so
    the draw ends after about as many events as there are steps in the span. It runs in a subprocess
    with a timeout, so a loop that stalls fails the test instead of hanging the suite, and the child
    checks that the realization it returns is a valid one.
    """
    script = (
        "import numpy as np\n"
        "from gwmock_noise.duty_cycle import DutyCycleModel, MaintenanceSchedule, simulate_duty_cycle\n"
        "start = 1e25\n"
        "model = DutyCycleModel(maintenance=MaintenanceSchedule(strategy='none'))\n"
        "r = simulate_duty_cycle(model, ['E1', 'E2', 'E3'], start, 500 * float(np.spacing(start)), seed=0)\n"
        "f = r.fraction_with_n_observing()\n"
        "assert np.isfinite(f).all() and abs(f.sum() - 1.0) < 1e-12, f\n"
        "assert all(r.observing[name].size for name in r.detectors)\n"
        "print('ok')\n"
    )
    # The child imports the package this process imports, a mutated copy included.
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(path for path in sys.path if path)}
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script], capture_output=True, text=True, check=False, timeout=60, env=env
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
