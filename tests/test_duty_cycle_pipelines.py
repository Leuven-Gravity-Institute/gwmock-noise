"""The duty-cycle state vector, read by the pipelines that read state vectors, with their current releases.

The bits follow LIGO's calibrated state vector so that a pipeline's own selection picks observing time.
That is a contract with other software, which can change on its side without any change here, so it is
checked against that software: the guide's examples are run as written, and PyCBC's low-latency state
buffer is driven over a frame file holding the state vector.

PyCBC is heavy and is not in the ``dev`` group: these tests are marked ``pipelines``, which the default
run excludes, and the ``pipelines`` CI job installs the pipelines and runs them. With
``GWMOCK_REQUIRE_PIPELINES=1`` a missing pipeline fails a test instead of skipping it, so that job cannot
pass by skipping everything.
"""

from __future__ import annotations

import importlib
import os
import re
from pathlib import Path

import numpy as np
import pytest

from gwmock_noise.duty_cycle import DAY, DutyCycleModel, simulate_duty_cycle

pytestmark = pytest.mark.pipelines

_GUIDE = Path(__file__).resolve().parents[1] / "docs" / "user_guide" / "duty_cycle.md"
_EXAMPLE = re.compile(r"<!-- example: (?P<name>[\w-]+) -->\s*```python\n(?P<code>.*?)```", re.DOTALL)


def _require(module: str):
    """Import ``module``, or skip -- unless the run requires the pipelines, when it fails."""
    if os.environ.get("GWMOCK_REQUIRE_PIPELINES") == "1":
        return importlib.import_module(module)
    return pytest.importorskip(module)


def _run_example(name: str) -> None:
    examples = {match["name"]: match["code"] for match in _EXAMPLE.finditer(_GUIDE.read_text(encoding="utf-8"))}
    assert name in examples, f"the guide no longer has the {name!r} example"
    exec(compile(examples[name], f"{_GUIDE.name}:{name}", "exec"), {"__name__": "__example__"})  # noqa: S102


def test_the_gwpy_example_runs() -> None:
    """The guide's gwpy example, as written."""
    _require("gwpy")
    _run_example("gwpy")


def test_the_pycbc_example_runs() -> None:
    """The guide's PyCBC example, as written."""
    _require("pycbc")
    _run_example("pycbc")


def test_pycbc_live_analyses_a_block_exactly_when_it_is_observing(tmp_path: Path) -> None:
    """PyCBC's own state buffer, over a frame file, with the flags the guide gives."""
    _require("pycbc")
    _require("gwpy")
    from gwpy.timeseries import TimeSeries
    from pycbc.frame import StatusBuffer, flag_names_to_bitmask

    start, rate, duration, block = 1_000_000_000, 16.0, 6 * 3600, 8
    realization = simulate_duty_cycle(
        DutyCycleModel(mean_uptime=1800.0, mean_downtime=600.0), ["E1", "E2", "E3"], start, duration, seed=4
    )
    samples = realization.state_vector("E1", start, rate, int(duration * rate))
    path = tmp_path / "E-E1_STATE.gwf"
    TimeSeries(samples, t0=start, sample_rate=rate, channel="E1:STATE_VECTOR", name="E1:STATE_VECTOR").write(
        str(path), format="gwf"
    )

    mask = flag_names_to_bitmask(["HOFT_OK", "SCIENCE_INTENT"])
    buffer = StatusBuffer(
        [str(path)], "E1:STATE_VECTOR", start, max_buffer=64, valid_mask=mask, force_update_cache=False
    )
    analysed = np.array([bool(buffer.advance(block)) for _ in range(duration // block)])

    # A block is analysable when every one of its samples is observing.
    times = start + np.arange(samples.size) / rate
    observing = realization.is_observing("E1", times).reshape(-1, int(block * rate)).all(axis=1)
    assert analysed.any()
    assert not analysed.all()
    np.testing.assert_array_equal(analysed, observing)


def test_a_two_day_default_realization_is_mostly_observing() -> None:
    """The examples above test something only if the realization has both observing time and gaps."""
    realization = simulate_duty_cycle(DutyCycleModel(), ["E1", "E2", "E3"], 1_000_000_000, 2 * DAY, seed=1)
    assert len(realization.observing["E1"]) > 1
    assert 0.5 < realization.observing_fraction("E1") < 1
