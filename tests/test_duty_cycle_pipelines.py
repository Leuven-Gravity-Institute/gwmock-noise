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
_EXAMPLE = re.compile(r"<!-- example: (?P<name>[\w-]+) -->\s*```(?P<language>\w+)\n(?P<code>.*?)```", re.DOTALL)


def _require(module: str):
    """Import ``module``, or skip -- unless the run requires the pipelines, when it fails."""
    if os.environ.get("GWMOCK_REQUIRE_PIPELINES") == "1":
        return importlib.import_module(module)
    return pytest.importorskip(module)


def _example(name: str) -> tuple[str, str]:
    """The guide's example called ``name``: its language and its code."""
    examples = {
        match["name"]: (match["language"], match["code"])
        for match in _EXAMPLE.finditer(_GUIDE.read_text(encoding="utf-8"))
    }
    assert name in examples, f"the guide no longer has the {name!r} example"
    return examples[name]


def _run_example(name: str) -> None:
    language, code = _example(name)
    assert language == "python"
    # The examples check their own results; a test that only ran them would pass once a check was deleted.
    assert "assert " in code, f"the {name!r} example no longer checks its result"
    exec(compile(code, f"{_GUIDE.name}:{name}", "exec"), {"__name__": "__example__"})  # noqa: S102


def test_the_gwpy_example_runs() -> None:
    """The guide's gwpy example, as written."""
    _require("gwpy")
    _run_example("gwpy")


def test_the_pycbc_example_runs() -> None:
    """The guide's PyCBC example, as written."""
    _require("pycbc")
    _run_example("pycbc")


def test_the_guides_pycbc_live_options_select_exactly_the_observing_blocks(tmp_path: Path) -> None:
    """The guide's ``pycbc_live`` options, parsed and applied by PyCBC's own classes, over a frame file.

    ``pycbc_live`` declares ``--state-channel`` with ``MultiDetMultiColonOptionAction`` and
    ``--analyze-flags`` with ``MultiDetOptionAction``, builds the state channel and the mask from them as
    ``StrainBuffer.from_cli`` does, and reads the state vector through ``StatusBuffer``: a block is
    analysed when its ``advance`` returns true. The program itself is not importable, so its option
    declarations are restated here.
    """
    _require("pycbc")
    _require("gwpy")
    import argparse
    import shlex

    from gwpy.timeseries import TimeSeries
    from pycbc.frame import StatusBuffer, flag_names_to_bitmask, read_frame
    from pycbc.types.optparse import MultiDetMultiColonOptionAction, MultiDetOptionAction

    language, command = _example("pycbc-live")
    assert language == "bash"
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-channel", action=MultiDetMultiColonOptionAction, nargs="+")
    parser.add_argument("--analyze-flags", action=MultiDetOptionAction, nargs="+")
    options = parser.parse_args(shlex.split(command)[1:])
    ifo = "E1"
    assert ifo in options.state_channel
    assert ifo in options.analyze_flags  # without both, pycbc_live reads no state vector
    channel = f"{ifo}:{options.state_channel[ifo]}"
    mask = flag_names_to_bitmask(options.analyze_flags[ifo].split(","))
    assert mask == 0b11

    start, rate, duration, block = 1_000_000_000, 16.0, 6 * 3600, 8
    realization = simulate_duty_cycle(
        DutyCycleModel(mean_uptime=1800.0, mean_downtime=600.0), ["E1", "E2", "E3"], start, duration, seed=4
    )
    samples = realization.state_vector(ifo, start, rate, int(duration * rate))
    path = tmp_path / f"E-{ifo}_STATE.gwf"
    TimeSeries(samples, t0=start, sample_rate=rate, channel=channel, name=channel).write(str(path), format="gwf")

    # Sample by sample, as PyCBC reads it back: exactly the observing samples.
    times = start + np.arange(samples.size) / rate
    observing = realization.is_observing(ifo, times)
    read = read_frame(str(path), channel).numpy().astype(np.int32)
    np.testing.assert_array_equal(np.bitwise_and(read, mask) == mask, observing)

    # Block by block, through the state buffer, with the guide's mask and with bit 2 required too.
    expected = observing.reshape(-1, int(block * rate)).all(axis=1)
    assert expected.any()
    assert not expected.all()
    for valid_mask in (mask, mask | flag_names_to_bitmask(["SCIENCE_QUALITY"])):
        buffer = StatusBuffer(
            [str(path)], channel, start, max_buffer=64, valid_mask=valid_mask, force_update_cache=False
        )
        analysed = np.array([bool(buffer.advance(block)) for _ in range(duration // block)])
        np.testing.assert_array_equal(analysed, expected)


def test_a_two_day_default_realization_is_mostly_observing() -> None:
    """The examples above test something only if the realization has both observing time and gaps."""
    realization = simulate_duty_cycle(DutyCycleModel(), ["E1", "E2", "E3"], 1_000_000_000, 2 * DAY, seed=1)
    assert len(realization.observing["E1"]) > 1
    assert 0.5 < realization.observing_fraction("E1") < 1
