# Copyright (C) 2026 Leuven Gravity Institute
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

"""What works with only the core dependencies: no h5py, no GWpy, no lalsuite.

This is the installation Python 3.15 gets today, where h5py publishes no wheel. CI runs this module
there with only the package and pytest installed; it also runs with everything installed, where it
checks the same paths never import the optional packages.

Each test runs its scenario in a fresh interpreter, so what the package imports can be observed:
``sys.modules`` in the test process already holds whatever other tests imported. The scenarios about a
missing h5py hide it with an import hook, so they test that path whether or not h5py is installed.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

#: Packages none of these paths may import.
_OPTIONAL = ("h5py", "gwpy", "lal", "lalframe", "gengli", "gwosc", "huggingface_hub")

#: Makes ``import h5py`` fail as it does where h5py is not installed.
_HIDE_H5PY = """
import importlib.abc, sys

class _HideH5py(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "h5py" or name.startswith("h5py."):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None

sys.meta_path.insert(0, _HideH5py())
"""


def _run(code: str, *, hide_h5py: bool = False) -> dict:
    """Run ``code`` in a fresh interpreter; it sets ``result``, printed as JSON with what it loaded."""
    script = (
        (_HIDE_H5PY if hide_h5py else "")
        + textwrap.dedent(code)
        + textwrap.dedent(
            f"""
        import json as _json, sys as _sys
        _loaded = sorted({{name.split(".")[0] for name in _sys.modules}} & set({_OPTIONAL!r}))
        print(_json.dumps({{"result": result, "optional_loaded": _loaded}}))
        """
        )
    )
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)  # noqa: S603
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_the_package_and_its_cli_import_without_optional_packages():
    """Importing the package, its CLI, the gengli module and the default simulator loads none of them."""
    outcome = _run(
        """
        import gwmock_noise, gwmock_noise.cli.main, gwmock_noise.glitches.gengli, gwmock_noise.simulators.default
        result = gwmock_noise.__version__
        """
    )

    assert outcome["optional_loaded"] == []


def test_a_default_run_writes_numpy_without_optional_packages(tmp_path: Path):
    """The default output format, ``npy``, needs no optional package, and h5py is never imported."""
    outcome = _run(
        f"""
        from pathlib import Path
        import numpy as np
        from gwmock_noise import DefaultNoiseSimulator, NoiseConfig, OutputConfig

        config = NoiseConfig(
            detectors=["H1", "L1"], duration=4.0, sampling_frequency=128.0, seed=7,
            output=OutputConfig(directory={str(tmp_path)!r}),
        )
        paths = DefaultNoiseSimulator().run(config).output_paths
        result = {{name: [Path(path).suffix, int(np.load(path).size)] for name, path in paths.items()}}
        """,
        hide_h5py=True,
    )

    assert outcome["optional_loaded"] == []
    assert outcome["result"] == {"H1": [".npy", 512], "L1": [".npy", 512]}


def test_hdf5_output_works_without_h5py(tmp_path: Path):
    """HDF5 output is written by gwmock-data's own writer and reads back with the grid and channel."""
    directory = tmp_path / "out"
    outcome = _run(
        f"""
        import numpy as np
        from gwmock_data import hdf5
        from gwmock_noise import DefaultNoiseSimulator, NoiseConfig, OutputConfig

        config = NoiseConfig(
            detectors=["H1"], duration=4.0, sampling_frequency=128.0, seed=7,
            output=OutputConfig(directory={str(directory)!r}, format="hdf5", gps_start=1000000000.0),
        )
        path = DefaultNoiseSimulator().run(config).output_paths["H1"]
        dataset = next(iter(hdf5.read(path).members.values()))
        result = [int(np.asarray(dataset.data).size), float(dataset.attrs["x0"]), dataset.attrs["channel"]]
        """,
        hide_h5py=True,
    )

    assert outcome["optional_loaded"] == []
    assert outcome["result"] == [512, 1000000000.0, "H1:MOCK_NOISE"]


def test_population_and_snr_files_work_without_h5py(tmp_path: Path):
    """A blip population file is written and read back, and read as an SNR table, without h5py."""
    population = tmp_path / "new" / "population.h5"
    outcome = _run(
        f"""
        import numpy as np
        from gwmock_noise.glitches.gengli import read_blip_population_file, write_blip_population_file
        from gwmock_noise.glitches.snr import load_snr_samples

        write_blip_population_file({str(population)!r}, snr_samples=np.array([8.0, 12.5]), metadata={{"source": "test"}})
        result = [read_blip_population_file({str(population)!r}).tolist(), load_snr_samples({str(population)!r}).tolist()]
        """,
        hide_h5py=True,
    )

    assert outcome["optional_loaded"] == []
    assert outcome["result"] == [[8.0, 12.5], [8.0, 12.5]]
