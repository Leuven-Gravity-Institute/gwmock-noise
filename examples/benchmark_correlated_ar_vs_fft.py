"""Benchmark correlated VMA and FFT-based noise simulators."""

from __future__ import annotations

import argparse
import tempfile
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np

from gwmock_noise.simulators import CorrelatedARNoiseSimulator, CorrelatedNoiseSimulator

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None


@dataclass(frozen=True)
class BenchmarkConfig:
    """Runtime configuration shared by both benchmarked simulators."""

    duration: float
    sampling_frequency: float
    detectors: list[str]
    seed: int
    psd_files: dict[str, Path]
    csd_files: dict[str, Path]
    low_frequency_cutoff: float
    high_frequency_cutoff: float | None
    order: int


@dataclass(frozen=True)
class BenchmarkResult:
    """Benchmark summary and spectral estimates for one simulator."""

    simulator: str
    wall_time_seconds: float
    peak_memory_mib: float
    setup_wall_time_seconds: float
    setup_peak_memory_mib: float
    generate_wall_time_seconds: float
    generate_peak_memory_mib: float
    mean_psd_relative_error: float
    mean_csd_relative_error: float
    frequencies: np.ndarray
    estimated_psd: np.ndarray
    estimated_csd: np.ndarray


def _write_default_spectral_inputs(
    base_dir: Path, sampling_frequency: float
) -> tuple[dict[str, Path], dict[str, Path]]:
    """Write flat PSD/CSD inputs when the user does not provide files."""
    detectors = ["H1", "L1"]
    frequencies = np.linspace(0.0, sampling_frequency / 2.0, 4097)
    psd_values = np.full_like(frequencies, 2.0e-3)
    csd_values = np.full(frequencies.shape, 8.0e-4, dtype=np.complex128)

    psd_files = {}
    for detector in detectors:
        path = base_dir / f"{detector}_psd.txt"
        np.savetxt(path, np.column_stack((frequencies, psd_values)))
        psd_files[detector] = path

    csd_files = {}
    for detector_a, detector_b in combinations(detectors, 2):
        path = base_dir / f"{detector_a}_{detector_b}_csd.npy"
        np.save(path, np.column_stack((frequencies, csd_values)))
        csd_files[f"{detector_a}-{detector_b}"] = path

    return psd_files, csd_files


def _estimate_one_sided_psd(strain: np.ndarray, sampling_frequency: float) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the one-sided periodogram."""
    n_samples = strain.size
    frequency_series = np.fft.rfft(strain)
    psd = (2.0 / (sampling_frequency * n_samples)) * np.abs(frequency_series) ** 2
    psd[0] /= 2.0
    if n_samples % 2 == 0:
        psd[-1] /= 2.0
    frequencies = np.fft.rfftfreq(n_samples, d=1.0 / sampling_frequency)
    return frequencies, psd


def _estimate_one_sided_csd(
    strain_a: np.ndarray,
    strain_b: np.ndarray,
    sampling_frequency: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the one-sided cross-spectral density."""
    n_samples = strain_a.size
    series_a = np.fft.rfft(strain_a)
    series_b = np.fft.rfft(strain_b)
    csd = (2.0 / (sampling_frequency * n_samples)) * (np.conj(series_a) * series_b)
    csd[0] /= 2.0
    if n_samples % 2 == 0:
        csd[-1] /= 2.0
    frequencies = np.fft.rfftfreq(n_samples, d=1.0 / sampling_frequency)
    return frequencies, csd


def _mean_relative_error(
    estimate: np.ndarray,
    target: np.ndarray,
    frequencies: np.ndarray,
    config: BenchmarkConfig,
) -> float:
    """Compute mean relative spectral error over the simulated band."""
    band_high = (
        config.sampling_frequency / 2.0 if config.high_frequency_cutoff is None else config.high_frequency_cutoff
    )
    band = (frequencies >= config.low_frequency_cutoff) & (frequencies <= band_high)
    floor = np.maximum(np.abs(target[band]), 1e-12)
    return float(np.mean(np.abs(estimate[band] - target[band]) / floor))


def _benchmark_simulator(
    label: str,
    build_simulator: Callable[[], CorrelatedARNoiseSimulator | CorrelatedNoiseSimulator],
    config: BenchmarkConfig,
) -> BenchmarkResult:
    """Benchmark one simulator and summarize PSD/CSD errors.

    Measures construction/setup (``CorrelatedARNoiseSimulator`` / ``CorrelatedNoiseSimulator`` ``__init__``)
    and ``generate()`` separately, each with its own wall time and tracemalloc peak.
    """
    tracemalloc.start()
    setup_started = time.perf_counter()
    simulator = build_simulator()
    setup_elapsed = time.perf_counter() - setup_started
    _, setup_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    generate_started = time.perf_counter()
    realization = simulator.generate(
        duration=config.duration,
        sampling_frequency=config.sampling_frequency,
        detectors=config.detectors,
        seed=config.seed,
    )
    generate_elapsed = time.perf_counter() - generate_started
    _, generate_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    total_elapsed = setup_elapsed + generate_elapsed
    setup_peak_mib = setup_peak / (1024.0**2)
    generate_peak_mib = generate_peak / (1024.0**2)
    peak_mib = max(setup_peak_mib, generate_peak_mib)

    frequencies, estimated_psd = _estimate_one_sided_psd(realization[config.detectors[0]], config.sampling_frequency)
    _, estimated_csd = _estimate_one_sided_csd(
        realization[config.detectors[0]],
        realization[config.detectors[1]],
        config.sampling_frequency,
    )

    psd_data = np.loadtxt(config.psd_files[config.detectors[0]])
    csd_data = np.load(config.csd_files[f"{config.detectors[0]}-{config.detectors[1]}"])
    target_psd = np.interp(frequencies, psd_data[:, 0], psd_data[:, 1], left=0.0, right=0.0)
    target_csd = np.interp(frequencies, csd_data[:, 0].real, csd_data[:, 1].real, left=0.0, right=0.0) + 1j * np.interp(
        frequencies,
        csd_data[:, 0].real,
        csd_data[:, 1].imag,
        left=0.0,
        right=0.0,
    )

    return BenchmarkResult(
        simulator=label,
        wall_time_seconds=total_elapsed,
        peak_memory_mib=peak_mib,
        setup_wall_time_seconds=setup_elapsed,
        setup_peak_memory_mib=setup_peak_mib,
        generate_wall_time_seconds=generate_elapsed,
        generate_peak_memory_mib=generate_peak_mib,
        mean_psd_relative_error=_mean_relative_error(
            estimated_psd,
            target_psd,
            frequencies,
            config,
        ),
        mean_csd_relative_error=_mean_relative_error(
            estimated_csd,
            target_csd,
            frequencies,
            config,
        ),
        frequencies=frequencies,
        estimated_psd=estimated_psd,
        estimated_csd=estimated_csd,
    )


def _print_table(rows: list[BenchmarkResult]) -> None:
    """Print a simple comparison table."""
    print(
        f"{'Simulator':<18} "
        f"{'Setup (s)':>10} {'Gen (s)':>10} {'Total (s)':>10} "
        f"{'Setup MiB':>10} {'Gen MiB':>10} {'Peak MiB':>10} "
        f"{'PSD rel. err':>14} {'CSD rel. err':>14}"
    )
    print(f"{'-' * 18} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 14} {'-' * 14}")
    for row in rows:
        print(
            f"{row.simulator:<18} "
            f"{row.setup_wall_time_seconds:>10.3f} "
            f"{row.generate_wall_time_seconds:>10.3f} "
            f"{row.wall_time_seconds:>10.3f} "
            f"{row.setup_peak_memory_mib:>10.2f} "
            f"{row.generate_peak_memory_mib:>10.2f} "
            f"{row.peak_memory_mib:>10.2f} "
            f"{row.mean_psd_relative_error:>14.5f} "
            f"{row.mean_csd_relative_error:>14.5f}"
        )


def _maybe_plot(rows: list[BenchmarkResult], output_path: Path | None) -> None:
    """Plot PSD/CSD estimates when matplotlib is available."""
    if output_path is None or plt is None:
        return

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for row in rows:
        axes[0].loglog(row.frequencies, row.estimated_psd, label=f"{row.simulator} PSD")
        axes[1].loglog(row.frequencies, np.abs(row.estimated_csd), label=f"{row.simulator} |CSD|")
    axes[0].set_xlabel("Frequency [Hz]")
    axes[0].set_ylabel("PSD")
    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].set_ylabel("|CSD|")
    axes[0].grid(True, which="both", alpha=0.2)
    axes[1].grid(True, which="both", alpha=0.2)
    axes[0].legend()
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_path)


def main() -> None:
    """Run the correlated AR-vs-FFT benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=100.0, help="Noise duration in seconds.")
    parser.add_argument("--sampling-frequency", type=float, default=4096.0, help="Sampling frequency in Hz.")
    parser.add_argument("--seed", type=int, default=1234, help="Seed used for both simulators.")
    parser.add_argument("--order", type=int, default=256, help="VMA model order.")
    parser.add_argument("--low-frequency-cutoff", type=float, default=2.0, help="Lower simulation band edge in Hz.")
    parser.add_argument(
        "--high-frequency-cutoff",
        type=float,
        default=None,
        help="Upper simulation band edge in Hz. Defaults to Nyquist.",
    )
    parser.add_argument("--plot", type=Path, default=None, help="Optional path for a saved benchmark plot.")
    args = parser.parse_args()

    temp_dir = tempfile.TemporaryDirectory()
    psd_files, csd_files = _write_default_spectral_inputs(Path(temp_dir.name), args.sampling_frequency)
    detectors = ["H1", "L1"]
    config = BenchmarkConfig(
        duration=args.duration,
        sampling_frequency=args.sampling_frequency,
        detectors=detectors,
        seed=args.seed,
        psd_files=psd_files,
        csd_files=csd_files,
        low_frequency_cutoff=args.low_frequency_cutoff,
        high_frequency_cutoff=args.high_frequency_cutoff,
        order=args.order,
    )

    rows = [
        _benchmark_simulator(
            "Correlated VMA",
            lambda: CorrelatedARNoiseSimulator(
                psd_files=psd_files,
                csd_files=csd_files,
                detectors=detectors,
                duration=args.duration,
                sampling_frequency=args.sampling_frequency,
                order=args.order,
                low_frequency_cutoff=args.low_frequency_cutoff,
                high_frequency_cutoff=args.high_frequency_cutoff,
            ),
            config,
        ),
        _benchmark_simulator(
            "Correlated FFT",
            lambda: CorrelatedNoiseSimulator(
                psd_files=psd_files,
                csd_files={("H1", "L1"): csd_files["H1-L1"]},
                detectors=detectors,
                duration=args.duration,
                sampling_frequency=args.sampling_frequency,
                low_frequency_cutoff=args.low_frequency_cutoff,
                high_frequency_cutoff=args.high_frequency_cutoff,
            ),
            config,
        ),
    ]
    _print_table(rows)
    _maybe_plot(rows, args.plot)
    temp_dir.cleanup()


if __name__ == "__main__":
    main()
