"""Benchmark AR and FFT-based noise simulators."""

from __future__ import annotations

import argparse
import tempfile
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gwmock_noise.simulators import ARNoiseSimulator, ColoredNoiseSimulator

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
    psd_file: Path
    low_frequency_cutoff: float
    high_frequency_cutoff: float | None


@dataclass(frozen=True)
class BenchmarkResult:
    """Benchmark summary and PSD estimate for one simulator."""

    simulator: str
    wall_time_seconds: float
    peak_memory_mib: float
    mean_relative_error: float
    frequencies: np.ndarray
    estimated_psd: np.ndarray


def _write_default_psd(path: Path, sampling_frequency: float) -> Path:
    """Write a flat PSD when the user does not provide one."""
    frequencies = np.linspace(0.0, sampling_frequency / 2.0, 4097)
    values = np.full_like(frequencies, 2.0e-3)
    np.savetxt(path, np.column_stack((frequencies, values)))
    return path


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


def _mean_relative_error(
    frequencies: np.ndarray,
    estimated_psd: np.ndarray,
    target_data: np.ndarray,
    config: BenchmarkConfig,
) -> float:
    """Compute mean relative PSD error over the simulated band."""
    target_psd = np.interp(frequencies, target_data[:, 0], target_data[:, 1], left=0.0, right=0.0)
    band_high = (
        config.sampling_frequency / 2.0 if config.high_frequency_cutoff is None else config.high_frequency_cutoff
    )
    band = (frequencies >= config.low_frequency_cutoff) & (frequencies <= band_high)
    floor = np.maximum(target_psd[band], 1e-12)
    return float(np.mean(np.abs(estimated_psd[band] - target_psd[band]) / floor))


def _benchmark_simulator(
    label: str,
    simulator: ARNoiseSimulator | ColoredNoiseSimulator,
    config: BenchmarkConfig,
) -> BenchmarkResult:
    """Benchmark one simulator and summarize its PSD error."""
    tracemalloc.start()
    started_at = time.perf_counter()
    realization = simulator.generate(
        duration=config.duration,
        sampling_frequency=config.sampling_frequency,
        detectors=config.detectors,
        seed=config.seed,
    )
    elapsed = time.perf_counter() - started_at
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    frequencies, estimated_psd = _estimate_one_sided_psd(realization[config.detectors[0]], config.sampling_frequency)
    target_data = np.loadtxt(config.psd_file)
    mean_relative_error = _mean_relative_error(
        frequencies,
        estimated_psd,
        target_data,
        config,
    )
    return BenchmarkResult(
        simulator=label,
        wall_time_seconds=elapsed,
        peak_memory_mib=peak / (1024.0**2),
        mean_relative_error=mean_relative_error,
        frequencies=frequencies,
        estimated_psd=estimated_psd,
    )


def _print_table(rows: list[BenchmarkResult]) -> None:
    """Print a simple comparison table."""
    print(f"{'Simulator':<12} {'Wall time (s)':>14} {'Peak MiB':>10} {'Mean rel. error':>17}")
    print(f"{'-' * 12} {'-' * 14} {'-' * 10} {'-' * 17}")
    for row in rows:
        print(
            f"{row.simulator:<12} "
            f"{row.wall_time_seconds:>14.3f} "
            f"{row.peak_memory_mib:>10.2f} "
            f"{row.mean_relative_error:>17.5f}"
        )


def _maybe_plot(
    rows: list[BenchmarkResult],
    config: BenchmarkConfig,
    output_path: Path | None,
) -> None:
    """Plot the target and simulated PSDs when matplotlib is available."""
    if output_path is None or plt is None:
        return

    target_data = np.loadtxt(config.psd_file)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.loglog(target_data[:, 0], target_data[:, 1], label="Target PSD", linewidth=2)
    for row in rows:
        axis.loglog(row.frequencies, row.estimated_psd, alpha=0.8, label=f"{row.simulator} estimate")

    axis.set_xlim(
        left=max(config.low_frequency_cutoff, 1e-3),
        right=config.high_frequency_cutoff or config.sampling_frequency / 2.0,
    )
    axis.set_xlabel("Frequency [Hz]")
    axis.set_ylabel("PSD")
    axis.set_title(f"AR vs FFT PSD benchmark ({config.duration:.0f} s @ {config.sampling_frequency:.0f} Hz)")
    axis.grid(True, which="both", alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path)


def main() -> None:
    """Run the AR-vs-FFT benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--psd-file", type=Path, default=None, help="Path to a two-column PSD file.")
    parser.add_argument("--duration", type=float, default=100.0, help="Noise duration in seconds.")
    parser.add_argument("--sampling-frequency", type=float, default=4096.0, help="Sampling frequency in Hz.")
    parser.add_argument("--seed", type=int, default=1234, help="Seed used for both simulators.")
    parser.add_argument("--order", type=int, default=256, help="AR model order.")
    parser.add_argument("--low-frequency-cutoff", type=float, default=2.0, help="Lower simulation band edge in Hz.")
    parser.add_argument(
        "--high-frequency-cutoff",
        type=float,
        default=None,
        help="Upper simulation band edge in Hz. Defaults to Nyquist.",
    )
    parser.add_argument("--plot", type=Path, default=None, help="Optional path for a saved benchmark plot.")
    args = parser.parse_args()

    if args.psd_file is not None:
        psd_file = args.psd_file
        temp_dir_cm = None
    else:
        temp_dir_cm = tempfile.TemporaryDirectory()
        psd_file = _write_default_psd(Path(temp_dir_cm.name) / "benchmark_psd.txt", args.sampling_frequency)

    detectors = ["H1"]
    ar_simulator = ARNoiseSimulator(
        psd_file=psd_file,
        detectors=detectors,
        duration=args.duration,
        sampling_frequency=args.sampling_frequency,
        order=args.order,
        low_frequency_cutoff=args.low_frequency_cutoff,
        high_frequency_cutoff=args.high_frequency_cutoff,
    )
    fft_simulator = ColoredNoiseSimulator(
        psd_file=psd_file,
        detectors=detectors,
        duration=args.duration,
        sampling_frequency=args.sampling_frequency,
        low_frequency_cutoff=args.low_frequency_cutoff,
        high_frequency_cutoff=args.high_frequency_cutoff,
    )

    config = BenchmarkConfig(
        duration=args.duration,
        sampling_frequency=args.sampling_frequency,
        detectors=detectors,
        seed=args.seed,
        psd_file=psd_file,
        low_frequency_cutoff=args.low_frequency_cutoff,
        high_frequency_cutoff=args.high_frequency_cutoff,
    )

    rows = [
        _benchmark_simulator("AR", ar_simulator, config),
        _benchmark_simulator("FFT", fft_simulator, config),
    ]

    _print_table(rows)
    _maybe_plot(rows, config, output_path=args.plot)

    if temp_dir_cm is not None:
        temp_dir_cm.cleanup()


if __name__ == "__main__":
    main()
