# Bundled PSD presets — provenance

Every file in this directory is a two-column ASCII table of

```text
<frequency / Hz>  <one-sided power spectral density / Hz^-1>
```

sorted by increasing frequency, resolvable by its bare stem (for example
`aLIGO_O3_actual_H1_psd`) anywhere a `psd_file` is accepted. The second column
is a **PSD**, not an amplitude spectral density: take the square root to get
strain/sqrt(Hz).

## Advanced LIGO

Source document: LIGO-T2000012-v2, _Noise curves used for simulations in the
update of the Observing Scenarios Paper_,
<https://dcc.ligo.org/LIGO-T2000012/public>. These are the curves underlying
Abbott et al., _Prospects for observing and localizing gravitational-wave
transients with Advanced LIGO, Advanced Virgo and KAGRA_, Living Reviews in
Relativity 23, 3 (2020).

The DCC files tabulate **amplitude** spectral density. Each bundled file was
produced by squaring the second column and leaving the frequency column
untouched — no resampling, smoothing, interpolation or line removal.

| Bundled preset                | Source file in LIGO-T2000012-v2 | SHA-256 of source file                                             |
| ----------------------------- | ------------------------------- | ------------------------------------------------------------------ |
| `aLIGO_O3_actual_H1_psd`      | `aligo_O3actual_H1.txt`         | `553d04ee6f8e4d737e004f57d383382f6bdda6774f56b8ceb16c317a38c3b632` |
| `aLIGO_O3_actual_L1_psd`      | `aligo_O3actual_L1.txt`         | `730b33339283a8a677af2f52d4aa9876f09ffe28a52c45627f1e57687c6b869b` |
| `aLIGO_O4_high_projected_psd` | `aligo_O4high.txt`              | `eb5ec9b081c3d86d2f4257b9aff6a57566d168b8a95e5e57b7909eebad021780` |
| `aLIGO_O4_low_projected_psd`  | `aligo_O4low.txt`               | `9a7449cda558e4f5c124c5693cf5149d464a9d9c5383acd24030fac1400b88bd` |

### What "actual" and "projected" mean here

- **`O3_actual`** — a _measured_ representative spectrum from the third
  observing run, one per LIGO site (H1 = Hanford, L1 = Livingston). It is real
  detector data on a uniform 0.25 Hz grid from 10 Hz to 5000 Hz and therefore
  still carries instrumental spectral lines (calibration lines, violin modes,
  mains harmonics). That is deliberate: the point of these curves is to
  reproduce the noise the O3 interferometers actually had. Callers that need a
  smooth curve should smooth it themselves.
- **`O4_projected`** — a _forecast_ of fourth-observing-run sensitivity,
  prepared before O4 began, on a 2736-point non-uniform grid from 10.2 Hz to
  4995 Hz. `high` and `low` bracket the anticipated range. These are design
  targets, **not** measurements of O4 data; no measured representative O4 curve
  was published as a single reference spectrum at the time these were bundled.

### Anchors

Each curve's 1.4+1.4 solar-mass BNS inspiral range (SNR 8, sky- and
orientation-averaged, integrated from 10 Hz to ISCO) was computed from the
bundled PSD and compared against the ranges quoted in the Observing Scenarios
paper for the corresponding run:

| Preset                        | Range from this file | Published reference |
| ----------------------------- | -------------------- | ------------------- |
| `aLIGO_O3_actual_H1_psd`      | 107 Mpc              | ~110 Mpc (O3, H1)   |
| `aLIGO_O3_actual_L1_psd`      | 133 Mpc              | ~135 Mpc (O3, L1)   |
| `aLIGO_O4_high_projected_psd` | 190 Mpc              | 190 Mpc (O4 upper)  |
| `aLIGO_O4_low_projected_psd`  | 168 Mpc              | 160–190 Mpc (O4)    |

The agreement confirms the ASD-to-PSD conversion and the units of the bundled
column.

## Einstein Telescope

`ET_D_psd`, `ET_10_HF_psd`, `ET_10_full_cryo_psd`, `ET_15_HF_psd`,
`ET_15_full_cryo_psd`, `ET_20_HF_psd` and `ET_20_full_cryo_psd` predate this
record and their originating document was not captured when they were added.
They are tabulated on a 3000-point logarithmic grid from 1 Hz to 10 kHz, and
`ET_D_psd` reproduces the ET-D design sensitivity to the precision of a read-off
(3.9e-25 strain/sqrt(Hz) at 100 Hz), but the exact data release each was drawn
from is not documented here.

### ET2026a

`ET2026a_10km_psd` and `ET2026a_15km_psd` are the ET2026a curves: one preset per
arm length, each the combination of that arm length's low-frequency (LF) and
high-frequency (HF) interferometers. The Einstein Telescope's page for
scientists, <https://einsteintelescope.eu/get-involved/for-scientists/>
(accessed 2026-11-09), recommends the ET2026a curve for new work: "We recommend
the use of this curve for future works".

Source: the public repository
[`pub/et-sensitivity-curves`](https://gitlab.et-gw.eu/pub/et-sensitivity-curves)
on the Einstein Telescope GitLab, commit
`66177ccf4626c77fee243ef21367ac934b2c0ccc` (2026-03-30). Its README describes
the curves as a snapshot of a work in progress, as of January 2026, computed for
a 90° opening angle between the arms.

| Source file               | SHA-256                                                            |
| ------------------------- | ------------------------------------------------------------------ |
| `10_km/LF/data/total.txt` | `2e4477e0d2dc30228a7cbc7ede804355b5b307486859a82ed9c6d4cd9bd16375` |
| `10_km/HF/data/total.txt` | `35bedf22facefd325148ef787caebb582348b91c1e37fd2b586de8510e7be355` |
| `15_km/LF/data/total.txt` | `ba58ade04eb74e9c2f148a16abf6d57da377c0b817d973d0f784a2b7ffa39854` |
| `15_km/HF/data/total.txt` | `b6f003e0224aaee67b62bb2aa788c87b3a93b100bf94a38f5019bc76932ba838` |

`scripts/build_et2026a_psd_presets.py` builds both presets from these files:

1. Column 3 of each file is read. It is the curve's default sensitivity, with
   the high-frequency correction for a signal 15° off zenith.
2. The column is squared. The README calls the files power spectral densities,
   but the values are amplitude spectral densities: about 4e-25 Hz^-1/2 at 100
   Hz, rising at 10 kHz to 9e-24 (10 km) and 1e-24 (15 km).
3. The LF and HF curves share one grid and are combined as two co-located
   detectors with independent noise, `S = 1 / (1 / S_LF + 1 / S_HF)`.

The grid is the 3000-point logarithmic grid from 1 Hz to 10 kHz that the other
ET presets use.

The combination is never above the better of the two curves. The LF curve is the
better one from 1.33 Hz to 23.2 Hz (10 km) or 21.7 Hz (15 km), and the HF curve
everywhere else, including the 93 rows below 1.33 Hz. There the two published
curves differ by up to a factor 2132 in amplitude, at both arm lengths, and the
preset follows the HF curve. Use the presets from 5 Hz, as the other ET presets
are.

Like the source curves, the presets tabulate no spectral lines; the other ET
presets carry one or two.

**Checks.** The integration test in `tests/test_psd_presets.py` downloads the
four files, checks their digests and rebuilds both presets byte for byte. It
also checks two properties of the source files the presets rely on:

- Below 50 Hz, column 3 is column 2 (the displacement noise) divided by the arm
  length, to 1.9e-5 (10 km) and 4.2e-5 (15 km). The column is therefore strain.
- Above 300 Hz, where the LF interferometer contributes nothing, the combination
  equals the HF curve, to 4.2e-6 (10 km) and 8.2e-4 (15 km).

Column 4 is the same correction for a signal at zenith. It departs from column 3
by 1.2e-6 (10 km) and 2.8e-6 (15 km) at 100 Hz, growing to 2.0e-3 and 4.0e-3 at
4096 Hz.

**Not anchored.** No published figure for the absolute level of these curves was
found to check them against. For comparison only: the 1.4+1.4 solar-mass
inspiral range, a Euclidean figure from 10 Hz with no cosmology computed by
`_bns_inspiral_range_mpc` in `tests/test_psd_presets.py` (which asserts the two
ET2026a values and so guards the presets' absolute scale), is 2824 Mpc for
`ET2026a_10km_psd` and 3745 Mpc for `ET2026a_15km_psd`, against 2468 Mpc and
3404 Mpc for `ET_10_full_cryo_psd` and `ET_15_full_cryo_psd`. Between 5 and 30
Hz the ET2026a PSDs are about half the `full_cryo` ones (median ratio 0.49 for
10 km, 0.54 for 15 km).
