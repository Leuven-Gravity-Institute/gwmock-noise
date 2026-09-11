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
