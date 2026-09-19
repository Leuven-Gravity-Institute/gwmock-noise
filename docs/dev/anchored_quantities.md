# Anchored quantities in the AR and ARMA fits

Four numbers in the AR (Levinson-Durbin) and ARMA / state-space simulators were
originally chosen by hand. Each is now either measured over the population it
decides, or stated as a claim with the evidence that supports it. This page is
the record: what was measured, over what data, and what the number came out to.

Everything here is reproduced by

```bash
uv run python scripts/measure_anchored_quantities.py
```

which takes about fifteen seconds and prints **every** number quoted below —
each value, the identity of the preset and frequency it came from where that
matters, and the population it was scored over. Nothing on this page is derived
by hand from a run that is not in the script. Re-run it after touching the fit
code, the bundled presets, or the band definition, and update this page with
whatever it prints. Where a number also has to keep holding — the two line
defaults, and the claim that the tightened fit bounds can actually fail — a test
in `tests/test_anchored_quantities.py` or `tests/test_ar_levinson_fit.py`
asserts it through the same functions the harness prints from, so the page, the
printout and the assertion cannot drift apart.

## Source data

The population is the eleven PSD presets bundled in
`src/gwmock_noise/data/psd/`, whose own provenance is recorded in
`src/gwmock_noise/data/psd/PROVENANCE.md`. They fall into two kinds that matter
here:

- the **measured** Advanced LIGO O3 curves (`aLIGO_O3_actual_H1_psd`,
  `aLIGO_O3_actual_L1_psd`), real detector spectra on a uniform 0.25 Hz grid
  which still carry instrumental lines; and
- the **design or forecast** curves (the O4 projections and the seven Einstein
  Telescope curves), smooth models that nevertheless tabulate a small number of
  lines each.

Every measurement is made at a 4096 Hz sampling frequency, over 20 Hz to 2000 Hz
for the Advanced LIGO curves and 5 Hz to 2000 Hz for the Einstein Telescope
curves, in the eight geometric bands `DEFAULT_FIT_BANDS` defines. Line
measurements are made on the design grid the fit interpolates the target onto,
because that -- not the raw table -- is what the detector and the pole placement
actually see.

### The reference line list

Three of the four items need to know which tabulated features are lines. A
tabulated sample counts as a line when it is a local maximum and stands at least
ten times above the median of the samples within 20 % of its own frequency
(widened to at least 15 samples each side). This is a _local_ criterion, unlike
the peak-to-median ratio the detector uses, so it is not confused by a baseline
that falls by two decades across the band.

It is an operational definition, not a published line list, and it is worth
being explicit about where it is soft. Narrow isolated features are labelled
identically by any reasonable window: the Advanced LIGO violin-mode fundamental
near 500 Hz and its harmonics near 1000, 1500 and 2000 Hz, the 60.01 Hz mains
harmonic and the 306.33 Hz calibration line in the O4 projections, the 166.75 Hz
line in every Einstein Telescope curve, the 420.28 Hz line in ET-D. Broad
low-frequency structure is not: a few of the 20-25 Hz features in the measured
O3 curves move in and out of the list as the window widens, and the numbers
below inherit that ambiguity. Over the eleven presets the criterion labels 96
lines, 94 of which have a measurable width.

## 1. The per-band fit tolerances

**What they decide.** Whether an AR or ARMA fit "passes" on the tabulated
Advanced LIGO O4-high and ET-D curves, on two paths: the deterministic fitted
model curve, and a Welch estimate of six 32-second generated realizations.

**What was measured.** The worst of the eight per-band relative errors, for each
of the four (model, preset) cases. The fitted-curve path is deterministic, so it
has a single value. The generated path was sampled over twenty independent seed
groups of the same configuration the tests use.

| Case                       | Fitted curve, worst band | Generated, worst band over 20 groups |
| -------------------------- | ------------------------ | ------------------------------------ |
| AR, aLIGO O4-high, 20 Hz   | 0.0439                   | 0.1061 ± 0.0260 (max 0.1736)         |
| AR, ET-D, 5 Hz             | 0.3203                   | 0.4429 ± 0.0210 (max 0.4933)         |
| ARMA, aLIGO O4-high, 20 Hz | 0.0645                   | 0.0409 ± 0.0252 (max 0.1038)         |
| ARMA, ET-D, 5 Hz           | 0.4314                   | 0.5306 ± 0.0175 (max 0.5724)         |

**The rule.** A tolerance is `1.25 × mean + 5 × standard deviation`, rounded up
to two decimal places. The 1.25 is headroom on the deterministic part, for
library and platform drift; the five standard deviations cover realization
scatter with a per-run false-alarm rate below `1e-6` under a Gaussian
approximation. On the deterministic path the standard deviation is zero and the
rule reduces to the measured value plus that headroom.

| Case                          | Was | Is       | Headroom over the observed maximum |
| ----------------------------- | --- | -------- | ---------------------------------- |
| AR fitted, aLIGO O4-high      | 0.5 | **0.06** | 1.37×                              |
| AR fitted, ET-D               | 0.5 | **0.41** | 1.28×                              |
| AR generated, aLIGO O4-high   | 0.5 | **0.27** | 1.56×                              |
| AR generated, ET-D            | 0.5 | **0.66** | 1.34×                              |
| ARMA fitted, aLIGO O4-high    | 0.6 | **0.09** | 1.40×                              |
| ARMA fitted, ET-D             | 0.6 | **0.54** | 1.25×                              |
| ARMA generated, aLIGO O4-high | 0.6 | **0.18** | 1.73×                              |
| ARMA generated, ET-D          | 0.6 | **0.76** | 1.33×                              |

Three consequences worth stating plainly.

- **Seven of the eight bounds tighten, one loosens.** The single shared number
  was far too loose for the Advanced LIGO cases -- an AR fit at order 224
  instead of 256 has a worst band of 0.1048 on O4-high, which the old 0.5 passed
  and the new 0.06 does not.
  `test_fitted_tolerance_rejects_an_underfitted_model` pins exactly that, so the
  bound cannot quietly be widened back to a value nothing can breach.
- **The ARMA ET-D generated bound had to loosen, and that is the honest
  reading.** `1.25 × 0.5306 + 5 × 0.0175 = 0.751` is above the old 0.6: the
  population that test asserts on sits closer to its bound than five standard
  deviations, so 0.6 was a latent flake rather than a tight bound.
- **The tolerances are scoped to these two presets.** Across all eleven, and
  both models, the worst fitted band ranges from 0.0183 to 0.9541, mostly
  because the high-frequency-only Einstein Telescope variants are being fitted
  from 5 Hz, well below where they have any sensitivity. A bound covering that
  spread would assert nothing.

## 2. The default line width

`DEFAULT_LINE_WIDTH_HZ` sets the radius `exp(-pi w / f_s)` of every placed
conjugate pole pair whose width the caller does not give.

**What was measured.** The full width at half the excess over the local
baseline, for each of the 94 measurable reference lines, on the design grid the
fit interpolates the target onto.

| Percentile | Width         |
| ---------- | ------------- |
| 5          | 0.3543 Hz     |
| 25         | 0.3956 Hz     |
| **50**     | **0.4613 Hz** |
| 75         | 0.6560 Hz     |
| 90         | 0.8385 Hz     |
| 95         | 1.2099 Hz     |

`DEFAULT_LINE_WIDTH_HZ` is therefore **0.46 Hz**, the median of the population,
where it was 0.5 Hz.

**What this number is and is not.** The tabulated lines are one to three samples
wide, on tables whose spacing is 0.25 Hz (O3), 0.46 Hz (O4) and 0.31 Hz at 5 Hz
on the Einstein Telescope logarithmic grid. The measured widths are therefore
the widths of the lines _as tabulated_ -- which is the right target, since the
fit's target is the tabulated curve -- and not the detectors' physical line
widths, which for suspension violin modes are smaller by many orders of
magnitude. The median also moves by about 0.05 Hz with the baseline convention
used to measure it, so the last digit is not meaningful.

**What the width costs, and what it cannot buy.** A placed pole's peak height is
fixed by its radius, and so by its width; it is not fixed by how far the
tabulated line rises above its surroundings. The two cannot both be matched with
one conjugate pair. Measured on the O3 Hanford curve at order 192/32:

| Configuration                                       | Worst band residual |
| --------------------------------------------------- | ------------------- |
| no placed lines                                     | 0.3618              |
| one placed line at 998.04 Hz, width 0.5 Hz          | 0.8055              |
| one placed line at 998.04 Hz, width 0.46 Hz         | 0.8040              |
| one placed line at 998.04 Hz, width 100 Hz          | 0.3607              |
| the eight lines the detector selects, width 0.46 Hz | 8.2e13              |

The last row does not improve when the moving-average order is raised from 32 to
1024 (8.17e13 at 32, 128 and 512; 8.05e13 at 1024), so it is not numerator
truncation: it is the placed poles themselves, whose resonances stand about
`1e3` above the tabulated line at its own peak and up to `1e7` above the target
in the wings. The practical consequence is that **pole placement suits lines
whose width is known and whose contrast is modest**, the default is to place
none (`detect_lines=False`), and a caller who does place them should check the
reported `fit_residual` rather than assume the lines were reproduced.

## 3. The line-detection prominence

`DEFAULT_LINE_PROMINENCE` is the minimum ratio of a local maximum to the median
of the whole fitted band for that maximum to be placed as a pole.

**What was measured.** Detections at the shipped `max_lines = 8`, scored against
the reference line list, over all eleven presets.

| Threshold | True   | False | Missed | Precision | Recall    | F1        |
| --------- | ------ | ----- | ------ | --------- | --------- | --------- |
| 2         | 30     | 14    | 1      | 0.682     | 0.968     | 0.800     |
| 4         | 30     | 10    | 1      | 0.750     | 0.968     | 0.845     |
| 6         | 30     | 6     | 1      | 0.833     | 0.968     | 0.896     |
| 8         | 29     | 4     | 2      | 0.879     | 0.935     | 0.906     |
| 10        | 29     | 2     | 2      | 0.935     | 0.935     | 0.935     |
| 12        | 29     | 1     | 2      | 0.967     | 0.935     | 0.951     |
| **13-15** | **29** | **0** | **2**  | **1.000** | **0.935** | **0.967** |
| 20        | 28     | 0     | 3      | 1.000     | 0.903     | 0.949     |
| 50        | 26     | 0     | 5      | 1.000     | 0.839     | 0.912     |
| 100       | 25     | 0     | 6      | 1.000     | 0.806     | 0.893     |

**Which candidates are scored, and why it matters.** Every number in this
section -- the table above and the gap below -- is scored over the candidates
that reach a preset's top `max_lines`, because those are the only ones the
detector can place. Ordering by value and ordering by peak-to-median ratio are
the same ordering on one preset, since its band median is a constant, so raising
the threshold can only drop members of that set and never admit a new one.
`selected_candidates()` in the harness is that definition, and
`prominence_gap()` is the statistic; the tests import both rather than
recomputing them, so the page, the printout and the assertion cannot drift.

The two populations separate cleanly. The strongest selected candidate that is
_not_ a tabulated line scores **12.18** (the 15.02 Hz ripple of
`ET_15_full_cryo_psd`); the weakest selected tabulated line standing above that
floor scores **18.63** (the 15.02 Hz line of `ET_10_full_cryo_psd`). Any
threshold in between gives precision 1.000 at recall 0.935, and F1 is maximal
there.

`DEFAULT_LINE_PROMINENCE` is therefore **15.0**, the round value inside that
gap, where it was 4.0. `tests/test_anchored_quantities.py` asserts the gap
property directly, so a new or re-sampled preset that closed it would fail
rather than silently degrade the default.

**A different statistic, and not the one that governs.** Scoring _every_ local
maximum instead of only the selected ones gives a much larger floor: the
strongest non-line local maximum anywhere in the eleven presets is the 24.0096
Hz feature of `aLIGO_O3_actual_H1_psd` at **46.30**, which 15.0 does not clear.
That is not a contradiction, and it does not weaken the anchor: that feature is
nowhere near its own preset's top eight -- the O3 Hanford curve's eight selected
candidates score between 1.807e4 and 1.678e6 -- so no threshold can make the
detector place it. It would matter only if `max_lines` were unbounded. The
harness prints both numbers, each labelled with the population it was scored
over, and `test_the_unselected_statistic_is_recorded_as_a_different_one` pins
the distinction so the two cannot be quoted for each other.

**What raising it fixes.** At 4.0 the detector placed poles on the ripple of the
steep low-frequency rise of five presets, and those poles wreck the fit. At the
ARMA default orders, worst band residual with `detect_lines=True`, at the old
defaults (4.0 and 0.5 Hz) and the new ones (15.0 and 0.46 Hz):

| Preset                        | Old     | New   | With no lines placed |
| ----------------------------- | ------- | ----- | -------------------- |
| `aLIGO_O4_high_projected_psd` | 1.55e11 | 1.95  | 0.0645               |
| `aLIGO_O4_low_projected_psd`  | 11.7    | 2.29  | 0.0435               |
| `ET_D_psd`                    | 2.41e8  | 0.483 | 0.4314               |
| `ET_10_full_cryo_psd`         | 6.05e9  | 197   | 0.6399               |
| `ET_15_full_cryo_psd`         | 7.73e9  | 0.881 | 0.6677               |
| `ET_20_full_cryo_psd`         | 8.88e9  | 0.892 | 0.7003               |

`ET_10_full_cryo_psd` is the exception that proves the point of the previous
section. Its 15.02 Hz feature is a genuine tabulated line, it scores 18.63 and
is correctly kept, and the fit is still two orders of magnitude worse for having
a pole placed on it. Detection accuracy is not what limits pole placement on
these curves; the pole's own contrast is.

**What it costs.** The 60 Hz mains harmonic both O4 projections tabulate now
falls below the threshold on each of them, so it is left to the smooth fit
instead of being placed. It scores **6.40** on `aLIGO_O4_high_projected_psd`,
where the reference line list does carry it, and **4.28** on
`aLIGO_O4_low_projected_psd`, where it is weak enough that the reference
criterion does not label it a line at all -- so on that curve it is not a
tabulated line being given up, only a feature that would not have been placed
either way. That is the intended trade: fabricating a resonance where the target
has none is worse for a mock-data generator than under-resolving one it does
have, and a caller who wants the mains line can ask for it by name with
`line_frequencies=[60.0]`.

**What is still not anchored.** The statistic is a ratio to the median of the
_whole fitted band_, which is not a local prominence. On a target whose baseline
spans orders of magnitude, ripple on the steep low-frequency rise scores like a
narrow line on the flat mid-band floor, and the same feature scores differently
when the band edges move: the ET-D ripple at 14.85 and 21.17 Hz clears 20 when
the fit is restricted to 5-400 Hz at 1024 Hz, while it scores 6.2 and 10.4 over
the full band at 4096 Hz. **The threshold quoted here is anchored at the full
band up to the Nyquist frequency and does not carry its measured precision over
a narrower band.** Anchoring it band-independently would mean replacing the
statistic with a local one, which is a change to what the detector computes, not
to the value of a constant.

## 4. The ET-D lowest-band residual

The lowest of the eight geometric bands of an ET-D fit from 5 Hz,
`[5.000, 10.574) Hz`, carries a much larger relative error than any other band:
0.3203 for the AR fit at order 256 and 0.4314 for the ARMA fit at 192/32,
against 0.0730 and 0.0406 for the worst of the remaining seven. Three
explanations were separated by measurement.

**It is not a floor of the tabulated curve.** Refining the target grid the fit
interpolates onto, from 0.683 Hz to 0.085 Hz -- eight times finer, and 65
samples in the band instead of 8 -- moves the residual from 0.4314 to 0.4078, a
5 % change:

| Target grid | Lowest band |
| ----------- | ----------- |
| 0.6829 Hz   | 0.4314      |
| 0.3414 Hz   | 0.4225      |
| 0.1707 Hz   | 0.4148      |
| 0.0853 Hz   | 0.4078      |

The ET-D table is sampled at 0.0154 Hz near 5 Hz, some forty times finer than
the fit's own design grid, and the band-mean of the target on that design grid
is within 9 % of its band-mean on the native table. The table is not the limit.

**It is a floor of the method at the order used, and it is not a hard floor.**
Holding the target grid fixed at 0.085 Hz and sweeping the autoregressive order
alone:

| AR order | Lowest band |
| -------- | ----------- |
| 64       | 0.5103      |
| 128      | 0.4927      |
| 192      | 0.4078      |
| 256      | 0.3151      |
| 384      | 0.0900      |
| 512      | 0.0994      |
| 768      | 0.0019      |

The residual falls by more than two orders of magnitude with order, so nothing
in the data or the band definition is holding it up. The reason it is the last
band to be fitted is that it carries **1.4 % of the in-band target power** while
spanning a factor of 9 in PSD, and the Levinson recursion minimises total
prediction error: the model puts 0.68 of the target's power in that band while
matching the rest of the spectrum to better than a per cent.

**Its magnitude, though, is set by the band definition.** The same fit reported
through a different number of geometric bands gives:

| Bands | Lowest band | Worst band |
| ----- | ----------- | ---------- |
| 4     | 0.1651      | 0.1651     |
| 8     | 0.3203      | 0.3203     |
| 16    | 0.7123      | 0.7123     |
| 32    | 0.8536      | 0.8536     |

Narrower bands isolate the steepest part of the rise and report a larger error
for the same model. The figure "0.32" is therefore a statement about an
order-256 AR fit _reported in eight geometric bands_, and it is not comparable
to a residual quoted under another band count.

**The statement.** The ET-D lowest-band residual is a **finite-order floor of
the method**, reported through a band definition that sets its magnitude. It is
not a floor of the tabulated PSD. Raising the order removes it, at a
proportionate cost in state size and per-sample work; the shipped orders are a
deliberate trade, not a limit.
