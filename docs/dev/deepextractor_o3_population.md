# The `deepextractor-o3-v1` glitch population

`deepextractor-o3-v1` injects the seven Gravity Spy classes the
[DeepExtractor dataset](https://huggingface.co/datasets/tomdooney/deepextractor-glitch-reconstructions)
reconstructs, each at its measured O3 rate and with its measured O3 SNR tail.
This page records how each registered number was measured and how well it
describes the data.

```bash
uv run python scripts/measure_glitch_population_anchors.py --population deepextractor-o3-v1
```

prints the measurement (about 560 MB of downloads on a cold cache);
`--write-summary` rewrites
`src/gwmock_noise/data/glitch_population/deepextractor_o3_summary.json`, against
which `tests/test_deepextractor_o3_population.py` asserts every registered
constant.

## Source data and selection

The data are those of [`et-o3-anchored-v1`](glitch_population.md#source-data):
the Gravity Spy O3a and O3b classifications (Zenodo record 5649212, v1.0.0) and
the GWOSC O3 observing segments, 42,952,865 s pooled over H1 and L1.

The selection differs in one rule. Every trigger labelled with the class above
the Omicron threshold of SNR 7.5 is counted, at any classification confidence. A
DeepExtractor class is defined by Gravity Spy's label, so its rate is the rate
of that label. A cut at confidence 0.9 would keep the following, counted in the
summary's `label_census`:

| Class               | All confidences | Confidence > 0.9 | Kept |
| ------------------- | --------------- | ---------------- | ---- |
| Blip                | 15,433          | 12,196           | 79 % |
| Fast_Scattering     | 85,330          | 63,617           | 75 % |
| Koi_Fish            | 26,661          | 18,593           | 70 % |
| Low_Frequency_Burst | 28,830          | 24,483           | 85 % |
| Scattered_Light     | 137,487         | 119,162          | 87 % |
| Tomte               | 53,613          | 45,258           | 84 % |
| Whistle             | 17,900          | 13,892           | 78 % |

## Rates

Each rate is the class's count over the pooled observing time, so each
interferometer glitches like an average O3 LIGO interferometer.

| Class               | Registered (Hz) | Per hour | H1 (Hz)  | L1 (Hz)  | H1/L1 |
| ------------------- | --------------- | -------- | -------- | -------- | ----- |
| Blip                | 3.5930e-4       | 1.29     | 4.351e-4 | 2.856e-4 | 1.52  |
| Fast_Scattering     | 1.9866e-3       | 7.15     | 1.292e-4 | 3.794e-3 | 0.034 |
| Koi_Fish            | 6.2070e-4       | 2.23     | 6.789e-4 | 5.641e-4 | 1.20  |
| Low_Frequency_Burst | 6.7120e-4       | 2.42     | 1.038e-3 | 3.139e-4 | 3.31  |
| Scattered_Light     | 3.2009e-3       | 11.52    | 3.556e-3 | 2.855e-3 | 1.25  |
| Tomte               | 1.2482e-3       | 4.49     | 1.316e-4 | 2.335e-3 | 0.056 |
| Whistle             | 4.1674e-4       | 1.50     | 4.105e-4 | 4.228e-4 | 0.97  |
| **Total**           | 8.5036e-3       | 30.61    |          |          |       |

The seven classes are 77 % of the O3 glitch rate of 1.104e-2 Hz, counted in the
summary's `label_census` over every label except `No_Glitch`, `Chirp` and
`None_of_the_Above`. The other fourteen classes have no DeepExtractor waveforms
and are absent.

The site ratios are the largest systematic: Fast_Scattering and Tomte are
Livingston phenomena, and the pooled rate is half the Livingston rate.

## SNR tails

Each class draws its target optimal SNR from a power law above 7.5, truncated at
the class's largest O3 SNR (rounded up to a hundredth), with survival exponent
α. α is the maximum-likelihood exponent of the truncated law. Its likelihood
equation states that the law's own draws have the Hill index the O3 sample has,
so the tests' check of that equality is a check on α and the cap, not an
independent test of the fit. For the steep classes the truncation barely matters
and α equals the Hill index; for Koi_Fish it matters most.

| Class               | α (registered) | Hill α | Cap       | Median, law | Median, O3 | Hill α above 30 |
| ------------------- | -------------- | ------ | --------- | ----------- | ---------- | --------------- |
| Blip                | 1.2613         | 1.2887 | 621.23    | 13.0        | 13.9       | 2.21            |
| Fast_Scattering     | 3.5052         | 3.5052 | 9,747.92  | 9.1         | 9.0        | 1.46            |
| Koi_Fish            | 0.2098         | 0.3610 | 11,841.49 | 81.2        | 114.0      | 0.65            |
| Low_Frequency_Burst | 1.3680         | 1.3714 | 2,781.22  | 12.4        | 12.4       | 1.50            |
| Scattered_Light     | 1.4139         | 1.4146 | 7,797.32  | 12.2        | 11.9       | 1.87            |
| Tomte               | 1.2492         | 1.3956 | 128.71    | 12.8        | 14.0       | 3.17            |
| Whistle             | 1.7077         | 1.7097 | 1,397.99  | 11.3        | 11.1       | 2.91            |

Only Koi_Fish's law median departs from the O3 median by more than 10 %: by −29
%. Tomte's departs by −9 %.

**The loud end.** The Hill index moves with threshold for every class, so one
power law from 7.5 upward does not describe it. Above SNR 150:

| Class               | O3 count | Law's count | Law / O3 |
| ------------------- | -------- | ----------- | -------- |
| Blip                | 22       | 295         | 13.4     |
| Fast_Scattering     | 105      | 2           | 0.0224   |
| Koi_Fish            | 10,505   | 10,848      | 1.03     |
| Low_Frequency_Burst | 430      | 470         | 1.09     |
| Scattered_Light     | 587      | 1,982       | 3.38     |
| Whistle             | 22       | 105         | 4.78     |

Tomte's largest O3 SNR is 128.7, so it has none. A study sensitive to loud
glitches should measure its own tail for the classes it cares about.

## What is not anchored

The population's `unanchored` field carries these, and so does its digest:

- the transfer from Advanced LIGO to another instrument, site and band;
- the site differences the pooled rate hides;
- the loud-end mismatch above;
- reading an Omicron SNR, measured against O3 LIGO noise over Omicron's band, as
  an optimal SNR against the configured noise curve and band;
- the fourteen absent classes, 23 % of the O3 glitch rate.

## The registered digest

With the default settings (`ET_10_full_cryo_psd`, every interferometer, 5 Hz,
the pinned dataset revision) the digest is pinned in
`tests/test_deepextractor_o3_population.py`. A run with its own settings pins
the digest those settings build, printed by
`get_glitch_population("deepextractor-o3-v1", **settings).digest()`.
