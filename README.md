# mRECIST reader reliability after TACE

Analysis code for a reader-agreement study of modified RECIST (mRECIST) in the public
HCC-TACE-Seg collection. The collection holds 105 patients with hepatocellular carcinoma
treated by transarterial chemoembolization, each with pre-treatment and follow-up multiphase
CT read independently by three abdominal radiologists. 93 patients have a complete
three-reader triplet and form the analysis set.

The code computes inter-reader agreement for the four-category mRECIST scale and for three
binary partitions of it, attributes categorical disagreement to one of the two timepoints the
criterion compares, and tests two alternative explanations for that attribution.

## Requirements

Python 3.11 or later and [uv](https://docs.astral.sh/uv/).

The source spreadsheets are not redistributed here. Download them from the collection and put
them in `data/raw/`. [`data/README.md`](data/README.md) lists the files, where they come from,
and their MD5s. Every script verifies those MD5s before doing anything and aborts on mismatch.

## Run

```bash
uv sync

uv run python analysis/tests/test_measurement_gate.py    # 75 assertions, prints "all tests passed"
uv run --with pytest pytest analysis/tests -q            # 39 tests

uv run python analysis/src/measurement_gate.py       --data data/raw --out results
uv run python analysis/src/size_confound.py          --data data/raw --out results/size_confound
uv run python analysis/src/embolic_stratification.py --data data/raw --out results/embolic_stratification
uv run python analysis/src/figures.py                --data data/raw --results results --out figures
```

Seed 20260803, 5,000 bootstrap resamples. Output is bit-identical across runs. The contents of
`results/` are committed, so the numbers can be checked without rerunning anything.

`test_measurement_gate.py` runs as a plain script as well as under pytest. `unittest discover`
collects none of it.

## Contents

```
analysis/src/measurement_gate.py         agreement coefficients, reader crossing, reference benchmark
analysis/src/size_confound.py            shrinkage controls
analysis/src/embolic_stratification.py   the same attribution within embolic class
analysis/src/figures.py                  figure rendering
analysis/tests/                          unit tests, including reverse-direction checks
results/                                 committed output of the four scripts
data/README.md                           provenance and checksums
figures/assets/                          CT tiles used in Figure 1, from the collection
```

## Method notes

Agreement is reported as Fleiss kappa and as Gwet's AC1, with AC2 for the ordinally weighted
four-category coefficient. AC1 is reported alongside kappa because the three partitions have
very different prevalence and kappa is sensitive to that.

Timepoint attribution crosses every observed baseline measurement with every observed follow-up
measurement for the same patient, giving nine combinations, and recomputes the mRECIST category
from each pair. No consensus or averaged value enters the comparison, so every input is a value
a radiologist actually recorded. Holding the baseline reader fixed while varying the follow-up
reader isolates the follow-up contribution, and the transpose isolates baseline. A
consensus-substitution variant, which replaces one timepoint by the three-reader mean, is
computed for comparison.

mRECIST thresholds the ratio (FU - BL) / BL, so the rule is arithmetically more sensitive to the
follow-up measurement once a lesion has shrunk. `size_confound.py` tests whether shrinkage alone
produces the asymmetry. It compares between-reader spread on the millimeter and logarithmic
scales, fits a reader-error law of the form var = tau^2 + sigma^2 * mu^2 and passes simulated
readers carrying that law, or a purely proportional or purely additive law, identically through
both timepoints, and repeats the crossing within tertiles of observed shrinkage.

`embolic_stratification.py` repeats the attribution separately in the two embolic classes, which
differ in radiopacity. The drug-eluting-bead arm is named explicitly in the collection
descriptor; the oil-based arm is inferred from the recorded drug combination, and 17 of the 105
released patients have no regimen recorded. The unit tests plant a between-arm difference and
require it to be recovered, so a null result cannot come from an insensitive estimator.

The reference-reader benchmark trains logistic-regression and random-forest models once against
the majority label, then scores the same fixed out-of-fold predictions against each reader in
turn. It measures how much a reported model score depends on the choice of reference reader, not
predictive utility.

## Data

HCC-TACE-Seg, The Cancer Imaging Archive, CC BY 4.0:
<https://doi.org/10.7937/TCIA.5FNA-0924>

The CT tiles in `figures/assets/` come from that collection and are used under the same license.

## License

MIT, see [LICENSE](LICENSE). The data are covered by the collection's own CC BY 4.0 license.
