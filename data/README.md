# Source data

Three spreadsheets. All public. **None are committed** — `raw/` is git-ignored, because
redistributing the source files is the data providers' call, not this repository's. Obtain them
from the links below and drop them in `raw/` under exactly these filenames.

`analysis/src/measurement_gate.py` verifies every MD5 before it reads anything and aborts on
mismatch, so a wrong or updated file fails loudly rather than silently changing a result.

| File | MD5 | Bytes | Source |
|---|---|---|---|
| `HCC-TACE-Seg_clinical_data-V2.xlsx` | `63af6363b9d0fe453c48e75402e9082c` | 128,507 | [TCIA](https://www.cancerimagingarchive.net/wp-content/uploads/HCC-TACE-Seg_clinical_data-V2.xlsx) |
| `clinical_data_wawtace_v2_15_07_2024.xlsx` | `fb7aa2803eae6d75745203602b6d385a` | — | [Zenodo record 12741586](https://zenodo.org/records/12741586) |
| `supplementary_table_s1_definitions_v2.xlsx` | `fd9fb1cd1c7279882ad7edf6563b1a8d` | — | [Zenodo record 12741586](https://zenodo.org/records/12741586) |

Verify after downloading:

```bash
md5sum data/raw/*.xlsx
```

## What each file is for

- **HCC-TACE-Seg clinical data** — the primary analysis set. 105 patients, TACE at a single
  tertiary centre 2002–2012. Carries per-reader baseline and follow-up sums of viable
  (mRECIST) and total (RECIST) target-lesion diameters for three independent abdominal
  radiologists, the released category codes, and overall survival / time to progression.
  93 patients have complete readings from all three readers. TCIA DOI 10.7937/TCIA.5FNA-0924.

- **WAW-TACE clinical data** — a second public TACE cohort, read only by the secondary audits
  (`M009_waw_survival`, `M010_waw_score_audit`). It releases investigator assessment rather than
  replicate reader measurements, so it cannot support the timepoint attribution and is not used
  for it.

- **Supplementary table S1 definitions** — WAW-TACE variable definitions, used by the checksum
  gate for completeness. Not read by any analysis.

## Imaging

The analysis is entirely on the released tabular measurements. **No CT is needed.** The ~26.6 GB
HCC-TACE-Seg image collection and the ~48 GB WAW-TACE archives are not used and were never
downloaded.

## Provenance ledger

`../results/source_ledger.json` records the original fetches — URLs, HTTP status, byte counts and
SHA-256 — made by `hcc_gate0.py` in the `gi-cancers-proposal` repo on 2026-07-24. Its
`local_files` paths are relative to that repo's layout and are historical; the MD5s in it are the
ones enforced here.
