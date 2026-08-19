#!/usr/bin/env python3
"""Does the follow-up excess depend on what was embolised?

A radiologist co-author raised the objection that decides this paper's reception:
disagreement after TACE is *expected*, because retained radiopaque ethiodized oil
is bright on CT and can obscure or mimic arterial enhancement, so of course the
post-treatment read is the harder one. If that is the whole story, the finding is
an artifact of one embolic agent rather than a property of response assessment.

HCC-TACE-Seg can answer this directly, because the two arms differ in exactly the
relevant way. Conventional TACE delivers chemotherapy in an ethiodized-oil
emulsion that is retained and radiopaque. Drug-eluting beads (LC Bead DEBDOX) are
radiolucent and leave no comparable attenuation. If oil attenuation drives the
asymmetry, the follow-up excess should be large under cTACE and absent, or much
smaller, under DEB-TACE.

Caveat carried into the manuscript: the data descriptor names the regimens but
never names ethiodized oil, so cTACE membership is inferred from the drug
combination, and 17 patients have no regimen recorded at all. The strata are
therefore small and the contrast is a check on a mechanism, not an estimate of a
treatment effect.

Run (from the repository root):
    .venv/bin/python analysis/src/embolic_stratification.py \
        --data data/raw --out results/embolic_stratification
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis" / "src"))

from measurement_gate import (  # noqa: E402
    RNG_SEED,
    READERS,
    _crossed_disagreement,
    _triple_complete,
    reader_frame,
    verify_inputs,
)


def tace_bucket(v: Any) -> str:
    """Regimen string -> embolic class.

    'doxorubicin LC beads' is the only bead regimen in the file; every other
    named regimen is a cisplatin/doxorubicin/mitomycin-C variant delivered by
    conventional technique. Missing stays missing and is never imputed.
    """
    if pd.isna(v):
        return "unrecorded"
    return "DEB-TACE" if "bead" in str(v).lower() else "cTACE"


def arm_stats(bl: np.ndarray, fu: np.ndarray, n_boot: int, rng: np.random.Generator
              ) -> dict[str, Any]:
    """Point estimate and patient bootstrap for one stratum."""
    stats = _crossed_disagreement(bl, fu)
    n = bl.shape[0]
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = _crossed_disagreement(bl[idx], fu[idx])["delta_orr_points"]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "n": int(n),
        "vary_followup_orr": stats["vary_followup_orr"],
        "vary_baseline_orr": stats["vary_baseline_orr"],
        "delta_orr_points": stats["delta_orr_points"],
        "delta_orr_ci95": [float(lo), float(hi)],
        "excludes_zero": bool(lo > 0.0),
        "_boots": boots,
    }


def stratify(hcc: pd.DataFrame, n_boot: int) -> dict[str, Any]:
    rf = reader_frame(hcc)
    rf["tace"] = hcc["chemotherapy"].map(tace_bucket).to_numpy()
    complete = _triple_complete(rf)

    bl_all = complete[[f"bl{r}" for r in READERS]].to_numpy(float)
    fu_all = complete[[f"fu{r}" for r in READERS]].to_numpy(float)
    tace = complete["tace"].to_numpy()

    rng = np.random.default_rng(RNG_SEED)
    out: dict[str, Any] = {
        "n_complete_readings": int(len(complete)),
        "strata_sizes": {k: int((tace == k).sum()) for k in
                         ("cTACE", "DEB-TACE", "unrecorded")},
    }

    arms: dict[str, dict[str, Any]] = {}
    for name in ("cTACE", "DEB-TACE", "unrecorded"):
        m = tace == name
        if m.sum() < 5:
            arms[name] = {"n": int(m.sum()), "skipped": "stratum too small"}
            continue
        arms[name] = arm_stats(bl_all[m], fu_all[m], n_boot, rng)

    # Contrast. Strata are disjoint patients, so resample each independently and
    # difference the paired draws; this is the CI on "oil explains the excess".
    a, b = arms["cTACE"], arms["DEB-TACE"]
    if "_boots" in a and "_boots" in b:
        diff = a["_boots"] - b["_boots"]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        out["cTACE_minus_DEB"] = {
            "difference_points": a["delta_orr_points"] - b["delta_orr_points"],
            "ci95": [float(lo), float(hi)],
            "excludes_zero": bool(lo > 0.0 or hi < 0.0),
        }

    # Lesion size by arm, to show the strata are not trivially different cohorts.
    bl_mean = complete[[f"bl{r}" for r in READERS]].to_numpy(float).mean(axis=1)
    fu_mean = complete[[f"fu{r}" for r in READERS]].to_numpy(float).mean(axis=1)
    out["baseline_diameter_mm_by_arm"] = {
        k: {"mean": float(bl_mean[tace == k].mean()),
            "median": float(np.median(bl_mean[tace == k])),
            "n": int((tace == k).sum())}
        for k in ("cTACE", "DEB-TACE", "unrecorded") if (tace == k).sum() > 0
    }
    out["followup_diameter_mm_by_arm"] = {
        k: {"mean": float(fu_mean[tace == k].mean()),
            "median": float(np.median(fu_mean[tace == k]))}
        for k in ("cTACE", "DEB-TACE", "unrecorded") if (tace == k).sum() > 0
    }
    out["complete_response_frac_by_arm"] = {
        k: float(np.mean((fu_mean == 0)[tace == k]))
        for k in ("cTACE", "DEB-TACE", "unrecorded") if (tace == k).sum() > 0
    }

    for v in arms.values():
        v.pop("_boots", None)
    out["arms"] = arms
    return out


def to_markdown(res: dict[str, Any]) -> str:
    L = ["# Embolic-agent stratification of the timepoint attribution", ""]
    L.append(f"Generated {res['generated_utc']}, seed {res['seed']}, "
             f"{res['n_boot']:,} bootstrap resamples.")
    L.append("")
    L.append("Question: is the follow-up excess an artifact of radiopaque ethiodized oil? "
             "If it were, it would be confined to the conventional-TACE arm, because "
             "drug-eluting beads are radiolucent.")
    L.append("")
    s = res["strata_sizes"]
    L.append(f"Of {res['n_complete_readings']} patients with complete readings: "
             f"{s['cTACE']} conventional, {s['DEB-TACE']} drug-eluting beads, "
             f"{s['unrecorded']} regimen unrecorded.")
    L.append("")
    L.append("| Stratum | n | Vary follow-up reader | Vary baseline reader | Excess (points) | 95% CI |")
    L.append("|---|---|---|---|---|---|")
    for k, v in res["arms"].items():
        if "skipped" in v:
            L.append(f"| {k} | {v['n']} | — | — | — | {v['skipped']} |")
            continue
        L.append(f"| {k} | {v['n']} | {v['vary_followup_orr']*100:.1f}% | "
                 f"{v['vary_baseline_orr']*100:.1f}% | {v['delta_orr_points']:+.1f} | "
                 f"{v['delta_orr_ci95'][0]:.1f} to {v['delta_orr_ci95'][1]:.1f} |")
    L.append("")
    if "cTACE_minus_DEB" in res:
        d = res["cTACE_minus_DEB"]
        L.append(f"**cTACE minus DEB-TACE: {d['difference_points']:+.1f} points "
                 f"({d['ci95'][0]:.1f} to {d['ci95'][1]:.1f}), "
                 f"{'excludes' if d['excludes_zero'] else 'includes'} zero.**")
        L.append("")
    L.append("| Arm | Baseline mean (mm) | Follow-up mean (mm) | Any-reader CR |")
    L.append("|---|---|---|---|")
    for k, v in res["baseline_diameter_mm_by_arm"].items():
        L.append(f"| {k} | {v['mean']:.1f} | "
                 f"{res['followup_diameter_mm_by_arm'][k]['mean']:.1f} | "
                 f"{res['complete_response_frac_by_arm'][k]*100:.1f}% |")
    L.append("")
    L.append("Regimen is inferred from the recorded drug combination; the data descriptor "
             "does not name ethiodized oil, and 17 of 105 patients have no regimen "
             "recorded. Strata are small and nothing is imputed.")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--boot", type=int, default=5000)
    args = ap.parse_args()

    provenance = verify_inputs(args.data)
    if not all(v["match"] for v in provenance.values()):
        raise SystemExit(f"CHECKSUM MISMATCH: {provenance}")

    hcc = pd.read_excel(args.data / "HCC-TACE-Seg_clinical_data-V2.xlsx",
                        sheet_name="data table")

    res = stratify(hcc, args.boot)
    res["generated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    res["seed"] = RNG_SEED
    res["n_boot"] = args.boot
    res["provenance"] = provenance

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "EMBOLIC_STRATIFICATION.json").write_text(json.dumps(res, indent=2))
    (args.out / "EMBOLIC_STRATIFICATION.md").write_text(to_markdown(res))
    print(to_markdown(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
