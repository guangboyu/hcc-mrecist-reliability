#!/usr/bin/env python3
"""Measurement gate: every load-bearing number in the manuscript.

Reproduces the results behind "objective response is the least reliable mRECIST
partition after TACE, and the disagreement originates at follow-up" directly
from the checksummed clinical spreadsheets. Nothing here touches the network.

Provenance note. This file began as a SIBLING of ``hcc_gate0.py`` in the
gi-cancers-proposal repo rather than an edit to it: that file carries an
external code-review verdict ("APPROVE AS A BLOCKED GATE") which mutating it
would have invalidated. ``hcc_gate0.py`` performed the original network fetches
and wrote ``results/source_ledger.json``, which is copied here unchanged; it
stays in the proposal repo and is not needed to reproduce anything below.

Run (from the repository root):
    uv run python analysis/src/measurement_gate.py --data data/raw --out results
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

RNG_SEED = 20260803
READERS = (1, 2, 3)

# mRECIST category codes as released.
CR, PR, SD, PD = 1, 2, 3, 4

EXPECTED_MD5 = {
    "HCC-TACE-Seg_clinical_data-V2.xlsx": "63af6363b9d0fe453c48e75402e9082c",
    "clinical_data_wawtace_v2_15_07_2024.xlsx": "fb7aa2803eae6d75745203602b6d385a",
    "supplementary_table_s1_definitions_v2.xlsx": "fd9fb1cd1c7279882ad7edf6563b1a8d",
}


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------
def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_inputs(data_dir: Path) -> dict[str, Any]:
    out = {}
    for name, expected in EXPECTED_MD5.items():
        path = data_dir / name
        actual = md5_file(path)
        out[name] = {"md5": actual, "expected": expected, "match": actual == expected}
    return out


# --------------------------------------------------------------------------
# the mRECIST rule g
# --------------------------------------------------------------------------
def mrecist_rule(bl: float, fu: float, abs_pd_mm: float = 0.0) -> int | None:
    """Map one reader's (baseline, follow-up) viable-diameter sum to a category.

    ``abs_pd_mm`` optionally imposes the RECIST 1.1 absolute-increase rule
    (PD also needs >= 5 mm growth). Passing 0.0 disables it.
    """
    if bl is None or fu is None or not np.isfinite(bl) or not np.isfinite(fu):
        return None
    if fu == 0:
        return CR
    if bl <= 0:
        return None
    pct = (fu - bl) / bl * 100.0
    if pct <= -30.0:
        return PR
    if pct >= 20.0 and (fu - bl) >= abs_pd_mm:
        return PD
    return SD


def reader_frame(hcc: pd.DataFrame) -> pd.DataFrame:
    """Long-ish frame: one row per patient, BL/FU/category per reader."""
    out = pd.DataFrame({"TCIA_ID": hcc["TCIA_ID"]})
    for r in READERS:
        out[f"bl{r}"] = pd.to_numeric(hcc[f"{r}_mRECIST_BL"], errors="coerce")
        out[f"fu{r}"] = pd.to_numeric(hcc[f"{r}_mRECIST_FU"], errors="coerce")
        out[f"cat{r}"] = pd.to_numeric(hcc[f"{r}_mRECIST"], errors="coerce")
    return out


def check_rule_reproduction(rf: pd.DataFrame) -> dict[str, Any]:
    """Is the released category a deterministic function of the two numbers?"""
    res: dict[str, Any] = {}
    for abs_rule, tag in ((0.0, "no_abs_rule"), (5.0, "with_5mm_abs_rule")):
        per_reader = {}
        for r in READERS:
            sub = rf[rf[f"cat{r}"].notna() & rf[f"bl{r}"].notna() & rf[f"fu{r}"].notna()]
            derived = [mrecist_rule(b, f, abs_rule) for b, f in zip(sub[f"bl{r}"], sub[f"fu{r}"])]
            actual = sub[f"cat{r}"].astype(int).tolist()
            n_match = sum(int(d == a) for d, a in zip(derived, actual))
            mismatches = [
                {
                    "TCIA_ID": str(sub["TCIA_ID"].iloc[i]),
                    "bl": float(sub[f"bl{r}"].iloc[i]),
                    "fu": float(sub[f"fu{r}"].iloc[i]),
                    "pct": round((sub[f"fu{r}"].iloc[i] - sub[f"bl{r}"].iloc[i])
                                 / sub[f"bl{r}"].iloc[i] * 100.0, 3),
                    "derived": derived[i],
                    "released": actual[i],
                }
                for i in range(len(sub))
                if derived[i] != actual[i]
            ]
            per_reader[f"reader{r}"] = {
                "n": len(sub),
                "n_match": n_match,
                "mismatches": mismatches[:10],
            }
        res[tag] = per_reader
    return res


# --------------------------------------------------------------------------
# H1 - counterfactual harmonisation
# --------------------------------------------------------------------------
def disagreement_stats(cats: np.ndarray) -> dict[str, float]:
    """cats: (n, 3) integer categories, no NaN."""
    n = cats.shape[0]
    four_cat_disagree = float(np.mean([len(set(row)) > 1 for row in cats]))
    binary_orr = (cats <= PR).astype(int)  # CR or PR = objective response
    orr_disagree = float(np.mean([len(set(row)) > 1 for row in binary_orr]))
    is_cr = (cats == CR).astype(int)
    n_cr = is_cr.sum(axis=1)
    split_cr = float(np.mean((n_cr > 0) & (n_cr < 3)))
    return {
        "n": n,
        "four_cat_disagreement": four_cat_disagree,
        "binary_orr_disagreement": orr_disagree,
        "split_cr_fraction": split_cr,
    }


def harmonisation_counterfactual(rf: pd.DataFrame, n_boot: int = 5000) -> dict[str, Any]:
    """Replace BL (or FU) with the 3-reader mean; recompute categories."""
    complete = rf.dropna(
        subset=[f"{p}{r}" for r in READERS for p in ("bl", "fu", "cat")]
    ).reset_index(drop=True)

    bl = complete[[f"bl{r}" for r in READERS]].to_numpy(float)
    fu = complete[[f"fu{r}" for r in READERS]].to_numpy(float)
    bl_mean = bl.mean(axis=1, keepdims=True)
    fu_mean = fu.mean(axis=1, keepdims=True)

    def cats_from(bl_m: np.ndarray, fu_m: np.ndarray) -> np.ndarray:
        out = np.empty(bl_m.shape, dtype=int)
        for i in range(bl_m.shape[0]):
            for j in range(bl_m.shape[1]):
                out[i, j] = mrecist_rule(bl_m[i, j], fu_m[i, j])
        return out

    observed = cats_from(bl, fu)
    bl_harm = cats_from(np.repeat(bl_mean, 3, axis=1), fu)   # readers differ only at FU
    fu_harm = cats_from(bl, np.repeat(fu_mean, 3, axis=1))   # readers differ only at BL

    scen = {
        "observed": disagreement_stats(observed),
        "baseline_harmonised": disagreement_stats(bl_harm),
        "followup_harmonised": disagreement_stats(fu_harm),
    }

    # Sanity: the released categories should equal `observed` where the rule holds.
    released = complete[[f"cat{r}" for r in READERS]].to_numpy(int)
    scen["released_vs_derived_identical"] = bool((released == observed).all())
    scen["released"] = disagreement_stats(released)

    # H1 effect: how much MORE does FU-harmonisation reduce disagreement than BL?
    obs = scen["observed"]["four_cat_disagreement"]
    delta_point = ((obs - scen["followup_harmonised"]["four_cat_disagreement"])
                   - (obs - scen["baseline_harmonised"]["four_cat_disagreement"])) * 100.0

    rng = np.random.default_rng(RNG_SEED)
    n = bl.shape[0]
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        o = disagreement_stats(observed[idx])["four_cat_disagreement"]
        b = disagreement_stats(bl_harm[idx])["four_cat_disagreement"]
        f = disagreement_stats(fu_harm[idx])["four_cat_disagreement"]
        boots.append(((o - f) - (o - b)) * 100.0)
    lo, hi = np.percentile(boots, [2.5, 97.5])

    scen["H1_delta_points"] = delta_point
    scen["H1_delta_ci95"] = [float(lo), float(hi)]
    scen["H1_passes_15pt_threshold"] = bool(lo > 15.0)
    return scen


# --------------------------------------------------------------------------
# kappa by dichotomy
# --------------------------------------------------------------------------
DICHOTOMIES = {
    "CR_vs_nonCR": lambda c: (c == CR).astype(int),
    "objective_response_CRPR_vs_SDPD": lambda c: (c <= PR).astype(int),
    "PD_vs_nonPD": lambda c: (c == PD).astype(int),
}


def fleiss_kappa(mat: np.ndarray, q: int) -> float:
    """mat: (n subjects, k raters) integer codes in [0, q)."""
    n, k = mat.shape
    cnt = np.zeros((n, q))
    for i in range(n):
        for j in range(k):
            cnt[i, int(mat[i, j])] += 1
    p = cnt.sum(axis=0) / (n * k)
    agree = ((cnt ** 2).sum(axis=1) - k) / (k * (k - 1))
    pe = (p ** 2).sum()
    return float((agree.mean() - pe) / (1 - pe))


def gwet_ac(mat: np.ndarray, q: int, weights: np.ndarray | None = None) -> float:
    """Gwet's AC1 (unweighted) or AC2 (weighted). mat: (n subjects, k raters)."""
    n, k = mat.shape
    cnt = np.zeros((n, q))
    for i in range(n):
        for j in range(k):
            cnt[i, int(mat[i, j])] += 1
    W = np.eye(q) if weights is None else weights
    pa = float(np.mean([
        (cnt[i] @ W @ cnt[i] - (cnt[i] * np.diag(W)).sum()) / (k * (k - 1))
        for i in range(n)
    ]))
    pi = cnt.sum(axis=0) / (n * k)
    pe = float((W.sum() / (q * (q - 1))) * np.sum(pi * (1 - pi)))
    return float((pa - pe) / (1 - pe))


def ordinal_weights(q: int) -> np.ndarray:
    """Gwet ordinal weights: 1 - C(|i-j|+1, 2) / C(q, 2)."""
    denom = q * (q - 1) / 2
    return np.array([[1 - ((abs(i - j) + 1) * abs(i - j) / 2) / denom
                      for j in range(q)] for i in range(q)])


def multirater_agreement(rf: pd.DataFrame, n_boot: int = 5000) -> dict[str, Any]:
    """Fleiss kappa + Gwet AC for the 4-category scale and each dichotomy.

    The weighted-vs-unweighted contrast on the 4-category scale is what
    reconciles this analysis with previously published agreement on the same
    readings, so both are reported explicitly.

    Every coefficient carries a percentile bootstrap CI over patients. The prior
    report on these readings publishes CIs on its coefficients, so a comparison
    against it needs them on both sides.
    """
    complete = rf.dropna(subset=[f"cat{r}" for r in READERS])
    cats = complete[[f"cat{r}" for r in READERS]].to_numpy(int)
    n = cats.shape[0]
    out: dict[str, Any] = {"n": int(n)}

    # every coefficient as a callable on a (n,3) category matrix, so the same
    # resamples drive the point estimates and the intervals
    stats: dict[str, dict[str, Any]] = {
        "four_category": {
            "fleiss_kappa_unweighted": lambda c: fleiss_kappa(c - 1, 4),
            "gwet_ac1_unweighted": lambda c: gwet_ac(c - 1, 4),
            "gwet_ac2_ordinal_weighted": lambda c: gwet_ac(c - 1, 4, ordinal_weights(4)),
        }
    }
    for name, fn in DICHOTOMIES.items():
        stats[name] = {
            "prevalence": lambda c, f=fn: float(f(c).mean()),
            "fleiss_kappa": lambda c, f=fn: fleiss_kappa(f(c), 2),
            "gwet_ac1": lambda c, f=fn: gwet_ac(f(c), 2),
            "at_least_one_discordant": lambda c, f=fn: float(
                np.mean([len(set(r)) > 1 for r in f(c)])
            ),
        }

    rng = np.random.default_rng(RNG_SEED)
    resamples = [cats[rng.integers(0, n, n)] for _ in range(n_boot)]
    draws: dict[str, dict[str, np.ndarray]] = {}
    for block, fns in stats.items():
        out[block] = {}
        draws[block] = {}
        for label, fn in fns.items():
            out[block][label] = fn(cats)
            b = np.asarray([fn(r) for r in resamples], dtype=float)
            draws[block][label] = b
            finite = b[np.isfinite(b)]
            # A coefficient is undefined on a resample containing none of the
            # rare category (PD prevalence here is 0.039). Recorded rather than
            # silently dropped, because it tells the reader how thin that
            # partition is. If a category is absent from the cohort entirely,
            # no resample defines it and there is no interval to report.
            out[block][f"{label}_ci95"] = (
                [float(x) for x in np.percentile(finite, [2.5, 97.5])]
                if finite.size else None
            )
            if finite.size < b.size:
                out[block][f"{label}_undefined_resamples"] = int(b.size - finite.size)

    # Paired contrasts. The partitions are computed on the SAME patients, so
    # comparing them by whether their marginal intervals overlap is the wrong
    # test and understates the evidence. These are bootstrapped differences.
    def paired(a_block: str, a_key: str, b_block: str, b_key: str) -> dict[str, Any]:
        d = draws[a_block][a_key] - draws[b_block][b_key]
        d = d[np.isfinite(d)]
        res: dict[str, Any] = {
            "difference": float(out[a_block][a_key] - out[b_block][b_key]),
            "n_defined_resamples": int(d.size),
        }
        if d.size:
            lo, hi = np.percentile(d, [2.5, 97.5])
            res["ci95"] = [float(lo), float(hi)]
            res["excludes_zero"] = bool(lo > 0.0 or hi < 0.0)
        else:
            res["ci95"] = None
            res["excludes_zero"] = False
        return res

    orr = "objective_response_CRPR_vs_SDPD"
    out["partition_contrasts"] = {
        # DECOMPOSITION. The reassuring published coefficient and the endpoint
        # coefficient differ in three ways at once: weighting, coefficient
        # family, and partition. Reporting only the combined gap attributes to
        # the partition an effect that is mostly weighting. Each step below
        # varies exactly one of the three.
        "step1_weighting_ac2_minus_ac1_four_category": paired(
            "four_category", "gwet_ac2_ordinal_weighted",
            "four_category", "gwet_ac1_unweighted"),
        "step2_family_ac1_minus_fleiss_four_category": paired(
            "four_category", "gwet_ac1_unweighted",
            "four_category", "fleiss_kappa_unweighted"),
        "step3_partition_fleiss_four_category_minus_orr": paired(
            "four_category", "fleiss_kappa_unweighted", orr, "fleiss_kappa"),
        "step3_partition_ac1_four_category_minus_orr": paired(
            "four_category", "gwet_ac1_unweighted", orr, "gwet_ac1"),
        # Within-cardinality: the only comparisons that hold family, weighting
        # AND cardinality fixed, so the only clean partition contrasts.
        "cr_kappa_minus_orr_kappa": paired(
            "CR_vs_nonCR", "fleiss_kappa", orr, "fleiss_kappa"),
        "cr_ac1_minus_orr_ac1": paired(
            "CR_vs_nonCR", "gwet_ac1", orr, "gwet_ac1"),
        "pd_kappa_minus_orr_kappa": paired(
            "PD_vs_nonPD", "fleiss_kappa", orr, "fleiss_kappa"),
        "pd_ac1_minus_orr_ac1": paired(
            "PD_vs_nonPD", "gwet_ac1", orr, "gwet_ac1"),
        # Retained for transparency only. NOT a partition contrast: it varies
        # weighting, family and partition simultaneously.
        "combined_reporting_gap_ac2_minus_orr_kappa": paired(
            "four_category", "gwet_ac2_ordinal_weighted", orr, "fleiss_kappa"),
    }
    out["partition_contrast_note"] = (
        "combined_reporting_gap_ac2_minus_orr_kappa varies weighting, coefficient "
        "family and partition at once and must not be cited as a partition effect. "
        "Use step1/step2/step3 for the decomposition and the within-cardinality "
        "contrasts for the partition claim."
    )
    return out


# --------------------------------------------------------------------------
# cohort characteristics
# --------------------------------------------------------------------------
CHARACTERISTICS_CATEGORICAL = [
    "Sex", "hepatitis", "Evidence_of_cirh", "CPS", "BCLC", "AFP_group",
    "tumor_nodul", "T_involvment", "Portal Vein Thrombosis", "Vascular invasion",
    "Metastasis", "CLIP", "Okuda",
]


def cohort_characteristics(hcc: pd.DataFrame, rf: pd.DataFrame) -> dict[str, Any]:
    """Patient, disease and reading characteristics, for the manuscript's Table 1.

    Reported for the full released collection and for the triple-complete
    analysis set, so a reader can see whether restricting to complete readings
    selected a different population.
    """
    analysis_ids = set(_triple_complete(rf)["TCIA_ID"])
    sets = {
        "full_collection": hcc,
        "analysis_set": hcc[hcc["TCIA_ID"].isin(analysis_ids)],
    }

    out: dict[str, Any] = {}
    for sname, df in sets.items():
        block: dict[str, Any] = {"n": int(len(df))}
        age = pd.to_numeric(df["age"], errors="coerce").dropna()
        block["age_years"] = {
            "median": float(age.median()),
            "q1": float(age.quantile(0.25)),
            "q3": float(age.quantile(0.75)),
        }
        for col in CHARACTERISTICS_CATEGORICAL:
            if col not in df.columns:
                continue
            counts = df[col].value_counts(dropna=False)
            block[col] = {
                str(k): {"n": int(v), "pct": float(v / len(df) * 100.0)}
                for k, v in counts.items()
            }
        os_ = pd.to_numeric(df["OS"], errors="coerce").dropna()
        block["overall_survival_weeks"] = {
            "median": float(os_.median()),
            "q1": float(os_.quantile(0.25)),
            "q3": float(os_.quantile(0.75)),
        }
        block["deaths"] = int(
            pd.to_numeric(df["Death_1_StillAliveorLostToFU_0"], errors="coerce").sum()
        )
        out[sname] = block

    # released mRECIST category distribution, per reader, on the analysis set
    complete = _triple_complete(rf)
    per_reader: dict[str, Any] = {}
    for r in READERS:
        c = complete[f"cat{r}"].astype(int)
        per_reader[f"reader{r}"] = {
            name: {"n": int((c == code).sum()),
                   "pct": float((c == code).mean() * 100.0)}
            for name, code in (("CR", CR), ("PR", PR), ("SD", SD), ("PD", PD))
        }
        per_reader[f"reader{r}"]["objective_response"] = {
            "n": int((c <= PR).sum()), "pct": float((c <= PR).mean() * 100.0)
        }
    out["mrecist_category_by_reader"] = per_reader
    return out


def structural_zero_robustness(rf: pd.DataFrame, n_boot: int = 5000) -> dict[str, Any]:
    """H1 re-run under the two prespecified structural-zero controls.

    Complete response is defined by follow-up == 0, so baseline harmonisation
    cannot alter a CR call by construction. These arms remove that component.
    """
    complete = rf.dropna(
        subset=[f"{p}{r}" for r in READERS for p in ("bl", "fu", "cat")]
    ).reset_index(drop=True)
    cats = complete[[f"cat{r}" for r in READERS]].to_numpy(int)

    out: dict[str, Any] = {}

    # Arm 1: drop every patient with any CR call.
    keep = (cats == CR).sum(axis=1) == 0
    sub = complete[keep].reset_index(drop=True)
    r1 = harmonisation_counterfactual(sub, n_boot)
    out["exclude_any_cr"] = {
        "n": int(keep.sum()),
        "n_excluded": int((~keep).sum()),
        "H1_delta_points": r1["H1_delta_points"],
        "H1_delta_ci95": r1["H1_delta_ci95"],
    }

    # Arm 2: replace follow-up zeros with 1 mm.
    alt = complete.copy()
    for r in READERS:
        alt[f"fu{r}"] = alt[f"fu{r}"].replace(0.0, 1.0)
        alt[f"cat{r}"] = [mrecist_rule(b, f) for b, f in zip(alt[f"bl{r}"], alt[f"fu{r}"])]
    r2 = harmonisation_counterfactual(alt, n_boot)
    out["zeros_to_1mm"] = {
        "n": int(len(alt)),
        "H1_delta_points": r2["H1_delta_points"],
        "H1_delta_ci95": r2["H1_delta_ci95"],
    }

    # Is the CR->0 split-CR result tautological? Verify directly.
    bl = complete[[f"bl{r}" for r in READERS]].to_numpy(float)
    fu = complete[[f"fu{r}" for r in READERS]].to_numpy(float)
    bl_harm = np.empty(bl.shape, dtype=int)
    bl_mean = bl.mean(axis=1, keepdims=True)
    for i in range(bl.shape[0]):
        for j in range(3):
            bl_harm[i, j] = mrecist_rule(bl_mean[i, 0], fu[i, j])
    out["split_cr_is_tautological"] = bool(
        np.array_equal(cats == CR, bl_harm == CR)
    )

    # Within-patient log variance by timepoint, positive values only.
    pos = (fu > 0).all(axis=1)
    vb = float(np.log(bl[pos]).var(axis=1, ddof=1).mean())
    vf = float(np.log(fu[pos]).var(axis=1, ddof=1).mean())
    rng = np.random.default_rng(RNG_SEED)
    m = int(pos.sum())
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, m, m)
        boots.append(np.log(fu[pos][idx]).var(axis=1, ddof=1).mean()
                     - np.log(bl[pos][idx]).var(axis=1, ddof=1).mean())
    out["log_variance_by_timepoint"] = {
        "n": m,
        "baseline": vb,
        "followup": vf,
        "difference": vf - vb,
        "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
    }
    return out


def kappa_table(rf: pd.DataFrame) -> dict[str, Any]:
    complete = rf.dropna(subset=[f"cat{r}" for r in READERS])
    cats = complete[[f"cat{r}" for r in READERS]].to_numpy(int)
    out: dict[str, Any] = {"n": int(cats.shape[0])}

    pairs = [(0, 1), (0, 2), (1, 2)]
    # 4-category, linear-weighted ordinal
    ordinal = {}
    for a, b in pairs:
        ordinal[f"reader{a+1}_reader{b+1}"] = float(
            cohen_kappa_score(cats[:, a], cats[:, b], weights="linear")
        )
    out["ordinal_linear_weighted"] = ordinal

    for name, fn in DICHOTOMIES.items():
        d = fn(cats)
        entry = {"prevalence": float(d.mean())}
        ks, aggs = [], []
        for a, b in pairs:
            ks.append(float(cohen_kappa_score(d[:, a], d[:, b])))
            aggs.append(float((d[:, a] == d[:, b]).mean()))
        entry["kappa_by_pair"] = dict(zip(["r1_r2", "r1_r3", "r2_r3"], ks))
        entry["kappa_range"] = [min(ks), max(ks)]
        entry["mean_observed_agreement"] = float(np.mean(aggs))
        out[name] = entry
    return out


# --------------------------------------------------------------------------
# H2 - co-reader oracle ceiling (assumption-free)
# --------------------------------------------------------------------------
def coreader_oracle(rf: pd.DataFrame) -> dict[str, Any]:
    complete = rf.dropna(
        subset=[f"{p}{r}" for r in READERS for p in ("bl", "fu", "cat")]
    ).reset_index(drop=True)
    bl = complete[[f"bl{r}" for r in READERS]].to_numpy(float)
    fu = complete[[f"fu{r}" for r in READERS]].to_numpy(float)
    cats = complete[[f"cat{r}" for r in READERS]].to_numpy(int)

    res: dict[str, Any] = {"n": int(bl.shape[0])}
    per_reader = {}
    for held in range(3):
        others = [j for j in range(3) if j != held]
        pred = np.array([
            mrecist_rule(bl[i, others].mean(), fu[i, others].mean())
            for i in range(bl.shape[0])
        ])
        truth = cats[:, held]
        acc4 = float((pred == truth).mean())
        accb = float(((pred <= PR) == (truth <= PR)).mean())
        per_reader[f"reader{held+1}"] = {
            "four_cat_accuracy": acc4,
            "binary_orr_accuracy": accb,
        }
    res["per_held_out_reader"] = per_reader
    res["mean_four_cat_accuracy"] = float(
        np.mean([v["four_cat_accuracy"] for v in per_reader.values()])
    )
    res["mean_binary_orr_accuracy"] = float(
        np.mean([v["binary_orr_accuracy"] for v in per_reader.values()])
    )

    # Majority-class baseline for the binary ORR task, pooled over readers.
    binary = (cats <= PR).astype(int)
    prev = float(binary.mean())
    res["binary_orr_prevalence"] = prev
    res["majority_class_baseline"] = max(prev, 1 - prev)
    res["oracle_gain_over_majority"] = res["mean_binary_orr_accuracy"] - max(prev, 1 - prev)
    return res


# --------------------------------------------------------------------------
# ICC of the continuous measurements
# --------------------------------------------------------------------------
def icc_two_way(mat: np.ndarray) -> dict[str, float]:
    """mat: (n subjects, k readers). Returns ICC(2,1) absolute + ICC(3,1) consistency."""
    n, k = mat.shape
    grand = mat.mean()
    ms_rows = k * ((mat.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    ms_cols = n * ((mat.mean(axis=0) - grand) ** 2).sum() / (k - 1)
    resid = mat - mat.mean(axis=1, keepdims=True) - mat.mean(axis=0, keepdims=True) + grand
    ms_err = (resid ** 2).sum() / ((n - 1) * (k - 1))
    icc21 = (ms_rows - ms_err) / (ms_rows + (k - 1) * ms_err + k * (ms_cols - ms_err) / n)
    icc31 = (ms_rows - ms_err) / (ms_rows + (k - 1) * ms_err)
    return {"ICC_2_1_absolute": float(icc21), "ICC_3_1_consistency": float(icc31)}


def measurement_variability(rf: pd.DataFrame) -> dict[str, Any]:
    complete = rf.dropna(
        subset=[f"{p}{r}" for r in READERS for p in ("bl", "fu", "cat")]
    ).reset_index(drop=True)
    bl = complete[[f"bl{r}" for r in READERS]].to_numpy(float)
    fu = complete[[f"fu{r}" for r in READERS]].to_numpy(float)
    cats = complete[[f"cat{r}" for r in READERS]].to_numpy(int)

    out: dict[str, Any] = {"n": int(bl.shape[0])}
    out["baseline_log_icc"] = icc_two_way(np.log(bl))
    out["baseline_within_patient_cv"] = float(
        np.mean(bl.std(axis=1, ddof=1) / bl.mean(axis=1))
    )

    # Follow-up ICC restricted to patients where NO reader called CR (log needs > 0).
    no_cr = (cats == CR).sum(axis=1) == 0
    fu_nc = fu[no_cr]
    out["followup_no_cr_n"] = int(no_cr.sum())
    out["followup_log_icc_no_cr"] = icc_two_way(np.log(fu_nc))
    out["followup_within_patient_cv_no_cr"] = float(
        np.mean(fu_nc.std(axis=1, ddof=1) / fu_nc.mean(axis=1))
    )

    # Spread of the between-reader ratio, mRECIST (viable) vs RECIST (total).
    ratio_bl = bl.max(axis=1) / bl.min(axis=1)
    out["baseline_ratio_ge_1p5_frac"] = float((ratio_bl >= 1.5).mean())
    out["baseline_ratio_ge_2_frac"] = float((ratio_bl >= 2.0).mean())
    out["baseline_ratio_max"] = float(ratio_bl.max())
    out["baseline_ratio_median"] = float(np.median(ratio_bl))

    # Threshold fragility: mean % change within one between-reader SD of a cutpoint.
    pct = (fu - bl) / bl * 100.0
    mean_pct = pct.mean(axis=1)
    sd_pct = pct.std(axis=1, ddof=1)
    near = (np.abs(mean_pct - (-30.0)) <= sd_pct) | (np.abs(mean_pct - 20.0) <= sd_pct)
    out["threshold_fragile_fraction"] = float(near.mean())
    return out


def recist_ratio_spread(hcc: pd.DataFrame) -> dict[str, Any]:
    cols_bl = [f"{r}_RECIST_BL" for r in READERS]
    sub = hcc[cols_bl].apply(pd.to_numeric, errors="coerce").dropna()
    m = sub.to_numpy(float)
    ratio = m.max(axis=1) / m.min(axis=1)
    return {
        "n": int(m.shape[0]),
        "ratio_ge_1p5_count": int((ratio >= 1.5).sum()),
        "ratio_ge_1p5_frac": float((ratio >= 1.5).mean()),
        "ratio_median": float(np.median(ratio)),
    }


# --------------------------------------------------------------------------
# EASL coding + TACE-type crosstab (the two fact corrections)
# --------------------------------------------------------------------------
def easl_audit(hcc: pd.DataFrame) -> dict[str, Any]:
    easl_cols = [c for c in hcc.columns if "EASL" in c]
    e = pd.to_numeric(hcc["1_EASL"], errors="coerce")
    fu = pd.to_numeric(hcc["1_EASL_FU"], errors="coerce")
    both = hcc[e.notna() & fu.notna()]
    e_b = pd.to_numeric(both["1_EASL"], errors="coerce")
    fu_b = pd.to_numeric(both["1_EASL_FU"], errors="coerce")
    return {
        "easl_columns_present": easl_cols,
        "n_readers_with_easl": len({c.split("_")[0] for c in easl_cols}),
        "value_counts_1_EASL": {str(k): int(v) for k, v in e.value_counts().items()},
        "n_distinct_categories": int(e.nunique()),
        "code_3_always_fu_zero": bool((fu_b[e_b == 3] == 0).all()),
        "code_4_never_fu_zero": bool((fu_b[e_b == 4] != 0).all()),
        "mrecist_uses_code_1_for_CR": bool(
            (pd.to_numeric(hcc["1_mRECIST"], errors="coerce") == 1).sum() > 0
        ),
    }


def tace_type_crosstab(hcc: pd.DataFrame) -> dict[str, Any]:
    chemo = hcc["chemotherapy"].astype("string").str.strip()

    def bucket(v: Any) -> str:
        if pd.isna(v):
            return "MISSING"
        s = str(v).lower()
        if "bead" in s:
            return "DEB-TACE"
        return "cTACE"

    grp = chemo.map(bucket)
    bclc = hcc["BCLC"].astype("string").str.strip()
    ct = pd.crosstab(grp, bclc)
    size = pd.to_numeric(hcc["Tr_Size"], errors="coerce")
    size_by_bclc = (
        pd.DataFrame({"bclc": bclc, "has_size": size.notna()})
        .groupby("bclc")["has_size"].agg(["sum", "count"])
    )
    return {
        "chemotherapy_raw_counts": {str(k): int(v) for k, v in chemo.value_counts(dropna=False).items()},
        "crosstab_tace_type_by_bclc": json.loads(ct.to_json(orient="index")),
        "n_missing_tace_type": int((grp == "MISSING").sum()),
        "missing_by_bclc": {
            str(k): int(v) for k, v in bclc[grp == "MISSING"].value_counts().items()
        },
        "Tr_Size_available_by_bclc": {
            str(idx): {"n_with_size": int(row["sum"]), "n_total": int(row["count"])}
            for idx, row in size_by_bclc.iterrows()
        },
    }


# --------------------------------------------------------------------------
# Cox proportional hazards (Breslow ties, Newton-Raphson)
# --------------------------------------------------------------------------
def cox_fit(X: np.ndarray, time: np.ndarray, event: np.ndarray,
            tol: float = 1e-9, max_iter: int = 100) -> dict[str, Any]:
    X = np.asarray(X, float)
    n, p = X.shape
    beta = np.zeros(p)
    order = np.argsort(time)
    X, time, event = X[order], time[order], event[order]

    for _ in range(max_iter):
        eta = X @ beta
        w = np.exp(eta)
        grad = np.zeros(p)
        hess = np.zeros((p, p))
        loglik = 0.0
        # Risk set = all j with time_j >= time_i. Iterate descending for cumulative sums.
        for t in np.unique(time[event == 1]):
            at_risk = time >= t
            dies = (time == t) & (event == 1)
            d = int(dies.sum())
            wr = w[at_risk]
            Xr = X[at_risk]
            s0 = wr.sum()
            s1 = (wr[:, None] * Xr).sum(axis=0)
            s2 = (wr[:, None, None] * Xr[:, :, None] * Xr[:, None, :]).sum(axis=0)
            xbar = s1 / s0
            loglik += X[dies].sum(axis=0) @ beta - d * np.log(s0)
            grad += X[dies].sum(axis=0) - d * xbar
            hess -= d * (s2 / s0 - np.outer(xbar, xbar))
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            return {"converged": False}
        beta_new = beta - step
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    cov = np.linalg.inv(-hess)
    se = np.sqrt(np.diag(cov))
    from scipy.stats import norm
    z = beta / se
    pvals = 2 * (1 - norm.cdf(np.abs(z)))
    return {
        "converged": True,
        "beta": beta.tolist(),
        "se": se.tolist(),
        "hr": np.exp(beta).tolist(),
        "hr_ci_lo": np.exp(beta - 1.96 * se).tolist(),
        "hr_ci_hi": np.exp(beta + 1.96 * se).tolist(),
        "p": pvals.tolist(),
        "loglik": float(loglik),
        "n": int(n),
        "events": int(event.sum()),
    }


def concordance(risk: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
    """Harrell's C. Higher risk should mean shorter survival."""
    num = den = 0.0
    n = len(time)
    for i in range(n):
        if event[i] != 1:
            continue
        for j in range(n):
            if time[j] <= time[i]:
                continue
            den += 1
            if risk[i] > risk[j]:
                num += 1
            elif risk[i] == risk[j]:
                num += 0.5
    return num / den if den else float("nan")


# --------------------------------------------------------------------------
# survival analyses (H4)
# --------------------------------------------------------------------------
def hcc_survival(hcc: pd.DataFrame, rf: pd.DataFrame) -> dict[str, Any]:
    df = hcc.copy()
    df["os"] = pd.to_numeric(df["OS"], errors="coerce")
    df["death"] = pd.to_numeric(df["Death_1_StillAliveorLostToFU_0"], errors="coerce")
    cats = rf[[f"cat{r}" for r in READERS]]
    n_cr = (cats == CR).sum(axis=1)
    df["n_readers_cr"] = n_cr.where(cats.notna().all(axis=1))
    df["majority_cr"] = (df["n_readers_cr"] >= 2).astype(float)
    df.loc[df["n_readers_cr"].isna(), "majority_cr"] = np.nan

    bclc = df["BCLC"].astype("string").str.strip()
    df["bclc_cd"] = bclc.isin(["Stage-C", "Stage-D", "C", "D"]).astype(float)
    df["log_afp"] = np.log(pd.to_numeric(df["AFP"], errors="coerce").clip(lower=1e-6))
    df["afp_ge400"] = (pd.to_numeric(df["AFP"], errors="coerce") >= 400).astype(float)
    df["log_size"] = np.log(pd.to_numeric(df["Tr_Size"], errors="coerce").clip(lower=1e-6))

    out: dict[str, Any] = {}

    # Median OS by number of readers calling CR.
    grad = {}
    for k in range(4):
        sel = df["n_readers_cr"] == k
        if sel.sum():
            grad[str(k)] = {
                "n": int(sel.sum()),
                "median_os_weeks": float(df.loc[sel, "os"].median()),
                "deaths": int(df.loc[sel, "death"].sum()),
            }
    out["median_os_by_n_readers_cr"] = grad

    # Adjusted Cox: majority CR + BCLC C/D + AFP>=400 (log size dropped: only 53/105).
    sub = df.dropna(subset=["os", "death", "majority_cr", "bclc_cd", "afp_ge400"])
    X = sub[["majority_cr", "bclc_cd", "afp_ge400"]].to_numpy(float)
    fit = cox_fit(X, sub["os"].to_numpy(float), sub["death"].to_numpy(int))
    out["cox_majority_cr_adjusted"] = {
        "terms": ["majority_cr", "bclc_cd", "afp_ge400"], **fit
    }

    # Per-reader CR, same adjustment.
    per_reader = {}
    for r in READERS:
        d2 = df.copy()
        d2["cr"] = (rf[f"cat{r}"] == CR).astype(float)
        d2.loc[rf[f"cat{r}"].isna(), "cr"] = np.nan
        s2 = d2.dropna(subset=["os", "death", "cr", "bclc_cd", "afp_ge400"])
        f2 = cox_fit(
            s2[["cr", "bclc_cd", "afp_ge400"]].to_numpy(float),
            s2["os"].to_numpy(float), s2["death"].to_numpy(int),
        )
        per_reader[f"reader{r}"] = {
            "hr_cr": f2["hr"][0], "ci": [f2["hr_ci_lo"][0], f2["hr_ci_hi"][0]],
            "p": f2["p"][0], "n": f2["n"], "events": f2["events"],
        }
    out["cox_per_reader_cr_adjusted"] = per_reader

    # Unadjusted dose-response in number of readers calling CR.
    s3 = df.dropna(subset=["os", "death", "n_readers_cr"])
    f3 = cox_fit(s3[["n_readers_cr"]].to_numpy(float),
                 s3["os"].to_numpy(float), s3["death"].to_numpy(int))
    out["cox_n_readers_cr_unadjusted"] = {
        "hr_per_reader": f3["hr"][0], "ci": [f3["hr_ci_lo"][0], f3["hr_ci_hi"][0]],
        "p": f3["p"][0], "n": f3["n"], "events": f3["events"],
    }

    # Is BCLC prognostic here?
    s4 = df.dropna(subset=["os", "death", "bclc_cd"])
    f4 = cox_fit(s4[["bclc_cd"]].to_numpy(float),
                 s4["os"].to_numpy(float), s4["death"].to_numpy(int))
    out["cox_bclc_cd_alone"] = {
        "hr": f4["hr"][0], "ci": [f4["hr_ci_lo"][0], f4["hr_ci_hi"][0]],
        "p": f4["p"][0], "n": f4["n"], "events": f4["events"],
    }

    # C-index of each response construct for OS.
    cidx = {}
    ok = df.dropna(subset=["os", "death"])
    for r in READERS:
        m = ok.index.intersection(rf[rf[f"cat{r}"].notna()].index)
        risk = (rf.loc[m, f"cat{r}"] != CR).astype(float).to_numpy()
        cidx[f"reader{r}_cr"] = concordance(
            risk, ok.loc[m, "os"].to_numpy(float), ok.loc[m, "death"].to_numpy(int)
        )
    m = ok.index.intersection(df[df["majority_cr"].notna()].index)
    cidx["majority_cr"] = concordance(
        (1 - df.loc[m, "majority_cr"]).to_numpy(float),
        ok.loc[m, "os"].to_numpy(float), ok.loc[m, "death"].to_numpy(int),
    )
    fu_mean = rf[[f"fu{r}" for r in READERS]].mean(axis=1)
    m = ok.index.intersection(fu_mean.dropna().index)
    cidx["mean_followup_viable_diameter"] = concordance(
        np.log1p(fu_mean.loc[m].to_numpy(float)),
        ok.loc[m, "os"].to_numpy(float), ok.loc[m, "death"].to_numpy(int),
    )
    out["c_index_for_os"] = cidx

    # MATCHED-SET C-index comparison. The block above computes each quantity on
    # whichever patients have that variable, so the five figures rest on five
    # different denominators and are not comparable. This block fixes one set,
    # adds the ordinal comparator the earlier version omitted (the paper argues
    # the objective-response partition matters, so a complete-response-only
    # comparator is not the relevant competitor), and attaches paired bootstrap
    # intervals to every difference against the continuous measurement.
    matched = df.dropna(subset=["os", "death"]).index
    matched = matched.intersection(
        rf.dropna(subset=[f"{p}{r}" for r in READERS for p in ("fu", "cat")]).index)
    t = df.loc[matched, "os"].to_numpy(float)
    e = df.loc[matched, "death"].to_numpy(float)
    cats_m = rf.loc[matched, [f"cat{r}" for r in READERS]].to_numpy(float)
    fu_m = rf.loc[matched, [f"fu{r}" for r in READERS]].to_numpy(float)

    preds = {"mean_followup_viable_diameter": np.log1p(fu_m.mean(axis=1))}
    for i, r in enumerate(READERS):
        preds[f"reader{r}_cr"] = (cats_m[:, i] != CR).astype(float)
        preds[f"reader{r}_four_category"] = cats_m[:, i]
        preds[f"reader{r}_orr"] = (cats_m[:, i] > PR).astype(float)
    preds["majority_cr"] = (((cats_m == CR).sum(axis=1) >= 2) == 0).astype(float)
    preds["mean_four_category"] = cats_m.mean(axis=1)
    preds["majority_orr"] = (((cats_m <= PR).sum(axis=1) >= 2) == 0).astype(float)

    ref = "mean_followup_viable_diameter"
    rng = np.random.default_rng(RNG_SEED)
    n_m = len(matched)
    idxs = [rng.integers(0, n_m, n_m) for _ in range(1000)]
    matched_block: dict[str, Any] = {
        "n": int(n_m),
        "events": int(e.sum()),
        "note": ("All predictors on one patient set. Differences are against the "
                 "continuous measurement, paired bootstrap over patients. Harrell's C "
                 "credits ties at 0.5, which compresses binary predictors toward 0.5; "
                 "the ordinal comparators are the fair competitors."),
    }
    base = concordance(preds[ref], t, e)
    matched_block["c_index"] = {k: concordance(v, t, e) for k, v in preds.items()}
    diffs: dict[str, Any] = {}
    for k, v in preds.items():
        if k == ref:
            continue
        boots = []
        for ix in idxs:
            if e[ix].sum() < 2:
                continue
            a = concordance(preds[ref][ix], t[ix], e[ix])
            b = concordance(v[ix], t[ix], e[ix])
            if np.isfinite(a) and np.isfinite(b):
                boots.append(a - b)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        diffs[k] = {
            "continuous_minus_this": float(base - concordance(v, t, e)),
            "ci95": [float(lo), float(hi)],
            "excludes_zero": bool(lo > 0.0 or hi < 0.0),
        }
    matched_block["difference_vs_continuous"] = diffs
    matched_block["any_categorisation_beaten_significantly"] = bool(
        all(d["excludes_zero"] for d in diffs.values()))
    out["c_index_matched_set"] = matched_block
    out["os_units"] = "weeks (per glossary)"
    return out


def waw_survival(waw: pd.DataFrame) -> dict[str, Any]:
    df = waw.copy()
    df["time"] = pd.to_numeric(df["survival_time"], errors="coerce")
    df["death"] = pd.to_numeric(df["death"], errors="coerce")
    lr = pd.to_numeric(df["initial_LR_TR"], errors="coerce")
    df["nonviable"] = (lr == 0).astype(float)
    df.loc[lr.isna(), "nonviable"] = np.nan
    df["log_afp"] = np.log(pd.to_numeric(df["lab_afp"], errors="coerce").clip(lower=1e-6))
    df["alb"] = pd.to_numeric(df["lab_albumin"], errors="coerce")
    df["bili"] = pd.to_numeric(df["lab_bilirubin"], errors="coerce")
    df["bclc_b"] = pd.to_numeric(df["bclc"], errors="coerce")

    out: dict[str, Any] = {}
    sub = df.dropna(subset=["time", "death", "nonviable", "bclc_b", "log_afp", "alb", "bili"])
    fit = cox_fit(
        sub[["nonviable", "bclc_b", "log_afp", "alb", "bili"]].to_numpy(float),
        sub["time"].to_numpy(float), sub["death"].to_numpy(int),
    )
    out["cox_nonviable_adjusted"] = {
        "terms": ["nonviable", "bclc_b", "log_afp", "albumin", "bilirubin"], **fit
    }

    s2 = df.dropna(subset=["time", "death", "bclc_b"])
    f2 = cox_fit(s2[["bclc_b"]].to_numpy(float),
                 s2["time"].to_numpy(float), s2["death"].to_numpy(int))
    out["cox_bclc_b_alone"] = {
        "hr": f2["hr"][0], "ci": [f2["hr_ci_lo"][0], f2["hr_ci_hi"][0]],
        "p": f2["p"][0], "n": f2["n"], "events": f2["events"],
    }

    s3 = df.dropna(subset=["time", "death", "alb"])
    f3 = cox_fit(s3[["alb"]].to_numpy(float),
                 s3["time"].to_numpy(float), s3["death"].to_numpy(int))
    out["cox_albumin_alone"] = {
        "hr_per_unit": f3["hr"][0], "ci": [f3["hr_ci_lo"][0], f3["hr_ci_hi"][0]],
        "p": f3["p"][0], "n": f3["n"], "events": f3["events"],
    }

    # 90-day landmark: drop anyone who died/was censored before day 90, reset clock.
    lm = 90.0
    land = df[(df["time"] > lm)].copy()
    land["time_lm"] = land["time"] - lm
    s4 = land.dropna(subset=["time_lm", "death", "nonviable", "bclc_b", "log_afp", "alb", "bili"])
    f4 = cox_fit(
        s4[["nonviable", "bclc_b", "log_afp", "alb", "bili"]].to_numpy(float),
        s4["time_lm"].to_numpy(float), s4["death"].to_numpy(int),
    )
    out["cox_nonviable_adjusted_90d_landmark"] = {
        "terms": ["nonviable", "bclc_b", "log_afp", "albumin", "bilirubin"],
        "n_excluded_by_landmark": int(len(df) - len(land)), **f4
    }

    ttn = pd.to_numeric(df["time_to_nonv"], errors="coerce")
    out["time_to_nonv"] = {
        "n": int(ttn.notna().sum()),
        "median": float(ttn.median()),
        "q1": float(ttn.quantile(0.25)),
        "q3": float(ttn.quantile(0.75)),
        "n_negative": int((ttn < 0).sum()),
        "min": float(ttn.min()),
    }
    pt = pd.to_numeric(df["progression_time"], errors="coerce")
    prog = pd.to_numeric(df["progression"], errors="coerce")
    out["progression"] = {
        "n_progression_time_nonmissing": int(pt.notna().sum()),
        "n_events": int((prog == 1).sum()),
        "n_rows": int(len(df)),
        "note": "PFS is restricted to rows with a progression_time; not 233",
    }
    out["survival_units"] = "days"
    return out


# --------------------------------------------------------------------------
# WAW released-score reproducibility defects
# --------------------------------------------------------------------------
def albi_grade(alb_g_per_l: pd.Series, bili_umol: pd.Series) -> pd.Series:
    score = np.log10(bili_umol) * 0.66 + alb_g_per_l * (-0.085)
    return pd.cut(score, [-np.inf, -2.60, -1.39, np.inf], labels=[1, 2, 3]).astype("float")


def waw_score_audit(waw: pd.DataFrame) -> dict[str, Any]:
    df = waw.copy()
    alb = pd.to_numeric(df["lab_albumin"], errors="coerce")
    bili_mgdl = pd.to_numeric(df["lab_bilirubin"], errors="coerce")
    afp = pd.to_numeric(df["lab_afp"], errors="coerce")
    nles = pd.to_numeric(df["lesions_number"], errors="coerce")
    diam = df[["lesion1_diameter", "lesion2_diameter", "lesion3_diameter"]].apply(
        pd.to_numeric, errors="coerce"
    )
    # lesion*_diameter is released in MILLIMETRES (range 7-180) while the derived
    # `6_12` column is in centimetres. The definitions sheet does not state either.
    dmax_cm = diam.max(axis=1) / 10.0
    bili_umol = bili_mgdl * 17.1

    out: dict[str, Any] = {}
    out["lesion_diameter_unit_finding"] = {
        "observed_range_mm": [float(diam.min().min()), float(diam.max().max())],
        "note": "lesion*_diameter is mm; the derived 6_12 column is cm. Neither unit is documented.",
    }
    out["lab_albumin_observed_range"] = [float(alb.min()), float(alb.max())]
    out["lab_albumin_documented_unit"] = "g/l (per supplementary_table_s1)"
    g_as_documented = albi_grade(alb, bili_umol)                # treat as g/L
    g_as_gdl = albi_grade(alb * 10.0, bili_umol)                # treat as g/dL -> g/L
    out["albi_grade_if_albumin_read_as_g_per_L"] = {
        str(k): int(v) for k, v in g_as_documented.value_counts().sort_index().items()
    }
    out["albi_grade_if_albumin_read_as_g_per_dL"] = {
        str(k): int(v) for k, v in g_as_gdl.value_counts().sort_index().items()
    }

    # HAP: 3-item (as documented) vs 4-item (standard, incl. size > 7 cm)
    alb_gl = alb * 10.0
    items3 = ((alb_gl < 36).astype(int) + (afp > 400).astype(int)
              + (bili_umol > 17).astype(int))
    items4 = items3 + (dmax_cm > 7).astype(int)
    released_hap = pd.to_numeric(df["hap_score"], errors="coerce")
    ok = released_hap.notna()
    out["hap_score"] = {
        "n_released": int(ok.sum()),
        "match_3_item": int((items3[ok] == released_hap[ok]).sum()),
        "match_4_item": int((items4[ok] == released_hap[ok]).sum()),
        "documented_criteria_count": 3,
    }

    released_mhap = pd.to_numeric(df["mhap_2"], errors="coerce")
    ok2 = released_mhap.notna()
    mhap = items4 + (nles >= 2).astype(int)
    out["mhap_2"] = {
        "n_released": int(ok2.sum()),
        "match_4_item_plus_multifocal": int((mhap[ok2] == released_mhap[ok2]).sum()),
    }

    # six-and-twelve: max diameter (cm) + lesion count
    released_612 = pd.to_numeric(df["6_12"], errors="coerce")
    derived_612 = dmax_cm + nles
    close = (derived_612 - released_612).abs() < 1e-6
    out["six_and_twelve_continuous"] = {
        "n": int(released_612.notna().sum()),
        "match_maxdiam_plus_count": int(close.sum()),
    }

    released_band = pd.to_numeric(df["6_12_score"], errors="coerce")
    published = pd.cut(released_612, [-np.inf, 6.0, 12.0, np.inf],
                       labels=[0, 1, 2]).astype("float")
    rounded = pd.cut(released_612.round(), [-np.inf, 6.0, 12.0, np.inf],
                     labels=[0, 1, 2]).astype("float")
    okb = released_band.notna() & released_612.notna()
    out["six_and_twelve_bands"] = {
        "n": int(okb.sum()),
        "match_published_cutoffs_le6_le12": int((published[okb] == released_band[okb]).sum()),
        "match_rounded_sum_cutoffs": int((rounded[okb] == released_band[okb]).sum()),
        "released_band_counts": {str(k): int(v) for k, v in released_band.value_counts().sort_index().items()},
    }
    return out


# --------------------------------------------------------------------------
# reader-crossed timepoint attribution (consensus-free)
# --------------------------------------------------------------------------
def categories_from(bl_m: np.ndarray, fu_m: np.ndarray) -> np.ndarray:
    """Elementwise mRECIST rule over two same-shaped measurement arrays."""
    out = np.empty(bl_m.shape, dtype=int)
    for i in range(bl_m.shape[0]):
        for j in range(bl_m.shape[1]):
            out[i, j] = mrecist_rule(bl_m[i, j], fu_m[i, j])
    return out


def _triple_complete(rf: pd.DataFrame) -> pd.DataFrame:
    return rf.dropna(
        subset=[f"{p}{r}" for r in READERS for p in ("bl", "fu", "cat")]
    ).reset_index(drop=True)


def _crossed_disagreement(bl: np.ndarray, fu: np.ndarray) -> dict[str, float]:
    """Disagreement when only one timepoint's reader is varied.

    Builds the full 3x3 grid of (baseline reader i, follow-up reader j) and asks
    two separate questions, neither of which uses a synthetic consensus value:

      vary_followup  hold the baseline reader fixed, swap the follow-up reader
      vary_baseline  hold the follow-up reader fixed, swap the baseline reader

    Each is averaged over the three choices of the held-fixed reader.
    """
    n, k = bl.shape
    grid = np.empty((n, k, k), dtype=int)  # [patient, baseline reader, follow-up reader]
    for i in range(k):
        for j in range(k):
            grid[:, i, j] = categories_from(bl[:, [i]], fu[:, [j]])[:, 0]

    def frac_disagree(rows: np.ndarray) -> float:
        return float(np.mean([len(set(r)) > 1 for r in rows]))

    def frac_disagree_orr(rows: np.ndarray) -> float:
        b = (rows <= PR).astype(int)
        return float(np.mean([len(set(r)) > 1 for r in b]))

    vary_fu = [grid[:, i, :] for i in range(k)]      # fixed baseline reader i
    vary_bl = [grid[:, :, j] for j in range(k)]      # fixed follow-up reader j

    four_fu = float(np.mean([frac_disagree(x) for x in vary_fu]))
    four_bl = float(np.mean([frac_disagree(x) for x in vary_bl]))
    orr_fu = float(np.mean([frac_disagree_orr(x) for x in vary_fu]))
    orr_bl = float(np.mean([frac_disagree_orr(x) for x in vary_bl]))

    all9 = grid.reshape(n, k * k)
    diag = np.stack([grid[:, i, i] for i in range(k)], axis=1)
    return {
        "vary_followup_four_cat": four_fu,
        "vary_baseline_four_cat": four_bl,
        "delta_four_cat_points": (four_fu - four_bl) * 100.0,
        "vary_followup_orr": orr_fu,
        "vary_baseline_orr": orr_bl,
        "delta_orr_points": (orr_fu - orr_bl) * 100.0,
        "all_nine_combinations_four_cat": frac_disagree(all9),
        "diagonal_observed_four_cat": frac_disagree(diag),
    }


def reader_crossed_attribution(rf: pd.DataFrame, n_boot: int = 5000) -> dict[str, Any]:
    """Attribute categorical disagreement to a timepoint WITHOUT a consensus value.

    The M002 counterfactual substitutes a synthetic three-reader mean, which a
    reviewer can object to on the grounds that the mean is not a reading any
    radiologist produced. This module answers the same question using only real
    readings: cross every baseline reader with every follow-up reader.

    Reported for the primary set, for the no-CR subset, and with follow-up zeros
    replaced by 1 mm, so the structural zero is separated from the rest.
    """
    complete = _triple_complete(rf)
    bl = complete[[f"bl{r}" for r in READERS]].to_numpy(float)
    fu = complete[[f"fu{r}" for r in READERS]].to_numpy(float)

    arms: dict[str, tuple[np.ndarray, np.ndarray]] = {"primary": (bl, fu)}
    keep = (fu > 0).all(axis=1)
    arms["exclude_any_cr"] = (bl[keep], fu[keep])
    arms["zeros_to_1mm"] = (bl, np.where(fu == 0.0, 1.0, fu))

    rng = np.random.default_rng(RNG_SEED)
    out: dict[str, Any] = {}
    for name, (b, f) in arms.items():
        stats = _crossed_disagreement(b, f)
        n = b.shape[0]
        boots_four, boots_orr = [], []
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            s = _crossed_disagreement(b[idx], f[idx])
            boots_four.append(s["delta_four_cat_points"])
            boots_orr.append(s["delta_orr_points"])
        lo4, hi4 = np.percentile(boots_four, [2.5, 97.5])
        lo_o, hi_o = np.percentile(boots_orr, [2.5, 97.5])
        stats["n"] = n
        stats["delta_four_cat_ci95"] = [float(lo4), float(hi4)]
        stats["delta_orr_ci95"] = [float(lo_o), float(hi_o)]
        stats["four_cat_excludes_zero"] = bool(lo4 > 0.0)
        out[name] = stats
    return out


# --------------------------------------------------------------------------
# ICC scale sensitivity (matched subsets)
# --------------------------------------------------------------------------
def icc_scale_sensitivity(rf: pd.DataFrame, n_boot: int = 5000) -> dict[str, Any]:
    """ICC by timepoint on the raw and log scales, on matched patient sets.

    M005 reports a baseline log ICC on n=93 next to a follow-up log ICC on the
    n=64 no-CR subset, which is not a matched comparison. This module reports
    every cell of {raw, log} x {baseline, follow-up} on both patient sets, plus a
    bootstrap CI for the within-scale timepoint contrast, so the scale choice and
    the subset choice are separated and auditable.

    The raw-scale, all-patients row is the setting used by Mohammadzadeh et al.
    (BMC Med Imaging 2025;25:148), reported there as 0.87 baseline / 0.86 follow-up.
    """
    complete = _triple_complete(rf)
    bl = complete[[f"bl{r}" for r in READERS]].to_numpy(float)
    fu = complete[[f"fu{r}" for r in READERS]].to_numpy(float)
    keep = (fu > 0).all(axis=1)

    sets = {"all": (bl, fu), "no_cr": (bl[keep], fu[keep])}
    rng = np.random.default_rng(RNG_SEED)
    out: dict[str, Any] = {}
    for sname, (b, f) in sets.items():
        cell: dict[str, Any] = {"n": int(b.shape[0])}
        for scale, fn in (("raw", lambda x: x), ("log", np.log)):
            if scale == "log" and (b <= 0).any():
                continue  # log undefined at a structural zero
            if scale == "log" and (f <= 0).any():
                continue
            icc_b = icc_two_way(fn(b))["ICC_2_1_absolute"]
            icc_f = icc_two_way(fn(f))["ICC_2_1_absolute"]
            n = b.shape[0]
            boots = []
            for _ in range(n_boot):
                idx = rng.integers(0, n, n)
                boots.append(icc_two_way(fn(b[idx]))["ICC_2_1_absolute"]
                             - icc_two_way(fn(f[idx]))["ICC_2_1_absolute"])
            lo, hi = np.percentile(boots, [2.5, 97.5])
            cell[scale] = {
                "baseline_icc": float(icc_b),
                "followup_icc": float(icc_f),
                "difference_bl_minus_fu": float(icc_b - icc_f),
                "difference_ci95": [float(lo), float(hi)],
                "excludes_zero": bool(lo > 0.0),
            }
        out[sname] = cell
    return out


# --------------------------------------------------------------------------
# reconciliation with the previously published coefficients on this cohort
# --------------------------------------------------------------------------
# Mohammadzadeh S, Mohebbi A, Abdi A, Mohammadi A. Inter-reader agreement of
# RECIST and mRECIST criteria for assessing response to TACE in HCC.
# BMC Med Imaging 2025;25(1):148. PMID 40319244. Table 4, mRECIST pairwise
# cross-tabulations, transcribed verbatim. Codes: 1 CR, 2 PR, 3 SD, 4 PD.
PUBLISHED_MRECIST_CROSSTABS = {
    "r1_r3": [[13, 6, 1, 0], [0, 29, 11, 0], [1, 9, 17, 1], [0, 1, 2, 2]],
    "r1_r2": [[17, 2, 2, 0], [6, 24, 10, 0], [3, 13, 11, 1], [0, 1, 2, 3]],
    "r2_r3": [[14, 8, 2, 1], [0, 29, 11, 0], [0, 8, 17, 0], [0, 0, 1, 2]],
}
PUBLISHED_COEFFICIENTS = {
    "mRECIST_gwet_four_category": 0.80,
    "RECIST_gwet_four_category": 0.90,
    "mRECIST_fleiss_four_category": 0.60,
    "RECIST_fleiss_four_category": 0.51,
}


def _cohen_kappa_2x2(a: float, b: float, c: float, d: float) -> float:
    n = a + b + c + d
    po = (a + d) / n
    pe = ((a + b) * (a + c) + (c + d) * (b + d)) / n ** 2
    return float((po - pe) / (1.0 - pe))


def published_table_reconciliation(rf: pd.DataFrame, hcc: pd.DataFrame) -> dict[str, Any]:
    """Reproduce the published coefficients, and recover ORR kappa from their tables.

    Two independent checks that this analysis and the prior report describe the
    same readings:

      1. The published Gwet coefficients (0.80 mRECIST, 0.90 RECIST) are the
         ORDINALLY WEIGHTED AC2. Recomputing them unweighted on the same readings
         gives materially lower values.
      2. Collapsing the published cross-tabulations to the objective-response
         dichotomy recovers this analysis's pairwise kappa. Two of three panels
         match to four decimals; the third is the n=95 pair-complete set, against
         this pipeline's n=93 triple-complete set.
    """
    out: dict[str, Any] = {"published": PUBLISHED_COEFFICIENTS}

    # 1. weighted vs unweighted, mRECIST and RECIST, on the triple-complete set
    coeffs: dict[str, Any] = {}
    for label, cols in (("mRECIST", [f"{r}_mRECIST" for r in READERS]),
                        ("RECIST", [f"{r}_RECIST" for r in READERS])):
        mat = hcc[cols].apply(pd.to_numeric, errors="coerce").dropna().astype(int).to_numpy()
        codes = sorted(set(mat.ravel()))
        remap = {c: i for i, c in enumerate(codes)}
        x = np.vectorize(remap.get)(mat)
        q = len(codes)
        # Ordinally weighted Fleiss kappa, needed to state which variants were
        # tried against the published value that does not reproduce.
        w = ordinal_weights(q)
        n_s, k_s = x.shape
        counts = np.zeros((n_s, q))
        for si in range(n_s):
            for rj in range(k_s):
                counts[si, x[si, rj]] += 1
        pbar = counts.sum(axis=0) / (n_s * k_s)
        po_w = float(np.mean([
            sum(w[a, b] * counts[si, a] * (counts[si, b] - (1 if a == b else 0))
                for a in range(q) for b in range(q)) / (k_s * (k_s - 1))
            for si in range(n_s)]))
        pe_w = float(sum(w[a, b] * pbar[a] * pbar[b] for a in range(q) for b in range(q)))
        coeffs[label] = {
            "n": int(mat.shape[0]),
            "codes_present": [int(c) for c in codes],
            "gwet_ac1_unweighted": gwet_ac(x, q),
            "gwet_ac2_ordinal_weighted": gwet_ac(x, q, ordinal_weights(q)),
            "fleiss_unweighted": fleiss_kappa(x, q),
            "fleiss_ordinal_weighted": (po_w - pe_w) / (1.0 - pe_w),
        }
    out["recomputed"] = coeffs

    # 2. ORR kappa recovered from the published cross-tabulations
    from_tables: dict[str, Any] = {}
    for pair, tab in PUBLISHED_MRECIST_CROSSTABS.items():
        m = np.array(tab, dtype=float)
        from_tables[pair] = {
            "n_in_published_table": int(m.sum()),
            "orr_kappa_from_published_table": _cohen_kappa_2x2(
                m[:2, :2].sum(), m[:2, 2:].sum(), m[2:, :2].sum(), m[2:, 2:].sum()
            ),
        }
    ours = kappa_table(rf)["objective_response_CRPR_vs_SDPD"]["kappa_by_pair"]
    for pair, v in from_tables.items():
        v["orr_kappa_this_pipeline_n93"] = float(ours[pair])
        v["agrees_to_4dp"] = bool(
            abs(v["orr_kappa_from_published_table"] - float(ours[pair])) < 5e-5
        )
    out["orr_from_published_tables"] = from_tables

    # 3. what each released code actually means, from the measurements themselves.
    #    Needed to read the published cross-tabulations, whose legends disagree
    #    with each other about whether code 3 is stable or progressive disease.
    pct: dict[int, list[float]] = {}
    for r in READERS:
        sub = rf[rf[f"cat{r}"].notna() & rf[f"bl{r}"].notna() & rf[f"fu{r}"].notna()]
        for b, f, c in zip(sub[f"bl{r}"], sub[f"fu{r}"], sub[f"cat{r}"]):
            if b > 0:
                pct.setdefault(int(c), []).append((f - b) / b * 100.0)
    out["percent_change_by_released_code"] = {
        f"code_{c}": {
            "n_reader_observations": len(v),
            "median_percent_change": float(np.median(v)),
            "min_percent_change": float(np.min(v)),
            "max_percent_change": float(np.max(v)),
        }
        for c, v in sorted(pct.items())
    }
    return out


# --------------------------------------------------------------------------
# equal-noise control: how much asymmetry does the rule itself manufacture?
# --------------------------------------------------------------------------
def equal_noise_simulation(rf: pd.DataFrame, n_sim: int = 2000) -> dict[str, Any]:
    """Push EQUAL multiplicative reader noise through the rule at both timepoints.

    If the mRECIST rule's geometry alone produced the M002 asymmetry, then
    simulated readers whose noise is identical at baseline and follow-up would
    still show a positive H1 delta. This is the control for that objection: the
    simulated delta is the share of the observed effect attributable to the rule
    rather than to reader behaviour.
    """
    complete = _triple_complete(rf)
    bl = complete[[f"bl{r}" for r in READERS]].to_numpy(float)
    fu = complete[[f"fu{r}" for r in READERS]].to_numpy(float)
    keep = (fu > 0).all(axis=1)
    b_pos, f_pos = bl[keep], fu[keep]

    # one pooled sigma, applied identically at both timepoints by construction
    sigma = float(np.sqrt(np.mean(
        np.concatenate([np.log(b_pos).var(axis=1, ddof=1),
                        np.log(f_pos).var(axis=1, ddof=1)])
    )))
    truth_bl = np.exp(np.log(b_pos).mean(axis=1))
    truth_fu = np.exp(np.log(f_pos).mean(axis=1))

    rng = np.random.default_rng(RNG_SEED)
    n = truth_bl.shape[0]
    deltas = []
    for _ in range(n_sim):
        sb = truth_bl[:, None] * np.exp(rng.normal(0.0, sigma, (n, 3)))
        sf = truth_fu[:, None] * np.exp(rng.normal(0.0, sigma, (n, 3)))
        obs = categories_from(sb, sf)
        bl_h = categories_from(np.repeat(sb.mean(axis=1, keepdims=True), 3, axis=1), sf)
        fu_h = categories_from(sb, np.repeat(sf.mean(axis=1, keepdims=True), 3, axis=1))
        o = disagreement_stats(obs)["four_cat_disagreement"]
        deltas.append((((o - disagreement_stats(fu_h)["four_cat_disagreement"])
                        - (o - disagreement_stats(bl_h)["four_cat_disagreement"])) * 100.0))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "n_patients": n,
        "n_sim": n_sim,
        "common_log_sd": sigma,
        "simulated_H1_delta_points_mean": float(np.mean(deltas)),
        "simulated_H1_delta_ci95": [float(lo), float(hi)],
        "note": ("Equal noise at both timepoints. A non-zero value is asymmetry "
                 "manufactured by the rule; compare against the observed no-CR "
                 "delta of 15.6 points from M003c."),
    }


# --------------------------------------------------------------------------
# within-cohort replication on the total-diameter (RECIST) measurements
# --------------------------------------------------------------------------
def recist_frame(hcc: pd.DataFrame) -> pd.DataFrame:
    """Reader frame built from the RECIST total-diameter columns.

    The same three readers recorded total target-lesion diameters on the same
    images at the same two timepoints. That gives a within-cohort replication
    of the timepoint attribution on a criterion that asks about tumour SIZE
    rather than tumour VIABILITY, with no structural zeros at follow-up.
    """
    out = pd.DataFrame({"TCIA_ID": hcc["TCIA_ID"]})
    for r in READERS:
        out[f"bl{r}"] = pd.to_numeric(hcc[f"{r}_RECIST_BL"], errors="coerce")
        out[f"fu{r}"] = pd.to_numeric(hcc[f"{r}_RECIST_FU"], errors="coerce")
        out[f"cat{r}"] = pd.to_numeric(hcc[f"{r}_RECIST"], errors="coerce")
    return out


def recist_replication(hcc: pd.DataFrame, rf: pd.DataFrame,
                       n_boot: int = 5000) -> dict[str, Any]:
    """Re-run the timepoint attribution on total diameters, paired against mRECIST.

    The mechanism claim is that readers agree about how large a tumour is and
    disagree about how much of it remains viable. That claim predicts the
    follow-up dominance should be WEAKER for the total-diameter criterion than
    for the viable-diameter criterion, measured by the same readers on the same
    images in the same patients.

    Both criteria are restricted to the patients complete on both, so the
    contrast is paired and the difference-of-differences is bootstrapped over
    patients rather than compared across two marginal intervals.
    """
    rrf = recist_frame(hcc)
    m_ok = rf.dropna(subset=[f"{p}{r}" for r in READERS for p in ("bl", "fu", "cat")])
    r_ok = rrf.dropna(subset=[f"{p}{r}" for r in READERS for p in ("bl", "fu", "cat")])
    ids = sorted(set(m_ok["TCIA_ID"]) & set(r_ok["TCIA_ID"]))
    m = m_ok[m_ok["TCIA_ID"].isin(ids)].sort_values("TCIA_ID").reset_index(drop=True)
    r = r_ok[r_ok["TCIA_ID"].isin(ids)].sort_values("TCIA_ID").reset_index(drop=True)

    out: dict[str, Any] = {
        "n_paired": len(ids),
        "n_mrecist_complete": int(len(m_ok)),
        "n_recist_complete": int(len(r_ok)),
    }

    m_bl = m[[f"bl{i}" for i in READERS]].to_numpy(float)
    m_fu = m[[f"fu{i}" for i in READERS]].to_numpy(float)
    r_bl = r[[f"bl{i}" for i in READERS]].to_numpy(float)
    r_fu = r[[f"fu{i}" for i in READERS]].to_numpy(float)

    out["followup_zeros"] = {
        "mrecist": int((m_fu == 0).sum()),
        "recist": int((r_fu == 0).sum()),
    }

    # Does the RECIST rule reproduce the released RECIST category, as it does
    # for mRECIST? Checked with and without the RECIST 1.1 5 mm absolute rule.
    repro: dict[str, Any] = {}
    for abs_mm, tag in ((0.0, "no_abs_rule"), (5.0, "with_5mm_abs_rule")):
        derived = categories_from(r_bl, r_fu) if abs_mm == 0.0 else np.array(
            [[mrecist_rule(b, f, abs_mm) for b, f in zip(rb, rf_)]
             for rb, rf_ in zip(r_bl, r_fu)])
        released = r[[f"cat{i}" for i in READERS]].to_numpy(int)
        repro[tag] = {"n_reader_observations": int(released.size),
                      "n_match": int((derived == released).sum())}
    out["rule_reproduction"] = repro

    stats = {"mrecist": _crossed_disagreement(m_bl, m_fu),
             "recist": _crossed_disagreement(r_bl, r_fu)}
    out["crossed"] = stats

    # paired difference-of-differences: is follow-up dominance criterion-specific?
    rng = np.random.default_rng(RNG_SEED)
    n = len(ids)
    for key, label in (("delta_four_cat_points", "four_cat"),
                       ("delta_orr_points", "orr")):
        obs = stats["mrecist"][key] - stats["recist"][key]
        boots = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            boots.append(_crossed_disagreement(m_bl[idx], m_fu[idx])[key]
                         - _crossed_disagreement(r_bl[idx], r_fu[idx])[key])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out[f"mrecist_minus_recist_{label}"] = {
            "difference_points": float(obs),
            "ci95": [float(lo), float(hi)],
            "excludes_zero": bool(lo > 0.0 or hi < 0.0),
        }
    return out


# --------------------------------------------------------------------------
# what reader disagreement does to a model benchmark
# --------------------------------------------------------------------------
REFERENCE_FEATURES: dict[str, Any] = {
    "age": None,
    "Sex": None,
    "Evidence_of_cirh": None,
    "CPS": {"A": 0, "B": 1, "C": 2},
    "BCLC": {"Stage-A": 0, "Stage-B": 1, "Stage-C": 2, "Stage-D": 3},
    "AFP_group": {"<400": 0, ">=400": 1},
    "tumor_nodul": {"uninodular": 0, "multinodular": 1},
    "T_involvment": {"< or = 50%": 0, ">50%": 1},
    "Portal Vein Thrombosis": None,
    "Vascular invasion": None,
    "Metastasis": None,
}


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    """Area under the ROC curve; NaN when the reference has a single class."""
    pos, neg = y == 1, y == 0
    if not pos.any() or not neg.any():
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), float)
    s = score[order]
    i = 0
    while i < len(s):                       # average ranks within ties
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def reference_dependence(hcc: pd.DataFrame, rf: pd.DataFrame,
                         n_boot: int = 5000) -> dict[str, Any]:
    """Score ONE set of out-of-fold predictions against every reader in turn.

    Seven published studies use this collection to fit or score imaging models,
    conventionally against a single reader's category. If readers disagree on
    the objective-response partition, then the benchmark itself moves with the
    choice of reference, independently of the model.

    The design isolates that: models are trained once, against the majority
    label, so the out-of-fold predictions are IDENTICAL across every scoring
    reference. Only the reference changes. Any movement in the reported figure
    is therefore attributable to the reference and not to the model.

    This is a statement about benchmark stability. It is not a claim that these
    clinical variables predict response well, and the absolute discrimination
    should not be read as one.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    complete = _triple_complete(rf)
    src = hcc[hcc["TCIA_ID"].isin(set(complete["TCIA_ID"]))].copy()
    order = complete.set_index("TCIA_ID").loc[src["TCIA_ID"]].reset_index()

    cols, X = [], []
    for col, mapping in REFERENCE_FEATURES.items():
        if col not in src.columns:
            continue
        v = src[col].map(mapping) if mapping else pd.to_numeric(src[col], errors="coerce")
        if v.notna().all():
            cols.append(col)
            X.append(v.to_numpy(float))
    X = np.column_stack(X)

    cats = order[[f"cat{r}" for r in READERS]].to_numpy(int)
    refs = {f"reader{r}": (cats[:, i] <= PR).astype(int) for i, r in enumerate(READERS)}
    refs["majority"] = ((cats <= PR).sum(axis=1) >= 2).astype(int)
    refs["unanimous_only"] = (cats <= PR).all(axis=1).astype(int)

    models = {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                                random_state=RNG_SEED),
        "bclc_alone": None,   # the clinical score card, a single released variable
    }

    y_train = refs["majority"]
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=RNG_SEED)
    oof: dict[str, np.ndarray] = {}
    for name, est in models.items():
        if est is None:
            oof[name] = X[:, cols.index("BCLC")].astype(float)
            continue
        acc = np.zeros(len(y_train))
        cnt = np.zeros(len(y_train))
        for tr, te in cv.split(X, y_train):
            from sklearn.base import clone
            f = clone(est).fit(X[tr], y_train[tr])
            acc[te] += f.predict_proba(X[te])[:, 1]
            cnt[te] += 1
        oof[name] = acc / cnt

    out: dict[str, Any] = {
        "n": int(X.shape[0]),
        "features": cols,
        "training_reference": "majority",
        "prevalence_by_reference": {k: float(v.mean()) for k, v in refs.items()},
        "note": ("Predictions are identical across scoring references by "
                 "construction; only the reference changes."),
    }

    per_model: dict[str, Any] = {}
    for name, score in oof.items():
        aucs = {ref: _auc(y, score) for ref, y in refs.items()}
        single = [aucs[f"reader{r}"] for r in READERS]
        per_model[name] = {
            "auc_by_reference": aucs,
            "single_reader_range": [float(min(single)), float(max(single))],
            "single_reader_spread": float(max(single) - min(single)),
        }
    out["models"] = per_model

    # Does the reference decide which model looks best?
    ranking = {ref: max(per_model, key=lambda m: per_model[m]["auc_by_reference"][ref])
               for ref in refs}
    out["best_model_by_reference"] = ranking
    out["ranking_depends_on_reference"] = bool(len(set(ranking.values())) > 1)

    # Bootstrap the single-reader spread for the primary model. NOTE: max-minus-min
    # is bounded below by zero, so an interval excluding zero is not evidence that
    # the references differ. It is reported as a magnitude only; the pairwise
    # contrasts below are the actual test, because a paired difference can take
    # either sign.
    rng = np.random.default_rng(RNG_SEED)
    n = X.shape[0]
    primary = "logistic_regression"
    boots = []
    pair_boots: dict[str, list[float]] = {}
    pairs = [(a, b) for i, a in enumerate(READERS) for b in READERS[i + 1:]]
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        a = {r: _auc(refs[f"reader{r}"][idx], oof[primary][idx]) for r in READERS}
        if not all(np.isfinite(list(a.values()))):
            continue
        boots.append(max(a.values()) - min(a.values()))
        for x, y in pairs:
            pair_boots.setdefault(f"reader{x}_minus_reader{y}", []).append(a[x] - a[y])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    out["primary_model_spread"] = {
        "model": primary,
        "spread": per_model[primary]["single_reader_spread"],
        "ci95": [float(lo), float(hi)],
        "n_defined_resamples": len(boots),
        "caveat": ("max-minus-min cannot be negative, so this interval measures "
                   "magnitude, not significance. See pairwise_auc_contrasts."),
    }

    aucs = per_model[primary]["auc_by_reference"]
    contrasts: dict[str, Any] = {}
    for x, y in pairs:
        key = f"reader{x}_minus_reader{y}"
        d = np.asarray(pair_boots[key], float)
        clo, chi = np.percentile(d, [2.5, 97.5])
        contrasts[key] = {
            "difference": float(aucs[f"reader{x}"] - aucs[f"reader{y}"]),
            "ci95": [float(clo), float(chi)],
            "excludes_zero": bool(clo > 0.0 or chi < 0.0),
        }
    out["pairwise_auc_contrasts"] = contrasts
    out["any_pairwise_contrast_excludes_zero"] = bool(
        any(v["excludes_zero"] for v in contrasts.values()))

    # Permutation null. The objection to any spread computed near AUC 0.5 is that
    # it could be noise amplification rather than a reader effect. Under the null
    # that the three readers relate identically to the score, their labels are
    # exchangeable WITHIN a patient. Permuting reader identity per patient
    # preserves every patient's label multiset - and therefore the whole
    # disagreement structure and the overall prevalence - while destroying any
    # reader-specific relationship to the predictions.
    labels = np.column_stack([refs[f"reader{r}"] for r in READERS])
    null_spreads = []
    for _ in range(n_boot):
        perm = np.apply_along_axis(rng.permutation, 1, labels)
        a = [_auc(perm[:, i], oof[primary]) for i in range(len(READERS))]
        if all(np.isfinite(a)):
            null_spreads.append(max(a) - min(a))
    null_spreads = np.asarray(null_spreads, float)
    observed = per_model[primary]["single_reader_spread"]
    out["permutation_null"] = {
        "model": primary,
        "observed_spread": float(observed),
        "null_spread_median": float(np.median(null_spreads)),
        "null_spread_p95": float(np.percentile(null_spreads, 95)),
        "p_value": float((1 + (null_spreads >= observed).sum()) / (1 + null_spreads.size)),
        "n_permutations": int(null_spreads.size),
        "design": ("reader identity permuted within each patient, preserving the "
                   "per-patient label multiset and the disagreement structure"),
    }
    # coarse histogram so the figure can draw the null without re-running it
    counts, edges = np.histogram(null_spreads, bins=30)
    out["permutation_null"]["histogram"] = {
        "counts": [int(c) for c in counts],
        "bin_edges": [float(e) for e in edges],
    }
    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--boot", type=int, default=5000)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    provenance = verify_inputs(args.data)
    if not all(v["match"] for v in provenance.values()):
        raise SystemExit(f"CHECKSUM MISMATCH: {provenance}")

    hcc = pd.read_excel(args.data / "HCC-TACE-Seg_clinical_data-V2.xlsx",
                        sheet_name="data table")
    waw = pd.read_excel(args.data / "clinical_data_wawtace_v2_15_07_2024.xlsx")
    rf = reader_frame(hcc)

    results = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": RNG_SEED,
        "provenance": provenance,
        "M001_rule_reproduction": check_rule_reproduction(rf),
        "M002_harmonisation_counterfactual": harmonisation_counterfactual(rf, args.boot),
        "M003_kappa_by_dichotomy": kappa_table(rf),
        "M003b_multirater_agreement": multirater_agreement(rf, args.boot),
        "M003c_structural_zero_robustness": structural_zero_robustness(rf, args.boot),
        "M004_coreader_oracle": coreader_oracle(rf),
        "M003d_published_table_reconciliation": published_table_reconciliation(rf, hcc),
        "M005_measurement_variability": measurement_variability(rf),
        "M005b_recist_ratio_spread": recist_ratio_spread(hcc),
        "M005c_icc_scale_sensitivity": icc_scale_sensitivity(rf, args.boot),
        "M006_easl_audit": easl_audit(hcc),
        "M007_tace_type_crosstab": tace_type_crosstab(hcc),
        "M008_hcc_survival": hcc_survival(hcc, rf),
        "M009_waw_survival": waw_survival(waw),
        "M010_waw_score_audit": waw_score_audit(waw),
        "M011_reader_crossed_attribution": reader_crossed_attribution(rf, args.boot),
        "M012_equal_noise_control": equal_noise_simulation(rf),
        "M013_cohort_characteristics": cohort_characteristics(hcc, rf),
        "M014_recist_replication": recist_replication(hcc, rf, args.boot),
        "M015_reference_dependence": reference_dependence(hcc, rf, args.boot),
    }

    path = args.out / "MEASUREMENT_GATE.json"
    path.write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
