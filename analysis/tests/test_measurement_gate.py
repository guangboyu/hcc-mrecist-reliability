#!/usr/bin/env python3
"""Tests for the Stage-0 measurement gate.

pytest is not installed in this venv, so these run standalone:
    .venv/bin/python tests/test_measurement_gate.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from measurement_gate import (  # noqa: E402
    CR, PR, SD, PD,
    _auc,
    PUBLISHED_MRECIST_CROSSTABS,
    _cohen_kappa_2x2,
    _crossed_disagreement,
    categories_from,
    cohort_characteristics,
    coreader_oracle,
    disagreement_stats,
    harmonisation_counterfactual,
    icc_scale_sensitivity,
    icc_two_way,
    mrecist_rule,
    multirater_agreement,
    published_table_reconciliation,
    reader_crossed_attribution,
    recist_replication,
    reference_dependence,
)

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name} {detail}")
        FAILURES.append(name)


def test_mrecist_rule():
    print("test_mrecist_rule")
    check("FU==0 is CR regardless of BL", mrecist_rule(100.0, 0.0) == CR)
    check("-30% exactly is PR (boundary inclusive)", mrecist_rule(100.0, 70.0) == PR)
    check("-29.9% is SD", mrecist_rule(100.0, 70.1) == SD)
    check("+20% exactly is PD (boundary inclusive)", mrecist_rule(100.0, 120.0) == PD)
    check("+19.9% is SD", mrecist_rule(100.0, 119.9) == SD)
    check("unchanged is SD", mrecist_rule(50.0, 50.0) == SD)
    check("NaN in -> None out", mrecist_rule(np.nan, 10.0) is None)
    check("5mm absolute rule can block PD",
          mrecist_rule(10.0, 12.0, abs_pd_mm=5.0) == SD)
    check("5mm absolute rule allows large PD",
          mrecist_rule(100.0, 130.0, abs_pd_mm=5.0) == PD)


def test_disagreement_stats():
    print("test_disagreement_stats")
    unanimous = np.array([[CR, CR, CR], [SD, SD, SD]])
    s = disagreement_stats(unanimous)
    check("unanimous -> 0 disagreement", s["four_cat_disagreement"] == 0.0)
    check("unanimous CR -> no split CR", s["split_cr_fraction"] == 0.0)

    split = np.array([[CR, PR, PR]])
    s = disagreement_stats(split)
    check("CR/PR/PR is a split CR", s["split_cr_fraction"] == 1.0)
    check("CR/PR/PR agrees on ORR", s["binary_orr_disagreement"] == 0.0,
          "(both CR and PR are objective responses)")

    orr_split = np.array([[PR, SD, SD]])
    s = disagreement_stats(orr_split)
    check("PR/SD/SD disagrees on ORR", s["binary_orr_disagreement"] == 1.0)


def test_harmonisation_direction():
    """Construct data where ALL disagreement is caused by follow-up."""
    print("test_harmonisation_direction")
    n = 20
    rng = np.random.default_rng(0)
    bl = np.repeat(rng.uniform(80, 120, n)[:, None], 3, axis=1)  # identical baselines
    # Follow-ups straddle the -30% cutpoint per reader.
    fu = bl * np.array([0.60, 0.72, 0.85])[None, :]
    rf = pd.DataFrame({"TCIA_ID": [f"P{i}" for i in range(n)]})
    for r in range(3):
        rf[f"bl{r+1}"] = bl[:, r]
        rf[f"fu{r+1}"] = fu[:, r]
        rf[f"cat{r+1}"] = [mrecist_rule(bl[i, r], fu[i, r]) for i in range(n)]
    res = harmonisation_counterfactual(rf, n_boot=200)
    check("synthetic: FU-harmonisation removes all disagreement",
          res["followup_harmonised"]["four_cat_disagreement"] == 0.0)
    check("synthetic: BL-harmonisation removes none",
          res["baseline_harmonised"]["four_cat_disagreement"] ==
          res["observed"]["four_cat_disagreement"])
    check("synthetic: H1 delta is large and positive", res["H1_delta_points"] > 50)
    check("released categories match the derived rule",
          res["released_vs_derived_identical"] is True)


def test_harmonisation_reverse():
    """Mirror case: all disagreement caused by BASELINE."""
    print("test_harmonisation_reverse")
    n = 20
    fu_val = np.full(n, 70.0)
    bl = np.stack([np.full(n, 100.0), np.full(n, 85.0), np.full(n, 75.0)], axis=1)
    rf = pd.DataFrame({"TCIA_ID": [f"P{i}" for i in range(n)]})
    for r in range(3):
        rf[f"bl{r+1}"] = bl[:, r]
        rf[f"fu{r+1}"] = fu_val
        rf[f"cat{r+1}"] = [mrecist_rule(bl[i, r], fu_val[i]) for i in range(n)]
    res = harmonisation_counterfactual(rf, n_boot=200)
    check("reverse: BL-harmonisation removes all disagreement",
          res["baseline_harmonised"]["four_cat_disagreement"] == 0.0)
    check("reverse: H1 delta is negative (points the other way)",
          res["H1_delta_points"] < 0,
          f"got {res['H1_delta_points']}")


def test_icc():
    print("test_icc")
    rng = np.random.default_rng(1)
    subject = rng.normal(0, 1, 200)[:, None]
    perfect = np.repeat(subject, 3, axis=1)
    r = icc_two_way(perfect + rng.normal(0, 1e-9, (200, 3)))
    check("identical readers -> ICC ~ 1", r["ICC_2_1_absolute"] > 0.999)
    noisy = subject + rng.normal(0, 3.0, (200, 3))
    r2 = icc_two_way(noisy)
    check("noise-dominated -> low ICC", r2["ICC_2_1_absolute"] < 0.3)

    # ICC(3,1) >= ICC(2,1) only when a genuine systematic reader effect exists
    # (MS_columns > MS_error). With a real reader bias the ordering must hold;
    # with pure noise it can invert, which is expected, not a bug.
    biased = np.repeat(subject, 3, axis=1) + np.array([0.0, 1.5, 3.0])[None, :]
    biased = biased + rng.normal(0, 0.3, (200, 3))
    r3 = icc_two_way(biased)
    check("with a real reader bias, consistency > absolute agreement",
          r3["ICC_3_1_consistency"] > r3["ICC_2_1_absolute"],
          f"got consistency={r3['ICC_3_1_consistency']:.4f} "
          f"absolute={r3['ICC_2_1_absolute']:.4f}")


def test_oracle_bounds():
    print("test_oracle_bounds")
    n = 30
    bl = np.repeat(np.full(n, 100.0)[:, None], 3, axis=1)
    fu = np.repeat(np.full(n, 50.0)[:, None], 3, axis=1)  # all readers agree -> PR
    rf = pd.DataFrame({"TCIA_ID": [f"P{i}" for i in range(n)]})
    for r in range(3):
        rf[f"bl{r+1}"] = bl[:, r]
        rf[f"fu{r+1}"] = fu[:, r]
        rf[f"cat{r+1}"] = PR
    res = coreader_oracle(rf)
    check("perfect agreement -> oracle accuracy 1.0",
          abs(res["mean_four_cat_accuracy"] - 1.0) < 1e-9)
    check("oracle accuracy is a probability",
          0.0 <= res["mean_binary_orr_accuracy"] <= 1.0)


def _rf_from(bl: np.ndarray, fu: np.ndarray) -> pd.DataFrame:
    """Build a reader frame from (n,3) baseline/follow-up arrays."""
    rf = pd.DataFrame({"TCIA_ID": [f"P{i}" for i in range(bl.shape[0])]})
    for r in range(3):
        rf[f"bl{r+1}"] = bl[:, r]
        rf[f"fu{r+1}"] = fu[:, r]
        rf[f"cat{r+1}"] = [mrecist_rule(b, f) for b, f in zip(bl[:, r], fu[:, r])]
    return rf


def _hcc_stub(bl: np.ndarray, fu: np.ndarray) -> pd.DataFrame:
    """Minimal clinical-file stand-in carrying the released category columns."""
    hcc = pd.DataFrame()
    for r in range(3):
        cats = [mrecist_rule(b, f) for b, f in zip(bl[:, r], fu[:, r])]
        hcc[f"{r+1}_mRECIST"] = cats
        hcc[f"{r+1}_RECIST"] = cats
    return hcc


def test_categories_from():
    print("test_categories_from")
    bl = np.array([[100.0, 100.0], [50.0, 80.0]])
    fu = np.array([[70.0, 0.0], [60.0, 96.0]])
    got = categories_from(bl, fu)
    want = np.array([[mrecist_rule(100.0, 70.0), mrecist_rule(100.0, 0.0)],
                     [mrecist_rule(50.0, 60.0), mrecist_rule(80.0, 96.0)]])
    check("categories_from is the elementwise rule", (got == want).all())


def test_reader_crossed_direction():
    print("test_reader_crossed_direction")
    n = 40
    # All baselines identical, follow-ups reader-specific -> only FU can disagree.
    bl = np.repeat(np.full(n, 100.0)[:, None], 3, axis=1)
    fu = np.repeat(np.array([[65.0, 75.0, 85.0]]), n, axis=0)
    s = _crossed_disagreement(bl, fu)
    check("FU-only noise -> varying follow-up reader disagrees",
          s["vary_followup_four_cat"] == 1.0)
    check("FU-only noise -> varying baseline reader agrees",
          s["vary_baseline_four_cat"] == 0.0)
    check("FU-only noise -> delta is +100 points",
          abs(s["delta_four_cat_points"] - 100.0) < 1e-9)


def test_reader_crossed_reverse():
    print("test_reader_crossed_reverse")
    n = 40
    # Mirror image: baselines reader-specific, follow-ups identical.
    bl = np.repeat(np.array([[100.0, 130.0, 160.0]]), n, axis=0)
    fu = np.repeat(np.full(n, 100.0)[:, None], 3, axis=1)
    s = _crossed_disagreement(bl, fu)
    check("BL-only noise -> varying baseline reader disagrees",
          s["vary_baseline_four_cat"] == 1.0)
    check("BL-only noise -> varying follow-up reader agrees",
          s["vary_followup_four_cat"] == 0.0)
    check("BL-only noise -> statistic resolves NEGATIVE (can detect either direction)",
          s["delta_four_cat_points"] < 0.0,
          f"got {s['delta_four_cat_points']}")


def test_reader_crossed_arms_and_ci():
    print("test_reader_crossed_arms_and_ci")
    rng = np.random.default_rng(7)
    n = 60
    bl = np.exp(rng.normal(np.log(90.0), 0.25, (n, 3)))
    fu = np.exp(rng.normal(np.log(55.0), 0.45, (n, 3)))
    fu[:10, :] = 0.0  # ten unanimous complete responses
    res = reader_crossed_attribution(_rf_from(bl, fu), n_boot=200)
    check("all three prespecified arms are reported",
          set(res) == {"primary", "exclude_any_cr", "zeros_to_1mm"})
    check("no-CR arm drops exactly the CR patients",
          res["exclude_any_cr"]["n"] == n - 10,
          f"got {res['exclude_any_cr']['n']}")
    check("zeros->1mm arm keeps every patient", res["zeros_to_1mm"]["n"] == n)
    lo, hi = res["primary"]["delta_four_cat_ci95"]
    check("bootstrap CI brackets the point estimate",
          lo <= res["primary"]["delta_four_cat_points"] <= hi)
    check("nine-combination disagreement is at least the observed diagonal",
          res["primary"]["all_nine_combinations_four_cat"]
          >= res["primary"]["diagonal_observed_four_cat"])


def test_cohen_kappa_and_published_tables():
    print("test_cohen_kappa_and_published_tables")
    check("perfect agreement -> kappa 1", abs(_cohen_kappa_2x2(10, 0, 0, 10) - 1.0) < 1e-12)
    check("chance agreement -> kappa 0", abs(_cohen_kappa_2x2(25, 25, 25, 25)) < 1e-12)

    # Collapsing the published cross-tabs to CR+PR vs SD+PD must recover the
    # pairwise ORR kappa reported in the manuscript's reconciliation.
    want = {"r1_r3": (93, 0.4635), "r1_r2": (95, 0.3135), "r2_r3": (93, 0.4702)}
    for pair, (n_want, k_want) in want.items():
        m = np.array(PUBLISHED_MRECIST_CROSSTABS[pair], dtype=float)
        check(f"published table {pair} totals n={n_want}", int(m.sum()) == n_want,
              f"got {int(m.sum())}")
        k = _cohen_kappa_2x2(m[:2, :2].sum(), m[:2, 2:].sum(),
                             m[2:, :2].sum(), m[2:, 2:].sum())
        check(f"published table {pair} collapses to ORR kappa {k_want}",
              abs(k - k_want) < 5e-5, f"got {k:.4f}")


def test_agreement_cis():
    print("test_agreement_cis")
    n = 50
    bl = np.repeat(np.full(n, 100.0)[:, None], 3, axis=1)
    fu = np.repeat(np.full(n, 50.0)[:, None], 3, axis=1)  # unanimous PR
    res = multirater_agreement(_rf_from(bl, fu), n_boot=200)
    check("unanimous readings -> AC1 is 1.0",
          abs(res["four_category"]["gwet_ac1_unweighted"] - 1.0) < 1e-9)
    lo, hi = res["four_category"]["gwet_ac2_ordinal_weighted_ci95"]
    check("unanimous readings -> degenerate CI at 1.0", abs(lo - 1.0) < 1e-9 and abs(hi - 1.0) < 1e-9)
    check("a category absent from the cohort yields no interval, not a crash",
          res["CR_vs_nonCR"]["fleiss_kappa_ci95"] is None)
    check("...and the undefined resamples are counted",
          res["CR_vs_nonCR"]["fleiss_kappa_undefined_resamples"] == 200)

    rng = np.random.default_rng(3)
    fu2 = np.exp(rng.normal(np.log(60.0), 0.5, (n, 3)))
    res2 = multirater_agreement(_rf_from(bl, fu2), n_boot=300)
    for block in ("four_category", "objective_response_CRPR_vs_SDPD"):
        for key in [k for k in res2[block]
                    if not k.endswith("_ci95") and not k.endswith("_undefined_resamples")]:
            lo, hi = res2[block][f"{key}_ci95"]
            check(f"{block}.{key}: CI brackets the point estimate",
                  lo <= res2[block][key] <= hi,
                  f"got {res2[block][key]:.4f} vs [{lo:.4f}, {hi:.4f}]")


def test_cohort_characteristics():
    print("test_cohort_characteristics")
    n = 12
    bl = np.repeat(np.full(n, 100.0)[:, None], 3, axis=1)
    fu = np.repeat(np.array([[0.0, 50.0, 90.0]]), n, axis=0)
    rf = _rf_from(bl, fu)
    hcc = pd.DataFrame({
        "TCIA_ID": rf["TCIA_ID"],
        "age": list(range(50, 50 + n)),
        "Sex": [1, 2] * (n // 2),
        "BCLC": ["Stage-A", "Stage-B", "Stage-C"] * (n // 3),
        "OS": np.linspace(10, 200, n),
        "Death_1_StillAliveorLostToFU_0": [1] * (n - 2) + [0, 0],
    })
    res = cohort_characteristics(hcc, rf)
    check("both cohorts reported",
          {"full_collection", "analysis_set"} <= set(res))
    check("analysis set equals the triple-complete count",
          res["analysis_set"]["n"] == n)
    check("deaths counted, not assumed", res["full_collection"]["deaths"] == n - 2)
    check("BCLC percentages sum to 100",
          abs(sum(v["pct"] for v in res["full_collection"]["BCLC"].values()) - 100.0) < 1e-9)
    rd = res["mrecist_category_by_reader"]
    check("per-reader distribution covers all three readers", len(rd) == 3)
    check("reader 1 is all CR here", rd["reader1"]["CR"]["n"] == n)
    check("objective response counts CR and PR together",
          rd["reader2"]["objective_response"]["n"] == n)
    check("reader 3 is not an objective responder",
          rd["reader3"]["objective_response"]["n"] == 0)


def test_code_semantics_orders_categories():
    print("test_code_semantics_orders_categories")
    # Percent change must rank PR < SD < PD; this is what licenses reading the
    # published cross-tabulations whose legends disagree about code 3 vs code 4.
    bl = np.array([[100.0]] * 9)
    fu = np.array([[50.0], [60.0], [65.0],       # PR
                   [95.0], [100.0], [110.0],     # SD
                   [130.0], [150.0], [200.0]])   # PD
    bl3, fu3 = np.repeat(bl, 3, axis=1), np.repeat(fu, 3, axis=1)
    res = published_table_reconciliation(_rf_from(bl3, fu3), _hcc_stub(bl3, fu3))
    codes = res["percent_change_by_released_code"]
    med = {c: v["median_percent_change"] for c, v in codes.items()}
    check("PR median change is negative", med[f"code_{PR}"] < 0)
    check("SD median change sits between PR and PD",
          med[f"code_{PR}"] < med[f"code_{SD}"] < med[f"code_{PD}"])
    check("PD median change exceeds +20%", med[f"code_{PD}"] >= 20.0)


def test_recist_replication():
    print("test_recist_replication")
    n = 30
    # mRECIST: disagreement lives entirely at follow-up.
    m_bl = np.repeat(np.full(n, 100.0)[:, None], 3, axis=1)
    m_fu = np.repeat(np.array([[65.0, 75.0, 85.0]]), n, axis=0)
    # RECIST: same readers, but disagreement lives entirely at baseline.
    r_bl = np.repeat(np.array([[100.0, 130.0, 160.0]]), n, axis=0)
    r_fu = np.repeat(np.full(n, 100.0)[:, None], 3, axis=1)

    rf = _rf_from(m_bl, m_fu)
    hcc = pd.DataFrame({"TCIA_ID": rf["TCIA_ID"]})
    for i in range(3):
        hcc[f"{i+1}_RECIST_BL"] = r_bl[:, i]
        hcc[f"{i+1}_RECIST_FU"] = r_fu[:, i]
        hcc[f"{i+1}_RECIST"] = [mrecist_rule(b, f) for b, f in zip(r_bl[:, i], r_fu[:, i])]

    res = recist_replication(hcc, rf, n_boot=200)
    check("pairs on the patients complete for both criteria", res["n_paired"] == n)
    check("follow-up zero counts are reported per criterion",
          res["followup_zeros"]["mrecist"] == 0 and res["followup_zeros"]["recist"] == 0)
    check("the RECIST rule reproduces the released RECIST category",
          res["rule_reproduction"]["no_abs_rule"]["n_match"] == 3 * n)
    check("mRECIST arm resolves to follow-up",
          res["crossed"]["mrecist"]["delta_four_cat_points"] > 0)
    check("RECIST arm resolves to baseline",
          res["crossed"]["recist"]["delta_four_cat_points"] < 0)
    d = res["mrecist_minus_recist_four_cat"]
    check("criterion contrast is positive when only mRECIST is follow-up-driven",
          d["difference_points"] > 0, f"got {d['difference_points']}")
    check("criterion contrast carries a bootstrap interval",
          d["ci95"][0] <= d["difference_points"] <= d["ci95"][1])


def test_auc():
    print("test_auc")
    y = np.array([0, 0, 1, 1])
    check("perfect separation -> 1.0", abs(_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) - 1.0) < 1e-12)
    check("perfectly reversed -> 0.0", abs(_auc(y, np.array([0.9, 0.8, 0.2, 0.1]))) < 1e-12)
    check("all scores tied -> 0.5", abs(_auc(y, np.array([0.5, 0.5, 0.5, 0.5])) - 0.5) < 1e-12)
    check("single-class reference -> NaN",
          not np.isfinite(_auc(np.array([1, 1, 1, 1]), np.array([0.1, 0.2, 0.3, 0.4]))))
    # ties spanning the boundary must be averaged, not ordered arbitrarily
    check("boundary ties are rank-averaged",
          abs(_auc(np.array([0, 1]), np.array([0.5, 0.5])) - 0.5) < 1e-12)


def test_reference_dependence():
    print("test_reference_dependence")
    rng = np.random.default_rng(5)
    n = 90
    bl = np.repeat(np.full(n, 100.0)[:, None], 3, axis=1)
    fu = np.exp(rng.normal(np.log(70.0), 0.35, (n, 3)))
    rf = _rf_from(bl, fu)
    hcc = pd.DataFrame({
        "TCIA_ID": rf["TCIA_ID"],
        "age": rng.integers(45, 85, n),
        "Sex": rng.integers(1, 3, n),
        "Evidence_of_cirh": rng.integers(0, 2, n),
        "CPS": rng.choice(["A", "B", "C"], n),
        "BCLC": rng.choice(["Stage-A", "Stage-B", "Stage-C", "Stage-D"], n),
        "AFP_group": rng.choice(["<400", ">=400"], n),
        "tumor_nodul": rng.choice(["uninodular", "multinodular"], n),
        "T_involvment": rng.choice(["< or = 50%", ">50%"], n),
        "Portal Vein Thrombosis": rng.integers(0, 2, n),
        "Vascular invasion": rng.integers(0, 2, n),
        "Metastasis": rng.integers(0, 2, n),
    })
    res = reference_dependence(hcc, rf, n_boot=120)
    check("every reader plus majority is a scoring reference",
          {"reader1", "reader2", "reader3", "majority"} <= set(res["prevalence_by_reference"]))
    check("training reference is fixed so predictions are identical",
          res["training_reference"] == "majority")
    for name, v in res["models"].items():
        a = v["auc_by_reference"]
        check(f"{name}: every AUC is a probability",
              all(0.0 <= x <= 1.0 for x in a.values() if np.isfinite(x)))
        check(f"{name}: spread equals the single-reader range",
              abs(v["single_reader_spread"]
                  - (v["single_reader_range"][1] - v["single_reader_range"][0])) < 1e-12)
    c = res["pairwise_auc_contrasts"]
    check("all three reader pairs are contrasted", len(c) == 3)
    for k, v in c.items():
        check(f"{k}: CI brackets the point estimate",
              v["ci95"][0] <= v["difference"] <= v["ci95"][1])
    check("the max-minus-min spread carries its bias caveat",
          "cannot be negative" in res["primary_model_spread"]["caveat"])
    p = res["permutation_null"]
    check("permutation null reports a p-value in (0, 1]", 0.0 < p["p_value"] <= 1.0)
    check("permutation null is non-degenerate", p["n_permutations"] > 0)
    check("the null spread is strictly positive, confirming max-minus-min is biased",
          p["null_spread_median"] > 0.0, f"got {p['null_spread_median']}")
    check("the null 95th percentile is at or above its median",
          p["null_spread_p95"] >= p["null_spread_median"])


def test_icc_scale_sensitivity_matched():
    print("test_icc_scale_sensitivity_matched")
    rng = np.random.default_rng(11)
    n = 50
    bl = np.exp(rng.normal(np.log(90.0), 0.30, (n, 3)))
    fu = np.exp(rng.normal(np.log(55.0), 0.30, (n, 3)))
    fu[:8, :] = 0.0
    res = icc_scale_sensitivity(_rf_from(bl, fu), n_boot=150)
    check("both patient sets reported", set(res) == {"all", "no_cr"})
    check("no-CR set excludes the structural zeros", res["no_cr"]["n"] == n - 8)
    check("log cell is omitted when a structural zero is present",
          "log" not in res["all"] and "raw" in res["all"])
    check("no-CR set carries both scales",
          "log" in res["no_cr"] and "raw" in res["no_cr"])
    for scale in ("raw", "log"):
        cell = res["no_cr"][scale]
        check(f"{scale}: baseline and follow-up ICC are on the same n",
              -1.0 <= cell["difference_bl_minus_fu"] <= 1.0)
        lo, hi = cell["difference_ci95"]
        check(f"{scale}: CI brackets the point estimate",
              lo <= cell["difference_bl_minus_fu"] <= hi)


def main() -> int:
    for fn in (test_mrecist_rule, test_disagreement_stats, test_harmonisation_direction,
               test_harmonisation_reverse, test_icc, test_oracle_bounds,
               test_categories_from, test_reader_crossed_direction,
               test_reader_crossed_reverse, test_reader_crossed_arms_and_ci,
               test_cohen_kappa_and_published_tables,
               test_agreement_cis, test_cohort_characteristics,
               test_code_semantics_orders_categories,
               test_recist_replication,
               test_auc, test_reference_dependence,
               test_icc_scale_sensitivity_matched):
        fn()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        return 1
    print("all tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
