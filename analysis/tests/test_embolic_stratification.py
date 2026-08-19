"""Tests for the embolic-agent stratification.

The claim the manuscript rests on is a null: the follow-up excess does not differ
between the oil-based and the radiolucent arm. A null is only worth reporting if
the machinery could have found a difference, so most of these tests plant one and
require it to be recovered.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embolic_stratification import arm_stats, stratify, tace_bucket


# -- regimen classification -------------------------------------------------

def test_bucket_reads_beads_as_deb_and_everything_named_as_conventional():
    assert tace_bucket("doxorubicin LC beads") == "DEB-TACE"
    assert tace_bucket("Cisplatin, doxorubicin, Mitomycin-C") == "cTACE"
    assert tace_bucket("Cisplatin, Mitomycin-C") == "cTACE"
    assert tace_bucket("Cisplastin") == "cTACE"


def test_bucket_never_imputes_a_missing_regimen():
    """17 patients have no regimen. They must form their own stratum, not be
    silently folded into the larger arm, which would fabricate the contrast."""
    assert tace_bucket(np.nan) == "unrecorded"
    assert tace_bucket(pd.NA) == "unrecorded"


def test_bucket_is_case_insensitive():
    assert tace_bucket("DOXORUBICIN LC BEADS") == "DEB-TACE"


# -- the estimator recovers a planted asymmetry ------------------------------

def _cohort(n, fu_noise, bl_noise, seed, proportional=False):
    """Readers agree at one timepoint and disagree at the other, by construction.

    With proportional=True the noise is a fixed fraction of the truth rather than
    a fixed number of millimetres. That distinction is the whole size-confound
    argument: only the proportional law is exchangeable between a large baseline
    and a smaller follow-up.
    """
    rng = np.random.default_rng(seed)
    truth_bl = rng.uniform(60.0, 120.0, n)
    truth_fu = truth_bl * 0.65
    if proportional:
        bl = truth_bl[:, None] * np.exp(rng.normal(0.0, bl_noise, (n, 3)))
        fu = truth_fu[:, None] * np.exp(rng.normal(0.0, fu_noise, (n, 3)))
    else:
        bl = truth_bl[:, None] + rng.normal(0.0, bl_noise, (n, 3))
        fu = truth_fu[:, None] + rng.normal(0.0, fu_noise, (n, 3))
    return np.clip(bl, 1.0, None), np.clip(fu, 0.0, None)


def test_follow_up_noise_produces_a_positive_excess():
    bl, fu = _cohort(300, fu_noise=12.0, bl_noise=1.0, seed=3)
    s = arm_stats(bl, fu, n_boot=300, rng=np.random.default_rng(0))
    assert s["delta_orr_points"] > 0
    assert s["excludes_zero"]


def test_baseline_noise_produces_a_negative_excess():
    """The reverse-direction check: move the disagreement and the sign follows."""
    bl, fu = _cohort(300, fu_noise=1.0, bl_noise=12.0, seed=4)
    s = arm_stats(bl, fu, n_boot=300, rng=np.random.default_rng(0))
    assert s["delta_orr_points"] < 0


def test_equal_proportional_error_produces_no_excess():
    """The genuine null. A proportional law applied identically at both timepoints
    is exchangeable between them, so the estimator must return nothing."""
    bl, fu = _cohort(300, fu_noise=0.18, bl_noise=0.18, seed=5, proportional=True)
    s = arm_stats(bl, fu, n_boot=300, rng=np.random.default_rng(0))
    assert not s["excludes_zero"]


def test_equal_additive_error_does_produce_an_excess_on_a_shrinking_cohort():
    """Not a defect: this is the size confound the objection rests on, and the
    reason the stratification below is read as a between-arm contrast. Both arms
    shrink, so the confound is common to them and differences it out."""
    bl, fu = _cohort(300, fu_noise=6.0, bl_noise=6.0, seed=5)
    s = arm_stats(bl, fu, n_boot=300, rng=np.random.default_rng(0))
    assert s["delta_orr_points"] > 0


# -- the stratified contrast ------------------------------------------------

def _frame(regimens, seed=7, fu_noise=(10.0, 10.0)):
    """Minimal HCC-TACE-Seg-shaped frame: the columns reader_frame() reads."""
    rng = np.random.default_rng(seed)
    n = len(regimens)
    bl = rng.uniform(60.0, 120.0, n)
    fu = bl * 0.65
    cols = {"TCIA_ID": [f"HCC_{i:03d}" for i in range(n)], "chemotherapy": regimens}
    for r in (1, 2, 3):
        noise = np.array([fu_noise[0] if "bead" in str(g).lower() else fu_noise[1]
                          for g in regimens])
        cols[f"{r}_mRECIST_BL"] = np.clip(bl + rng.normal(0, 1.0, n), 1.0, None)
        cols[f"{r}_mRECIST_FU"] = np.clip(fu + rng.normal(0, 1.0, n) * noise / 10.0, 0.0, None)
        cols[f"{r}_mRECIST"] = 3
    return pd.DataFrame(cols)


def test_strata_partition_the_cohort_without_loss():
    regimens = ["doxorubicin LC beads"] * 20 + ["Cisplatin, doxorubicin, Mitomycin-C"] * 30 + [None] * 10
    res = stratify(_frame(regimens), n_boot=100)
    s = res["strata_sizes"]
    assert s["DEB-TACE"] + s["cTACE"] + s["unrecorded"] == res["n_complete_readings"] == 60


def test_contrast_recovers_a_planted_between_arm_difference():
    """If oil really did drive the excess, this is the shape the result would take.
    The test exists so that the reported null cannot be an insensitive estimator."""
    regimens = ["doxorubicin LC beads"] * 45 + ["Cisplatin, doxorubicin, Mitomycin-C"] * 45
    res = stratify(_frame(regimens, fu_noise=(1.0, 30.0)), n_boot=400)
    assert res["cTACE_minus_DEB"]["difference_points"] > 0
    assert res["cTACE_minus_DEB"]["excludes_zero"]


def test_a_stratum_smaller_than_five_is_skipped_not_estimated():
    regimens = ["doxorubicin LC beads"] * 3 + ["Cisplatin, doxorubicin, Mitomycin-C"] * 40
    res = stratify(_frame(regimens), n_boot=50)
    assert res["arms"]["DEB-TACE"]["skipped"]
    assert "cTACE_minus_DEB" not in res


def test_results_are_deterministic_under_a_fixed_seed():
    regimens = ["doxorubicin LC beads"] * 25 + ["Cisplatin, doxorubicin, Mitomycin-C"] * 35
    a = stratify(_frame(regimens), n_boot=200)
    b = stratify(_frame(regimens), n_boot=200)
    assert a["arms"]["cTACE"]["delta_orr_ci95"] == b["arms"]["cTACE"]["delta_orr_ci95"]
