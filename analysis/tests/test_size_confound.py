import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from size_confound import (
    calibrate_noise,
    dispersion_by_timepoint,
    noise_control,
    shrinkage_strata,
)


def _shrinking_cohort(n=60, shrink=0.6, tau=0.0, sigma=0.0, seed=11):
    """Baseline/follow-up arrays whose error law is identical at both timepoints."""
    rng = np.random.default_rng(seed)
    truth_bl = rng.uniform(50.0, 120.0, n)
    truth_fu = truth_bl * shrink
    def draw(truth):
        out = truth[:, None] * np.exp(rng.normal(0.0, sigma, (n, 3))) if sigma else np.repeat(
            truth[:, None], 3, axis=1
        )
        if tau:
            out = out + rng.normal(0.0, tau, (n, 3))
        return np.clip(out, 0.1, None)
    return draw(truth_bl), draw(truth_fu)


# -- check 1: the discriminating behaviour of the two scales -----------------

def test_additive_error_holds_absolute_spread_and_inflates_relative_spread():
    """A fixed-millimetre error is the objection's error law. It must show up as
    equal absolute spread and larger relative spread on the smaller timepoint."""
    bl, fu = _shrinking_cohort(n=400, shrink=0.5, tau=8.0)
    d = dispersion_by_timepoint(bl, fu, n_boot=200)
    assert abs(d["absolute_sd_mm"]["difference"]) < 1.0        # absolute spread flat
    assert d["log_sd"]["difference"] > 0.1                     # relative spread up
    assert d["log_sd"]["difference_ci95"][0] > 0


def test_multiplicative_error_holds_relative_spread_and_shrinks_absolute_spread():
    """The mirror image, and the reason the original control could not see the
    objection: under multiplicative error the relative scale is the flat one."""
    bl, fu = _shrinking_cohort(n=400, shrink=0.5, sigma=0.25)
    d = dispersion_by_timepoint(bl, fu, n_boot=200)
    assert abs(d["log_sd"]["difference"]) < 0.05               # relative spread flat
    assert d["absolute_sd_mm"]["difference"] < -1.0            # absolute spread down


def test_dispersion_reverses_when_timepoints_are_swapped():
    bl, fu = _shrinking_cohort(n=200, shrink=0.5, tau=6.0)
    forward = dispersion_by_timepoint(bl, fu, n_boot=100)
    reverse = dispersion_by_timepoint(fu, bl, n_boot=100)
    np.testing.assert_allclose(
        forward["absolute_sd_mm"]["difference"],
        -reverse["absolute_sd_mm"]["difference"],
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        forward["log_sd"]["difference"], -reverse["log_sd"]["difference"], rtol=1e-10
    )


def test_nonpositive_measurements_fail():
    with pytest.raises(ValueError, match="positive"):
        dispersion_by_timepoint(np.array([[10.0, 11.0, 12.0]]), np.array([[0.0, 4.0, 5.0]]))


def test_single_reader_fails():
    with pytest.raises(ValueError, match="two readers"):
        dispersion_by_timepoint(np.array([[10.0]]), np.array([[8.0]]))


# -- check 2: calibration and the controls ----------------------------------

def test_calibration_recovers_a_known_additive_term():
    bl, fu = _shrinking_cohort(n=800, shrink=0.6, tau=10.0)
    cal = calibrate_noise(bl, fu)
    assert 7.0 < cal["tau_mixed_mm"] < 13.0
    assert cal["sigma_mixed"] < 0.10          # no multiplicative component present


def test_calibration_recovers_a_known_multiplicative_term():
    bl, fu = _shrinking_cohort(n=800, shrink=0.6, sigma=0.20)
    cal = calibrate_noise(bl, fu)
    assert 0.15 < cal["sigma_mixed"] < 0.26
    assert 0.14 < cal["sigma_multiplicative"] < 0.26


def test_multiplicative_control_is_centred_on_zero_but_additive_is_not():
    """The load-bearing test. Both models apply ONE error law identically at both
    timepoints, so any positive contrast is manufactured by the rule from
    shrinkage alone. Multiplicative error balances the rule's two sensitivities;
    additive error does not, which is exactly the gap the original control left."""
    bl, fu = _shrinking_cohort(n=120, shrink=0.55, tau=9.0, sigma=0.18)
    out = noise_control(bl, fu, n_sim=60)
    mult = out["multiplicative"]["reader_crossed_orr"]["mean"]
    add = out["additive"]["reader_crossed_orr"]["mean"]
    assert abs(mult) < 5.0
    assert add > mult


def test_controls_vanish_when_the_lesion_does_not_shrink():
    """With no shrinkage there is no geometric asymmetry for any error law."""
    bl, fu = _shrinking_cohort(n=120, shrink=1.0, tau=9.0)
    out = noise_control(bl, fu, n_sim=60)
    for model in ("multiplicative", "additive", "mixed"):
        assert abs(out[model]["reader_crossed_orr"]["mean"]) < 6.0


# -- check 3: strata ---------------------------------------------------------

def test_shrinkage_strata_partition_the_cohort_and_order_by_change():
    rng = np.random.default_rng(3)
    n = 90
    bl = rng.uniform(50.0, 120.0, (n, 3))
    fu = bl * rng.uniform(0.2, 1.1, (n, 1))
    out = shrinkage_strata(bl, fu, n_boot=50)
    counts = [out[k]["n"] for k in ("most_shrinkage", "middle", "least_shrinkage")]
    assert sum(counts) == n
    assert (
        out["most_shrinkage"]["median_percent_change"]
        < out["middle"]["median_percent_change"]
        < out["least_shrinkage"]["median_percent_change"]
    )
