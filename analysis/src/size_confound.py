#!/usr/bin/env python3
"""Is the follow-up asymmetry a reader effect, or just smaller numbers?

mRECIST thresholds a ratio, (FU - BL) / BL. The sensitivity of that ratio to a
perturbation of the follow-up measurement is 1/BL; to a perturbation of the
baseline measurement it is FU/BL^2. Their ratio is BL/FU, which exceeds one
whenever the lesion shrank. So after a working treatment the rule is
arithmetically more sensitive to the follow-up reading, and an asymmetry could
in principle be manufactured without any change in reader behaviour.

The existing control in measurement_gate.equal_noise_simulation draws
MULTIPLICATIVE noise, truth * exp(N(0, sigma)). Under that model a reader's
error scales with the measurement, the two sensitivities balance exactly, and
the control correctly returns approximately zero. It therefore cannot see the
case the objection actually rests on: a fixed few-millimetre boundary
uncertainty, which is a small relative error on a 45 mm baseline and a large one
on a 15 mm residual.

Three checks, cheapest and most decisive first.

  dispersion_by_timepoint   Absolute (mm) against relative (log) between-reader
                            spread at each timepoint. Model-free. If readers
                            disagree by MORE MILLIMETRES on the physically
                            smaller post-treatment target, no shrinkage
                            geometry explains it.

  noise_control             The equal-noise simulation under additive and mixed
                            error, not only multiplicative, pushed through the
                            reader-crossed statistic so the control matches the
                            estimand the paper reports.

  shrinkage_strata          The reader-crossed contrast within tertiles of
                            observed shrinkage. Model-free. Underpowered by
                            construction; read as direction, not as an estimate.

Patients, never readings, are the resampling unit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "analysis" / "src"))

from measurement_gate import (  # noqa: E402
    RNG_SEED,
    READERS,
    _crossed_disagreement,
    _triple_complete,
    categories_from,
    disagreement_stats,
    reader_frame,
)


# --------------------------------------------------------------------------
# 1. absolute against relative dispersion
# --------------------------------------------------------------------------
def dispersion_by_timepoint(
    baseline: np.ndarray, followup: np.ndarray, n_boot: int = 5000, seed: int = RNG_SEED
) -> dict[str, Any]:
    """Between-reader SD at each timepoint, on the millimetre and log scales.

    The log rows reproduce the variances already reported in the manuscript.
    The millimetre rows are the discriminating quantity: relative spread must
    rise when a lesion shrinks under any fixed-error model, but absolute spread
    need not, and a rise in absolute spread cannot be produced by shrinkage.
    """
    if baseline.shape != followup.shape or baseline.ndim != 2:
        raise ValueError("baseline and followup must be same-shaped 2-D arrays")
    if baseline.shape[1] < 2:
        raise ValueError("at least two readers are required")
    if (baseline <= 0).any() or (followup <= 0).any():
        raise ValueError("log-scale comparison requires positive measurements")

    abs_bl = baseline.std(axis=1, ddof=1)
    abs_fu = followup.std(axis=1, ddof=1)
    log_bl = np.log(baseline).std(axis=1, ddof=1)
    log_fu = np.log(followup).std(axis=1, ddof=1)
    mean_bl = baseline.mean(axis=1)
    mean_fu = followup.mean(axis=1)

    n = len(abs_bl)
    rng = np.random.default_rng(seed)
    boot_abs, boot_log = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_abs.append(float(np.mean(abs_fu[idx] - abs_bl[idx])))
        boot_log.append(float(np.mean(log_fu[idx] - log_bl[idx])))

    def ci(values: list[float]) -> list[float]:
        return [float(v) for v in np.percentile(values, [2.5, 97.5])]

    return {
        "n": int(n),
        "mean_diameter_mm": {
            "baseline": float(mean_bl.mean()),
            "followup": float(mean_fu.mean()),
        },
        "absolute_sd_mm": {
            "baseline": float(abs_bl.mean()),
            "followup": float(abs_fu.mean()),
            "difference": float(np.mean(abs_fu - abs_bl)),
            "difference_ci95": ci(boot_abs),
        },
        "log_sd": {
            "baseline": float(log_bl.mean()),
            "followup": float(log_fu.mean()),
            "difference": float(np.mean(log_fu - log_bl)),
            "difference_ci95": ci(boot_log),
        },
        "log_variance": {
            "baseline": float(np.log(baseline).var(axis=1, ddof=1).mean()),
            "followup": float(np.log(followup).var(axis=1, ddof=1).mean()),
        },
    }


# --------------------------------------------------------------------------
# 2. noise controls under three error models
# --------------------------------------------------------------------------
def calibrate_noise(baseline: np.ndarray, followup: np.ndarray) -> dict[str, float]:
    """Pool an additive and a multiplicative error term over both timepoints.

    For the mixed model the within-patient between-reader variance is taken as

        var_i = tau^2 + sigma^2 * mean_i^2

    and tau^2, sigma^2 are recovered by regressing observed variance on squared
    mean across patients and both timepoints. Negative fitted components are
    clipped to zero, which can happen when one term carries all the signal.
    """
    var = np.concatenate([baseline.var(axis=1, ddof=1), followup.var(axis=1, ddof=1)])
    mean = np.concatenate([baseline.mean(axis=1), followup.mean(axis=1)])
    log_var = np.concatenate(
        [np.log(baseline).var(axis=1, ddof=1), np.log(followup).var(axis=1, ddof=1)]
    )

    design = np.column_stack([np.ones(len(mean)), mean**2])
    fit = np.linalg.lstsq(design, var, rcond=None)[0]
    tau2_mixed = max(float(fit[0]), 0.0)
    sigma2_mixed = max(float(fit[1]), 0.0)

    return {
        # pooled identically across both timepoints, by construction
        "sigma_multiplicative": float(np.sqrt(log_var.mean())),
        "tau_additive_mm": float(np.sqrt(var.mean())),
        "tau_mixed_mm": float(np.sqrt(tau2_mixed)),
        "sigma_mixed": float(np.sqrt(sigma2_mixed)),
    }


def _simulate(
    truth_bl: np.ndarray,
    truth_fu: np.ndarray,
    model: str,
    cal: dict[str, float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw three simulated readers whose error law is identical at both timepoints."""
    n = truth_bl.shape[0]
    if model == "multiplicative":
        s = cal["sigma_multiplicative"]
        sb = truth_bl[:, None] * np.exp(rng.normal(0.0, s, (n, 3)))
        sf = truth_fu[:, None] * np.exp(rng.normal(0.0, s, (n, 3)))
    elif model == "additive":
        t = cal["tau_additive_mm"]
        sb = truth_bl[:, None] + rng.normal(0.0, t, (n, 3))
        sf = truth_fu[:, None] + rng.normal(0.0, t, (n, 3))
    elif model == "mixed":
        t, s = cal["tau_mixed_mm"], cal["sigma_mixed"]
        sb = truth_bl[:, None] * np.exp(rng.normal(0.0, s, (n, 3))) + rng.normal(0.0, t, (n, 3))
        sf = truth_fu[:, None] * np.exp(rng.normal(0.0, s, (n, 3))) + rng.normal(0.0, t, (n, 3))
    else:
        raise ValueError(f"unknown noise model: {model}")
    # a diameter cannot be negative; clip rather than resample so the error law
    # stays identical at the two timepoints
    return np.clip(sb, 0.1, None), np.clip(sf, 0.1, None)


def noise_control(
    baseline: np.ndarray,
    followup: np.ndarray,
    n_sim: int = 2000,
    seed: int = RNG_SEED,
) -> dict[str, Any]:
    """Equal-noise control under three error models, on both estimands.

    A positive delta here is asymmetry the rule manufactures from shrinkage
    alone. Compare against the observed reader-crossed contrasts.
    """
    cal = calibrate_noise(baseline, followup)
    truth_bl = np.exp(np.log(baseline).mean(axis=1))
    truth_fu = np.exp(np.log(followup).mean(axis=1))

    out: dict[str, Any] = {"calibration": cal, "n_patients": int(len(truth_bl)), "n_sim": n_sim}
    for model in ("multiplicative", "additive", "mixed"):
        rng = np.random.default_rng(seed)
        harm, crossed_four, crossed_orr = [], [], []
        for _ in range(n_sim):
            sb, sf = _simulate(truth_bl, truth_fu, model, cal, rng)
            obs = categories_from(sb, sf)
            bl_h = categories_from(np.repeat(sb.mean(axis=1, keepdims=True), 3, axis=1), sf)
            fu_h = categories_from(sb, np.repeat(sf.mean(axis=1, keepdims=True), 3, axis=1))
            o = disagreement_stats(obs)["four_cat_disagreement"]
            harm.append(
                ((o - disagreement_stats(fu_h)["four_cat_disagreement"])
                 - (o - disagreement_stats(bl_h)["four_cat_disagreement"])) * 100.0
            )
            c = _crossed_disagreement(sb, sf)
            crossed_four.append(c["delta_four_cat_points"])
            crossed_orr.append(c["delta_orr_points"])

        def summarise(values: list[float]) -> dict[str, Any]:
            lo, hi = np.percentile(values, [2.5, 97.5])
            return {"mean": float(np.mean(values)), "ci95": [float(lo), float(hi)]}

        out[model] = {
            "harmonisation_four_cat": summarise(harm),
            "reader_crossed_four_cat": summarise(crossed_four),
            "reader_crossed_orr": summarise(crossed_orr),
        }
    return out


# --------------------------------------------------------------------------
# 3. reader-crossed contrast within shrinkage strata
# --------------------------------------------------------------------------
def shrinkage_strata(
    baseline: np.ndarray,
    followup: np.ndarray,
    n_boot: int = 5000,
    seed: int = RNG_SEED,
) -> dict[str, Any]:
    """Reader-crossed contrast by tertile of observed shrinkage.

    If the follow-up excess is only shrinkage geometry it must fall away in the
    tertile whose lesions barely changed. Tertiles hold about a third of the
    no-complete-response set each, so the intervals are wide by construction.
    """
    change = np.log(followup).mean(axis=1) - np.log(baseline).mean(axis=1)
    cuts = np.percentile(change, [100 / 3, 200 / 3])
    # tertile 0 is the most shrinkage (most negative change), 2 the least
    which = np.digitize(change, cuts)

    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {
        "tertile_cutpoints_log_change": [float(c) for c in cuts],
        "median_log_change": float(np.median(change)),
    }
    labels = {0: "most_shrinkage", 1: "middle", 2: "least_shrinkage"}
    for t, label in labels.items():
        sel = which == t
        b, f = baseline[sel], followup[sel]
        n = int(sel.sum())
        stats = _crossed_disagreement(b, f)
        boots_four, boots_orr = [], []
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            s = _crossed_disagreement(b[idx], f[idx])
            boots_four.append(s["delta_four_cat_points"])
            boots_orr.append(s["delta_orr_points"])
        out[label] = {
            "n": n,
            "median_percent_change": float(
                (np.exp(np.median(change[sel])) - 1.0) * 100.0
            ),
            "delta_four_cat_points": stats["delta_four_cat_points"],
            "delta_four_cat_ci95": [
                float(v) for v in np.percentile(boots_four, [2.5, 97.5])
            ],
            "delta_orr_points": stats["delta_orr_points"],
            "delta_orr_ci95": [float(v) for v in np.percentile(boots_orr, [2.5, 97.5])],
        }
    return out


# --------------------------------------------------------------------------
def analyse(
    data: pd.DataFrame, n_boot: int = 5000, n_sim: int = 2000, seed: int = RNG_SEED
) -> dict[str, Any]:
    readings = _triple_complete(reader_frame(data))
    baseline = readings[[f"bl{r}" for r in READERS]].to_numpy(float)
    followup = readings[[f"fu{r}" for r in READERS]].to_numpy(float)
    positive = (baseline > 0).all(axis=1) & (followup > 0).all(axis=1)
    b, f = baseline[positive], followup[positive]

    return {
        "source_cohort_n": int(len(readings)),
        "analysis_n": int(positive.sum()),
        "excluded_structural_zero_n": int((~positive).sum()),
        "dispersion": dispersion_by_timepoint(b, f, n_boot=n_boot, seed=seed),
        "noise_control": noise_control(b, f, n_sim=n_sim, seed=seed),
        "shrinkage_strata": shrinkage_strata(b, f, n_boot=n_boot, seed=seed),
    }


def render(r: dict[str, Any]) -> str:
    d = r["dispersion"]
    nc = r["noise_control"]
    ss = r["shrinkage_strata"]
    lines = [
        "# Size-confound checks",
        "",
        f"**Patients:** {r['analysis_n']} with three positive readings at both timepoints "
        f"({r['excluded_structural_zero_n']} structural-zero patients excluded from "
        f"{r['source_cohort_n']}).",
        "",
        "## 1. Between-reader spread by timepoint, absolute and relative",
        "",
        "| Quantity | Baseline | Follow-up | Difference | 95% CI |",
        "|---|---:|---:|---:|---:|",
        f"| Mean diameter, mm | {d['mean_diameter_mm']['baseline']:.1f} | "
        f"{d['mean_diameter_mm']['followup']:.1f} | | |",
        f"| Between-reader SD, mm | {d['absolute_sd_mm']['baseline']:.2f} | "
        f"{d['absolute_sd_mm']['followup']:.2f} | {d['absolute_sd_mm']['difference']:+.2f} | "
        f"{d['absolute_sd_mm']['difference_ci95'][0]:.2f} to "
        f"{d['absolute_sd_mm']['difference_ci95'][1]:.2f} |",
        f"| Between-reader SD, log | {d['log_sd']['baseline']:.3f} | "
        f"{d['log_sd']['followup']:.3f} | {d['log_sd']['difference']:+.3f} | "
        f"{d['log_sd']['difference_ci95'][0]:.3f} to "
        f"{d['log_sd']['difference_ci95'][1]:.3f} |",
        "",
        "Relative spread must rise when a lesion shrinks under any fixed-error model.",
        "Absolute spread need not, and a rise in it cannot be produced by shrinkage.",
        "",
        "## 2. Equal-noise control under three error models",
        "",
        f"Calibration: sigma_multiplicative {nc['calibration']['sigma_multiplicative']:.3f}, "
        f"tau_additive {nc['calibration']['tau_additive_mm']:.2f} mm, "
        f"mixed (tau {nc['calibration']['tau_mixed_mm']:.2f} mm, "
        f"sigma {nc['calibration']['sigma_mixed']:.3f}).",
        "",
        "| Error model | Harmonisation, 4-cat | Reader-crossed, 4-cat | Reader-crossed, ORR |",
        "|---|---:|---:|---:|",
    ]
    for model in ("multiplicative", "additive", "mixed"):
        m = nc[model]
        cells = []
        for key in ("harmonisation_four_cat", "reader_crossed_four_cat", "reader_crossed_orr"):
            v = m[key]
            cells.append(f"{v['mean']:+.1f} ({v['ci95'][0]:.1f} to {v['ci95'][1]:.1f})")
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Percentage points. A positive value is asymmetry the rule manufactures from",
        "shrinkage alone under an error law applied identically at both timepoints.",
        "",
        "## 3. Reader-crossed contrast by tertile of observed shrinkage",
        "",
        "| Tertile | n | Median change | 4-cat difference | ORR difference |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in ("most_shrinkage", "middle", "least_shrinkage"):
        s = ss[label]
        lines.append(
            f"| {label.replace('_', ' ')} | {s['n']} | {s['median_percent_change']:+.0f}% | "
            f"{s['delta_four_cat_points']:+.1f} "
            f"({s['delta_four_cat_ci95'][0]:.1f} to {s['delta_four_cat_ci95'][1]:.1f}) | "
            f"{s['delta_orr_points']:+.1f} "
            f"({s['delta_orr_ci95'][0]:.1f} to {s['delta_orr_ci95'][1]:.1f}) |"
        )
    lines += [
        "",
        "Tertiles carry about a third of the set each, so these intervals are wide by",
        "construction and are read as direction rather than as estimates.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--boot", type=int, default=5000)
    parser.add_argument("--sim", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    args = parser.parse_args()

    data = pd.read_excel(
        args.data / "HCC-TACE-Seg_clinical_data-V2.xlsx", sheet_name="data table"
    )
    result = analyse(data, n_boot=args.boot, n_sim=args.sim, seed=args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "SIZE_CONFOUND.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.out / "SIZE_CONFOUND.md").write_text(render(result), encoding="utf-8")
    print(render(result))


if __name__ == "__main__":
    main()
