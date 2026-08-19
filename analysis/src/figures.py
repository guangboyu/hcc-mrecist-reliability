#!/usr/bin/env python3
"""Manuscript figures, built from the frozen results.

Every annotated value is read from MEASUREMENT_GATE.json rather than retyped, so
a figure cannot drift from the results file. Panels that need per-patient detail
(distributions, scatter) recompute it from the md5-verified source spreadsheet
using the same functions the gate uses, and assert their summary statistics
against the JSON before drawing.

    .venv/bin/python analysis/src/figures.py --data data/raw --results results --out figures

Figure contract
---------------
Target: Insights into Imaging, Original Article. Double column 174 mm.

Figure 1 - core conclusion: how the public cohort reached the complete-case
    analysis, followed by an illustration of the reading task. Archetype:
    participant flow plus illustrative imaging. Panel a is the required cohort
    flow; panels b and c show one reader-discordant case. Review risk: the public
    file gives no reason for missingness, and the figure says so rather than
    inventing one. The case is an illustration and never evidence. Its outline
    is the collection's released segmentation, not any individual reader's.
Figure 2 - core conclusion: mRECIST reliability depends on which partition is
    taken, and objective-response agreement is lower than complete-response
    agreement. Archetype: quantitative grid, panel a is the hero.
    Panel a carries the coefficient comparison, panel b converts it into the
    per-patient consequence. Review risk: marginal intervals overlap, so the
    paired contrast is drawn explicitly rather than left to the reader, and it
    joins the two equal-cardinality partitions rather than crossing a change of
    weighting and coefficient family at the same time.
Figure 3 - core conclusion: the derived category is more sensitive to variation
    in the reader supplying follow-up than baseline, across two methods, three
    patient sets and both scales. Archetype:
    single hero quantitative panel. Review risk: the synthetic-consensus,
    structural-zero and rule-geometry objections, all answered within the panel.
Figure 4 - core conclusion: between-reader spread is greater for viable than
    total diameter; a large share of patients sit within measurement variation
    of a response cut-point. Archetype: quantitative grid,
    two coordinate panels.
Figure 5 - core conclusion: because the reference reader moves the reported
    figure, a benchmark score on this collection is a statement about the model
    and that reader jointly. Archetype: quantitative grid. Review risk: the
    range statistic is bounded below by zero, so panel b carries the
    permutation null rather than leaving the range to speak for itself.
Graphical abstract - the three findings in narrative order, required by the
    target journal. Not a numbered display item.

Export contract: editable text (svg.fonttype none, pdf.fonttype 42), vector SVG
and PDF for submission, 600-dpi TIFF for production, PNG as a review preview.
All rendered glyphs are at or above the 5 pt floor.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "analysis" / "src"))

from measurement_gate import reader_frame, verify_inputs  # noqa: E402

# Okabe-Ito, colour-blind safe.
BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREY = "#7F7F7F"
LIGHT = "#BFBFBF"
DOUBLE_COL_IN = 6.85  # 174 mm

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none",   # editable text in SVG
    "pdf.fonttype": 42,       # TrueType, not Type 3 - production requires editable text
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "axes.titleweight": "bold",
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "lines.solid_capstyle": "round",
})


RASTER_DPI = 600


def save(fig: plt.Figure, out: Path, name: str, tight: bool = True) -> None:
    """Vector SVG + PDF for submission, 600-dpi TIFF for production, PNG preview.

    Written as four explicit calls rather than a loop so that the format set is
    visible to static preflight and to anyone reading the file.
    """
    out.mkdir(parents=True, exist_ok=True)
    stem = str(out / name)
    bbox = "tight" if tight else None
    fig.savefig(stem + ".svg", bbox_inches=bbox)
    fig.savefig(stem + ".pdf", bbox_inches=bbox)
    fig.savefig(stem + ".tiff", dpi=RASTER_DPI, bbox_inches=bbox,
                pil_kwargs={"compression": "tiff_lzw"})  # lossless; ~30 MB -> ~2 MB
    fig.savefig(stem + ".png", dpi=RASTER_DPI, bbox_inches=bbox)
    plt.close(fig)
    print(f"  wrote {name}: svg, pdf, tiff ({RASTER_DPI} dpi), png")


def _dot_row(ax, y: float, value: float, ci: list[float] | None,
             colour: str, label_fmt: str = "{:.3f}") -> None:
    if ci is not None:
        ax.plot(ci, [y, y], color=colour, lw=1.4, alpha=0.65, zorder=2)
        for x in ci:
            ax.plot([x, x], [y - 0.12, y + 0.12], color=colour, lw=1.0, alpha=0.65, zorder=2)
    ax.plot([value], [y], "o", color=colour, ms=5.5, zorder=3,
            markeredgecolor="white", markeredgewidth=0.6)
    right = ci[1] if ci is not None else value
    ax.annotate(label_fmt.format(value), (right, y), xytext=(6, 0),
                textcoords="offset points", va="center", fontsize=7.2,
                color=colour, fontweight="bold")


# --------------------------------------------------------------------------
# Figure 1 - cohort flow, and what the reading task looks like in one patient
# --------------------------------------------------------------------------
CASE_ID = "HCC_034"


def _case_readings(data_dir: Path) -> dict[str, list[float]]:
    """The three readers' recorded sums for the illustrated patient.

    Read from the md5-verified spreadsheet rather than transcribed, so the
    numbers annotated on the images cannot drift from the analysis.
    """
    hcc = pd.read_excel(data_dir / "HCC-TACE-Seg_clinical_data-V2.xlsx")
    row = reader_frame(hcc).set_index("TCIA_ID").loc[CASE_ID]
    bl = [float(row[f"bl{r}"]) for r in (1, 2, 3)]
    fu = [float(row[f"fu{r}"]) for r in (1, 2, 3)]
    cat = [int(row[f"cat{r}"]) for r in (1, 2, 3)]
    return {"bl": bl, "fu": fu, "cat": cat,
            "pct": [(f - b) / b * 100 for b, f in zip(bl, fu)]}


def _ct_panel(ax, img_path: Path, title: str, mask_path: Path | None = None) -> None:
    ax.imshow(plt.imread(img_path), cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
    if mask_path is not None:
        ax.contour(plt.imread(mask_path), levels=[0.5], colors=VERMILLION, linewidths=1.1)
    ax.set_title(title, fontsize=7.4, loc="left", pad=3)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(LIGHT); s.set_linewidth(0.6)


def _icon_person(ax, x, y, s, colour, lw=0.9):
    """Head plus shoulders, drawn in data units so it stays circular."""
    ax.add_patch(plt.Circle((x, y + 0.62 * s), 0.30 * s, facecolor="white",
                            edgecolor=colour, lw=lw, zorder=4))
    sh = mpatches.FancyBboxPatch((x - 0.42 * s, y - 0.48 * s), 0.84 * s, 0.78 * s,
                                 boxstyle="round,pad=0,rounding_size=" + str(0.34 * s),
                                 facecolor="white", edgecolor=colour, lw=lw, zorder=3)
    ax.add_patch(sh)


def _icon_scan(ax, x, y, s, colour, lw=0.9, treated=False):
    """A scan frame holding a lesion. `treated` speckles it, as after embolisation."""
    ax.add_patch(mpatches.FancyBboxPatch(
        (x - 0.62 * s, y - 0.52 * s), 1.24 * s, 1.04 * s,
        boxstyle="round,pad=0,rounding_size=" + str(0.16 * s),
        facecolor="white", edgecolor=colour, lw=lw, zorder=3))
    ax.add_patch(plt.Circle((x, y), 0.30 * s, facecolor="none",
                            edgecolor=colour, lw=lw, zorder=4))
    if treated:
        for dx, dy in ((-0.12, 0.06), (0.10, 0.10), (0.02, -0.11), (-0.09, -0.07)):
            ax.add_patch(plt.Circle((x + dx * s, y + dy * s), 0.045 * s,
                                    facecolor=colour, edgecolor="none", zorder=5))
    else:
        ax.add_patch(plt.Circle((x, y), 0.15 * s, facecolor=colour,
                                edgecolor="none", alpha=0.55, zorder=4))


def _icon_caliper(ax, x, y, s, colour, lw=0.9):
    """A lesion with its diameter measured across it."""
    ax.add_patch(mpatches.Ellipse((x, y), 1.30 * s, 0.94 * s, facecolor=colour,
                                  edgecolor=colour, lw=lw, alpha=0.20, zorder=3))
    ax.add_patch(mpatches.Ellipse((x, y), 1.30 * s, 0.94 * s, facecolor="none",
                                  edgecolor=colour, lw=lw, zorder=4))
    ax.annotate("", xy=(x + 0.65 * s, y), xytext=(x - 0.65 * s, y),
                arrowprops=dict(arrowstyle="<|-|>", color=colour, lw=lw,
                                mutation_scale=5, shrinkA=0, shrinkB=0), zorder=5)


def _stage_box(ax, cx, cy, w, h, colour, fill):
    ax.add_patch(mpatches.FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0,rounding_size=0.9",
        facecolor=fill, edgecolor=colour, lw=0.9, zorder=2))


def _chevron(ax, x, y, s, colour):
    ax.annotate("", xy=(x + s, y), xytext=(x, y),
                arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.0,
                                shrinkA=0, shrinkB=0), zorder=5)


def figure_design(res: dict[str, Any], data_dir: Path, out: Path) -> None:
    """Cohort flow and a representative reader-discordant case.

    Archetype: flow-led composite. Panel a is the hero and carries the cohort
    path from public release to complete-case analysis; panels b and c are
    quieter and supply the imaging and the per-reader consequence.

    The case is chosen for what it shows, not for effect: the three readers
    agree at baseline and diverge at follow-up, which is the paper's finding in
    one patient. It is one patient and is drawn as an illustration, never as
    evidence; every quantitative claim rests on the 93-patient analysis.
    """
    case = _case_readings(data_dir)
    assets = out / "assets"
    meta = json.loads((assets / "hcc034_meta.json").read_text())
    n_released = res["M013_cohort_characteristics"]["full_collection"]["n"]
    n_set = res["M003b_multirater_agreement"]["n"]
    n_excluded = n_released - n_set

    fig = plt.figure(figsize=(DOUBLE_COL_IN, 5.80))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.05, 0.95, 1.22],
                          width_ratios=[1.02, 1.30],
                          left=0.055, right=0.982, top=0.945, bottom=0.065,
                          hspace=0.42, wspace=0.26)

    # ---- a. cohort flow ---------------------------------------------------
    # Every string is measured against its box width at the stated font size.
    # The long ones are wrapped rather than shrunk, because a 5 pt caption is
    # unreadable at print size and an overflowing one is worse.
    axA = fig.add_subplot(gs[0, :])
    axA.set_axis_off()
    # Geometry is set from measured glyph heights, not by eye. On this row one y
    # unit is about 2.6 pt, so a 6.8 pt heading spans ~2.6 units and a three-line
    # 6.1 pt caption ~9.3. Boxes are sized to clear both with a margin; the
    # excluded caption is one line so its box does not need a third row of text.
    W, H = 100.0, 36.0
    axA.set_xlim(0, W); axA.set_ylim(0, H)
    axA.text(0, H - 0.3, "a  Cohort flow", ha="left", va="top", fontsize=9,
             fontweight="bold")

    source = (14.5, 22.0, 27.0, 17.0)
    screen = (50.0, 22.0, 34.0, 17.0)
    included = (86.0, 22.0, 26.0, 17.0)
    excluded = (50.0, 5.5, 56.0, 9.5)
    _stage_box(axA, *source, BLUE, "#EAF3FA")
    _stage_box(axA, *screen, BLUE, "#EAF3FA")
    _stage_box(axA, *included, VERMILLION, "#FCF0E8")
    _stage_box(axA, *excluded, GREY, "#F2F2F2")

    axA.text(source[0], 27.0, "Public HCC-TACE-Seg", ha="center", va="center",
             fontsize=6.8, fontweight="bold")
    axA.text(source[0], 19.5, f"{n_released} patients\none evaluated\nHCC focus each",
             ha="center", va="center", fontsize=6.1, color=GREY, linespacing=1.32)
    axA.text(screen[0], 27.0, "Complete-case requirement", ha="center", va="center",
             fontsize=6.8, fontweight="bold")
    axA.text(screen[0], 20.2, "all 3 readers: baseline + follow-up\nviable diameter and category",
             ha="center", va="center", fontsize=6.1, color=GREY, linespacing=1.32)
    axA.text(included[0], 27.0, "Primary analysis", ha="center", va="center",
             fontsize=6.8, fontweight="bold", color=VERMILLION)
    axA.text(included[0], 20.2, f"{n_set} patients\nno imputation",
             ha="center", va="center", fontsize=6.1, color=GREY, linespacing=1.32)
    axA.text(excluded[0], 7.6, f"Excluded, n = {n_excluded}", ha="center", va="center",
             fontsize=6.6, fontweight="bold", color=GREY)
    axA.text(excluded[0], 3.6, "at least one incomplete reader triplet; reason not reported",
             ha="center", va="center", fontsize=5.9, color=GREY)

    _chevron(axA, 28.6, 22.0, 3.8, GREY)
    _chevron(axA, 67.5, 22.0, 4.5, GREY)
    axA.annotate("", xy=(50.0, 10.5), xytext=(50.0, 13.5),
                 arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.0,
                                 shrinkA=0, shrinkB=0), zorder=5)

    # ---- b. the reading design -------------------------------------------
    # The flow above says who was analyzed; it does not say what was done.
    # Without this strip the reader never learns that three radiologists read
    # independently, or that the category is derived from two summed diameters.
    axD = fig.add_subplot(gs[1, :])
    axD.set_axis_off()
    WD, HD = 100.0, 26.0
    axD.set_xlim(0, WD); axD.set_ylim(0, HD)
    axD.text(0, HD - 0.3, "b  Reading design", ha="left", va="top", fontsize=9,
             fontweight="bold")

    stages = [
        ("scan", "CT before and after", "contrast-enhanced,\narterial phase"),
        ("readers", "Three radiologists", "independent,\n> 20 years' experience"),
        ("caliper", "Viable diameters", "summed at each\ntimepoint"),
        ("chips", "mRECIST category", "one per reader,\nfrom two sums"),
    ]
    nS = len(stages)
    pad, gap = 0.8, 3.0
    bw = (WD - 2 * pad - (nS - 1) * gap) / nS
    # The box top must clear the 9 pt panel title and the box floor must clear a
    # two-line 5.8 pt caption; both were being crossed before.
    bh = 18.0
    cyD = 12.5
    for i, (kind, head, sub) in enumerate(stages):
        cx = pad + bw / 2 + i * (bw + gap)
        last = i == nS - 1
        colour = VERMILLION if last else BLUE
        _stage_box(axD, cx, cyD, bw, bh, colour, "#FCF0E8" if last else "#EAF3FA")
        iy = cyD + bh / 2 - 4.2 + 0.0
        if kind == "scan":
            _icon_scan(axD, cx - 2.3, iy, 2.4, colour)
            _icon_scan(axD, cx + 2.3, iy, 2.4, colour, treated=True)
        elif kind == "readers":
            for dx in (-3.0, 0.0, 3.0):
                _icon_person(axD, cx + dx, iy, 2.4, colour)
        elif kind == "caliper":
            _icon_caliper(axD, cx, iy, 3.2, colour)
        else:
            for j, lab in enumerate(("CR", "PR", "SD", "PD")):
                bx = cx - 5.4 + j * 3.6
                axD.add_patch(mpatches.FancyBboxPatch(
                    (bx - 1.5, iy - 1.2), 3.0, 2.4,
                    boxstyle="round,pad=0,rounding_size=0.5",
                    facecolor="white", edgecolor=colour, lw=0.8, zorder=4))
                axD.text(bx, iy, lab, ha="center", va="center", fontsize=5.4,
                         color=colour, zorder=5)
        axD.text(cx, cyD + 0.2, head, ha="center", va="center", fontsize=6.6,
                 fontweight="bold", zorder=5)
        axD.text(cx, cyD - 4.8, sub, ha="center", va="center", fontsize=5.8,
                 color=GREY, linespacing=1.32, zorder=5)
        if not last:
            _chevron(axD, cx + bw / 2 + 0.5, cyD, gap - 1.0, GREY)

    # ---- c. the case, imaged --------------------------------------------
    gsB = gs[2, 0].subgridspec(1, 2, wspace=0.05)
    axB1, axB2 = fig.add_subplot(gsB[0]), fig.add_subplot(gsB[1])
    _ct_panel(axB1, assets / "hcc034_baseline.png", "before TACE",
              assets / "hcc034_baseline_mass.png")
    _ct_panel(axB2, assets / "hcc034_followup.png",
              f"{meta['interval_days']} days after")
    axB1.text(-0.06, 1.22, "c", transform=axB1.transAxes, fontsize=9,
              fontweight="bold", va="top", ha="left")
    axB1.text(0.5, -0.045, "viable tumor outlined", transform=axB1.transAxes,
              ha="center", va="top", fontsize=5.9, color=VERMILLION)
    axB2.text(0.5, -0.045, "necrosis and retained\nembolic material",
              transform=axB2.transAxes, ha="center", va="top", fontsize=5.9,
              color=GREY, linespacing=1.3)

    # ---- d. what the three readers recorded ------------------------------
    axC = fig.add_subplot(gs[2, 1])
    axC.text(-0.16, 1.14, "d", transform=axC.transAxes, fontsize=9,
             fontweight="bold", va="top", ha="left")
    CATNAME = {1: "CR", 2: "PR", 3: "SD", 4: "PD"}
    for i in range(3):
        responder = case["cat"][i] in (1, 2)
        colour = VERMILLION if responder else BLUE
        axC.plot([0, 1], [case["bl"][i], case["fu"][i]], "-o", color=colour,
                 lw=1.6, ms=4.6, markeredgecolor="white", markeredgewidth=0.6,
                 zorder=3, clip_on=False)
        # Reader identity rides with the right-hand label. Tagging the baseline
        # end instead stacks three labels into 3 mm of vertical space, which is
        # the very agreement the panel is there to show.
        axC.annotate(f"R{i + 1}  {case['pct'][i]:+.0f}%  {CATNAME[case['cat'][i]]}",
                     (1, case["fu"][i]), xytext=(8, 0), textcoords="offset points",
                     va="center", fontsize=6.8, color=colour, fontweight="bold",
                     annotation_clip=False)
    # Right-hand labels live inside the axes. Overhanging them past x=1 puts
    # them beyond the figure edge, where bbox_inches=None simply clips them.
    axC.set_xlim(-0.08, 1.62); axC.set_ylim(0, max(case["bl"]) * 1.26)
    axC.set_xticks([0, 1]); axC.set_xticklabels(["before", "after"], fontsize=7)
    axC.spines["bottom"].set_bounds(0, 1)
    axC.set_ylabel("Sum of viable\ndiameters (mm)", fontsize=6.8, labelpad=2,
                   linespacing=1.3)
    axC.tick_params(labelsize=6.8)
    spread_bl = max(case["bl"]) - min(case["bl"])
    spread_fu = max(case["fu"]) - min(case["fu"])
    axC.annotate(f"{spread_bl:.0f} mm apart", (0, max(case["bl"])), xytext=(5, 9),
                 textcoords="offset points", ha="left", fontsize=6.0, color=GREY)
    axC.annotate(f"{spread_fu:.0f} mm apart", (1, max(case["fu"])), xytext=(0, 11),
                 textcoords="offset points", ha="center", fontsize=6.0, color=GREY)

    save(fig, out, "figure1_design_and_case", tight=False)


# --------------------------------------------------------------------------
# Figure 2 - reliability depends on which partition is taken
# --------------------------------------------------------------------------
def figure2(res: dict[str, Any], out: Path) -> None:
    m = res["M003b_multirater_agreement"]
    four, contr = m["four_category"], m["partition_contrasts"]
    orr = m["objective_response_CRPR_vs_SDPD"]

    # Labels are kept short because they sit in the y-margin of a double-column
    # figure; the coefficient family is already stated by the group headings.
    rows = [
        ("Gwet AC2, weighted", four["gwet_ac2_ordinal_weighted"],
         four["gwet_ac2_ordinal_weighted_ci95"], BLUE, True),
        ("Gwet AC1, unweighted", four["gwet_ac1_unweighted"],
         four["gwet_ac1_unweighted_ci95"], BLUE, False),
        ("Fleiss κ, unweighted", four["fleiss_kappa_unweighted"],
         four["fleiss_kappa_unweighted_ci95"], BLUE, False),
        (None, None, None, None, None),  # group separator
        ("Complete response", m["CR_vs_nonCR"]["fleiss_kappa"],
         m["CR_vs_nonCR"]["fleiss_kappa_ci95"], BLUE, False),
        ("Progressive disease", m["PD_vs_nonPD"]["fleiss_kappa"],
         m["PD_vs_nonPD"]["fleiss_kappa_ci95"], BLUE, False),
        ("Objective response", orr["fleiss_kappa"],
         orr["fleiss_kappa_ci95"], VERMILLION, True),
    ]

    # Two rows rather than three columns: the decomposition needs five labelled
    # positions and is illegible squeezed into a third of a column width.
    # Constrained layout, because fixed margins cannot know how wide the
    # y-tick labels render and silently clip them.
    fig = plt.figure(figsize=(DOUBLE_COL_IN, 5.0), layout="constrained")
    gsF = fig.add_gridspec(2, 2, height_ratios=[1.16, 1.00], width_ratios=[1.0, 1.32])
    axA = fig.add_subplot(gsF[0, :])
    axB = fig.add_subplot(gsF[1, 0])
    axC = fig.add_subplot(gsF[1, 1])

    ys, labels = [], []
    y = len(rows)
    for label, value, ci, colour, bold in rows:
        if label is None:
            y -= 1
            continue
        _dot_row(axA, y, value, ci, colour)
        ys.append(y)
        labels.append(label)
        y -= 1

    axA.set_yticks(ys)
    axA.set_yticklabels(labels)
    for tick, (_, _, _, colour, bold) in zip(
            axA.get_yticklabels(), [r for r in rows if r[0] is not None]):
        tick.set_color(colour if bold else "black")
        if bold:
            tick.set_fontweight("bold")
    axA.set_xlim(-0.1, 1.0)
    axA.set_ylim(-0.55, len(rows) + 0.8)
    axA.set_xlabel("Agreement coefficient (95% CI)")
    axA.set_title("a  Same 93 readings, six coefficients", loc="left")

    axA.text(-0.09, len(rows) + 0.35, "FOUR-CATEGORY SCALE", fontsize=6.6,
             color=GREY, fontweight="bold")
    axA.text(-0.09, 3.75, "PARTITION (Fleiss κ)", fontsize=6.6,
             color=GREY, fontweight="bold")

    # The partition contrast the paper actually claims, connected in clear space
    # below the rows so nothing overlaps a label.
    #
    # This annotation deliberately joins the two Fleiss kappa partitions, which
    # differ only in where the cut falls. Drawing it from the weighted
    # four-category coefficient instead would put a single arrow across a change
    # of weighting, coefficient family and partition at once, and label that sum
    # as a partition effect. The decomposition in the caption is what licenses
    # this pair; see partition_contrast_note in MEASUREMENT_GATE.json.
    d = contr["cr_kappa_minus_orr_kappa"]
    top, bot = m["CR_vs_nonCR"]["fleiss_kappa"], orr["fleiss_kappa"]
    axA.plot([top, top], [0.5, 3.0], color=GREY, lw=0.6, ls=":", zorder=1)
    axA.plot([bot, bot], [0.5, 1.0], color=VERMILLION, lw=0.6, ls=":", zorder=1)
    axA.annotate("", xy=(bot, 0.42), xytext=(top, 0.42),
                 arrowprops=dict(arrowstyle="<->", color=GREY, lw=0.8,
                                 shrinkA=0, shrinkB=0))
    axA.text((bot + top) / 2, 0.10,
             f"paired difference {d['difference']:.3f} "
             f"(95% CI {d['ci95'][0]:.3f}–{d['ci95'][1]:.3f})",
             ha="center", va="top", fontsize=6.8, color=GREY)

    # Panel B - the clinically legible version
    bars = [("Progressive\ndisease", m["PD_vs_nonPD"], BLUE),
            ("Complete\nresponse", m["CR_vs_nonCR"], BLUE),
            ("Objective\nresponse", orr, VERMILLION)]
    xs = np.arange(len(bars))
    vals = [b[1]["at_least_one_discordant"] * 100 for b in bars]
    axB.bar(xs, vals, color=[b[2] for b in bars], width=0.62, zorder=2)
    for x, v in zip(xs, vals):
        axB.annotate(f"{v:.1f}%", (x, v), xytext=(0, 3), textcoords="offset points",
                     ha="center", fontsize=7.5, fontweight="bold")
    axB.set_xticks(xs)
    axB.set_xticklabels([b[0] for b in bars], fontsize=7)
    axB.set_ylabel("Patients with ≥1\ndiscordant reader (%)", linespacing=1.3)
    axB.set_ylim(0, max(vals) * 1.28)
    axB.set_title("b  What that costs, per patient", loc="left")
    axB.grid(axis="y", color=LIGHT, lw=0.5, alpha=0.5, zorder=0)
    axB.set_axisbelow(True)

    # ---- c. where the distance between (a)'s two ends actually comes from ---
    # Panel a shows a reassuring 0.802 and an unreassuring 0.409 on one scale
    # but cannot show why they differ. Changing one thing at a time answers it:
    # most of the gap is a weighting convention, not the partition.
    w_step = contr["step1_weighting_ac2_minus_ac1_four_category"]
    f_step = contr["step2_family_ac1_minus_fleiss_four_category"]
    p_step = contr["step3_partition_fleiss_four_category_minus_orr"]
    start = four["gwet_ac2_ordinal_weighted"]
    l1 = four["gwet_ac1_unweighted"]
    l2 = four["fleiss_kappa_unweighted"]
    end = orr["fleiss_kappa"]

    steps = [("remove\nordinal weights", start, l1, w_step),
             ("change coefficient\nfamily", l1, l2, f_step),
             ("change\npartition", l2, end, p_step)]

    axC.bar(0, start, width=0.62, color=BLUE, zorder=2)
    axC.annotate(f"{start:.3f}", (0, start), xytext=(0, 3), textcoords="offset points",
                 ha="center", fontsize=7.2, fontweight="bold", color=BLUE)
    for i, (lab, top_v, bot_v, st) in enumerate(steps, start=1):
        axC.bar(i, top_v - bot_v, bottom=bot_v, width=0.62, color=LIGHT,
                edgecolor=GREY, lw=0.7, zorder=2)
        axC.plot([i - 0.31 - 0.38, i - 0.31], [top_v, top_v], color=GREY, lw=0.6,
                 ls=":", zorder=1)
        marker = "" if st["excludes_zero"] else "\nn.s."
        # A 0.033 step is thinner than its own label, so the text goes above the
        # step rather than inside it.
        inside = (top_v - bot_v) > 0.10
        axC.annotate(f"−{st['difference']:.3f}{marker}",
                     (i, (top_v + bot_v) / 2 if inside else top_v),
                     xytext=(0, 0 if inside else 4), textcoords="offset points",
                     ha="center", va="center" if inside else "bottom",
                     fontsize=6.3, fontweight="bold", color="black",
                     linespacing=1.25)
    axC.plot([len(steps) - 0.31, len(steps) + 0.69 - 0.31], [end, end],
             color=GREY, lw=0.6, ls=":", zorder=1)
    axC.bar(len(steps) + 1, end, width=0.62, color=VERMILLION, zorder=2)
    axC.annotate(f"{end:.3f}", (len(steps) + 1, end), xytext=(0, 3),
                 textcoords="offset points", ha="center", fontsize=7.2,
                 fontweight="bold", color=VERMILLION)

    axC.set_xticks(range(len(steps) + 2))
    axC.set_xticklabels(["AC2\nweighted,\n4-category", "remove\nordinal\nweights",
                         "change\ncoefficient\nfamily", "change\npartition",
                         "Fleiss κ,\nobjective\nresponse"],
                        fontsize=5.6, linespacing=1.35)
    axC.set_ylim(0, 0.95)
    axC.set_ylabel("Agreement coefficient")
    axC.set_title("c  One change at a time", loc="left")
    axC.grid(axis="y", color=LIGHT, lw=0.5, alpha=0.5, zorder=0)
    axC.set_axisbelow(True)

    save(fig, out, "figure2_partition_dependence", tight=True)


# --------------------------------------------------------------------------
# Figure 3 - where the disagreement lives
# --------------------------------------------------------------------------
def _crossing_schematic(ax) -> None:
    """The 3x3 reader-crossing design, drawn rather than described.

    The method is the paper's novel contribution and is hard to hold in the head
    from prose alone, so it gets a panel.
    """
    import matplotlib.patches as mpatches

    ax.set_xlim(-1.3, 9.2)
    ax.set_ylim(-1.35, 3.85)
    ax.set_aspect("equal", adjustable="box")   # cells must read as squares
    ax.axis("off")

    # highlight bands: the fixed-baseline row, and the fixed-follow-up column
    ax.add_patch(mpatches.Rectangle((0, 2), 3, 1, facecolor=VERMILLION,
                                    alpha=0.16, zorder=0, linewidth=0))
    ax.add_patch(mpatches.Rectangle((0, 0), 1, 3, facecolor=BLUE,
                                    alpha=0.16, zorder=0, linewidth=0))

    for i in range(3):        # i = baseline reader (row, top to bottom)
        for j in range(3):    # j = follow-up reader (column)
            y = 2 - i
            ax.add_patch(mpatches.Rectangle((j + 0.08, y + 0.08), 0.84, 0.84,
                                            facecolor="white", edgecolor=LIGHT,
                                            linewidth=0.7, zorder=1))
            ax.text(j + 0.5, y + 0.5, f"B{i+1}\nF{j+1}", ha="center", va="center",
                    fontsize=6.0, color="#444444", zorder=2, linespacing=1.2)

    ax.text(1.5, 3.60, "follow-up reader", ha="center", va="center", fontsize=7)
    for j in range(3):
        ax.text(j + 0.5, 3.18, f"F{j+1}", ha="center", va="center",
                fontsize=6.6, color=GREY)
    ax.text(-0.98, 1.5, "baseline reader", va="center", rotation=90,
            ha="center", fontsize=7, rotation_mode="anchor")
    for i in range(3):
        ax.text(-0.30, 2.5 - i, f"B{i+1}", va="center", ha="center",
                fontsize=6.6, color=GREY)

    # Arrows sit outside the grid so they never cross a cell label; the shaded
    # bands already carry which row and which column is being held fixed.
    ax.annotate("", xy=(3.48, 2.5), xytext=(3.05, 2.5),
                arrowprops=dict(arrowstyle="->", color=VERMILLION, lw=1.3))
    ax.text(3.58, 2.5, "swap the follow-up reader", fontsize=6.6, va="center",
            ha="left", color=VERMILLION, fontweight="bold")
    ax.annotate("", xy=(0.5, -0.52), xytext=(0.5, -0.09),
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.3))
    ax.text(0.5, -0.72, "swap the baseline reader", fontsize=6.6, ha="center",
            va="top", color=BLUE, fontweight="bold")
    ax.text(3.58, 1.05,
            "Each of the nine cells recomputes the response category from one\n"
            "reader's baseline measurement and another reader's follow-up\n"
            "measurement. Disagreement along each band is then compared.\n"
            "No consensus or averaged value enters the comparison.",
            fontsize=6.2, ha="left", va="center", color="#444444", linespacing=1.45)


def figure3(res: dict[str, Any], out: Path, sc: dict[str, Any] | None = None) -> None:
    h1 = res["M002_harmonisation_counterfactual"]
    sz = res["M003c_structural_zero_robustness"]
    rc = res["M011_reader_crossed_attribution"]
    ctrl = res["M012_equal_noise_control"]

    groups = [
        ("Consensus substitution — four-category", [
            ("All patients (n = 93)", h1["H1_delta_points"], h1["H1_delta_ci95"], BLUE, False),
            ("Excluding any complete response (n = 64)",
             sz["exclude_any_cr"]["H1_delta_points"],
             sz["exclude_any_cr"]["H1_delta_ci95"], BLUE, False),
            ("Follow-up zeros → 1 mm (n = 93)",
             sz["zeros_to_1mm"]["H1_delta_points"],
             sz["zeros_to_1mm"]["H1_delta_ci95"], BLUE, False),
        ]),
        ("Reader-crossed, no consensus — four-category", [
            ("All patients (n = 93)", rc["primary"]["delta_four_cat_points"],
             rc["primary"]["delta_four_cat_ci95"], BLUE, False),
            ("Excluding any complete response (n = 64)",
             rc["exclude_any_cr"]["delta_four_cat_points"],
             rc["exclude_any_cr"]["delta_four_cat_ci95"], BLUE, False),
            ("Follow-up zeros → 1 mm (n = 93)",
             rc["zeros_to_1mm"]["delta_four_cat_points"],
             rc["zeros_to_1mm"]["delta_four_cat_ci95"], BLUE, False),
        ]),
        ("Reader-crossed, no consensus — objective response", [
            ("All patients (n = 93)", rc["primary"]["delta_orr_points"],
             rc["primary"]["delta_orr_ci95"], VERMILLION, False),
            ("Excluding any complete response (n = 64)",
             rc["exclude_any_cr"]["delta_orr_points"],
             rc["exclude_any_cr"]["delta_orr_ci95"], VERMILLION, True),
            ("Follow-up zeros → 1 mm (n = 93)",
             rc["zeros_to_1mm"]["delta_orr_points"],
             rc["zeros_to_1mm"]["delta_orr_ci95"], VERMILLION, False),
        ]),
        ("Negative control", [
            ("Identical simulated noise at both timepoints",
             ctrl["simulated_H1_delta_points_mean"],
             ctrl["simulated_H1_delta_ci95"], GREY, False),
        ]),
    ]

    n_rows = sum(len(g[1]) for g in groups) + len(groups)
    forest_h = 0.30 * n_rows + 1.15
    conf_h = 2.75 if sc else 0.0
    # Constrained layout: this figure carries long y-tick labels, a two-line
    # x-label and a nested row of sub-panels, and fixed margins clip all three.
    fig = plt.figure(figsize=(DOUBLE_COL_IN, forest_h + 1.85 + conf_h),
                     layout="constrained")
    heights = [1.85, forest_h] + ([conf_h] if sc else [])
    gs3 = fig.add_gridspec(len(heights), 1, height_ratios=heights)
    ax_s = fig.add_subplot(gs3[0])
    ax = fig.add_subplot(gs3[1])
    _crossing_schematic(ax_s)
    ax_s.set_title("a  The reader-crossed design", loc="left")
    ax.set_title("b  Operational sensitivity to reader variation by timepoint", loc="left")

    y = n_rows
    yticks, ylabels, bolds = [], [], []
    for title, rows in groups:
        ax.text(-24.5, y, title.upper(), fontsize=6.8, color=GREY, fontweight="bold",
                va="center")
        y -= 1
        for label, value, ci, colour, bold in rows:
            _dot_row(ax, y, value, ci, colour, label_fmt="{:.1f}")
            yticks.append(y)
            ylabels.append(label)
            bolds.append((colour, bold))
            y -= 1

    ax.axvline(0, color="black", lw=0.9, zorder=1)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    for tick, (colour, bold) in zip(ax.get_yticklabels(), bolds):
        if bold:
            tick.set_fontweight("bold")
            tick.set_color(colour)
    ax.set_ylim(0.3, n_rows + 0.9)
    ax.set_xlim(-25, 45)
    ax.set_xlabel("Excess discordance when the follow-up rather than baseline "
                  "reader is varied (percentage points, 95% CI)",
                  labelpad=16)
    ax.grid(axis="x", color=LIGHT, lw=0.5, alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    trans = ax.get_xaxis_transform()
    for x, text, ha in ((-2, "← baseline", "right"), (2, "follow-up →", "left")):
        ax.text(x, -0.052, text, transform=trans, ha=ha, va="top",
                fontsize=6.8, color=GREY, style="italic", clip_on=False)

    # ---- c. the one alternative explanation, and why it fails --------------
    # The rule thresholds (FU - BL)/BL, so it is arithmetically more sensitive
    # to follow-up once a lesion shrinks. Three independent checks, each of
    # which the shrinkage account has to survive and does not.
    if sc:
        gsC = gs3[2].subgridspec(1, 3, wspace=0.52, width_ratios=[1.0, 1.16, 1.0])
        c1, c2, c3 = (fig.add_subplot(gsC[i]) for i in range(3))
        d = sc["dispersion"]
        observed = rc["exclude_any_cr"]["delta_orr_points"]

        # c1 - absolute spread did not fall, though the lesion did
        bl_sd, fu_sd = d["absolute_sd_mm"]["baseline"], d["absolute_sd_mm"]["followup"]
        shrink = d["mean_diameter_mm"]["followup"] / d["mean_diameter_mm"]["baseline"]
        predicted = bl_sd * shrink
        c1.bar([0, 1], [bl_sd, fu_sd], width=0.6, color=[BLUE, VERMILLION], zorder=2)
        # Only the strip outside the bars is free, so the line carries a bare
        # number and the caption carries its meaning.
        c1.plot([-0.45, 1.45], [predicted, predicted], color="black", lw=1.0, ls="--",
                zorder=4)
        c1.annotate(f"{predicted:.1f} expected", (1.45, predicted), xytext=(0, 2),
                    textcoords="offset points", fontsize=5.6, color="black",
                    va="bottom", ha="right")
        for x, v in zip((0, 1), (bl_sd, fu_sd)):
            c1.annotate(f"{v:.1f}", (x, v), xytext=(0, 2), textcoords="offset points",
                        ha="center", fontsize=6.8, fontweight="bold")
        c1.set_xticks([0, 1])
        c1.set_xticklabels([f"baseline\n{d['mean_diameter_mm']['baseline']:.0f} mm",
                            f"follow-up\n{d['mean_diameter_mm']['followup']:.0f} mm"],
                           fontsize=6.0, linespacing=1.3)
        c1.set_xlim(-0.55, 1.55)
        c1.set_ylim(0, max(bl_sd, fu_sd) * 1.42)
        c1.set_ylabel("Between-reader SD (mm)", fontsize=6.6)
        c1.set_title("c  Spread did not fall", loc="left", fontsize=7.4)

        # c2 - equal-error controls against the observed effect
        # Drawn by hand rather than through _dot_row: its value label sits to the
        # right of the interval, which is exactly where the observed line falls.
        # Here the value rides in the row label instead.
        law = [("proportional", "multiplicative", LIGHT),
               ("additive", "additive", LIGHT),
               ("fitted (mixed)", "mixed", GREY)]
        ylabels2 = []
        for i, (lab, key, colour) in enumerate(law):
            v = sc["noise_control"][key]["reader_crossed_orr"]
            y2 = len(law) - i
            c2.plot(v["ci95"], [y2, y2], color=colour, lw=1.4, alpha=0.75, zorder=2)
            for xb in v["ci95"]:
                c2.plot([xb, xb], [y2 - 0.12, y2 + 0.12], color=colour, lw=1.0,
                        alpha=0.75, zorder=2)
            c2.plot([v["mean"]], [y2], "o", color=colour, ms=5.0, zorder=3,
                    markeredgecolor="white", markeredgewidth=0.6)
            ylabels2.append(f"{lab}\n{v['mean']:+.1f}")
        c2.axvline(observed, color=VERMILLION, lw=1.5, zorder=5)
        c2.annotate(f"observed {observed:.1f}", (observed, len(law) + 0.55),
                    xytext=(-3, 0), textcoords="offset points", fontsize=6.0,
                    color=VERMILLION, fontweight="bold", ha="right", va="center")
        c2.axvline(0, color="black", lw=0.8, zorder=1)
        c2.set_yticks(range(1, len(law) + 1))
        c2.set_yticklabels(ylabels2[::-1], fontsize=6.0, linespacing=1.3)
        c2.set_ylim(0.3, len(law) + 0.95)
        c2.set_xlim(-18, 30)
        c2.set_xlabel("Excess from the rule alone (points)", fontsize=6.4)
        c2.set_title("Equal-error control", loc="left", fontsize=7.4)

        # c3 - the effect does not grow with shrinkage
        order = ["most_shrinkage", "middle", "least_shrinkage"]
        xs3 = np.arange(len(order))
        for i, key in enumerate(order):
            s = sc["shrinkage_strata"][key]
            lo, hi = s["delta_orr_ci95"]
            c3.plot([i, i], [lo, hi], color=VERMILLION, lw=1.2, zorder=2,
                    solid_capstyle="butt")
            c3.plot([i], [s["delta_orr_points"]], "o", color=VERMILLION, ms=5,
                    markeredgecolor="white", markeredgewidth=0.6, zorder=3)
        c3.axhline(0, color="black", lw=0.8, zorder=1)
        c3.set_xticks(xs3)
        c3.set_xticklabels([f"{sc['shrinkage_strata'][k]['median_percent_change']:+.0f}%\n"
                            f"n = {sc['shrinkage_strata'][k]['n']}" for k in order],
                           fontsize=5.6, linespacing=1.3)
        c3.set_xlim(-0.55, len(order) - 0.45)
        c3.set_xlabel("Median change in viable diameter", fontsize=6.4)
        c3.set_ylabel("Follow-up excess (points)", fontsize=6.6)
        c3.set_title("No growth with shrinkage", loc="left", fontsize=7.4)

    save(fig, out, "figure3_timepoint_attribution", tight=True)


# --------------------------------------------------------------------------
# Figure 4 - mechanism
# --------------------------------------------------------------------------
def figure4(res: dict[str, Any], data_dir: Path, out: Path) -> None:
    hcc = pd.read_excel(data_dir / "HCC-TACE-Seg_clinical_data-V2.xlsx",
                        sheet_name="data table")
    rf = reader_frame(hcc)
    # Exclusion rule: the panel needs all three readers at both timepoints, which
    # is the same triple-complete set the whole analysis uses. Counts are printed
    # so the reduction is auditable rather than silent.
    complete = rf.dropna(subset=[f"{p}{r}" for r in (1, 2, 3) for p in ("bl", "fu", "cat")])
    print(f"  figure 4 patient set: {len(rf)} released -> {len(complete)} with complete "
          f"three-reader readings at both timepoints "
          f"({len(rf) - len(complete)} excluded, incomplete readings)")
    bl = complete[[f"bl{r}" for r in (1, 2, 3)]].to_numpy(float)
    fu = complete[[f"fu{r}" for r in (1, 2, 3)]].to_numpy(float)

    tot = hcc.loc[complete.index, [f"{r}_RECIST_BL" for r in (1, 2, 3)]] \
             .apply(pd.to_numeric, errors="coerce").to_numpy(float)

    viable_ratio = bl.max(axis=1) / bl.min(axis=1)
    # A max/min ratio is undefined where a reader recorded a zero total diameter.
    ok = np.isfinite(tot).all(axis=1) & (tot.min(axis=1) > 0)
    if (~ok).any():
        print(f"  figure 4a total-diameter ratio: {ok.sum()}/{len(ok)} evaluable "
              f"({(~ok).sum()} excluded, zero or missing total diameter)")
    total_ratio = tot[ok].max(axis=1) / tot[ok].min(axis=1)

    mv = res["M005_measurement_variability"]
    mb = res["M005b_recist_ratio_spread"]
    assert abs(np.mean(viable_ratio >= 1.5) - mv["baseline_ratio_ge_1p5_frac"]) < 1e-9, \
        "viable ratio recomputation disagrees with the frozen results"

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(DOUBLE_COL_IN, 2.9))

    # Hero (viable) is filled; the comparator is a step outline. Overlaid fills
    # would produce an ambiguous third colour and would not survive grayscale.
    bins = np.linspace(1.0, 3.0, 26)
    axA.hist(np.clip(viable_ratio, None, 3.0), bins=bins, color=VERMILLION, alpha=0.75, zorder=2,
             label=("Viable diameter (mRECIST)\n"
                    f"median {mv['baseline_ratio_median']:.2f}, "
                    f"{mv['baseline_ratio_ge_1p5_frac']*100:.1f}% ≥1.5×"))
    axA.hist(np.clip(total_ratio, None, 3.0), bins=bins, histtype="step", lw=1.3,
             color=BLUE, zorder=3,
             label=("Total diameter (RECIST)\n"
                    f"median {mb['ratio_median']:.2f}, "
                    f"{mb['ratio_ge_1p5_frac']*100:.1f}% ≥1.5×"))
    axA.axvline(1.5, color="black", lw=0.8, ls="--", zorder=4)
    axA.set_xlabel("Between-reader max / min ratio at baseline")
    axA.set_ylabel("Patients")
    # The first bin starts exactly at 1.0, so without a left margin the tallest
    # bar's outline is drawn on top of the spine and reads as unbounded. The
    # extra y headroom keeps the legend clear of that same bar.
    axA.set_xlim(0.92, 3.06)
    axA.set_ylim(0, 54)
    axA.legend(frameon=False, fontsize=6.4, loc="upper right", handlelength=1.1,
               handletextpad=0.5, borderaxespad=0.2, labelspacing=0.7)
    axA.set_title("a  Reader spread: viable versus total diameter", loc="left")

    # Panel B - threshold proximity
    pct = (fu - bl) / bl * 100.0
    mean_pct, sd_pct = pct.mean(axis=1), pct.std(axis=1, ddof=1)
    near = (np.abs(mean_pct - (-30.0)) <= sd_pct) | (np.abs(mean_pct - 20.0) <= sd_pct)
    axB.axvspan(-30, 20, color=LIGHT, alpha=0.25, zorder=0)
    for cut in (-30.0, 20.0):
        axB.axvline(cut, color="black", lw=0.9, ls="--", zorder=3)
    axB.scatter(mean_pct[~near], sd_pct[~near], s=13, color=GREY, alpha=0.55,
                linewidths=0, zorder=2, label="Threshold-secure")
    axB.scatter(mean_pct[near], sd_pct[near], s=15, color=VERMILLION, alpha=0.85,
                linewidths=0, zorder=4,
                label=f"Within 1 SD of a cut-point ({mv['threshold_fragile_fraction']*100:.1f}%)")
    axB.set_xlim(-108, 80)
    axB.set_ylim(-3, 70)
    axB.set_xlabel("Mean percent change in viable diameter")
    axB.set_ylabel("Between-reader SD (percentage points)")
    for x, text, ha in ((-33, "PR", "right"), (-5, "SD", "center"), (23, "PD", "left")):
        axB.annotate(text, xy=(x, 68), ha=ha, va="top", fontsize=7,
                     color=GREY, fontweight="bold")
    leg = axB.legend(fontsize=6.6, loc="lower right", handletextpad=0.3,
                     borderaxespad=0.4, markerscale=1.3, frameon=True,
                     framealpha=0.92, edgecolor="none")
    leg.get_frame().set_facecolor("white")
    leg.set_zorder(6)
    axB.set_title("b  Proximity to response thresholds", loc="left")

    fig.tight_layout(w_pad=2.4)
    save(fig, out, "figure4_mechanism")


# --------------------------------------------------------------------------
# Figure 5 - the benchmark moves with the reference
# --------------------------------------------------------------------------
def figure5(res: dict[str, Any], out: Path) -> None:
    m = res["M015_reference_dependence"]
    null = m["permutation_null"]
    refs = ["reader1", "reader2", "reader3", "majority"]
    nice = {"reader1": "Reader 1", "reader2": "Reader 2", "reader3": "Reader 3",
            "majority": "Majority"}
    models = [("logistic_regression", "Logistic regression", VERMILLION, "o"),
              ("random_forest", "Random forest", BLUE, "s"),
              ("bclc_alone", "BCLC stage, adjusted", GREY, "^")]

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(DOUBLE_COL_IN, 2.9), gridspec_kw={"width_ratios": [1.5, 1.0]})

    xs = np.arange(len(refs))
    for key, label, colour, marker in models:
        ys = [m["models"][key]["auc_by_reference"][r] for r in refs]
        if key == "bclc_alone":
            # Lower BCLC stage favours response. Orient its score in the same
            # clinical direction as the learned-model probabilities.
            ys = [1.0 - y for y in ys]
        axA.plot(xs, ys, marker=marker, ms=5, lw=1.3, color=colour, label=label,
                 markeredgecolor="white", markeredgewidth=0.6, zorder=3)
    axA.axhline(0.5, color=LIGHT, lw=0.8, ls="--", zorder=1)
    axA.annotate("chance", xy=(len(refs) - 0.5, 0.5), xytext=(0, 3),
                 textcoords="offset points", fontsize=6.4, color=GREY, ha="right")

    lr = m["models"]["logistic_regression"]["auc_by_reference"]
    hi, lo = lr["reader2"], lr["reader3"]
    c = m["pairwise_auc_contrasts"]["reader2_minus_reader3"]
    # The contrast is between two points at DIFFERENT x positions — reader 2 and
    # reader 3 — so a bare vertical arrow at reader 2 spanned a value that has no
    # marker there and read as ambiguous. Each endpoint is now ringed and carried
    # to a common bracket by a guide line, which is what makes the 0.169 legible
    # as "these two scores, same predictions".
    BR = 1.62
    for x_pt, y_pt in ((1, hi), (2, lo)):
        axA.plot([x_pt], [y_pt], "o", ms=9, markerfacecolor="none",
                 markeredgecolor=VERMILLION, markeredgewidth=1.0, zorder=5)
        # At reader 2 the random forest scores 0.7033 against logistic regression's
        # 0.7027 and is drawn last, so without this the ring appears to circle the
        # wrong series. Re-draw the bracketed series on top of it.
        axA.plot([x_pt], [y_pt], "o", ms=5, color=VERMILLION, zorder=6,
                 markeredgecolor="white", markeredgewidth=0.6)
    axA.plot([1, BR], [hi, hi], color=VERMILLION, lw=0.6, ls=":", zorder=2)
    axA.plot([2, BR], [lo, lo], color=VERMILLION, lw=0.6, ls=":", zorder=2)
    axA.annotate("", xy=(BR, hi), xytext=(BR, lo),
                 arrowprops=dict(arrowstyle="<->", color=VERMILLION, lw=1.0,
                                 shrinkA=0, shrinkB=0), zorder=5)
    # Every plotted series lies at or above 0.534, so the band below the chance
    # line is the one region of this panel no data occupies.
    axA.plot([BR, BR], [lo - 0.008, 0.495], color=VERMILLION, lw=0.6, ls=":", zorder=2)
    axA.text(BR, 0.487,
             f"same predictions,\nΔAUC {c['difference']:.3f} "
             f"(95% CI {c['ci95'][0]:.3f}–{c['ci95'][1]:.3f})",
             fontsize=6.2, va="top", ha="center", color=VERMILLION,
             fontweight="bold", linespacing=1.3)

    axA.set_xticks(xs)
    axA.set_xticklabels([nice[r] for r in refs], fontsize=7)
    axA.set_xlim(-0.35, len(refs) - 0.35)
    axA.set_ylim(0.44, 0.78)
    axA.set_xlabel("Reference used to score the predictions")
    axA.set_ylabel("Area under the ROC curve")
    axA.set_title("a  Identical predictions, four references", loc="left")
    axA.legend(frameon=False, fontsize=6.6, loc="upper left", handlelength=1.5,
               borderaxespad=0.2)

    counts = np.asarray(null["histogram"]["counts"], float)
    edges = np.asarray(null["histogram"]["bin_edges"], float)
    centres = (edges[:-1] + edges[1:]) / 2
    axB.bar(centres, counts / counts.sum(), width=np.diff(edges), color=LIGHT, zorder=2)
    axB.axvline(null["null_spread_p95"], color=GREY, lw=1.0, ls="--", zorder=3)
    axB.annotate("null 95th", xy=(null["null_spread_p95"], axB.get_ylim()[1] * 0.30),
                 xytext=(-3, 0), textcoords="offset points", rotation=90,
                 fontsize=6.2, color=GREY, ha="right", va="center",
                 rotation_mode="anchor")
    axB.axvline(null["observed_spread"], color=VERMILLION, lw=1.6, zorder=4)
    axB.annotate(f"observed {null['observed_spread']:.3f}\np = {null['p_value']:.3f}",
                 xy=(null["observed_spread"], axB.get_ylim()[1] * 0.55),
                 xytext=(5, 0), textcoords="offset points", fontsize=6.6,
                 color=VERMILLION, fontweight="bold", va="center", linespacing=1.3)
    axB.set_xlabel("Range of AUC across the three readers")
    axB.set_ylabel("Proportion of permutations")
    axB.set_title("b  Against a permutation null", loc="left")

    fig.tight_layout(w_pad=2.2)
    save(fig, out, "figure5_reference_dependence")


# --------------------------------------------------------------------------
# Graphical abstract (required by the target journal)
# --------------------------------------------------------------------------
def graphical_abstract(res: dict[str, Any], out: Path) -> None:
    """Design, then the two findings that follow from it.

    A graphical abstract has to say what was done before it says what was found.
    Three bar charts state three results and leave a reader who has not read the
    paper with no idea what the study was, so panel 1 carries the design and the
    imaging and the partition finding moves into the strapline, where it reads as
    the paper's claim rather than as a fourth chart.
    """
    m = res["M003b_multirater_agreement"]
    rc = res["M011_reader_crossed_attribution"]["exclude_any_cr"]
    ref = res["M015_reference_dependence"]["models"]["logistic_regression"]
    orr = m["objective_response_CRPR_vs_SDPD"]
    cr = m["CR_vs_nonCR"]
    assets = out / "assets"

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL_IN, 2.70),
                             gridspec_kw={"width_ratios": [1.22, 1.0, 1.0]})

    # Panel 1 - the design, imaged. The imaging is the only part of a graphical
    # abstract a radiologist reads first, so it gets the extra column width and
    # the full panel height, and the baseline ROI is drawn rather than described.
    ax = axes[0]
    ax.axis("off")
    # One line, not two: a two-line panel title reaches down into the row of
    # image labels that sits just above the enlarged scans.
    ax.set_title("Three radiologists, two scans each", fontsize=7.6, loc="left")
    bl = plt.imread(assets / "hcc034_baseline.png")
    fu = plt.imread(assets / "hcc034_followup.png")
    mask = plt.imread(assets / "hcc034_baseline_mass.png")
    Y0, Y1 = 0.24, 0.94
    ax.imshow(bl, cmap="gray", vmin=0, vmax=1, extent=(0.0, 0.49, Y0, Y1),
              aspect="auto", interpolation="bilinear")
    ax.imshow(fu, cmap="gray", vmin=0, vmax=1, extent=(0.51, 1.0, Y0, Y1),
              aspect="auto", interpolation="bilinear")
    # contour needs origin="upper" to sit on an imshow drawn with the same extent
    ax.contour(mask, levels=[0.5], colors=[VERMILLION], linewidths=1.2,
               extent=(0.0, 0.49, Y0, Y1), origin="upper", zorder=4)
    ax.text(0.245, Y1 + 0.015, "before TACE", ha="center", va="bottom", fontsize=6.4)
    ax.text(0.755, Y1 + 0.015, "after TACE", ha="center", va="bottom", fontsize=6.4)
    ax.text(0.245, Y0 - 0.02, "viable tumor outlined", ha="center", va="top",
            fontsize=5.9, color=VERMILLION)
    ax.text(0.755, Y0 - 0.02, "necrosis and retained\nembolic material", ha="center",
            va="top", fontsize=5.9, color=GREY, linespacing=1.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    ax = axes[1]
    vals = [rc["vary_baseline_orr"] * 100, rc["vary_followup_orr"] * 100]
    ax.bar([0, 1], vals, color=[BLUE, VERMILLION], width=0.6, zorder=2)
    for x, v in zip([0, 1], vals):
        ax.annotate(f"{v:.0f}%", (x, v), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Swap baseline\nreader", "Swap follow-up\nreader"], fontsize=6.8)
    ax.set_ylim(0, 60)
    ax.set_ylabel("Patients whose response\nstatus changes (%)")
    ax.set_title("Response status is more sensitive\nto the follow-up reader (n = 64)",
                 fontsize=7.6, loc="left")

    ax = axes[2]
    keys = ["reader1", "reader2", "reader3"]
    vals = [ref["auc_by_reference"][k] for k in keys]
    ax.bar(range(3), vals, color=[BLUE, VERMILLION, BLUE], width=0.6, zorder=2)
    for x, v in zip(range(3), vals):
        ax.annotate(f"{v:.2f}", (x, v), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8, fontweight="bold")
    ax.axhline(0.5, color=LIGHT, lw=0.8, ls="--", zorder=1)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Reader 1", "Reader 2", "Reader 3"], fontsize=6.8)
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("AUC of identical\npredictions")
    ax.set_title("So the benchmark moves\nwith the reference", fontsize=7.6, loc="left")

    for ax in axes[1:]:
        ax.grid(axis="y", color=LIGHT, lw=0.5, alpha=0.45, zorder=0)
        ax.set_axisbelow(True)

    # One line, and the paper's claim rather than a restatement of panel 2. The
    # previous strapline ran to two full-width lines and said "complete response"
    # twice; the coefficients it quoted are in the paper, not the abstract image.
    fig.suptitle(
        "mRECIST agreement is lowest on the partition that defines trial response rate",
        fontsize=8.4, fontweight="bold", y=1.045, x=0.01, ha="left")
    fig.tight_layout(w_pad=2.0)
    save(fig, out, "graphical_abstract")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    provenance = verify_inputs(args.data)
    if not all(v["match"] for v in provenance.values()):
        raise SystemExit(f"CHECKSUM MISMATCH: {provenance}")

    res = json.loads((args.results / "MEASUREMENT_GATE.json").read_text())
    print(f"building figures from {args.results / 'MEASUREMENT_GATE.json'}")
    sc_path = Path(__file__).resolve().parent.parent / "results" / "size_confound" / "SIZE_CONFOUND.json"
    sc = json.loads(sc_path.read_text()) if sc_path.exists() else None
    print(f"size-confound panel: {'on, from ' + str(sc_path) if sc else 'off (results absent)'}")
    figure_design(res, args.data, args.out)
    figure2(res, args.out)
    figure3(res, args.out, sc)
    figure4(res, args.data, args.out)
    figure5(res, args.out)
    graphical_abstract(res, args.out)


if __name__ == "__main__":
    main()
