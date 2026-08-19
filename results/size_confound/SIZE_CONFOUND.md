# Size-confound checks

**Patients:** 64 with three positive readings at both timepoints (29 structural-zero patients excluded from 93).

## 1. Between-reader spread by timepoint, absolute and relative

| Quantity | Baseline | Follow-up | Difference | 95% CI |
|---|---:|---:|---:|---:|
| Mean diameter, mm | 82.8 | 58.8 | | |
| Between-reader SD, mm | 12.41 | 13.57 | +1.16 | -1.27 to 3.71 |
| Between-reader SD, log | 0.162 | 0.293 | +0.130 | 0.069 to 0.198 |

Relative spread must rise when a lesion shrinks under any fixed-error model.
Absolute spread need not, and a rise in it cannot be produced by shrinkage.

## 2. Equal-noise control under three error models

Calibration: sigma_multiplicative 0.335, tau_additive 17.76 mm, mixed (tau 9.66 mm, sigma 0.181).

| Error model | Harmonisation, 4-cat | Reader-crossed, 4-cat | Reader-crossed, ORR |
|---|---:|---:|---:|
| multiplicative | -3.5 (-15.6 to 9.4) | -0.0 (-10.9 to 10.4) | +0.1 (-11.5 to 11.5) |
| additive | +7.1 (-4.7 to 18.8) | +8.2 (-2.1 to 18.8) | +13.1 (2.6 to 24.0) |
| mixed | +2.3 (-9.4 to 14.1) | +3.5 (-7.3 to 14.1) | +5.8 (-5.2 to 16.1) |

Percentage points. A positive value is asymmetry the rule manufactures from
shrinkage alone under an error law applied identically at both timepoints.

## 3. Reader-crossed contrast by tertile of observed shrinkage

| Tertile | n | Median change | 4-cat difference | ORR difference |
|---|---:|---:|---:|---:|
| most shrinkage | 21 | -56% | +7.9 (-4.8 to 22.2) | +7.9 (-4.8 to 22.2) |
| middle | 22 | -37% | +33.3 (10.6 to 56.1) | +37.9 (13.6 to 60.6) |
| least shrinkage | 21 | -14% | +9.5 (-9.5 to 28.6) | +19.0 (4.8 to 34.9) |

Tertiles carry about a third of the set each, so these intervals are wide by
construction and are read as direction rather than as estimates.
