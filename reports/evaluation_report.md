# MedPredict Research — Evaluation Report

> ⚕️ **Educational research prototype. Not a diagnostic device, clinical decision system or substitute for a licensed clinician. Do not use for real patient-care decisions.**

*Generated 2026-11-17T13:56:42+00:00 · synthetic de-identified cohort · temporal + patient-level split*

**Target.** Unplanned readmission within 30 days of discharge (binary). Prediction time is the moment of discharge; only information available at or before discharge may be used as a feature.

**Selected on validation log loss:** `logistic_regression` + `isotonic` · threshold 0.209 (target sensitivity 0.80 on validation)

## Discrimination & calibration (test period)

| Model | Calibration | AUROC (95% CI) | PR-AUC | Brier | ECE | Cal. slope | Accuracy@0.5 |
|---|---|---|---|---|---|---|---|
| logistic_regression | uncalibrated | 0.683 (0.649–0.718) | 0.506 | 0.1939 | 0.042 | 0.90 | 0.708 |
| logistic_regression | sigmoid | 0.683 (0.649–0.718) | 0.506 | 0.1925 | 0.030 | 0.95 | 0.705 |
| logistic_regression | isotonic | 0.678 (0.645–0.712) | 0.481 | 0.1940 | 0.029 | 0.76 | 0.705 |
| random_forest | uncalibrated | 0.694 (0.661–0.731) | 0.516 | 0.1935 | 0.044 | 1.35 | 0.712 |
| random_forest | sigmoid | 0.694 (0.661–0.731) | 0.516 | 0.1906 | 0.023 | 1.03 | 0.713 |
| random_forest | isotonic | 0.692 (0.659–0.726) | 0.491 | 0.1910 | 0.022 | 0.93 | 0.712 |
| hist_gradient_boosting | uncalibrated | 0.682 (0.647–0.718) | 0.491 | 0.1958 | 0.058 | 0.76 | 0.705 |
| hist_gradient_boosting | sigmoid | 0.682 (0.647–0.718) | 0.491 | 0.1936 | 0.033 | 1.00 | 0.708 |
| hist_gradient_boosting | isotonic | 0.680 (0.646–0.716) | 0.475 | 0.1947 | 0.017 | 0.67 | 0.714 |

**At the frozen threshold (test):** sensitivity 0.790 · specificity 0.417 · PPV 0.375 · flag rate 64.6%

**Accuracy-only view:** the highest accuracy@0.5 belongs to `hist_gradient_boosting` + `isotonic` (0.714) — accuracy barely separates models while calibration metrics differ substantially.

## Subgroups (test; groups with n < 50 suppressed)

| Grouping | Group | n | Positives | AUROC (95% CI) | Sensitivity | Mean predicted | Observed |
|---|---|---|---|---|---|---|---|
| sex | F | 629 | 207 | 0.671 (0.624–0.713) | 0.802 | 0.308 | 0.329 |
| sex | M | 522 | 146 | 0.683 (0.635–0.733) | 0.774 | 0.290 | 0.280 |
| age_band | 18-44 | 99 | 24 | 0.624 (0.517–0.727) | 0.167 | 0.126 | 0.242 |
| age_band | 45-64 | 415 | 83 | 0.594 (0.521–0.659) | 0.530 | 0.225 | 0.200 |
| age_band | 65-79 | 427 | 144 | 0.624 (0.571–0.668) | 0.903 | 0.332 | 0.337 |
| age_band | 80+ | 210 | 102 | 0.708 (0.636–0.773) | 0.990 | 0.463 | 0.486 |
| site | A | 518 | 125 | 0.674 (0.631–0.729) | 0.776 | 0.283 | 0.241 |
| site | B | 408 | 127 | 0.674 (0.614–0.723) | 0.772 | 0.286 | 0.311 |
| site | C | 225 | 101 | 0.646 (0.579–0.723) | 0.832 | 0.362 | 0.449 |

## Shift tests

| Test | Condition | AUROC | Brier | ECE | Mean predicted | Observed |
|---|---|---|---|---|---|---|
| missingness | 0% | 0.678 | 0.1940 | 0.029 | 0.300 | 0.307 |
| missingness | 10% | 0.676 | 0.1936 | 0.022 | 0.305 | 0.307 |
| missingness | 20% | 0.679 | 0.1930 | 0.029 | 0.311 | 0.307 |
| missingness | 30% | 0.674 | 0.1933 | 0.027 | 0.315 | 0.307 |
| missingness | 40% | 0.672 | 0.1942 | 0.026 | 0.321 | 0.307 |
| covariate | older_population | 0.680 | 0.1948 | 0.049 | 0.349 | 0.307 |
| covariate | higher_glucose | 0.678 | 0.1942 | 0.034 | 0.310 | 0.307 |
| site | in-distribution sites | 0.677 | 0.1832 | 0.038 | 0.250 | 0.272 |
| site | held-out site C | 0.658 | 0.2506 | 0.161 | 0.299 | 0.449 |
| temporal | validation period | 0.701 | 0.1873 | 0.000 | 0.304 | 0.304 |
| temporal | test period (later) | 0.678 | 0.1940 | 0.029 | 0.300 | 0.307 |

## Limitations
- Synthetic cohort; no clinical validity. Real use would require independent clinical validation and governance.
- Subgroup CIs are wide; small groups are suppressed rather than over-interpreted.
- Bootstrap intervals capture estimation uncertainty only — not label noise, confounding or shift.
- Explanations are associations in the model, not causal effects.
