# Model Card — MedPredict Research Risk Model

> ⚕️ **Educational research prototype. Not a diagnostic device, clinical decision system or substitute for a
> licensed clinician. Do not use for real patient-care decisions.**

| Item | Detail |
|---|---|
| **Intended use** | Teaching and research on evaluation methodology: calibration, subgroup analysis, shift testing |
| **Out of scope** | Any real clinical use; triage; diagnosis; treatment selection; resource allocation |
| **Model** | Selected on validation log loss among LR / RF / HGB × {uncalibrated, sigmoid, isotonic} |
| **Training data** | Synthetic cohort (see `DATA_CARD.md`), discharges up to 2024-06-30 |
| **Threshold** | Chosen on validation for sensitivity ≥ 0.80 — a stated research objective, not a clinical one |
| **Outputs** | Calibrated probability + bootstrap model-uncertainty band + top model associations |

## Evaluation (synthetic test period)
See [`reports/evaluation_report.md`](../reports/evaluation_report.md) for AUROC/PR-AUC with CIs, Brier, ECE,
calibration slope, threshold metrics, subgroup results with sample sizes, and shift tests.

## Known failure modes
* **Site transfer:** calibration degrades sharply on a site not seen in training.
* **Shared threshold:** sensitivity differs strongly across age bands.
* **Missing-indicator features** encode care-process differences that may not transfer.

## Ethical considerations
Subgroup attributes are used for evaluation. Choosing group-specific thresholds or using sensitive
attributes as inputs is a governance decision requiring clinical, ethical and legal review.
