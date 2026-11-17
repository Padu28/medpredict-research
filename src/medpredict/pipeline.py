"""End-to-end evaluation study.

audit → leakage checks → temporal patient-level split → 3 model families × 3
calibration methods → discrimination + calibration + threshold metrics with CIs →
model selection on VALIDATION → subgroup analysis → shift tests (missingness,
covariate, site hold-out, temporal) → bootstrap uncertainty → explanations → report.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

from medpredict import DISCLAIMER, __version__
from medpredict.calibration.calibrate import calibrate, reliability
from medpredict.common.config import get_settings
from medpredict.common.logging import get_logger
from medpredict.common.utils import dump_json, to_jsonable
from medpredict.data.cohort import data_audit, load_cohort
from medpredict.evaluation.metrics import at_threshold, bootstrap_ci, core_metrics, threshold_for_sensitivity
from medpredict.evaluation.shift import run_shift_tests
from medpredict.explainability.explain import lr_coefficients, permutation
from medpredict.fairness.subgroups import disparity_summary, subgroup_report
from medpredict.models.train import BootstrapEnsemble, fit, make_model
from medpredict.preprocessing.split import check_no_leakage, check_no_patient_overlap, temporal_patient_split
from medpredict.storage.database import Database, utcnow

log = get_logger("medpredict.pipeline")
MODELS = ("logistic_regression", "random_forest", "hist_gradient_boosting")


def run_pipeline() -> dict:
    s = get_settings()
    ev = s.evaluation
    db = Database(s.db_path)
    db.reset()
    s.models_dir.mkdir(parents=True, exist_ok=True)

    df = load_cohort()
    audit = data_audit(df)
    offending = check_no_leakage(s.features, s.target_cfg["leakage_columns"], s.target)
    if offending:
        raise ValueError(f"Leakage: features include post-outcome columns {offending}")
    tr, va, te = temporal_patient_split(df, s.dataset["id_col"], s.dataset["time_col"],
                                        ev["split"]["train_end"], ev["split"]["validation_end"])
    assert check_no_patient_overlap(tr, va, te, id_col=s.dataset["id_col"])
    splits = {n: {"n": len(d), "patients": int(d.patient_id.nunique()), "prevalence": float(d[s.target].mean()),
                  "start": str(d[s.dataset["time_col"]].min().date()), "end": str(d[s.dataset["time_col"]].max().date())}
              for n, d in [("train", tr), ("validation", va), ("test", te)]}
    log.info("Split %s", {k: (v["n"], round(v["prevalence"], 3)) for k, v in splits.items()})

    F, y_va, y_te = s.features, va[s.target].to_numpy(), te[s.target].to_numpy()
    results, curves, fitted = [], [], {}
    for name in MODELS:
        base = fit(name, tr)
        for method in ev["calibration_methods"]:
            cal = calibrate(base, va[F], y_va, method)
            p_va, p_te = cal.predict_proba(va[F])[:, 1], cal.predict_proba(te[F])[:, 1]
            thr = threshold_for_sensitivity(y_va, p_va, ev["threshold"]["target_sensitivity"])
            m = core_metrics(y_te, p_te)
            m["auroc_ci"] = bootstrap_ci(y_te, p_te, roc_auc_score, ev["bootstrap"]["n"])
            m["brier_ci"] = bootstrap_ci(y_te, p_te, brier_score_loss, ev["bootstrap"]["n"])
            m["accuracy_at_0.5"] = float(accuracy_score(y_te, p_te >= 0.5))
            m["val_log_loss"] = core_metrics(y_va, p_va)["log_loss"]
            m["val_auroc"] = float(roc_auc_score(y_va, p_va))
            m["at_threshold"] = at_threshold(y_te, p_te, thr)
            results.append({"model": name, "calibration": method, "metrics_json": json.dumps(to_jsonable(m))})
            rc = reliability(y_te, p_te, ev["calibration_bins"])
            curves.append({"model": name, "calibration": method, "observed": json.dumps(rc["observed"]),
                           "predicted": json.dumps(rc["predicted"]),
                           "hist": json.dumps(np.histogram(p_te, bins=20, range=(0, 1))[0].tolist())})
            fitted[(name, method)] = (cal, thr, p_te)
            log.info("%-22s %-12s AUROC %.3f · Brier %.4f · ECE %.3f · slope %.2f", name, method, m["auroc"],
                     m["brier"], m["ece"], m["cal_slope"])
    db.write_df("model_results", pd.DataFrame(results))
    db.write_df("calibration_curves", pd.DataFrame(curves))

    # ---- model selection on VALIDATION log loss (proper scoring rule), never on test
    res_df = pd.DataFrame([{"model": r["model"], "calibration": r["calibration"], **json.loads(r["metrics_json"])} for r in results])
    best = res_df.sort_values("val_log_loss").iloc[0]
    primary = (best.model, best.calibration)
    model, thr, p_te = fitted[primary]
    log.info("Selected on validation: %s + %s (threshold %.3f)", *primary, thr)

    # ---- subgroups
    sub = subgroup_report(te, p_te, s.target, s.target_cfg["subgroups"], thr, ev["fairness"]["min_group_size"])
    db.write_df("subgroups", sub)
    disp = disparity_summary(sub)

    # ---- shift tests
    sh = run_shift_tests(model, te, F, s.target, ev["shift_tests"])
    hold = ev["shift_tests"]["site_holdout"]
    trv = pd.concat([tr, va])
    src_tr, src_va = trv[trv.site != hold], trv[trv.site != hold]
    site_model = calibrate(fit(primary[0], src_tr), src_va[F], src_va[s.target].to_numpy(), primary[1])
    for lbl, part in (("in-distribution sites", te[te.site != hold]), (f"held-out site {hold}", te[te.site == hold])):
        sh.loc[len(sh)] = {"test": "site", "level": lbl,
                           **core_metrics(part[s.target].to_numpy(), site_model.predict_proba(part[F])[:, 1])}
    sh.loc[len(sh)] = {"test": "temporal", "level": "validation period", **core_metrics(y_va, model.predict_proba(va[F])[:, 1])}
    sh.loc[len(sh)] = {"test": "temporal", "level": "test period (later)", **core_metrics(y_te, p_te)}
    db.write_df("shift_results", sh)

    # ---- uncertainty (bootstrap LR ensemble) & predictions
    ens = BootstrapEnsemble(make_model("logistic_regression"), ev["bootstrap"]["ensemble_models"], ev["random_state"]).fit(tr[F], tr[s.target])
    _, med, _ = ens.interval(te[F])
    lo, hi = ens.band_around(p_te, te[F])
    preds = te[["patient_id", s.dataset["time_col"], "site", "sex", "age_band", s.target]].copy()
    preds = preds.rename(columns={s.dataset["time_col"]: "discharge_date", s.target: "outcome"})
    preds["discharge_date"] = preds.discharge_date.astype(str)
    preds["risk"], preds["ens_lo"], preds["ens_median"], preds["ens_hi"] = p_te, lo, med, hi
    preds["flagged"] = (p_te >= thr).astype(int)
    db.write_df("predictions", preds)

    # ---- explanations
    lr = fit("logistic_regression", tr)
    coefs = lr_coefficients(lr).assign(kind="lr_coefficient")
    perm = permutation(model, te[F], te[s.target], ev["random_state"]).rename(columns={"importance": "coefficient"}).assign(kind="permutation")
    db.write_df("explanations", pd.concat([coefs[["feature", "coefficient", "kind"]], perm[["feature", "coefficient", "kind"]]]))

    joblib.dump({"model": model, "threshold": thr, "primary": primary, "lr": lr, "ensemble": ens,
                 "background": tr[F].sample(min(500, len(tr)), random_state=0), "features": F,
                 "numeric": s.numeric, "categorical": s.categorical}, s.models_dir / "medpredict_bundle.joblib")

    acc_only = res_df.sort_values("accuracy_at_0.5", ascending=False).iloc[0]
    summary = {
        "version": __version__, "created_at": utcnow(), "disclaimer": DISCLAIMER, "audit": audit, "splits": splits,
        "primary": {"model": primary[0], "calibration": primary[1], "threshold": thr},
        "accuracy_only_choice": {"model": acc_only.model, "calibration": acc_only.calibration,
                                 "accuracy": float(acc_only["accuracy_at_0.5"])},
        "results": to_jsonable(res_df.drop(columns=["at_threshold"]).to_dict(orient="records")),
        "primary_metrics": to_jsonable(res_df[(res_df.model == primary[0]) & (res_df.calibration == primary[1])].iloc[0].to_dict()),
        "disparities": disp, "target_definition": s.target_cfg["definition"].strip(),
        "uncertainty": {"mean_interval_width": float(np.mean(hi - lo))},
    }
    db.put_meta("summary", to_jsonable(summary))
    dump_json(to_jsonable(summary), s.reports_dir / "evaluation_summary.json")
    write_report(summary, res_df, sub, sh, s.reports_dir / "evaluation_report.md")
    try:
        from medpredict.evaluation.figures import make_figures
        make_figures(db, s.reports_dir / "figures")
    except Exception as exc:
        log.warning("Figures skipped: %s", exc)
    return summary


def _f(v, d=3):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


def write_report(summary: dict, res: pd.DataFrame, sub: pd.DataFrame, sh: pd.DataFrame, path) -> None:
    pm = summary["primary_metrics"]
    L = ["# MedPredict Research — Evaluation Report", "",
         f"> ⚕️ **{summary['disclaimer']}**", "",
         f"*Generated {summary['created_at']} · synthetic de-identified cohort · temporal + patient-level split*", "",
         f"**Target.** {summary['target_definition']}", "",
         f"**Selected on validation log loss:** `{summary['primary']['model']}` + `{summary['primary']['calibration']}` · "
         f"threshold {summary['primary']['threshold']:.3f} (target sensitivity 0.80 on validation)", "",
         "## Discrimination & calibration (test period)", "",
         "| Model | Calibration | AUROC (95% CI) | PR-AUC | Brier | ECE | Cal. slope | Accuracy@0.5 |", "|---|---|---|---|---|---|---|---|"]
    for r in res.to_dict(orient="records"):
        ci = r["auroc_ci"]
        L.append(f"| {r['model']} | {r['calibration']} | {r['auroc']:.3f} ({ci[0]:.3f}–{ci[1]:.3f}) | {r['pr_auc']:.3f} | "
                 f"{r['brier']:.4f} | {r['ece']:.3f} | {r['cal_slope']:.2f} | {r['accuracy_at_0.5']:.3f} |")
    t = pm["at_threshold"]
    acc = summary["accuracy_only_choice"]
    L += ["", f"**At the frozen threshold (test):** sensitivity {t['sensitivity']:.3f} · specificity {t['specificity']:.3f} · "
          f"PPV {t['precision']:.3f} · flag rate {t['flag_rate']:.1%}", "",
          f"**Accuracy-only view:** the highest accuracy@0.5 belongs to `{acc['model']}` + `{acc['calibration']}` "
          f"({acc['accuracy']:.3f}) — accuracy barely separates models while calibration metrics differ substantially.", "",
          "## Subgroups (test; groups with n < 50 suppressed)", "",
          "| Grouping | Group | n | Positives | AUROC (95% CI) | Sensitivity | Mean predicted | Observed |", "|---|---|---|---|---|---|---|---|"]
    for r in sub.itertuples():
        if r.suppressed:
            L.append(f"| {r.grouping} | {r.group} | {r.n} | {r.n_positive} | suppressed | | | |")
        else:
            L.append(f"| {r.grouping} | {r.group} | {r.n} | {r.n_positive} | {r.auroc:.3f} ({_f(r.auroc_lo)}–{_f(r.auroc_hi)}) | "
                     f"{r.sensitivity:.3f} | {r.mean_predicted:.3f} | {r.observed_rate:.3f} |")
    L += ["", "## Shift tests", "", "| Test | Condition | AUROC | Brier | ECE | Mean predicted | Observed |", "|---|---|---|---|---|---|---|"]
    for r in sh.itertuples():
        L.append(f"| {r.test} | {r.level} | {r.auroc:.3f} | {r.brier:.4f} | {r.ece:.3f} | {r.mean_predicted:.3f} | {r.observed_rate:.3f} |")
    L += ["", "## Limitations",
          "- Synthetic cohort; no clinical validity. Real use would require independent clinical validation and governance.",
          "- Subgroup CIs are wide; small groups are suppressed rather than over-interpreted.",
          "- Bootstrap intervals capture estimation uncertainty only — not label noise, confounding or shift.",
          "- Explanations are associations in the model, not causal effects."]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
