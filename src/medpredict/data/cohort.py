"""Synthetic de-identified readmission cohort + loaders.

The generator creates patients with 1–3 discharges between 2022 and 2025 at three
sites. Risk depends non-linearly on age, glycaemic control, kidney function,
anaemia, prior admissions, length of stay and discharge disposition. Site C has
a different case-mix and slightly different coding (a deliberate site shift),
and the outcome prevalence drifts upward over time (temporal shift).

Two post-outcome columns (`followup_calls_30d`, `readmission_cost`) are included
ON PURPOSE to demonstrate leakage detection; they must never be features.

SYNTHETIC — contains no real patient information.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from medpredict.common.config import PROJECT_ROOT, get_settings


def generate_cohort(cfg: dict) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.get("seed", 42))
    n_pat = cfg["n_patients"]
    sites = list(cfg["sites"])
    site_p = np.array(list(cfg["sites"].values()), dtype=float)
    site_p /= site_p.sum()
    t0, t1 = pd.Timestamp(cfg["start_date"]), pd.Timestamp(cfg["end_date"])
    span = (t1 - t0).days

    pat_site = rng.choice(sites, n_pat, p=site_p)
    pat_sex = rng.choice(["F", "M"], n_pat, p=[0.52, 0.48])
    base_age = np.clip(rng.normal(62, 15, n_pat) + np.where(pat_site == "C", 6, 0), 18, 95)
    smoking = rng.choice(["never", "former", "current"], n_pat, p=[0.5, 0.32, 0.18])
    diabetic = rng.random(n_pat) < 0.28 + 0.1 * (pat_site == "C")
    n_enc = rng.integers(1, cfg["max_encounters_per_patient"] + 1, n_pat)

    rows = []
    for p in range(n_pat):
        first = rng.integers(0, span - 60)
        days = np.sort(np.r_[first, first + np.cumsum(rng.integers(40, 400, n_enc[p] - 1))])
        for k, d in enumerate(days[days < span]):
            rows.append((p, k, d))
    enc = pd.DataFrame(rows, columns=["pid", "k", "day"])
    n = len(enc)
    pid = enc.pid.to_numpy()
    site = pat_site[pid]
    age = base_age[pid] + enc.day.to_numpy() / 365
    diab = diabetic[pid]
    hba1c = np.where(diab, rng.normal(7.9, 1.3, n), rng.normal(5.5, 0.4, n))
    glucose = 18 * hba1c - 10 + rng.normal(0, 18, n)
    creat = np.clip(rng.lognormal(np.log(0.95) + 0.006 * (age - 60), 0.32, n), 0.4, 8)
    hemo = np.clip(rng.normal(np.where(pat_sex[pid] == "F", 12.6, 14.0), 1.6, n), 6, 19)
    bmi = np.clip(rng.normal(28 + 2 * diab, 5.5, n), 15, 60)
    sbp = np.clip(rng.normal(132 + 0.25 * (age - 60), 17, n), 80, 220)
    prior = rng.poisson(0.4 + 0.6 * enc.k.to_numpy() + 0.3 * diab, n)
    adm = rng.choice(["emergency", "elective", "urgent"], n, p=[0.55, 0.25, 0.20])
    los = np.clip(rng.lognormal(1.3 + 0.3 * (adm == "emergency"), 0.55, n), 1, 60).round()
    meds = np.clip(rng.poisson(4 + 2.5 * diab + 0.04 * (age - 50), n), 0, 30)
    dispo = rng.choice(["home", "home_with_services", "skilled_nursing"], n,
                       p=[0.62, 0.24, 0.14])
    t = enc.day.to_numpy() / span

    logit = (-2.6 + 0.022 * (age - 62) + 0.0006 * (age - 62) ** 2 + 0.25 * np.clip(hba1c - 6.5, 0, None)
             + 0.75 * np.log(creat / 0.95) + 0.17 * np.clip(12 - hemo, 0, None) + 0.32 * np.sqrt(prior)
             + 0.03 * np.clip(los - 4, 0, 25) + 0.05 * meds + 0.45 * (dispo == "skilled_nursing")
             + 0.20 * (dispo == "home_with_services") + 0.30 * (adm == "emergency")
             + 0.20 * (smoking[pid] == "current") + 0.45 * t + 0.25 * (site == "C"))
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    # Site C codes HbA1c less often (missing-not-at-random style) — a realistic data-collection shift
    hba1c = np.where((site == "C") & (rng.random(n) < 0.35), np.nan, hba1c)
    hba1c = np.where(rng.random(n) < 0.05, np.nan, hba1c)
    creat = np.where(rng.random(n) < 0.03, np.nan, creat)

    df = pd.DataFrame({
        "patient_id": [f"P{p:05d}" for p in pid], "encounter": enc.k.to_numpy(),
        "discharge_date": t0 + pd.to_timedelta(enc.day.to_numpy(), unit="D"), "site": site,
        "age": age.round(1), "sex": pat_sex[pid], "bmi": bmi.round(1), "systolic_bp": sbp.round(),
        "glucose": glucose.round(), "hba1c": np.round(hba1c, 1), "creatinine": np.round(creat, 2),
        "hemoglobin": hemo.round(1), "length_of_stay": los, "prior_admissions_12m": prior,
        "medication_count": meds, "admission_type": adm, "smoking_status": smoking[pid],
        "discharge_disposition": dispo, "readmit_30d": y,
        # ---- post-outcome leakage columns (never features) ----
        "followup_calls_30d": rng.poisson(0.5 + 2.5 * y),
        "readmission_cost": np.where(y == 1, rng.lognormal(9, 0.5, n), 0).round(0),
        "is_synthetic": True,
    })
    df["age_band"] = pd.cut(df.age, [0, 45, 65, 80, 200], labels=["18-44", "45-64", "65-79", "80+"], right=False).astype(str)
    return df.sort_values("discharge_date", kind="stable").reset_index(drop=True)


def load_breast_cancer_frame() -> pd.DataFrame:
    from sklearn.datasets import load_breast_cancer

    d = load_breast_cancer(as_frame=True)
    df = d.frame.copy()
    df["readmit_30d"] = (df.pop("target") == 0).astype(int)   # malignant = positive class
    df["patient_id"] = [f"B{i:04d}" for i in range(len(df))]
    return df


def load_cohort() -> pd.DataFrame:
    s = get_settings()
    src = s.dataset["source"]
    if src == "breast_cancer":
        return load_breast_cancer_frame()
    if src == "csv":
        df = pd.read_csv(PROJECT_ROOT / s.dataset["csv_path"], parse_dates=[s.dataset["time_col"]])
        return df.sort_values(s.dataset["time_col"]).reset_index(drop=True)
    return generate_cohort(s.dataset["synthetic"])


def data_audit(df: pd.DataFrame) -> dict:
    s = get_settings()
    feats = [c for c in s.features if c in df]
    num = [c for c in s.numeric if c in df]
    return {
        "n_rows": len(df), "n_patients": int(df[s.dataset["id_col"]].nunique()),
        "prevalence": float(df[s.target].mean()),
        "missingness": {c: float(df[c].isna().mean()) for c in feats if df[c].isna().any()},
        "duplicate_rows": int(df.duplicated().sum()),
        "ranges": {c: [float(df[c].min()), float(df[c].quantile(0.5)), float(df[c].max())] for c in num},
        "leakage_columns_present": [c for c in s.target_cfg["leakage_columns"] if c in df],
        "date_range": [str(df[s.dataset["time_col"]].min()), str(df[s.dataset["time_col"]].max())]
        if s.dataset["time_col"] in df else None,
        "prevalence_by_site": df.groupby("site")[s.target].mean().round(4).to_dict() if "site" in df else {},
    }
