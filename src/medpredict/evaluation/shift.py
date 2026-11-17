"""Controlled dataset-shift tests: missingness, covariate shift and site hold-out."""
from __future__ import annotations

import numpy as np
import pandas as pd

from medpredict.evaluation.metrics import core_metrics


def inject_missingness(df: pd.DataFrame, cols: list[str], rate: float, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    for c in cols:
        out.loc[rng.random(len(out)) < rate, c] = np.nan
    return out


def shift_covariates(df: pd.DataFrame, deltas: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    for c, d in deltas.items():
        out[c] = out[c] + d
    return out


def run_shift_tests(model, test: pd.DataFrame, features: list[str], target: str, cfg: dict) -> pd.DataFrame:
    rows = []
    y = test[target].to_numpy()
    for r in cfg["missingness_rates"]:
        X = inject_missingness(test, cfg["missingness_columns"], r)[features]
        rows.append({"test": "missingness", "level": f"{int(r * 100)}%", **core_metrics(y, model.predict_proba(X)[:, 1])})
    for name, deltas in cfg["covariate_shifts"].items():
        X = shift_covariates(test, deltas)[features]
        rows.append({"test": "covariate", "level": name, **core_metrics(y, model.predict_proba(X)[:, 1])})
    return pd.DataFrame(rows)
