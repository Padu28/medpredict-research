"""Subgroup evaluation with sample sizes, CIs and small-group suppression."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from medpredict.evaluation.metrics import at_threshold, bootstrap_ci


def subgroup_report(df: pd.DataFrame, p: np.ndarray, target: str, groups: dict[str, str], thr: float,
                    min_size: int = 50, n_boot: int = 200) -> pd.DataFrame:
    rows = []
    d = df.assign(_p=np.asarray(p))
    for gname, col in groups.items():
        if col not in d:
            continue
        for val, g in d.groupby(col):
            n, n_pos = len(g), int(g[target].sum())
            row = {"grouping": gname, "group": str(val), "n": n, "n_positive": n_pos}
            if n < min_size or n_pos < 5 or n_pos == n:
                rows.append({**row, "suppressed": True})
                continue
            y, pp = g[target].to_numpy(), g._p.to_numpy()
            t = at_threshold(y, pp, thr)
            lo, hi = bootstrap_ci(y, pp, roc_auc_score, n_boot)
            rows.append({**row, "suppressed": False, "auroc": float(roc_auc_score(y, pp)), "auroc_lo": lo, "auroc_hi": hi,
                         "sensitivity": t["sensitivity"], "specificity": t["specificity"], "precision": t["precision"],
                         "flag_rate": t["flag_rate"], "mean_predicted": float(pp.mean()), "observed_rate": float(y.mean()),
                         "calibration_gap": float(pp.mean() - y.mean())})
    return pd.DataFrame(rows)


def disparity_summary(rep: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Largest between-group gaps per grouping (equal-opportunity gap = sensitivity range)."""
    out = {}
    for gname, g in rep[~rep.suppressed.astype(bool)].groupby("grouping"):
        out[gname] = {"sensitivity_gap": float(g.sensitivity.max() - g.sensitivity.min()),
                      "auroc_gap": float(g.auroc.max() - g.auroc.min()),
                      "max_abs_calibration_gap": float(g.calibration_gap.abs().max())}
    return out
