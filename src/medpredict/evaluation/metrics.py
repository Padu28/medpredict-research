"""Discrimination, calibration and threshold metrics with bootstrap confidence intervals."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from medpredict.calibration.calibrate import calibration_slope_intercept, expected_calibration_error


def threshold_for_sensitivity(y: np.ndarray, p: np.ndarray, target: float = 0.8) -> float:
    """Largest threshold that still achieves the target sensitivity (on validation data)."""
    pos = np.sort(np.asarray(p)[np.asarray(y) == 1])[::-1]
    k = int(np.ceil(target * len(pos))) - 1
    return float(pos[max(k, 0)])


def at_threshold(y: np.ndarray, p: np.ndarray, thr: float) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    pred = np.asarray(p) >= thr
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    sens = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    return {"threshold": float(thr), "sensitivity": sens, "specificity": tn / max(tn + fp, 1), "precision": prec,
            "f1": 2 * prec * sens / max(prec + sens, 1e-9), "flag_rate": float(pred.mean()),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def core_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    slope, intercept = calibration_slope_intercept(y, p)
    return {"auroc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p)),
            "ece": expected_calibration_error(y, p), "cal_slope": slope, "cal_intercept": intercept,
            "mean_predicted": float(p.mean()), "observed_rate": float(y.mean())}


def bootstrap_ci(y: np.ndarray, p: np.ndarray, fn: Callable[[np.ndarray, np.ndarray], float], n: int = 300,
                 seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        vals.append(fn(y[i], p[i]))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)
