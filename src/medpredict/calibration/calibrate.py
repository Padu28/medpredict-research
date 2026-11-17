"""Post-hoc calibration on VALIDATION data only, plus calibration diagnostics."""
from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve


class Identity:
    def __init__(self, model) -> None:
        self.model = model

    def predict_proba(self, X):
        return self.model.predict_proba(X)


def calibrate(model, X_val, y_val, method: str):
    if method == "uncalibrated":
        return Identity(model)
    try:
        from sklearn.frozen import FrozenEstimator

        cal = CalibratedClassifierCV(FrozenEstimator(model), method=method)
    except ImportError:  # pragma: no cover
        cal = CalibratedClassifierCV(model, method=method, cv="prefit")
    return cal.fit(X_val, y_val)


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Equal-width ECE: weighted mean |observed − predicted| across probability bins."""
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def calibration_slope_intercept(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Logistic recalibration: logit(P(y)) = a + b·logit(p). Ideal: a = 0, b = 1."""
    from sklearn.linear_model import LogisticRegression

    p = np.clip(p, 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(lp, y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> dict[str, list[float]]:
    frac, mean_pred = calibration_curve(y, p, n_bins=bins, strategy="quantile")
    return {"observed": frac.tolist(), "predicted": mean_pred.tolist()}
