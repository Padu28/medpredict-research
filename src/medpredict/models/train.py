"""Risk models: interpretable baseline (logistic regression) vs tree ensembles, plus a
bootstrap ensemble for per-patient uncertainty intervals."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from medpredict.common.config import get_settings
from medpredict.features.pipeline import build_preprocessor

ESTIMATORS = {"logistic_regression": LogisticRegression, "random_forest": RandomForestClassifier,
              "hist_gradient_boosting": HistGradientBoostingClassifier}


def make_model(name: str, numeric: list[str] | None = None, categorical: list[str] | None = None) -> Pipeline:
    s = get_settings()
    params: dict[str, Any] = dict(s.evaluation["models"][name])
    if "random_state" in ESTIMATORS[name]().get_params():
        params["random_state"] = s.evaluation["random_state"]
    num = s.numeric if numeric is None else numeric
    cat = s.categorical if categorical is None else categorical
    return Pipeline([("prep", build_preprocessor(num, cat)), ("clf", ESTIMATORS[name](**params))])


def fit(name: str, df: pd.DataFrame, features: list[str] | None = None) -> Pipeline:
    s = get_settings()
    feats = features or s.features
    num = [c for c in s.numeric if c in feats]
    cat = [c for c in s.categorical if c in feats]
    return make_model(name, num, cat).fit(df[num + cat], df[s.target])


class BootstrapEnsemble:
    """B logistic regressions on bootstrap resamples → 5th/95th percentile of predicted risk.

    This captures *model (estimation) uncertainty* only; it does not cover label noise,
    unmeasured confounding or dataset shift.
    """

    def __init__(self, base: Pipeline, n_models: int = 30, seed: int = 0) -> None:
        self.base, self.n, self.seed = base, n_models, seed
        self.models: list[Pipeline] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> BootstrapEnsemble:
        rng = np.random.default_rng(self.seed)
        for _ in range(self.n):
            idx = rng.integers(0, len(X), len(X))
            self.models.append(clone(self.base).fit(X.iloc[idx], y.iloc[idx]))
        return self

    def interval(self, X: pd.DataFrame, lo: float = 5, hi: float = 95) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        P = np.stack([m.predict_proba(X)[:, 1] for m in self.models])
        return np.percentile(P, lo, 0), np.median(P, 0), np.percentile(P, hi, 0)

    def band_around(self, p: np.ndarray, X: pd.DataFrame, lo: float = 5, hi: float = 95) -> tuple[np.ndarray, np.ndarray]:
        """Transfer the ensemble's log-odds spread onto a (calibrated) probability ``p``.

        The bootstrap models are uncalibrated, so their raw interval may not contain the calibrated
        estimate; shifting the spread in log-odds space keeps the band centred on what is reported.
        """
        def logit(q):
            q = np.clip(q, 1e-6, 1 - 1e-6)
            return np.log(q / (1 - q))

        def sigm(z):
            return 1 / (1 + np.exp(-z))

        a, m, b = self.interval(X, lo, hi)
        z = logit(np.asarray(p))
        return sigm(z + logit(a) - logit(m)), sigm(z + logit(b) - logit(m))
