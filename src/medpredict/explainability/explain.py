"""Model explanations — associations within the model, NOT causal effects."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def lr_coefficients(model) -> pd.DataFrame:
    """Standardised log-odds coefficients of the logistic-regression pipeline."""
    names = model.named_steps["prep"].get_feature_names_out()
    coef = model.named_steps["clf"].coef_[0]
    df = pd.DataFrame({"feature": names, "coefficient": coef, "odds_ratio_per_sd": np.exp(coef)})
    return df.reindex(df.coefficient.abs().sort_values(ascending=False).index).reset_index(drop=True)


def permutation(model, X: pd.DataFrame, y: pd.Series, seed: int = 0) -> pd.DataFrame:
    r = permutation_importance(model, X, y, scoring="roc_auc", n_repeats=8, random_state=seed, n_jobs=-1)
    return (pd.DataFrame({"feature": X.columns, "importance": r.importances_mean, "std": r.importances_std})
            .sort_values("importance", ascending=False).reset_index(drop=True))


def local_contributions(model, row: pd.DataFrame, background: pd.DataFrame) -> pd.DataFrame:
    """Linear log-odds contributions relative to the average background patient (LR only)."""
    prep, clf = model.named_steps["prep"], model.named_steps["clf"]
    x = prep.transform(row)[0]
    base = prep.transform(background).mean(0)
    names = prep.get_feature_names_out()
    c = clf.coef_[0] * (x - base)
    df = pd.DataFrame({"feature": names, "contribution": c})
    return df.reindex(df.contribution.abs().sort_values(ascending=False).index).reset_index(drop=True)
