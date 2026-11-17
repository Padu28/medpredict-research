import numpy as np
import pytest

from medpredict.calibration.calibrate import calibrate, calibration_slope_intercept, expected_calibration_error
from medpredict.evaluation.metrics import at_threshold, core_metrics, threshold_for_sensitivity
from medpredict.evaluation.shift import inject_missingness, run_shift_tests
from medpredict.explainability.explain import local_contributions, lr_coefficients
from medpredict.fairness.subgroups import disparity_summary, subgroup_report
from medpredict.models.train import BootstrapEnsemble, fit, make_model


@pytest.fixture(scope="module")
def lr(splits):
    return fit("logistic_regression", splits[0])


def test_ece_perfect_vs_bad():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 20000)
    y = (rng.random(20000) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.02
    assert expected_calibration_error(y, np.clip(p * 0.4, 0, 1)) > 0.2
    slope, _ = calibration_slope_intercept(y, p)
    assert slope == pytest.approx(1.0, abs=0.1)


def test_threshold_hits_target_sensitivity():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000)
    p = np.clip(0.3 * y + rng.uniform(0, 0.7, 2000), 0, 1)
    thr = threshold_for_sensitivity(y, p, 0.8)
    assert at_threshold(y, p, thr)["sensitivity"] >= 0.8


def test_model_beats_chance_and_calibration_helps_brier(lr, splits, settings):
    _, va, te = splits
    F = settings.features
    raw = core_metrics(te[settings.target], lr.predict_proba(te[F])[:, 1])
    cal = calibrate(lr, va[F], va[settings.target], "sigmoid")
    m = core_metrics(te[settings.target], cal.predict_proba(te[F])[:, 1])
    assert raw["auroc"] > 0.6
    assert m["auroc"] == pytest.approx(raw["auroc"], abs=1e-9)    # sigmoid calibration is monotone
    assert m["brier"] <= raw["brier"] + 0.005


def test_subgroups_suppress_small_groups(lr, splits, settings):
    _, _, te = splits
    p = lr.predict_proba(te[settings.features])[:, 1]
    rep = subgroup_report(te, p, settings.target, settings.target_cfg["subgroups"], 0.25, min_size=10_000, n_boot=20)
    assert rep.suppressed.all()
    rep2 = subgroup_report(te, p, settings.target, {"sex": "sex"}, 0.25, min_size=50, n_boot=20)
    assert (~rep2.suppressed.astype(bool)).all() and "sex" in disparity_summary(rep2)


def test_shift_tests_and_missingness(lr, splits, settings):
    _, _, te = splits
    m = inject_missingness(te, ["hba1c"], 0.5, seed=0)
    assert m.hba1c.isna().mean() > 0.45
    out = run_shift_tests(lr, te, settings.features, settings.target, settings.evaluation["shift_tests"])
    assert set(out.test) == {"missingness", "covariate"}


def test_uncertainty_interval_contains_median(splits, settings):
    tr, _, te = splits
    ens = BootstrapEnsemble(make_model("logistic_regression"), n_models=8).fit(tr[settings.features], tr[settings.target])
    lo, med, hi = ens.interval(te[settings.features].head(50))
    assert (lo <= med).all() and (med <= hi).all()
    p = np.full(50, 0.3)
    blo, bhi = ens.band_around(p, te[settings.features].head(50))
    assert (blo <= p + 1e-9).all() and (p <= bhi + 1e-9).all()


def test_explanations(lr, splits, settings):
    tr, _, te = splits
    c = lr_coefficients(lr)
    assert {"feature", "coefficient", "odds_ratio_per_sd"} <= set(c.columns)
    lc = local_contributions(lr, te[settings.features].head(1), tr[settings.features].head(200))
    assert len(lc) == len(c)
