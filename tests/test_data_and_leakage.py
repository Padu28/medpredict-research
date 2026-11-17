from medpredict.data.cohort import data_audit, load_breast_cancer_frame
from medpredict.preprocessing.split import check_no_leakage, check_no_patient_overlap


def test_leakage_columns_never_features(settings):
    assert check_no_leakage(settings.features, settings.target_cfg["leakage_columns"], settings.target) == []
    assert check_no_leakage(["age", "followup_calls_30d"], settings.target_cfg["leakage_columns"], settings.target) == ["followup_calls_30d"]


def test_leakage_columns_are_actually_leaky(cohort):
    # sanity: the excluded column is strongly associated with the outcome — that is why it must be excluded
    assert cohort.groupby("readmit_30d").followup_calls_30d.mean().diff().iloc[-1] > 1


def test_patient_level_split(splits):
    tr, va, te = splits
    assert check_no_patient_overlap(tr, va, te)
    assert tr.discharge_date.min() < te.discharge_date.min()


def test_test_patients_first_seen_later(splits, settings):
    _, _, te = splits
    first = te.groupby("patient_id").discharge_date.min()
    assert (first > settings.evaluation["split"]["validation_end"]).all()


def test_audit(cohort):
    a = data_audit(cohort)
    assert 0.1 < a["prevalence"] < 0.5
    assert "hba1c" in a["missingness"]
    assert a["prevalence_by_site"]["C"] > a["prevalence_by_site"]["A"]


def test_breast_cancer_loader():
    df = load_breast_cancer_frame()
    assert len(df) == 569 and set(df.readmit_30d.unique()) == {0, 1}
