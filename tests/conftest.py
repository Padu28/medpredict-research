import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def settings():
    from medpredict.common.config import get_settings

    return get_settings()


@pytest.fixture(scope="session")
def cohort(settings):
    from medpredict.data.cohort import generate_cohort

    cfg = {**settings.dataset["synthetic"], "n_patients": 2500}
    return generate_cohort(cfg)


@pytest.fixture(scope="session")
def splits(cohort, settings):
    from medpredict.preprocessing.split import temporal_patient_split

    e = settings.evaluation["split"]
    return temporal_patient_split(cohort, "patient_id", "discharge_date", e["train_end"], e["validation_end"])
