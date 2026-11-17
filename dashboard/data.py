"""Cached access to MedPredict results (no hard-coded metrics)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from medpredict.common.config import get_settings  # noqa: E402
from medpredict.storage.database import Database  # noqa: E402

settings = get_settings()


def db() -> Database:
    return Database(settings.db_path)


def ensure_pipeline() -> None:
    if db().get_meta("summary") is None or not (settings.models_dir / "medpredict_bundle.joblib").exists():
        with st.status("First launch — running the evaluation study (≈20 s)…", expanded=True) as s:
            from medpredict.cli import generate_sample
            from medpredict.pipeline import run_pipeline

            st.write("Generating synthetic cohort · 3 models × 3 calibration methods · subgroups · shift tests…")
            generate_sample()
            run_pipeline()
            s.update(label="Study complete", state="complete")
        st.cache_data.clear()
        st.cache_resource.clear()


@st.cache_data(ttl=300)
def summary() -> dict:
    return db().get_meta("summary", {})


@st.cache_data(ttl=300)
def results() -> pd.DataFrame:
    r = db().query_df("SELECT * FROM model_results")
    m = r.metrics_json.apply(json.loads)
    return pd.concat([r[["model", "calibration"]], pd.json_normalize(m)], axis=1)


@st.cache_data(ttl=300)
def curves() -> pd.DataFrame:
    c = db().query_df("SELECT * FROM calibration_curves")
    for col in ("observed", "predicted", "hist"):
        c[col] = c[col].apply(json.loads)
    return c


@st.cache_data(ttl=300)
def subgroups() -> pd.DataFrame:
    return db().query_df("SELECT * FROM subgroups")


@st.cache_data(ttl=300)
def shifts() -> pd.DataFrame:
    return db().query_df("SELECT * FROM shift_results")


@st.cache_data(ttl=300)
def predictions() -> pd.DataFrame:
    return db().query_df("SELECT * FROM predictions")


@st.cache_data(ttl=300)
def explanations() -> pd.DataFrame:
    return db().query_df("SELECT * FROM explanations")


@st.cache_resource
def bundle() -> dict:
    return joblib.load(settings.models_dir / "medpredict_bundle.joblib")
