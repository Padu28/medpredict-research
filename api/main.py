"""MedPredict Research API — EDUCATIONAL, NON-CLINICAL.

Run:  uvicorn api.main:app --reload --port 8003     → docs at http://localhost:8003/docs

Every response carries the safety disclaimer. The API returns a calibrated
probability with an uncertainty band; it never returns a diagnosis or a
treatment recommendation.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from medpredict import DISCLAIMER, __version__  # noqa: E402
from medpredict.common.config import get_settings  # noqa: E402
from medpredict.common.utils import to_jsonable  # noqa: E402
from medpredict.explainability.explain import local_contributions  # noqa: E402
from medpredict.storage.database import Database, utcnow  # noqa: E402

app = FastAPI(title="MedPredict Research API (educational, non-clinical)", version=__version__, description=DISCLAIMER)
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
                   allow_methods=["*"], allow_headers=["*"])
settings = get_settings()
db = Database(settings.db_path)
_cache: dict[str, Any] = {}


def bundle() -> dict:
    if "b" not in _cache:
        p = settings.models_dir / "medpredict_bundle.joblib"
        if not p.exists():
            raise HTTPException(503, "Model bundle missing — run `python -m medpredict.cli pipeline`.")
        _cache["b"] = joblib.load(p)
    return _cache["b"]


class Profile(BaseModel):
    """A hypothetical, synthetic profile. Do not submit real patient data."""
    age: float = Field(68, ge=18, le=110)
    sex: Literal["F", "M"] = "F"
    bmi: float | None = Field(29.0, ge=10, le=80)
    systolic_bp: float | None = Field(135, ge=60, le=260)
    glucose: float | None = Field(120, ge=30, le=700)
    hba1c: float | None = Field(6.2, ge=3, le=20)
    creatinine: float | None = Field(1.0, ge=0.2, le=15)
    hemoglobin: float | None = Field(13.0, ge=4, le=22)
    length_of_stay: float = Field(5, ge=0, le=365)
    prior_admissions_12m: int = Field(1, ge=0, le=50)
    medication_count: int = Field(6, ge=0, le=60)
    admission_type: Literal["emergency", "elective", "urgent"] = "emergency"
    smoking_status: Literal["never", "former", "current"] = "never"
    discharge_disposition: Literal["home", "home_with_services", "skilled_nursing"] = "home"


class RiskOut(BaseModel):
    probability: float
    uncertainty_band: tuple[float, float]
    research_threshold: float
    above_research_threshold: bool
    model: str
    top_associations: list[dict[str, Any]]
    disclaimer: str = DISCLAIMER


@app.get("/health", tags=["system"])
def health() -> dict[str, Any]:
    return {"status": "ok", "version": __version__, "time": utcnow(), "disclaimer": DISCLAIMER,
            "bundle_ready": (settings.models_dir / "medpredict_bundle.joblib").exists()}


@app.post("/risk", response_model=RiskOut, tags=["research"])
def risk(profile: Profile) -> RiskOut:
    b = bundle()
    row = pd.DataFrame([profile.model_dump()])[b["features"]]
    p = float(b["model"].predict_proba(row)[:, 1][0])
    lo, hi = (float(x[0]) for x in b["ensemble"].band_around([p], row))
    lc = local_contributions(b["lr"], row, b["background"]).head(5)
    return RiskOut(probability=round(p, 4), uncertainty_band=(round(lo, 4), round(hi, 4)),
                   research_threshold=round(b["threshold"], 4), above_research_threshold=p >= b["threshold"],
                   model=f"{b['primary'][0]}+{b['primary'][1]}",
                   top_associations=[{"feature": f, "log_odds": round(float(c), 4)} for f, c in zip(lc.feature, lc.contribution)])


@app.get("/metrics", tags=["research"])
def metrics() -> dict[str, Any]:
    s = db.get_meta("summary")
    if not s:
        raise HTTPException(404, "Run the pipeline first.")
    return {"disclaimer": DISCLAIMER, "primary": s["primary"], "primary_metrics": s["primary_metrics"],
            "results": s["results"], "splits": s["splits"]}


@app.get("/subgroups", tags=["research"])
def subgroups() -> list[dict[str, Any]]:
    return to_jsonable(db.query_df("SELECT * FROM subgroups").to_dict(orient="records"))


@app.get("/shift-tests", tags=["research"])
def shift_tests() -> list[dict[str, Any]]:
    return to_jsonable(db.query_df("SELECT * FROM shift_results").to_dict(orient="records"))
